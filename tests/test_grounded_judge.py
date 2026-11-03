"""Transport, blinding and recovery contracts for image-batched silver labels."""
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from urllib import error

import pytest

from mvpg_run.claim_feedback import grounded_judge as g
from mvpg_run.claim_feedback import diagnostics as d


def cases(tmp_path, images=2, claims=2):
    rows = []
    for i in range(images):
        image = tmp_path / f"image-{i}.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nfixture" + str(i).encode())
        for j in range(claims):
            rows.append({"case_id": f"{i}:{j}", "record_id": str(i), "claim_id": str(j),
                         "image": str(image), "image_group": str(image),
                         "split": "dev" if i % 2 else "test", "question": "Describe.",
                         "response": "A cat and a dog.", "claim_text": "cat" if j % 2 == 0 else "dog",
                         "source_spans": [[2, 5]] if j % 2 == 0 else [[12, 15]],
                         "kind": "object", "judge_label": None, "scores": {}})
    return rows


def args(tmp_path, **overrides):
    values = dict(input=str(tmp_path / "cases.jsonl"), output=str(tmp_path / "judged.jsonl"),
                  model="deepseek-flash", image_root=str(tmp_path), api_key_env="TEST_BATCH_KEY",
                  base_url_env="TEST_BATCH_URL", timeout=1, max_retries=0, workers=2,
                  max_output_tokens=4096, image_detail="original", log_every=1, resume=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def write_input(tmp_path, **kwargs):
    rows = cases(tmp_path, **kwargs)
    d._write_rows(tmp_path / "cases.jsonl", rows)
    return rows


@pytest.fixture(autouse=True)
def judge_environment(monkeypatch):
    monkeypatch.setenv("TEST_BATCH_KEY", "secret-not-in-artifacts")
    monkeypatch.setenv("TEST_BATCH_URL", "https://judge.example.test/v1")
    monkeypatch.setenv("JUDGE_THINKING", "disabled")


def judgments(ids, label="supported"):
    return [{"case_id": cid, "label": label, "reason": "Concise reason",
             "evidence": "Visible object"} for cid in ids]


def result(ids, label="supported"):
    rows = judgments(ids, label)
    content = json.dumps({"judgments": rows})
    return {"judgments": rows, "judge_raw_text": content,
            "judge_raw_response": json.dumps({"provider": "actual", "content": content}),
            "judge_error": None, "judge_response_model": "provider-version",
            "judge_request_id": "request-id", "judge_usage": {"total_tokens": 123},
            "judge_request_attempts": 1, "judge_fatal_error": False}


class Response:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self):
        return self.body.encode()


def response(ids, *, finish="stop", content=None):
    return Response(json.dumps({"id": "id", "model": "actual-model",
        "usage": {"total_tokens": 30}, "choices": [{"finish_reason": finish,
        "message": {"content": content or json.dumps({"judgments": judgments(ids)})}}]}))


def test_payload_has_one_actual_image_and_no_weak_labels_or_scores(tmp_path):
    rows = cases(tmp_path, images=1)
    for row in rows:
        row.update(scores={"global_raw": -99}, judge_label="unsupported",
                   source_provenance={"threshold": -0.01, "old_label": "-"})
    rows[1]["record_id"] = "another-caption"
    payload = g.judge_payload(rows, model="deepseek-flash", thinking="disabled")
    blocks = payload["messages"][1]["content"]
    assert len(blocks) == 2 and blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert blocks[1]["image_url"]["detail"] == "original"
    visible = json.loads(blocks[0]["text"])
    assert len(visible["answers"]) == 2 and len(visible["claims"]) == 2
    assert visible["claims"][0]["source_excerpt"] == ["cat"]
    assert not any(x in blocks[0]["text"] for x in ["global_raw", "threshold", "old_label", "unsupported"])
    assert payload["thinking"] == {"type": "disabled"} and payload["max_tokens"] == 4096
    other = copy.deepcopy(rows)
    other[1]["image"] = str(tmp_path / "another.png")
    with pytest.raises(ValueError, match="one actual image"):
        g.judge_payload(other, model="vision")


def test_strict_parser_requires_exact_ids_and_keeps_uncertain():
    ids = ["a", "b"]
    rows = judgments(ids, "uncertain")
    assert g.parse_judgments(json.dumps({"judgments": rows[::-1]}), ids) == rows
    invalid = [{"judgments": rows[:1]}, {"judgments": [rows[0], rows[0]]},
               {"judgments": [dict(rows[0], case_id="other"), rows[1]]},
               {"judgments": [dict(rows[0], label="false"), rows[1]]},
               {"judgments": [dict(rows[0], score=.1), rows[1]]},
               {"judgments": [dict(rows[0], evidence=""), rows[1]]},
               {"judgments": rows, "extra": True}]
    for value in invalid:
        with pytest.raises(ValueError):
            g.parse_judgments(json.dumps(value), ids)


def test_grouping_separates_answers_and_actual_views_from_split_group(tmp_path):
    rows = cases(tmp_path, images=2, claims=1)
    rows[1]["image_group"] = rows[0]["image_group"]
    rows[1]["split"] = rows[0]["split"]
    repeated_image = dict(rows[0], case_id="different-answer", record_id="different-answer",
                          response="Another answer.", source_spans=[[0, 7]])
    groups = g.group_cases(rows + [repeated_image])
    assert len(groups) == 3
    assert len({x["image_key"] for x in groups}) == 3


def test_transport_retries_and_never_converts_failures_to_labels():
    seen = []
    def transient(req, **kwargs):
        seen.append(1)
        if len(seen) == 1:
            raise error.HTTPError(req.full_url, 429, "busy", {}, None)
        return response(["a"])
    answer = g.ask_image({}, "https://example.test", "key", ["a"], max_retries=1,
                         opener=transient, sleep=lambda _: None)
    assert answer["judge_request_attempts"] == 2 and answer["judgments"][0]["label"] == "supported"
    for body in [response(["a"], finish="length"), response(["a"], content='{"judgments":[]}')]:
        answer = g.ask_image({}, "https://example.test", "key", ["a"], max_retries=0,
                             opener=lambda *a, **k: body)
        assert answer["judgments"] is None and answer["judge_error"] == "ValueError"
    def denied(*a, **k):
        raise error.HTTPError("https://example.test", 401, "secret-error-body", {}, None)
    answer = g.ask_image({}, "https://example.test", "key", ["a"], max_retries=3,
                         opener=denied, sleep=lambda _: pytest.fail("fatal request must not retry"))
    assert answer["judge_fatal_error"] and answer["judge_request_attempts"] == 1
    assert "secret-error-body" not in json.dumps(answer)


def test_one_request_per_image_ordered_claim_output_and_raw_saved_once(tmp_path, monkeypatch):
    original = write_input(tmp_path, images=3, claims=3)
    calls = []
    def ask(payload, endpoint, key, ids, **kwargs):
        calls.append(ids)
        assert key == "secret-not-in-artifacts"
        return result(ids, "uncertain")
    monkeypatch.setattr(g, "ask_image", ask)
    summary = g.judge_cases(args(tmp_path))
    assert summary["status"] == "complete" and summary["images_judged"] == 3 and summary["cases_judged"] == 9
    assert len(calls) == 3
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert [x["case_id"] for x in rows] == [x["case_id"] for x in original]
    assert all(x["judge_label"] == "uncertain" and x["judge_response_model"] == "provider-version" for x in rows)
    assert all("judge_raw_response" not in x and "judge_usage" not in x for x in rows)
    journal = d._read_rows(tmp_path / "judged.jsonl.images.progress.jsonl")
    assert len(journal) == 3 and sum(x["judge_usage"]["total_tokens"] for x in journal) == 369
    assert all(x["judge_raw_response"] for x in journal)
    assert "secret-not-in-artifacts" not in "".join(p.read_text() for p in tmp_path.glob('judged*') if p.is_file())
    monkeypatch.setattr(g, "ask_image", lambda *a, **k: pytest.fail("Already judged images must not request again"))
    assert g.judge_cases(args(tmp_path, resume=True, workers=1))["status"] == "complete"


def test_failed_images_retry_on_resume_without_repeating_successes(tmp_path, monkeypatch):
    write_input(tmp_path)
    calls = []
    def first(payload, endpoint, key, ids, **kwargs):
        calls.append(ids)
        return result(ids) if ids[0].startswith("0:") else {
            "judgments": None, "judge_error": "HTTP 503", "judge_fatal_error": False,
            "judge_request_attempts": 1}
    monkeypatch.setattr(g, "ask_image", first)
    assert g.judge_cases(args(tmp_path))["status"] == "incomplete"
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert rows[2]["judge_label"] is None and rows[2]["judge_error"] == "HTTP 503"
    calls.clear()
    def recovered(payload, endpoint, key, ids, **kwargs):
        calls.append(ids)
        return result(ids, "unsupported")
    monkeypatch.setattr(g, "ask_image", recovered)
    assert g.judge_cases(args(tmp_path, resume=True))["status"] == "complete"
    assert calls == [["1:0", "1:1"]]
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert rows[2]["judge_attempt"] == 2 and rows[0]["judge_attempt"] == 1
    assert len(d._read_rows(tmp_path / "judged.jsonl.images.progress.jsonl")) == 3


def test_fatal_error_stops_new_requests_and_keeps_all_case_rows(tmp_path, monkeypatch):
    write_input(tmp_path, images=3)
    calls = []
    def denied(payload, endpoint, key, ids, **kwargs):
        calls.append(ids)
        return {"judgments": None, "judge_error": "HTTP 400", "judge_fatal_error": True,
                "judge_request_attempts": 1}
    monkeypatch.setattr(g, "ask_image", denied)
    summary = g.judge_cases(args(tmp_path, workers=1))
    assert summary["status"] == "incomplete" and len(calls) == 1
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert len(rows) == 6 and all(x["judge_label"] is None for x in rows)
    assert rows[-1]["judge_error"] == "not_attempted_after_fatal_error"


def test_fatal_error_drains_inflight_success_without_submitting_new_images(tmp_path, monkeypatch):
    write_input(tmp_path, images=3)
    barrier = threading.Barrier(2)
    calls = []
    def ask(payload, endpoint, key, ids, **kwargs):
        calls.append(ids)
        barrier.wait(timeout=3)
        if ids[0].startswith("0:"):
            return {"judgments": None, "judge_error": "HTTP 401", "judge_fatal_error": True,
                    "judge_request_attempts": 1}
        time.sleep(.03)
        return result(ids)
    monkeypatch.setattr(g, "ask_image", ask)
    summary = g.judge_cases(args(tmp_path, workers=2))
    assert summary["status"] == "incomplete" and summary["cases_judged"] == 2
    assert len(calls) == 2
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert rows[2]["judge_label"] == "supported"
    assert rows[-1]["judge_error"] == "not_attempted_after_fatal_error"


@pytest.mark.parametrize("change", ["model", "tokens", "image", "input"])
def test_resume_rejects_changed_protocol_or_inputs(tmp_path, monkeypatch, change):
    write_input(tmp_path)
    monkeypatch.setattr(g, "ask_image", lambda p, e, k, ids, **kw: result(ids))
    g.judge_cases(args(tmp_path))
    updated = args(tmp_path, resume=True)
    if change == "model": updated.model = "another-model"
    if change == "tokens": updated.max_output_tokens = 8192
    if change == "image": (tmp_path / "image-0.png").write_bytes(b"changed")
    if change == "input":
        path = tmp_path / "cases.jsonl"
        path.write_text(path.read_text().replace("Describe.", "Describe!"))
    with pytest.raises(ValueError, match="manifest differs"):
        g.judge_cases(updated)


def test_torn_journal_tail_is_preserved_and_completed_calls_not_repeated(tmp_path, monkeypatch):
    write_input(tmp_path)
    monkeypatch.setattr(g, "ask_image", lambda p, e, k, ids, **kw: result(ids))
    g.judge_cases(args(tmp_path))
    progress = tmp_path / "judged.jsonl.images.progress.jsonl"
    with progress.open("ab") as handle:
        handle.write(b'{"interrupted":')
    monkeypatch.setattr(g, "ask_image", lambda *a, **kw: pytest.fail("Successful images must be reused"))
    assert g.judge_cases(args(tmp_path, resume=True))["status"] == "complete"
    backups = list(tmp_path.glob("*.truncated-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == b'{"interrupted":'
    assert len(d._read_rows(progress)) == 2


def test_output_lock_excludes_another_writer(tmp_path):
    path = tmp_path / "lock"
    with g.output_lock(path):
        with pytest.raises(ValueError, match="Another visual judge"):
            with g.output_lock(path):
                pass


def test_request_aliases_are_bijective_clones_with_no_semantic_edits(tmp_path):
    original = cases(tmp_path, images=1, claims=3)
    for index, row in enumerate(original):
        row["case_id"] = ("abcdef0123456789" * 4)[:-1] + str(index)
    snapshot = copy.deepcopy(original)
    encoded, mapping = g.encode_request_cases(original, "request_alias")
    assert mapping == {f"c{i:04d}": row["case_id"] for i, row in enumerate(original, 1)}
    assert [row["case_id"] for row in encoded] == ["c0001", "c0002", "c0003"]
    for source, request_row in zip(original, encoded):
        assert {k: v for k, v in source.items() if k != "case_id"} == {
            k: v for k, v in request_row.items() if k != "case_id"}
    encoded[0]["scores"]["should_not_mutate_source"] = 1
    assert original == snapshot
    canonical, identity = g.encode_request_cases(original)
    assert canonical == original and canonical is not original
    assert identity == {row["case_id"]: row["case_id"] for row in original}
    with pytest.raises(ValueError, match="unique"):
        g.encode_request_cases([original[0], original[0]], "request_alias")
    with pytest.raises(ValueError, match="encoding"):
        g.encode_request_cases(original, "unsupported_encoding")


def test_default_manifest_and_request_fingerprints_remain_legacy_exact(tmp_path, monkeypatch):
    original = write_input(tmp_path, images=1)
    monkeypatch.setattr(g, "ask_image", lambda p, e, k, ids, **kw: result(ids))
    g.judge_cases(args(tmp_path))  # Old callers have no id_encoding attribute.
    manifest = json.loads((tmp_path / "judged.jsonl.manifest.json").read_text())
    assert manifest == {
        "schema": "image_batch_claim_judge_v1", "grouping": "record_id+resolved_image_path",
        "input_sha256": d.file_sha256(tmp_path / "cases.jsonl"), "model": "deepseek-flash",
        "endpoint": d._endpoint("https://judge.example.test/v1"), "prompt_sha256": g._digest(g.BATCH_PROMPT),
        "image_sha256": {original[0]["image"]: d.file_sha256(original[0]["image"])},
        "temperature": 0, "max_tokens": 4096, "thinking": "disabled", "image_detail": "original"}
    journal = d._read_rows(tmp_path / "judged.jsonl.images.progress.jsonl")
    assert journal[0]["request_fingerprint"] == g._digest({
        "image_sha256": d.file_sha256(original[0]["image"]), "visible_text": g._visible_text(original)})
    assert "judge_id_encoding" not in journal[0] and "judge_case_id_map" not in journal[0]
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert all("judge_request_case_id" not in row and row["judge_protocol"] == "image_batch_claim_judge_v1" for row in rows)


def test_alias_requests_restore_canonical_ids_and_preserve_actual_raw_responses(tmp_path, monkeypatch):
    original = write_input(tmp_path, images=2, claims=2)
    requests = []
    def ask(payload, endpoint, key, ids, **kwargs):
        requests.append((payload, list(ids)))
        assert ids == ["c0001", "c0002"]
        return result(ids, "uncertain")
    monkeypatch.setattr(g, "ask_image", ask)
    assert g.judge_cases(args(tmp_path, id_encoding="request_alias"))["status"] == "complete"
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert [row["case_id"] for row in rows] == [row["case_id"] for row in original]
    assert [row["judge_request_case_id"] for row in rows] == ["c0001", "c0002", "c0001", "c0002"]
    assert all(row["judge_label"] == "uncertain" and row["judge_id_encoding"] == "request_alias"
               and row["judge_protocol"] == "image_batch_claim_judge_v1_request_alias" for row in rows)
    journal = d._read_rows(tmp_path / "judged.jsonl.images.progress.jsonl")
    for entry in journal:
        assert set(entry["judge_case_id_map"]) == {"c0001", "c0002"}
        assert set(entry["judge_case_id_map"].values()) == set(entry["case_ids"])
        assert [j["case_id"] for j in entry["judgments"]] == entry["case_ids"]
        assert [j["case_id"] for j in json.loads(entry["judge_raw_text"])["judgments"]] == ["c0001", "c0002"]
        assert entry["judge_raw_response"] == json.dumps({"provider": "actual", "content": entry["judge_raw_text"]})
    for payload, _ in requests:
        visible = json.loads(payload["messages"][1]["content"][0]["text"])
        image_cases = [row for row in original if row["record_id"] == visible["answers"][0]["record_id"]]
        encoded, _ = g.encode_request_cases(image_cases, "request_alias")
        assert payload == g.judge_payload(encoded, model="deepseek-flash", thinking="disabled")
    assert json.loads((tmp_path / "judged.jsonl.manifest.json").read_text())["case_id_encoding"] == "request_alias"


@pytest.mark.parametrize("bad_ids", [["c0001", "c0003"], ["c0001", "c0001"], ["0:0", "0:1"], ["c0001"]])
def test_bad_returned_aliases_are_never_fuzzily_matched_or_labeled(tmp_path, monkeypatch, bad_ids):
    write_input(tmp_path, images=1)
    real_ask = g.ask_image
    def ask(payload, endpoint, key, ids, **kwargs):
        return real_ask(payload, endpoint, key, ids, opener=lambda *a, **k: response(bad_ids), **kwargs)
    monkeypatch.setattr(g, "ask_image", ask)
    summary = g.judge_cases(args(tmp_path, id_encoding="request_alias"))
    assert summary["status"] == "incomplete" and summary["cases_judged"] == 0
    assert all(row["judge_label"] is None and row["judge_error"] == "ValueError"
               for row in d.load_cases(tmp_path / "judged.jsonl"))


def test_alias_resume_validates_encoding_mapping_and_reuses_successes(tmp_path, monkeypatch):
    write_input(tmp_path, images=1)
    monkeypatch.setattr(g, "ask_image", lambda p, e, k, ids, **kw: result(ids))
    g.judge_cases(args(tmp_path, id_encoding="request_alias"))
    monkeypatch.setattr(g, "ask_image", lambda *a, **k: pytest.fail("Completed alias request must be reused"))
    assert g.judge_cases(args(tmp_path, resume=True, id_encoding="request_alias"))["status"] == "complete"
    with pytest.raises(ValueError, match="manifest differs"):
        g.judge_cases(args(tmp_path, resume=True))
    path = tmp_path / "judged.jsonl.images.progress.jsonl"
    entries = d._read_rows(path)
    entries[0]["judge_case_id_map"]["c0001"] = "another-canonical-id"
    path.write_text("".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping differs"):
        g.judge_cases(args(tmp_path, resume=True, id_encoding="request_alias"))


def test_composite_repaired_master_cannot_be_overwritten_by_legacy_resume(tmp_path, monkeypatch):
    write_input(tmp_path)
    monkeypatch.setattr(g, "ask_image", lambda p, e, k, ids, **kw: result(ids))
    g.judge_cases(args(tmp_path))
    master = tmp_path / "judged.jsonl"
    before = master.read_bytes()
    Path(str(master) + ".repair_manifest.json").write_text('{"composite":true}', encoding="utf-8")
    monkeypatch.setattr(g, "ask_image", lambda *a, **k: pytest.fail("Read-only composite must not request"))
    with pytest.raises(ValueError, match="Composite-repaired"):
        g.judge_cases(args(tmp_path, resume=True))
    assert master.read_bytes() == before
