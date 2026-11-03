"""Offline integration contracts for the supplemental visual judge experiment."""
import copy
import json
from types import SimpleNamespace
from urllib import error

import pytest

from mvpg_run.claim_feedback import diagnostics as d
from mvpg_run.claim_feedback.data import validate_record, write_jsonl


def record(rid, image, label="?", score=None):
    return validate_record({"id": rid, "image": image, "question": "Describe the image.",
                            "response": "A cat.", "claims": [
                                {"id": "cat", "text": "cat", "kind": "object",
                                 "source_spans": [[2, 5]], "label": label,
                                 "support_score": score, "weight": 0}]})


def prepared(tmp_path):
    return d.prepare_cases([record("a", "a.png"), record("b", "b.png")], image_root=tmp_path)


def write_images(tmp_path):
    # Image bytes are never decoded in these transport-contract tests.
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"\x89PNG\r\n\x1a\nfixture")


def judge_args(tmp_path, **kwargs):
    values = dict(input=str(tmp_path / "cases.jsonl"), output=str(tmp_path / "judged.jsonl"),
                  model="vision-model", image_root=str(tmp_path), api_key_env="TEST_JUDGE_KEY",
                  base_url_env="TEST_JUDGE_URL", timeout=1, max_retries=0, resume=False)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_prepare_groups_aliases_and_ignores_old_weak_labels(tmp_path):
    records = [record("a", "images/../a.png", "+", 0.8),
               record("b", str(tmp_path / "a.png"), "-", -0.9),
               record("c", "b.png", "?", None)]
    audit = {}
    cases = d.prepare_cases(records, image_root=tmp_path, audit=audit)
    assert len(cases) == 3
    assert cases[0]["split"] == cases[1]["split"]
    assert cases[0]["image_group"] == cases[1]["image_group"]
    assert cases[0]["split"] != cases[2]["split"]
    assert all(c["judge_label"] is None and c["scores"] == {} for c in cases)
    assert audit["eligible_candidates"] == 3
    assert d.prepare_cases(records, image_root=tmp_path) == cases


def test_prepare_audits_unaligned_and_non_object_claims(tmp_path):
    rows = [record("a", "a.png"), record("b", "b.png")]
    rows[0]["claims"] += [dict(rows[0]["claims"][0], id="no_span", source_spans=[]),
                          dict(rows[0]["claims"][0], id="ambiguous", ambiguous=True),
                          dict(rows[0]["claims"][0], id="relation", kind="relation")]
    audit = {}
    cases = d.prepare_cases(rows, image_root=tmp_path, audit=audit)
    assert len(cases) == 2
    assert audit["total_candidates"] == 5
    assert sum(audit["excluded_candidates"].values()) == 3
    with pytest.raises(ValueError, match="two image groups"):
        d.prepare_cases(rows[:1], image_root=tmp_path)


def test_image_selection_is_independent_of_parser_success(tmp_path):
    records = [record(str(i), f"{i}.png") for i in range(6)]
    audit_before, audit_after = {}, {}
    d.prepare_cases(records, image_root=tmp_path, audit=audit_before)
    records[0]["claims"] = []
    d.prepare_cases(records, image_root=tmp_path, audit=audit_after)
    assert audit_before["selected_image_splits"] == audit_after["selected_image_splits"]
    assert audit_after["selected_images"] == 6
    assert len(audit_after["empty_candidate_images"]) == 1


def test_payload_contains_actual_image_and_blinds_weak_feedback(tmp_path):
    write_images(tmp_path)
    case = prepared(tmp_path)[0]
    case.update(scores={"global_raw": 0.1}, judge_label="unsupported",
                metadata={"old_feedback_label": "-"})
    payload = d.judge_payload(case, model="vision-model", image_root=tmp_path)
    content = payload["messages"][1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    text = json.loads(content[0]["text"])
    assert text["object_mention"] == "cat"
    assert not {"scores", "judge_label", "metadata", "old_feedback_label"} & set(text)
    assert "scope_mismatch" in payload["messages"][0]["content"]
    assert d.judge_payload(case, model="deepseek-flash", image_root=tmp_path,
                           thinking="disabled")["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("label", ["supported", "unsupported", "uncertain"])
def test_strict_judgment_keeps_three_distinct_labels(label):
    value = d.parse_judgment(json.dumps({"label": label, "reason": "reason", "evidence": "visible region"}))
    assert value["label"] == label


@pytest.mark.parametrize("content", [
    '{"label":"false","reason":"r","evidence":"e"}',
    '```json\n{"label":"uncertain","reason":"r","evidence":"e"}\n```',
    '{"label":"supported","reason":"r"}',
    '{"label":"supported","reason":"r","evidence":"e","score":0.9}',
    '{"label":[],"reason":"r","evidence":"e"}',
])
def test_invalid_judgment_cannot_become_a_negative(content):
    with pytest.raises(ValueError):
        d.parse_judgment(content)


class Response:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None
    def read(self):
        return self.body.encode("utf-8")


def response_body(label="uncertain"):
    return json.dumps({"id": "response-id", "model": "actual-model", "usage": {"total_tokens": 12},
                       "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(
                           {"label": label, "reason": "Insufficient resolution", "evidence": "Occluded area"})}}]})


def test_request_errors_are_not_judge_abstention_or_false_claims():
    def denied(*args, **kwargs):
        raise error.HTTPError("https://example.test", 400, "denied", {}, None)
    result = d.ask_judge({}, "https://example.test", "secret", max_retries=2, opener=denied,
                         sleep=lambda _: None)
    assert result["judge_label"] is None
    assert result["judge_error"] == "HTTP 400"
    assert result["judge_request_attempts"] == 1
    result = d.ask_judge({}, "https://example.test", "secret", max_retries=0,
                         opener=lambda *a, **k: Response(response_body()))
    assert result["judge_label"] == "uncertain"
    assert result["judge_raw_response"] == response_body()
    assert result["judge_response_model"] == "actual-model"


def test_parse_failures_retry_but_remain_errors():
    seen = []
    def malformed(*args, **kwargs):
        seen.append(1)
        return Response('{"choices": []}')
    result = d.ask_judge({}, "https://example.test", "secret", max_retries=1,
                         opener=malformed, sleep=lambda _: None)
    assert len(seen) == 2 and result["judge_label"] is None
    assert result["judge_error"] == "ValueError"


def test_judge_journal_resumes_successes_and_retries_errors(tmp_path, monkeypatch):
    write_images(tmp_path)
    cases = prepared(tmp_path)
    d._write_rows(tmp_path / "cases.jsonl", cases)
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    monkeypatch.setenv("TEST_JUDGE_URL", "https://example.test/v1")
    calls = []
    def first(payload, *args, **kwargs):
        calls.append(payload)
        return {"judge_label": "uncertain" if len(calls) == 1 else None,
                "judge_error": None if len(calls) == 1 else "HTTP 503"}
    monkeypatch.setattr(d, "ask_judge", first)
    args = judge_args(tmp_path)
    assert d.judge_cases(args)["status"] == "incomplete"
    assert len(calls) == 2
    args.resume = True
    monkeypatch.setattr(d, "ask_judge", lambda *a, **k: {"judge_label": "supported", "judge_error": None})
    assert d.judge_cases(args)["status"] == "complete"
    rows = d.load_cases(args.output)
    assert [r["judge_attempt"] for r in rows] == [1, 2]
    assert all(row["judge_finished_utc"] >= row["judge_started_utc"] for row in rows)
    assert len(d._read_rows(str(args.output) + ".progress.jsonl")) == 3
    assert "secret" not in (tmp_path / "judged.jsonl.manifest.json").read_text()
    monkeypatch.setattr(d, "ask_judge", lambda *a, **k: pytest.fail("Completed cases must not request again"))
    assert d.judge_cases(args)["status"] == "complete"


def test_fatal_judge_error_stops_after_persisting_one_failure(tmp_path, monkeypatch):
    write_images(tmp_path)
    d._write_rows(tmp_path / "cases.jsonl", prepared(tmp_path))
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    monkeypatch.setenv("TEST_JUDGE_URL", "https://example.test/v1")
    calls = []
    def denied(*args, **kwargs):
        calls.append(1)
        return {"judge_label": None, "judge_error": "HTTP 401", "judge_fatal_error": True}
    monkeypatch.setattr(d, "ask_judge", denied)
    result = d.judge_cases(judge_args(tmp_path))
    assert len(calls) == 1
    assert result["status"] == "incomplete" and result["errors"] == 2
    rows = d.load_cases(tmp_path / "judged.jsonl")
    assert rows[0]["judge_error"] == "HTTP 401"
    assert rows[1]["judge_error"] == "not_attempted_after_fatal_error"
    assert len(d._read_rows(tmp_path / "judged.jsonl.progress.jsonl")) == 1


def test_score_keeps_uncertain_labels_and_uses_locked_template(tmp_path, monkeypatch):
    from mvpg_run.claim_feedback import backends
    cases = prepared(tmp_path)
    for case in cases:
        case["judge_label"] = "uncertain"
    d._write_rows(tmp_path / "cases.jsonl", cases)
    root = tmp_path / "fgclip"
    root.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.model"):
        (root / name).write_bytes(b"fixture")
    calls = []
    class Scorer:
        def __init__(self, *args):
            pass
        def __call__(self, image, texts):
            calls.append(texts)
            return [0.2] * len(texts)
    monkeypatch.setattr(backends, "FGClipVerifier", Scorer)
    args = SimpleNamespace(input=str(tmp_path / "cases.jsonl"), output=str(tmp_path / "scored.jsonl"),
                           fgclip_root=str(root), device="cpu", image_root=str(tmp_path),
                           score_mode="global", text_form="presence")
    result = d.score_cases(args)
    assert result["score_key"] == "global_presence"
    assert all(texts == ["A photo of cat."] for texts in calls)
    scored = d.load_cases(args.output)
    assert all(c["judge_label"] == "uncertain" and c["scores"]["global_presence"] == 0.2 for c in scored)


@pytest.mark.parametrize("change", ["model", "image", "input", "endpoint", "prompt"])
def test_resume_rejects_provenance_changes_before_network(tmp_path, monkeypatch, change):
    write_images(tmp_path)
    d._write_rows(tmp_path / "cases.jsonl", prepared(tmp_path))
    monkeypatch.setenv("TEST_JUDGE_KEY", "secret")
    monkeypatch.setenv("TEST_JUDGE_URL", "https://example.test/v1")
    monkeypatch.setattr(d, "ask_judge", lambda *a, **k: {"judge_label": "uncertain", "judge_error": None})
    args = judge_args(tmp_path)
    d.judge_cases(args)
    args.resume = True
    if change == "model":
        args.model = "different-model"
    elif change == "image":
        (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nchanged")
    elif change == "input":
        with (tmp_path / "cases.jsonl").open("a") as handle:
            handle.write("\n")
    elif change == "endpoint":
        monkeypatch.setenv("TEST_JUDGE_URL", "https://different.test/v1")
    else:
        monkeypatch.setattr(d, "JUDGE_PROMPT", "changed prompt")
    monkeypatch.setattr(d, "ask_judge", lambda *a, **k: pytest.fail("Resume mismatch must precede request"))
    with pytest.raises(ValueError, match="resume manifest differs"):
        d.judge_cases(args)


def test_case_validation_rejects_group_leakage_and_duplicates(tmp_path):
    cases = prepared(tmp_path)
    duplicate = copy.deepcopy(cases[0])
    with pytest.raises(ValueError, match="Duplicate"):
        d.validate_cases(cases + [duplicate])
    cases[1]["image_group"] = cases[0]["image_group"]
    with pytest.raises(ValueError, match="both dev and test"):
        d.validate_cases(cases)


def test_cli_prepare_uses_real_schema_without_models(tmp_path):
    source, output = tmp_path / "claims.jsonl", tmp_path / "cases.jsonl"
    write_jsonl(source, [record("a", "a.png", "+", 0.1), record("b", "b.png", "-", -0.1)])
    assert d.main(["prepare", "--input", str(source), "--output", str(output),
                   "--image-root", str(tmp_path)]) == 0
    assert len(d.load_cases(output)) == 2
    with pytest.raises(FileExistsError):
        d.main(["prepare", "--input", str(source), "--output", str(output)])


def test_endpoint_rejects_credential_or_query_leakage():
    assert d._endpoint("https://api.deepseek.com") == "https://api.deepseek.com/chat/completions"
    for bad in ("https://user:password@example.test", "https://example.test?key=secret", "file:///tmp/a"):
        with pytest.raises(ValueError):
            d._endpoint(bad)
