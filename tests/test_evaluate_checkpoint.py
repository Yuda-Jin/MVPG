"""The friendly entry must retain raw evidence and reject mixed experiments."""
import hashlib
import json
from pathlib import Path

import pytest

from run import evaluate_checkpoint as ec


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in values), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    monkeypatch.setattr(ec, "REPO", repo)
    model, checkpoint = tmp_path / "llava7b", tmp_path / "train/checkpoint-005000"
    model.mkdir()
    checkpoint.mkdir(parents=True)
    dump(checkpoint / "adapter_config.json", {"r": 16})
    (checkpoint / "adapter_model.bin").write_bytes(b"adapter")
    dump(checkpoint / "trainer_state.json", {"counts": {"updated": 100},
         "settings": {"max_updates": 5000}, "signature": {"model_name": str(model)}})
    for name in ["coco/train2017", "coco/val2014", "AMBER/image", "coco/annotations"]:
        (data / name).mkdir(parents=True)
    source, suffix = tmp_path / "original", "verified_fixture"
    paths = ec.sources(data)
    mm = [{"question": str(i), "image_src": str(i), "gt_answer": "dog"} for i in range(96)]
    dump(paths["mmhal"], mm)
    dump(source / f"mmhal_bench/answer/answer-file-{suffix}.jsonl", [dict(x, model_answer="dog") for x in mm])
    qa = [{"question_id": i} for i in range(90)]
    jsonl(paths["qa"], qa)
    jsonl(source / f"llava_bench/answer/answer-file-{suffix}.jsonl", [dict(x, text="yes") for x in qa])
    jsonl(source / f"llava_bench/review/review-file-{suffix}.jsonl", [dict(x, tuple=[8, 7]) for x in qa])
    for path in [repo / "eval_llava_rlhf_coco/table/caps_boxes_coco2014_val_80.jsonl",
                 repo / "eval_llava_rlhf_coco/llava/qa90_gpt4_answer.jsonl", repo / "eval_llava_rlhf_coco/table/rule.json"]:
        dump(path, {})
    for c in ["random", "popular", "adversarial"]:
        rows = [{"question_id": i} for i in range(3000)]
        jsonl(paths["pope_" + c], rows)
        jsonl(source / f"pop_bench/answer/answer-file-{suffix}_{c}.jsonl", [dict(x, text="yes") for x in rows])
    amber = [{"id": i} for i in range(15220)]
    dump(paths["amber"], amber)
    jsonl(source / f"AMBER_bench/answer/answer-file-{suffix}-all.jsonl", [dict(x, response="yes") for x in amber])
    objects = [{"image_id": i, "question": "Describe"} for i in range(300)]
    jsonl(paths["object"], objects)
    jsonl(source / f"objhal_bench/answer/answer-file-{suffix}.jsonl",
          [{"image_id": x["image_id"], "prompt": x["question"], "text": "dog"} for x in objects])
    for mode in ["local", "gpt"]:
        dump(source / f"objhal_bench/review-{mode}/review-file-{suffix}.jsonl",
             {"sentences": [{"image_id": x["image_id"], "caption": "dog"} for x in objects], "overall_metrics": {"sentence_num": 300}})
    review = source / f"mmhal_bench/review/review-file-{suffix}.jsonl"
    dump(review, ["Rating: 3"] + ["Rating: 4"] * 95)
    dump(Path(str(review) + ".meta.json"), {"judge_model": "deepseek-flash",
         "schema": "mmhal_ordered_reviews_v1", "judge_request_options": {"extra_body": {"thinking": {"type": "disabled"}}},
         "response_sha256": ec.sha(source / f"mmhal_bench/answer/answer-file-{suffix}.jsonl")})
    info = ec.checkpoint_info(checkpoint)
    settings = {"GPT_MODEL": "deepseek-flash", "OPENAI_ENDPOINT": "https://api.deepseek.com", "JUDGE_THINKING": "disabled"}
    identity = ec.make_identity(info, settings, data, 16)
    dump(source / f"llava_bench/review/review-file-{suffix}.jsonl.meta.json",
         {"inputs_sha256": ec.qa_fingerprint(source / f"llava_bench/answer/answer-file-{suffix}.jsonl"),
          "judge_model": "deepseek-flash", "max_tokens": 1024, "schema": "llava_bench_reviews_v1",
          "judge_request_options": ec.judge_options(identity)})
    logs = source / "logs" / suffix
    logs.mkdir(parents=True)
    dump(logs / "manifest.json", {"model": str(model), "adapter": str(checkpoint), "adapter_sha256": info["adapter_sha256"],
         "judge_model": "deepseek-flash", "judge_endpoint": "https://api.deepseek.com", "judge_thinking": "disabled",
         "batch_size": "16", "sources_sha256": identity["sources_sha256"]})
    texts = {"mmhal_score": "Average score: 3.99\nHallucination rate: 0.00\n",
             "llava_score": "all 80.0\n", "llava_review": "review complete\n",
             "amber_score": "CHAIR: 8.4\nCover: 52.6\nHal: 38.5\nCog: 4.5\n" +
             "".join(h + "\nAccuracy: 80\nPrecision: 81\nRecall: 79\nF1: 80\n" for h in
                     ["Discriminative Task:", "Existence:", "Attribute:", "Relation:"])}
    for c in ["random", "popular", "adversarial"]:
        texts[f"pope_{c}_score"] = "Accuracy: 0.85\nPrecision: 0.90\nRecall: 0.79\nF1 score: 0.84\nYes ratio: 0.44\n"
    for mode in ["local", "gpt"]:
        texts[f"object_score_{mode}"] = "CHAIRs: 53.33\nCHAIRi: 27.39\nCHAIRs_refine: 55.36\n"
    for name in ec.STAGES:
        (logs / (name + ".log")).write_text(texts.get(name, "generated\n"))
    (logs / "stages.tsv").write_text("stage\tstatus\texit_code\tmessage\tlog\n" +
         "".join(f"{name}\tcompleted\t0\tok\t{logs / (name + '.log')}\n" for name in ec.STAGES))
    dump(source / f"summary_{suffix}.json", {"model_suffix": suffix})
    (source / ".ipynb_checkpoints").mkdir()
    (source / ".ipynb_checkpoints/junk").write_text("cache")
    (source / ".env").write_text("OPENAI_API_KEY=secret-do-not-copy")
    return {"source": source, "output": tmp_path / "clean", "checkpoint": checkpoint, "logs": logs,
            "data": data, "settings": settings, "identity": identity, "suffix": suffix}


def import_args(e):
    return ["--checkpoint", str(e["checkpoint"]), "--import-existing", str(e["source"]),
            "--output-dir", str(e["output"])]


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_import_keeps_old_evidence_and_publishes_only_paper_metrics(experiment):
    e = experiment
    before = inventory(e["source"])
    assert ec.main(import_args(e)) == 0
    assert inventory(e["source"]) == before
    assert {p.name for p in e["output"].iterdir()} == {"results.md", "results.json", "details"}
    result = ec.read(e["output"] / "results.json")
    assert result["status"] == "complete" and result["step"] == 100
    assert result["completed_stages"] == 16 and sum(result["counts"].values()) == 24706
    assert set(result["metrics"]) == {k for k, _, _ in ec.PAPER_COLUMNS}
    assert result["metrics"]["mmhal_avg_score"] == 383 / 96
    assert result["object_scoring"] == "gpt"
    assert not (e["output"] / "details/.env").exists()
    assert not (e["output"] / "details/.ipynb_checkpoints").exists()
    assert ec.main(import_args(e)) == 0  # Idempotent, no engine calls.


def test_incomplete_stage_cannot_publish_complete(experiment):
    e = experiment
    status = e["logs"] / "stages.tsv"
    status.write_text(status.read_text().replace("object_score_local\tcompleted", "object_score_local\tfailed"))
    assert ec.main(import_args(e)) == 1
    result = ec.read(e["output"] / "results.json")
    assert result["status"] == "incomplete"
    assert "object_hal_chairs" not in result["metrics"]


def test_wrong_answer_order_cannot_publish_score(experiment):
    e = experiment
    path = e["source"] / f"pop_bench/answer/answer-file-{e['suffix']}_adversarial.jsonl"
    rows = path.read_text().splitlines()
    path.write_text("\n".join(reversed(rows)) + "\n")
    assert ec.main(import_args(e)) == 1
    assert "pope_adv_accuracy" not in ec.read(e["output"] / "results.json")["metrics"]


def test_refresh_rejects_tampered_evaluation_manifest(experiment):
    e = experiment
    assert ec.main(import_args(e)) == 0
    manifest = e["output"] / f"details/logs/{e['suffix']}/manifest.json"
    value = ec.read(manifest); value["judge_model"] = "other-judge"; dump(manifest, value)
    args = ["--checkpoint", str(e["checkpoint"]), "--output-dir", str(e["output"]), "--refresh-only"]
    assert ec.main(args) == 1
    result = ec.read(e["output"] / "results.json")
    assert result["status"] == "incomplete" and result["metrics"] == {}


def test_imported_partial_run_can_resume_without_repeating_name_or_data(experiment, monkeypatch):
    e = experiment
    status = e["logs"] / "stages.tsv"
    status.write_text(status.read_text().replace("object_score_local\tcompleted", "object_score_local\tfailed"))
    assert ec.main(import_args(e)) == 1
    monkeypatch.setattr(ec, "env_settings", lambda _: dict(e["settings"], key_present=True))
    calls = []
    monkeypatch.setattr(ec, "run_engine", lambda *args: calls.append(args) or 9)
    assert ec.main(["--checkpoint", str(e["checkpoint"]), "--output-dir", str(e["output"])]) == 9
    assert calls[0][0]["name"] == e["suffix"]
    assert calls[0][0]["identity"]["data_root"] == str(e["data"])


@pytest.mark.parametrize("target", ["inside", "foreign"])
def test_import_refuses_recursive_or_foreign_output(experiment, target):
    e = experiment
    e["output"] = e["source"] / "nested" if target == "inside" else e["output"]
    if target == "foreign":
        e["output"].mkdir(); (e["output"] / "user.txt").write_text("keep")
    assert ec.main(import_args(e)) == 2
    if target == "foreign":
        assert (e["output"] / "user.txt").read_text() == "keep"


def test_engine_exit_preserved_and_protocol_drift_rejected(experiment, monkeypatch):
    e = experiment
    settings = dict(e["settings"], key_present=True, OPENAI_API_KEY="secret-never-persist")
    monkeypatch.setattr(ec, "env_settings", lambda _: settings)
    calls = []
    monkeypatch.setattr(ec, "run_engine", lambda *args: calls.append(args) or 7)
    args = ["--checkpoint", str(e["checkpoint"]), "--data-root", str(e["data"]), "--output-dir", str(e["output"])]
    assert ec.main(args) == 7
    assert len(calls) == 1
    assert b"secret-never-persist" not in b"".join(p.read_bytes() for p in e["output"].rglob("*") if p.is_file())
    assert ec.main(args + ["--batch-size", "32"]) == 2
    settings["GPT_MODEL"] = "different-judge"
    assert ec.main(args) == 2
    assert len(calls) == 1


def test_legacy_intermediate_does_not_inherit_parent_final_count(tmp_path):
    checkpoint = tmp_path / "train/checkpoint-000025"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.bin").write_bytes(b"weights")
    dump(checkpoint / "adapter_config.json", {})
    dump(checkpoint.parent / "manifest.json", {"counts": {"updated": 100}, "backend": {"model_name": "/model"}})
    with pytest.raises(ValueError, match="Legacy intermediate"):
        ec.checkpoint_info(checkpoint)


def test_per_run_lock_rejects_concurrent_writer(tmp_path):
    with ec.run_lock(tmp_path):
        with pytest.raises(ValueError, match="Another evaluation"):
            with ec.run_lock(tmp_path):
                pass
