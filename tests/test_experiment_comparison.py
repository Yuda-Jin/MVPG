"""Result registries must preserve experiment identity, units and source data."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from run.compare_experiments import (REQUIRED_SCORE_STAGES, display_metric,
    local_row, main, render_markdown, training_row, update_registry, upsert, validate_row)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def entry(run_id="ours-100", **kwargs):
    result = {"run_id": run_id, "source_type": "local_run", "method": "claim_feedback/modulated",
        "model": "llava-v1.5-7b", "steps": 100, "judge": "deepseek-flash", "status": "complete",
        "metrics": {"pope_adv_accuracy": .8506666667, "mmhal_hallucination_rate": .57,
                    "amber_chair": 7.7, "object_hal_chairs": 47.33, "object_hal_chairi": 25.83},
        "source": {"summary_sha256": "test"}, "protocol": {"object_hal_scoring": "gpt"}}
    result.update(kwargs)
    return result


def summary(tmp_path, *, complete=True):
    return dump(tmp_path / "summary_ours.json", {"model_suffix": "ours", "all_benchmarks_scored": complete,
        "mmhal_avg_score": 2.43, "mmhal_hallucination_rate": .57, "pope_adv_accuracy": .8506666667,
        "object_hal_gpt_chairs": 47.33, "object_hal_chairs": 47.33,
        "object_hal_scoring": "gpt", "amber_yes_no_parsing": "strict",
        "stages": {stage: {"status": "completed"} for stage in REQUIRED_SCORE_STAGES}})


def test_units_missing_and_zero_are_not_confused():
    row = entry()
    assert display_metric(row, "pope_adv_accuracy") == "85.07"
    assert display_metric(row, "mmhal_hallucination_rate") == "57.00"
    assert display_metric(row, "amber_chair") == "7.70"
    assert display_metric(row, "object_hal_chairi") == "25.83"
    assert display_metric(row, "llava_bench_gpt_score") == "-"
    row["metrics"]["pope_adv_f1"] = 0
    assert display_metric(row, "pope_adv_f1") == "0.00"


@pytest.mark.parametrize("key,value", [("pope_adv_accuracy", 85.1), ("mmhal_hallucination_rate", 57),
    ("amber_chair", float("nan")), ("mmhal_avg_score", True)])
def test_invalid_units_and_nonfinite_values_rejected(key, value):
    row = entry()
    row["metrics"][key] = value
    with pytest.raises(ValueError):
        validate_row(row)


@pytest.mark.parametrize("key,new", [("judge", "gpt-4"), ("source_type", "paper_reported"),
    ("model", "llava-v1.5-13b"), ("method", "different")])
def test_run_identity_cannot_be_relabelled(key, new):
    rows = [entry(), entry("other")]
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError, match="use a new run_id"):
        upsert(rows, entry(**{key: new}))
    assert rows == before


def test_complete_run_cannot_be_silently_downgraded(tmp_path):
    out = tmp_path / "comparison"
    update_registry(out, entry(), reference_file=None)
    old_json = (out / "registry.json").read_bytes()
    old_md = (out / "comparison.md").read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        update_registry(out, entry(status="incomplete"), reference_file=None)
    assert (out / "registry.json").read_bytes() == old_json
    assert (out / "comparison.md").read_bytes() == old_md
    assert not (out / ".registry.lock").exists()
    registry = update_registry(out, entry(status="incomplete"), reference_file=None,
                               allow_incomplete_update=True)
    assert registry["rows"][0]["status"] == "incomplete"
    assert registry["rows"][0]["history"][-1]["status"] == "complete"


def test_upsert_preserves_other_runs_and_prior_snapshot():
    before = [entry(), entry("other")]
    new = entry()
    new["metrics"]["amber_chair"] = 6.5
    result = upsert(before, new)
    assert len(result) == 2 and result[1] == before[1]
    assert result[0]["metrics"]["amber_chair"] == 6.5
    assert result[0]["history"][-1]["metrics"]["amber_chair"] == 7.7
    again = upsert(result, entry(metrics=copy.deepcopy(new["metrics"])))
    assert len(again[0]["history"]) == 1


@pytest.mark.parametrize("key,previous,current", [("batch_size", "1", "16"),
    ("sources_sha256", {"questions": "old"}, {"questions": "new"}),
    ("adapter_sha256", {"weights": "old"}, {"weights": "new"})])
def test_known_protocol_drift_requires_a_new_run_id(key, previous, current):
    row, new = entry(), entry()
    row["protocol"][key] = previous
    new["protocol"][key] = current
    with pytest.raises(ValueError, match="changed protocol"):
        upsert([row], new)


def test_local_source_records_hashes_and_actual_update_count(tmp_path):
    data = summary(tmp_path)
    train = dump(tmp_path / "train.json", {"status": "complete", "counts": {"attempted": 109,
        "updated": 100, "skipped": 9}, "settings": {"mode": "modulated"}, "pool_sha256": "pool"})
    evaluation = dump(tmp_path / "eval.json", {"judge_model": "deepseek-flash", "batch_size": "16",
        "judge_thinking": "disabled", "judge_endpoint": "https://user:secret@example.com/v1?key=secret",
        "sources_sha256": {"questions": "q"}, "adapter_sha256": {"weights": "w"}})
    row = local_row(data, run_id="ours", method="ours", model="7B", judge="deepseek-flash",
                    training_manifest=train, evaluation_manifest=evaluation)
    assert row["steps"] == 100 and row["status"] == "complete"
    assert row["protocol"]["training_counts"]["attempted"] == 109
    assert row["protocol"]["judge_endpoint"] == "https://example.com/v1"
    assert row["protocol"]["batch_size"] == "16"
    assert len(row["source"]["summary_sha256"]) == 64
    with pytest.raises(ValueError, match="actual successful"):
        local_row(data, run_id="ours", method="ours", model="7B", judge="deepseek-flash",
                  steps=109, training_manifest=train)
    with pytest.raises(ValueError, match="judge differs"):
        local_row(data, run_id="ours", method="ours", model="7B", judge="gpt-4",
                  evaluation_manifest=evaluation)


def test_exact_mmhal_rates_come_from_all_reviews(tmp_path):
    data = summary(tmp_path)
    reviews = dump(tmp_path / "mmhal_bench/review/review-file-ours.jsonl",
                   ["Reason. Rating: 1"] * 55 + ["Reason. Rating: 6"] * 41)
    row = local_row(data, run_id="ours", method="ours", model="7B", judge="deepseek-flash")
    assert row["metrics"]["mmhal_hallucination_rate"] == 55 / 96
    assert row["metrics"]["mmhal_avg_score"] == (55 + 6 * 41) / 96
    assert display_metric(row, "mmhal_hallucination_rate") == "57.29"
    dump(reviews, ["Rating: 1"] * 95)
    with pytest.raises(ValueError, match="all 96"):
        local_row(data, run_id="ours", method="ours", model="7B", judge="deepseek-flash")


def test_checkpoint_state_uses_its_own_counter_and_mode(tmp_path):
    data = summary(tmp_path)
    state = dump(tmp_path / "checkpoint-000500/trainer_state.json", {
        "counts": {"attempted": 550, "updated": 500, "skipped": 50},
        "settings": {"mode": "fixed"}, "cursor": {"epoch": 2, "record_index": 38}})
    row = local_row(data, run_id="at-500", method="claim_feedback", model="7B",
                    judge="deepseek-flash", training_manifest=state)
    assert row["steps"] == 500 and row["method"] == "claim_feedback/fixed"


def test_flag_without_successful_stage_evidence_does_not_claim_complete(tmp_path):
    path = dump(tmp_path / "summary.json", {"all_benchmarks_scored": True, "stages": {}})
    row = local_row(path, run_id="ours", method="ours", model="7B", judge="deepseek-flash")
    assert row["status"] == "incomplete"


def test_paper_and_local_rows_render_in_separate_sections_and_sizes(tmp_path):
    paper = entry("paper:7b:base", source_type="paper_reported", judge="GPT-4 / ObjectHal unspecified",
                  status="paper_reported_not_reproduced", steps=None,
                  metrics={"mmhal_avg_score": 2.01, "pope_adv_accuracy": .8493})
    paper13 = dict(paper, run_id="paper:13b:base", model="llava-v1.5-13b")
    source = dump(tmp_path / "reference.json", {"schema_version": 1, "rows": [paper, paper13]})
    registry = update_registry(tmp_path / "table", entry(), reference_file=source)
    text = render_markdown(registry)
    assert text.index("## 论文原报（未复现）") < text.index("## 本项目实测")
    assert "### llava-v1.5-13b" in text
    assert "ObjectHal unspecified" in text
    assert "84.93" in text and "论文原报" in text
    assert registry["rows"][0]["status"] == "paper_reported_not_reproduced"
    assert len(registry["rows"]) == 3


def test_cli_registers_current_and_keeps_previous_run(tmp_path, monkeypatch):
    data = summary(tmp_path)
    reference = dump(tmp_path / "reference.json", {"schema_version": 1, "rows": []})
    out = tmp_path / "table"
    for run_id in ("ours-100", "ours-500"):
        monkeypatch.setattr("sys.argv", ["compare_experiments.py", "--summary", str(data),
            "--run-id", run_id, "--method", "ours", "--model", "7B", "--judge", "deepseek-flash",
            "--output-dir", str(out), "--reference-file", str(reference)])
        assert main() == 0
    assert len(json.loads((out / "registry.json").read_text())["rows"]) == 2
    assert (out / "comparison.md").is_file()


def training_manifest(tmp_path, *, updated=200, status="running", **overrides):
    manifest = {"status": status, "pool": "/data/feedback.jsonl", "pool_sha256": "pool",
        "pool_records": 256, "counts": {"attempted": updated + 16, "updated": updated, "skipped": 16},
        "settings": {"mode": "modulated", "lr": .001, "max_updates": 5000, "epochs": 100},
        "backend": {"model": "/cache/llava-v1.5-7b"}, "signature": {"lora_rank": 16},
        "resume": {"source": "/original-100/final"},
        "counts_at_start": {"attempted": 109, "updated": 100, "skipped": 9}}
    manifest.update(overrides)
    return dump(tmp_path / "training/manifest.json", manifest)


def progress(path, **kwargs):
    return training_row(path, run_id="ours-5000", method="claim_feedback", model="llava-v1.5-7b", **kwargs)


def test_training_only_uses_actual_counts_and_never_inherits_scores(tmp_path):
    path = training_manifest(tmp_path)
    row = progress(path, training_data="SA1B fixed feedback pool", feedback_source="DiscoSG + FGCLIP")
    assert row["steps"] == 200
    assert row["protocol"]["training_target_updates"] == 5000
    assert row["status"] == "training_running"
    assert row["metrics"] == {} and row["judge"] == "not_evaluated"
    assert row["method"] == "claim_feedback/modulated"
    registry = update_registry(tmp_path / "table", entry())
    old_rows = copy.deepcopy(registry["rows"])
    registry = update_registry(tmp_path / "table", row)
    assert len(registry["rows"]) == 20  # 18 references, original 100-step scores, new progress
    assert registry["rows"][:-1] == old_rows
    assert display_metric(registry["rows"][-1], "mmhal_avg_score") == "-"
    text = render_markdown(registry)
    assert "训练中·未评测" in text and "目标累计更新：5000" in text
    assert "manifest.json" in text
    with pytest.raises(ValueError, match="actual successful"):
        progress(path, steps=5000)


def test_training_progress_can_advance_to_final_without_protocol_drift(tmp_path):
    path = training_manifest(tmp_path)
    rows = [progress(path)]
    path = training_manifest(tmp_path, updated=500, status="running")
    rows = upsert(rows, progress(path))
    assert rows[0]["steps"] == 500 and rows[0]["history"][-1]["steps"] == 200
    path = training_manifest(tmp_path, updated=5000, status="running")
    assert progress(path)["status"] == "training_running"  # final save still pending
    path = training_manifest(tmp_path, updated=5000, status="complete", target_updates_reached=True)
    rows = upsert(rows, progress(path))
    assert rows[0]["status"] == "trained_not_evaluated" and rows[0]["metrics"] == {}
    assert "训练完成·未评测" in render_markdown({"rows": rows})
    training_manifest(tmp_path, updated=5000, status="running")
    with pytest.raises(ValueError, match="cannot revert"):
        upsert(rows, progress(path))


@pytest.mark.parametrize("status,updated,reached,expected", [
    ("complete", 4999, True, "incomplete"), ("complete", 5000, False, "incomplete"),
    ("failed", 400, None, "failed"), ("interrupted", 400, None, "incomplete"),
    (None, 5000, None, "incomplete")])
def test_training_completion_requires_finished_job_and_actual_target(tmp_path, status, updated, reached, expected):
    path = training_manifest(tmp_path, updated=updated, status=status, target_updates_reached=reached)
    assert progress(path)["status"] == expected


@pytest.mark.parametrize("change", [
    {"settings": {"mode": "modulated", "lr": .002, "max_updates": 5000}},
    {"signature": {"lora_rank": 32}}, {"pool_sha256": "different-pool"},
    {"resume": {"source": "/different-start/final"}}])
def test_training_configuration_cannot_change_under_same_id(tmp_path, change):
    path = training_manifest(tmp_path)
    old = progress(path)
    training_manifest(tmp_path, updated=500, **change)
    with pytest.raises(ValueError, match="changed protocol training_config_sha256"):
        upsert([old], progress(path))


def test_training_progress_cannot_decrease_or_replace_scored_run(tmp_path):
    path = training_manifest(tmp_path, updated=500)
    old = progress(path)
    training_manifest(tmp_path, updated=400)
    with pytest.raises(ValueError, match="decrease"):
        upsert([old], progress(path))
    for allow in (False, True):
        with pytest.raises(ValueError, match="new run_id"):
            upsert([entry("ours-5000")], old, allow_incomplete_update=allow)
        with pytest.raises(ValueError, match="new run_id"):
            upsert([old], entry("ours-5000"), allow_incomplete_update=allow)


def test_training_manifest_hash_matches_the_single_parsed_snapshot(tmp_path, monkeypatch):
    path = training_manifest(tmp_path)
    payload = path.read_bytes()
    real_read = Path.read_bytes
    reads = []
    def read_then_advance(self):
        data = real_read(self)
        if self == path:
            reads.append(self)
            training_manifest(tmp_path, updated=500)
        return data
    monkeypatch.setattr(Path, "read_bytes", read_then_advance)
    row = progress(path)
    assert len(reads) == 1 and row["steps"] == 200
    assert row["source"]["training_manifest_sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("bad_counts", [
    {"attempted": 200, "updated": 200, "skipped": 16},
    {"attempted": 200, "updated": True, "skipped": 16},
    {"updated": 200}])
def test_training_requires_consistent_successful_update_evidence(tmp_path, bad_counts):
    path = training_manifest(tmp_path, counts=bad_counts)
    with pytest.raises(ValueError, match="count"):
        progress(path)


def test_cli_training_only_preserves_reference_and_scored_rows(tmp_path, monkeypatch):
    out = tmp_path / "table"
    update_registry(out, entry())
    path = training_manifest(tmp_path)
    argv = ["compare_experiments.py", "--training-only", "--training-manifest", str(path),
        "--run-id", "ours-5000", "--method", "claim_feedback", "--model", "llava-v1.5-7b",
        "--output-dir", str(out)]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    registry = json.loads((out / "registry.json").read_text(encoding="utf-8"))
    assert len(registry["rows"]) == 20 and registry["rows"][-1]["metrics"] == {}
    for extra in (["--summary", "any.json"], ["--judge", "deepseek-flash"]):
        monkeypatch.setattr("sys.argv", argv + extra)
        with pytest.raises(SystemExit) as caught:
            main()
        assert caught.value.code == 2
