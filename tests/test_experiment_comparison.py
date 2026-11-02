"""Result registries must preserve experiment identity, units and source data."""
import copy
import json
from pathlib import Path

import pytest

from run.compare_experiments import (REQUIRED_SCORE_STAGES, display_metric,
    local_row, main, render_markdown, update_registry, upsert, validate_row)


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
