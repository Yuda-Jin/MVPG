"""Result publication and frozen image-group boundaries in long experiments."""
import json

import pytest

from run.run_grounding_calibration import Runner, frozen_groups, sha, validate_scores


def test_analysis_publication_survives_polling_and_restart(tmp_path):
    details = tmp_path / "details"
    details.mkdir()
    (details / "plan.json").write_text("{}")
    (details / "inputs.jsonl").write_text("")
    runner = Runner(tmp_path, {})
    report = {"schema": "grounding_calibration_results_v1", "status": "complete", "thresholds": {"negative": .2}}
    (tmp_path / "results.json").write_text(json.dumps(report))
    (tmp_path / "results.md").write_text("Final threshold report")
    runner.state["stage"] = "analysis"
    runner.save()
    assert json.loads((tmp_path / "results.json").read_text()) == report
    assert (tmp_path / "results.md").read_text() == "Final threshold report"
    Runner(tmp_path, {}).save()
    assert json.loads((tmp_path / "results.json").read_text()) == report


def test_polling_does_not_overwrite_markdown_before_final_json(tmp_path):
    details = tmp_path / "details"
    details.mkdir()
    (details / "plan.json").write_text("{}")
    (details / "inputs.jsonl").write_text("")
    runner = Runner(tmp_path, {})
    runner.state["stage"] = "analysis"
    (tmp_path / "results.md").write_text("New final markdown")
    runner.save()
    assert (tmp_path / "results.md").read_text() == "New final markdown"


def test_reused_scores_detect_changed_model_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.model"):
        (model / name).write_bytes(b"original")
    case = dict(case_id="c", record_id="r", claim_id="c", image="image.png", image_group="g", split="dev",
        question="Describe", response="car", claim_text="car", source_spans=[[0, 3]], kind="object", judge_label=None, scores={})
    (tmp_path / "cases.jsonl").write_text(json.dumps(case) + "\n")
    scored = dict(case, scores={"global_raw": .2}, score_provenance={"global_raw": {
        "root": str(model.resolve()), "score_mode": "global", "text_form": "raw", "walk_type": "short",
        "input_sha256": sha(tmp_path / "cases.jsonl"), "weights_sha256": sha(model / "model.safetensors"),
        "config_sha256": sha(model / "config.json"), "tokenizer_sha256": sha(model / "tokenizer.model")}})
    (tmp_path / "scored.jsonl").write_text(json.dumps(scored) + "\n")
    assert validate_scores(tmp_path, model) == 1
    (model / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="different FG-CLIP"):
        validate_scores(tmp_path, model)


def test_controlled_interruption_reaps_owned_children(tmp_path):
    details = tmp_path / "details"
    details.mkdir()
    (details / "plan.json").write_text("{}")
    (details / "inputs.jsonl").write_text("")
    runner = Runner(tmp_path, {})
    process, log = runner.command("judge", ["-c", "import time; time.sleep(60)"])
    try:
        runner.stop_owned_jobs()
        assert process.poll() is not None
        assert log.closed
        assert runner.state["stages"]["judge"]["status"] == "interrupted"
        assert not runner.active
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("changed", [{"split": "test"}, {"analysis_role": "retrospective"}])
def test_content_duplicates_cannot_cross_frozen_boundaries(changed):
    meta = {"image_group": "sha256:identical", "split": "dev", "analysis_role": "formal"}
    records = [{"id": "a", "metadata": meta}, {"id": "b", "metadata": dict(meta, **changed)}]
    with pytest.raises(ValueError, match="crosses"):
        frozen_groups(records)
