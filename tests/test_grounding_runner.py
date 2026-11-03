"""Result publication and frozen image-group boundaries in long experiments."""
import json

import pytest

from run.run_grounding_calibration import Runner, frozen_groups


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


@pytest.mark.parametrize("changed", [{"split": "test"}, {"analysis_role": "retrospective"}])
def test_content_duplicates_cannot_cross_frozen_boundaries(changed):
    meta = {"image_group": "sha256:identical", "split": "dev", "analysis_role": "formal"}
    records = [{"id": "a", "metadata": meta}, {"id": "b", "metadata": dict(meta, **changed)}]
    with pytest.raises(ValueError, match="crosses"):
        frozen_groups(records)
