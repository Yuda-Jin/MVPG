"""Only incomplete, nonfatal visual judgments may trigger bounded recovery."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from run import recover_grounding_judge as recovery


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def failed_run(tmp_path, entries=None):
    details = tmp_path / "details"
    details.mkdir()
    (details / "logs").mkdir()
    write_json(details / "plan.json", {"judge_model": "test-vision"})
    (details / "cases.jsonl").write_text("{}\n", encoding="utf-8")
    state = {"status": "failed", "stage": "judge_and_score", "stages": {
        "generation": {"status": "complete", "exit_code": 0},
        "decomposition": {"status": "complete", "exit_code": 0},
        "prepare": {"status": "complete"},
        "score": {"status": "complete", "exit_code": 0},
        "judge": {"status": "failed", "exit_code": 1}}}
    write_json(details / "status.json", state)
    write_json(details / "judged.jsonl.status.json", {"status": "incomplete"})
    if entries is None:
        entries = [{"image_key": "retry-me", "judge_error": "HTTP 503", "judge_fatal_error": False}]
    (details / "judged.jsonl.images.progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in entries), encoding="utf-8")
    return tmp_path


def modify_state(root, callback):
    path = root / "details/status.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    callback(state)
    write_json(path, state)


@pytest.mark.parametrize("error", ["HTTP 503", "ValueError", "JSONDecodeError", "UnicodeDecodeError",
                                  "TimeoutError", "ConnectionResetError", "HTTP 429"])
def test_incomplete_nonfatal_judge_failures_are_retryable(tmp_path, error):
    root = failed_run(tmp_path, [{"image_key": "retry-me", "judge_error": error,
                                  "judge_fatal_error": False}])
    eligible, reason = recovery.retry_decision(root)
    assert eligible
    assert "1 failed image" in reason


@pytest.mark.parametrize("error", ["HTTP 401", "ImageChanged"])
def test_fatal_latest_image_attempt_is_not_retried(tmp_path, error):
    root = failed_run(tmp_path, [{"image_key": "stop-me", "judge_error": error,
                                  "judge_fatal_error": True}])
    eligible, reason = recovery.retry_decision(root)
    assert not eligible
    assert "fatal" in reason


def test_unknown_nonfatal_error_is_not_automatically_retried(tmp_path):
    root = failed_run(tmp_path, [{"image_key": "stop-me", "judge_error": "UnexpectedFailure",
                                  "judge_fatal_error": False}])
    assert not recovery.retry_decision(root)[0]


def test_judge_interface_exit_two_is_not_retried(tmp_path):
    root = failed_run(tmp_path)
    modify_state(root, lambda state: state["stages"]["judge"].update(exit_code=2))
    assert not recovery.retry_decision(root)[0]


@pytest.mark.parametrize("stage", ["score", "decomposition", "generation", "prepare"])
def test_other_failed_or_incomplete_stage_blocks_recovery(tmp_path, stage):
    root = failed_run(tmp_path)
    modify_state(root, lambda state: state["stages"][stage].update(status="failed", exit_code=2))
    eligible, reason = recovery.retry_decision(root)
    assert not eligible
    assert "Another pipeline stage" in reason


@pytest.mark.parametrize("stage", ["score", "decomposition", "generation", "prepare"])
def test_missing_required_stage_blocks_recovery(tmp_path, stage):
    root = failed_run(tmp_path)
    modify_state(root, lambda state: state["stages"].pop(stage))
    assert not recovery.retry_decision(root)[0]


def test_running_pipeline_is_never_restarted(tmp_path):
    root = failed_run(tmp_path)
    modify_state(root, lambda state: state.update(status="running"))
    assert not recovery.retry_decision(root)[0]


def test_superseded_failed_attempt_does_not_trigger_a_new_request(tmp_path):
    root = failed_run(tmp_path, [
        {"image_key": "done", "judge_error": "HTTP 503", "judge_fatal_error": False},
        {"image_key": "done", "judge_error": None, "judge_fatal_error": False},
    ])
    assert not recovery.retry_decision(root)[0]


def test_superseded_fatal_attempt_does_not_block_other_retryable_images(tmp_path):
    root = failed_run(tmp_path, [
        {"image_key": "done", "judge_error": "HTTP 401", "judge_fatal_error": True},
        {"image_key": "done", "judge_error": None, "judge_fatal_error": False},
        {"image_key": "retry-me", "judge_error": "HTTP 503", "judge_fatal_error": False},
    ])
    eligible, reason = recovery.retry_decision(root)
    assert eligible
    assert "1 failed image" in reason


def test_complete_judge_status_is_not_retried(tmp_path):
    root = failed_run(tmp_path)
    write_json(root / "details/judged.jsonl.status.json", {"status": "complete"})
    assert not recovery.retry_decision(root)[0]


def mock_supervisor(monkeypatch):
    monkeypatch.setattr(recovery, "process_running", lambda pid: False)
    sleeps = []
    monkeypatch.setattr(recovery.time, "sleep", sleeps.append)
    return sleeps


def test_supervisor_obeys_resume_budget_and_records_every_pass(tmp_path, monkeypatch):
    root = failed_run(tmp_path)
    sleeps = mock_supervisor(monkeypatch)
    calls = []
    def launch(command, **options):
        calls.append((command, options))
        return SimpleNamespace(pid=9000 + len(calls), returncode=1, poll=lambda: 1)
    monkeypatch.setattr(recovery.subprocess, "Popen", launch)
    assert recovery.supervise(root, 111, max_resume_passes=2) == 1
    assert len(calls) == 2
    assert sleeps == [20, 60]
    assert all(cmd[-2:] == ["--run", str(root.resolve())] for cmd, _ in calls)
    assert all(cmd[2] == "run/run_grounding_calibration.py" for cmd, _ in calls)
    status = json.loads((root / "details/judge_recovery.json").read_text(encoding="utf-8"))
    assert status["status"] == "exhausted"
    assert len(status["attempts"]) == 2
    assert all(row["exit_code"] == 1 and row["finished_utc"] for row in status["attempts"])


def test_resume_budget_survives_supervisor_restart(tmp_path, monkeypatch):
    root = failed_run(tmp_path)
    sleeps = mock_supervisor(monkeypatch)
    calls = []
    def launch(command, **options):
        calls.append(command)
        return SimpleNamespace(pid=9000 + len(calls), returncode=1, poll=lambda: 1)
    monkeypatch.setattr(recovery.subprocess, "Popen", launch)
    assert recovery.supervise(root, 111, max_resume_passes=1) == 1
    assert len(calls) == 1
    assert recovery.supervise(root, 111, max_resume_passes=1) == 1
    assert len(calls) == 1 and sleeps == [20]
    status = json.loads((root / "details/judge_recovery.json").read_text(encoding="utf-8"))
    assert len(status["attempts"]) == 1 and status["status"] == "exhausted"


def test_changed_frozen_inputs_cannot_reuse_recovery_history(tmp_path, monkeypatch):
    root = failed_run(tmp_path)
    mock_supervisor(monkeypatch)
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **k: pytest.fail("Zero resume budget"))
    assert recovery.supervise(root, 111, max_resume_passes=0) == 1
    (root / "details/cases.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="different frozen experiment"):
        recovery.supervise(root, 111, max_resume_passes=0)


def test_supervisor_stops_after_successful_recovery(tmp_path, monkeypatch):
    root = failed_run(tmp_path)
    sleeps = mock_supervisor(monkeypatch)
    calls = []
    def launch(command, **options):
        calls.append(command)
        modify_state(root, lambda state: state.update(status="complete", stage="complete"))
        return SimpleNamespace(pid=9001, returncode=0, poll=lambda: 0)
    monkeypatch.setattr(recovery.subprocess, "Popen", launch)
    assert recovery.supervise(root, 111, max_resume_passes=3) == 0
    assert len(calls) == 1 and sleeps == [20]
    status = json.loads((root / "details/judge_recovery.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["attempts"][0]["exit_code"] == 0


def test_zero_budget_makes_no_new_requests(tmp_path, monkeypatch):
    root = failed_run(tmp_path)
    sleeps = mock_supervisor(monkeypatch)
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **k: pytest.fail("No resume budget"))
    assert recovery.supervise(root, 111, max_resume_passes=0) == 1
    assert not sleeps


def test_supervisor_stops_for_attention_without_launching_on_fatal_error(tmp_path, monkeypatch):
    root = failed_run(tmp_path, [{"image_key": "fatal", "judge_error": "ImageChanged",
                                  "judge_fatal_error": True}])
    sleeps = mock_supervisor(monkeypatch)
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **k: pytest.fail("Fatal error must stop"))
    assert recovery.supervise(root, 111) == 1
    assert not sleeps
    status = json.loads((root / "details/judge_recovery.json").read_text(encoding="utf-8"))
    assert status["status"] == "needs_attention"
