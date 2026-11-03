"""Wait for an existing calibration and retry only incomplete visual judgments.

Successful image judgments are immutable within resume. No prompt, image,
model, output limit, or label-validation rule changes between sweeps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from run.build_grounding_corpus import read_json, sha, write_json
from mvpg_run.claim_feedback.grounded_judge import output_lock


def retry_decision(root):
    details = Path(root) / "details"
    state = read_json(details / "status.json")
    if state.get("status") != "failed":
        return False, "Pipeline is not failed"
    stages = state.get("stages", {})
    judge = stages.get("judge", {})
    if (state.get("stage") != "judge_and_score" or judge.get("status") != "failed"
            or judge.get("exit_code") != 1):
        return False, "Failure is not an incomplete visual-judge sweep"
    if (any(stages.get(name, {}).get("status") != "complete"
            for name in ("generation", "decomposition", "prepare", "score"))
            or any(name != "judge" and item.get("status") != "complete" for name, item in stages.items())):
        return False, "Another pipeline stage is incomplete or failed"
    status = read_json(details / "judged.jsonl.status.json")
    if status.get("status") != "incomplete":
        return False, "Visual-judge status is not incomplete"
    latest = {}
    with (details / "judged.jsonl.images.progress.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                latest[row["image_key"]] = row
    errors = [row for row in latest.values() if row.get("judge_error") is not None]
    if any(row.get("judge_fatal_error") for row in errors):
        return False, "A latest image attempt has a fatal error"
    allowed = {"HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "URLError",
               "TimeoutError", "OSError", "UnicodeError", "ValueError", "KeyError",
               "TypeError", "IndexError", "AttributeError", "JSONDecodeError", "UnicodeDecodeError",
               "ConnectionError", "ConnectionResetError", "ConnectionAbortedError", "ConnectionRefusedError",
               "BrokenPipeError", "SSLError"}
    if not errors or any(row.get("judge_error") not in allowed for row in errors):
        return False, "No recognized transient or response-validation failures to retry"
    return True, f"Retry {len(errors)} failed image groups using the unchanged judge manifest"


def process_running(pid):
    # Deployment is Linux; avoid treating an unreaped zombie as a live writer.
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def supervise(root, wait_pid, *, max_resume_passes=3):
    root = Path(root).resolve()
    details = root / "details"
    record = {"schema": "grounding_judge_recovery_v1", "status": "waiting", "pid": os.getpid(),
              "watched_pipeline_pid": wait_pid, "max_resume_passes": max_resume_passes,
              "plan_sha256": sha(details / "plan.json"), "cases_sha256": sha(details / "cases.jsonl"),
              "semantic_configuration": "unchanged; retry only failed images via judge --resume",
              "attempts": []}
    def save(status, **fields):
        record.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        write_json(details / "judge_recovery.json", record)
    with output_lock(str(details / "judge_recovery.lock")):
        previous_path = details / "judge_recovery.json"
        if previous_path.exists():
            previous = read_json(previous_path)
            if any(previous.get(k) != record[k] for k in ("schema", "plan_sha256", "cases_sha256")):
                raise ValueError("Recovery history belongs to a different frozen experiment")
            record["attempts"] = previous["attempts"]
        if process_running(wait_pid):
            command = Path(f"/proc/{wait_pid}/cmdline").read_bytes()
            if b"run/run_grounding_calibration.py" not in command or str(root).encode() not in command:
                raise ValueError("The monitored PID is not this calibration pipeline")
        save("waiting")
        while process_running(wait_pid):
            save("waiting")
            time.sleep(15)
        while True:
            state = read_json(details / "status.json")
            if state.get("status") == "complete":
                save("complete")
                return 0
            eligible, reason = retry_decision(root)
            if not eligible:
                save("needs_attention", reason=reason)
                return 1
            if len(record["attempts"]) >= max_resume_passes:
                save("exhausted", reason=reason)
                return 1
            delay = 20 if not record["attempts"] else 60
            save("backoff", reason=reason, backoff_seconds=delay)
            time.sleep(delay)
            with (details / "logs/recovery_pipeline.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-u", "run/run_grounding_calibration.py",
                    "--run", str(root)], cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT)
                (details / "pipeline.pid").write_text(str(process.pid) + "\n")
                attempt = {"pipeline_pid": process.pid, "started_utc": datetime.now(timezone.utc).isoformat()}
                record["attempts"].append(attempt)
                save("resuming")
                while process.poll() is None:
                    save("resuming")
                    time.sleep(15)
                attempt.update(exit_code=process.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
                judge_status = read_json(details / "judged.jsonl.status.json")
                attempt.update(cases_judged=judge_status.get("cases_judged"),
                               cases_failed=judge_status.get("cases_failed_attempt"))
                save("checking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--max-resume-passes", type=int, default=3)
    args = parser.parse_args()
    if args.wait_pid < 1 or args.max_resume_passes < 0:
        parser.error("PID must be positive and resume-pass budget nonnegative")
    return supervise(args.run, args.wait_pid, max_resume_passes=args.max_resume_passes)


if __name__ == "__main__":
    raise SystemExit(main())
