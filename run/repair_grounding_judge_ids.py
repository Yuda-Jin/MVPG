"""Requery only failed images with short request IDs, then finish calibration.

This is a documented transport change, not fuzzy matching of invalid IDs.
The original full-corpus journal and failed output remain backed up intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mvpg_run.claim_feedback import grounded_judge as g
from mvpg_run.claim_feedback.diagnostics import JUDGE_LABELS, _endpoint, load_cases, validate_cases
from run.build_grounding_corpus import atomic_bytes, read_json, sha, stage_lock, write_json
from run.run_grounding_calibration import Runner, immutable_bytes, join, now, rows_bytes


def merge_repaired_rows(cases, original, repaired, repair_reference):
    for rows in (cases, original, repaired):
        validate_cases(rows)
    expected = {c["case_id"] for c in cases}
    original_by_id = {c["case_id"]: c for c in original}
    if set(original_by_id) != expected:
        raise ValueError("Original judgments differ from the full frozen corpus")
    failed = {c["case_id"] for c in original
              if c.get("judge_error") or c.get("judge_label") not in JUDGE_LABELS}
    fixed = {c["case_id"]: c for c in repaired}
    if set(fixed) != failed:
        raise ValueError("Repair IDs must exactly match the original failed cases")
    result = []
    for case in cases:
        cid = case["case_id"]
        old = original_by_id[cid]
        candidate = fixed.get(cid, old)
        for row in (old, candidate):
            if any(row.get(k) != v for k, v in case.items() if k != "judge_label"):
                raise ValueError("A judgment changed frozen case fields")
        if candidate.get("judge_error") or candidate.get("judge_label") not in JUDGE_LABELS:
            raise ValueError("Repair has an invalid or unfinished judgment")
        row = copy.deepcopy(candidate)
        if cid in failed:
            row["judge_repair"] = dict(copy.deepcopy(repair_reference), original_judge_error=old.get("judge_error"))
        result.append(row)
    return result, {"total_cases": len(result), "preserved_successful_cases": len(result) - len(failed),
                    "repaired_cases": len(failed)}


def prepare(root):
    details = root / "details"
    repair = details / "repairs/case_id_transport"
    repair.mkdir(parents=True, exist_ok=True)
    master = details / "judged.jsonl"
    backup = repair / "original_judged.jsonl"
    if not backup.exists():
        immutable_bytes(backup, master.read_bytes())
        immutable_bytes(repair / "original_judge_status.json", (details / "judged.jsonl.status.json").read_bytes())
        immutable_bytes(repair / "original_pipeline_status.json", (details / "status.json").read_bytes())
    cases, original = load_cases(details / "cases.jsonl"), load_cases(backup)
    failed_ids = {row["case_id"] for row in original if row.get("judge_error") or row.get("judge_label") not in JUDGE_LABELS}
    if not failed_ids:
        raise ValueError("No original failed cases need ID-transport repair")
    subset = [row for row in cases if row["case_id"] in failed_ids]
    failed_images = {(r["record_id"], r["image"]) for r in subset}
    if {r["case_id"] for r in cases if (r["record_id"], r["image"]) in failed_images} != failed_ids:
        raise ValueError("Repair must requery complete failed image groups")
    manifest = read_json(details / "judged.jsonl.manifest.json")
    if manifest["input_sha256"] != sha(details / "cases.jsonl") or manifest["prompt_sha256"] != g._digest(g.BATCH_PROMPT):
        raise ValueError("Original cases or judge prompt changed")
    endpoint = _endpoint(os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    thinking = os.getenv("JUDGE_THINKING") or None
    if endpoint != manifest["endpoint"] or thinking != manifest["thinking"]:
        raise ValueError("Endpoint or thinking settings differ from the original judge")
    for row in subset:
        path = Path(row["image"]).resolve()
        if manifest["image_sha256"].get(str(path)) != sha(path):
            raise ValueError("A repair image changed")
    immutable_bytes(repair / "cases.jsonl", rows_bytes(subset))
    selection = {"schema": "judge_id_transport_selection_v1", "source_cases_sha256": sha(details / "cases.jsonl"),
                 "original_judged_sha256": sha(backup), "failed_case_ids": sorted(failed_ids),
                 "failed_image_groups": len(failed_images), "original_judge_manifest_sha256": sha(details / "judged.jsonl.manifest.json"),
                 "change": "Request-local short IDs; image, question, answer, claim text, judge instructions/model and validation unchanged"}
    immutable_bytes(repair / "selection.json", (json.dumps(selection, indent=2) + "\n").encode())
    marker = Path(str(master) + ".repair_manifest.json")
    if marker.exists():
        saved = read_json(marker)
        if sha(master) not in (saved["original_judged_sha256"], saved["merged_judged_sha256"]):
            raise ValueError("Composite judgment output changed outside the repair")
    elif sha(master) != sha(backup):
        raise ValueError("Original judgment output changed after backup")
    return repair, cases, original, manifest


def run(root):
    root = Path(root).resolve()
    details = root / "details"
    with stage_lock(details), g.output_lock(str(details / "judged.jsonl") + ".lock"):
        repair, cases, original, manifest = prepare(root)
        plan = read_json(details / "plan.json")
        old_state = read_json(repair / "original_pipeline_status.json")
        runner = Runner(root, plan)
        runner.state.update(stages=copy.deepcopy(old_state["stages"]), stage="judge_id_repair")
        runner.state["stages"]["judge"].update(status="repairing")
        runner.save()
        try:
            output = repair / "judged.jsonl"
            args = ["-m", "mvpg_run.claim_feedback.grounded_judge", "--input", repair / "cases.jsonl",
                    "--output", output, "--model", manifest["model"], "--workers", "2",
                    "--max-output-tokens", str(manifest["max_tokens"]), "--image-detail", manifest["image_detail"],
                    "--id-encoding", "request_alias"]
            if Path(str(output) + ".manifest.json").exists():
                args.append("--resume")
            runner.await_jobs({"judge_id_repair": runner.command("judge_id_repair", args)})
            repaired = load_cases(output)
            ref = {"protocol": "request_local_short_ids_v1", "repair_manifest": str(output) + ".manifest.json",
                   "repair_manifest_sha256": sha(str(output) + ".manifest.json"),
                   "repair_raw_journal": str(output) + ".images.progress.jsonl",
                   "repair_raw_journal_sha256": sha(str(output) + ".images.progress.jsonl")}
            merged, audit = merge_repaired_rows(cases, original, repaired, ref)
            payload = rows_bytes(merged)
            master = details / "judged.jsonl"
            marker = {"schema": "composite_judge_id_repair_v1", "status": "prepared", **audit,
                      "original_judged_sha256": sha(repair / "original_judged.jsonl"),
                      "merged_judged_sha256": hashlib.sha256(payload).hexdigest(), "repair_reference": ref}
            write_json(str(master) + ".repair_manifest.json", marker)
            atomic_bytes(master, payload)
            marker["status"] = "complete"
            write_json(str(master) + ".repair_manifest.json", marker)
            write_json(repair / "merge_report.json", marker)
            status = read_json(repair / "original_judge_status.json")
            status.update(status="complete", images_judged=status["images_total"], cases_judged=len(merged),
                          cases_failed_attempt=0, cases_not_attempted=0,
                          labels=dict(Counter(r["judge_label"] for r in merged)), updated_utc=now(),
                          composite_repair_manifest=str(master) + ".repair_manifest.json")
            write_json(str(master) + ".status.json", status)
            runner.state["stages"]["judge"].update(status="complete", exit_code=0, finished_utc=now(),
                                                  repair_manifest=str(master) + ".repair_manifest.json")
            runner.state.update(stage="join"); runner.save()
            runner.state["stages"]["join"] = {"status": "complete", **join(root)}
            runner.await_jobs({"analysis": runner.command("analysis", ["run/analyze_grounding_calibration.py",
                "--input", details / "joined.jsonl", "--output-dir", root, "--score-key", plan.get("score_key", "global_raw"),
                "--target-precision", str(plan.get("target_precision", .95)), "--min-selected", str(plan.get("min_selected", 30)),
                "--bootstrap-samples", str(plan.get("bootstrap_samples", 500)), "--seed", str(plan.get("seed", 42)), "--plots"])})
            if read_json(root / "results.json").get("status") != "complete":
                raise ValueError("Final analysis is incomplete")
            runner.final_published = True
            runner.state.update(status="complete", stage="complete", finished_utc=now())
            runner.save()
            return runner.state
        except BaseException as error:
            runner.stop_owned_jobs()
            runner.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            runner.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.run)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Judge ID repair: {error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
