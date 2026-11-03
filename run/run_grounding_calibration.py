"""Run the frozen full-image grounding calibration plan with recoverable stages."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from run.build_grounding_corpus import atomic_bytes, digest, read_json, sha, stage_lock, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode() for row in rows)


def immutable_bytes(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing artifact differs from frozen inputs: {path.name}")
    else:
        atomic_bytes(path, payload)


def frozen_groups(records):
    mapping, groups = {}, {}
    for record in records:
        meta = record.get("metadata", {})
        group, split, role = (meta.get(k) for k in ("image_group", "split", "analysis_role"))
        if not isinstance(group, str) or not group or split not in ("dev", "test") or role not in ("formal", "retrospective"):
            raise ValueError("Each input needs frozen metadata image_group, dev/test split, and formal/retrospective role")
        if group in groups and groups[group] != (split, role):
            raise ValueError("A duplicate image group crosses split or formal/retrospective roles")
        groups[group] = split, role
        if record["id"] in mapping:
            raise ValueError("Duplicated input record ID")
        mapping[record["id"]] = {"image_group": group, "split": split, "analysis_role": role}
    return mapping


def prepare(root, plan):
    from mvpg_run.claim_feedback.data import read_jsonl
    from mvpg_run.claim_feedback.diagnostics import prepare_cases, validate_cases
    details = root / "details"
    claims = details / "decomposition/claims.jsonl"
    records = list(read_jsonl(claims))
    source = list(read_jsonl(details / "inputs.jsonl", validate=False))
    mapping = frozen_groups(source)
    if [r["id"] for r in records] != [r["id"] for r in source]:
        raise ValueError("Decomposed records are not the complete, ordered image corpus")
    # Projection is label-blind; the previously frozen grouping overrides the
    # helper's incidental random split before any judgment or score is requested.
    initial_audit = {}
    cases = prepare_cases(records, dev_fraction=.4, seed=plan.get("seed", 42),
        image_root=plan.get("image_root", "."), input_sha256=sha(claims), audit=initial_audit)
    for case in cases:
        case.update(mapping[case["record_id"]])
    validate_cases(cases)
    formal = [c for c in cases if c["analysis_role"] == "formal"]
    if {c["split"] for c in formal} != {"dev", "test"}:
        raise ValueError("Formal claims require nonempty frozen dev and test splits")
    per_role = {}
    for role in ("formal", "retrospective"):
        rr = [r for r in records if mapping[r["id"]]["analysis_role"] == role]
        cc = [c for c in cases if c["analysis_role"] == role]
        present = {c["record_id"] for c in cc}
        per_role[role] = {"records": len(rr), "image_groups": len({mapping[r['id']]['image_group'] for r in rr}),
            "raw_claims": sum(len(r["claims"]) for r in rr), "eligible_claims": len(cc),
            "eligible_images": len(present), "empty_candidate_record_ids": [r["id"] for r in rr if r["id"] not in present],
            "cases_per_split": dict(Counter(c["split"] for c in cc))}
    audit = {"claims_sha256": sha(claims), "plan_sha256": sha(details / "plan.json"),
             "input_sha256": sha(details / "inputs.jsonl"), "records": len(records), "cases": len(cases),
             "raw_claims": sum(len(r["claims"]) for r in records),
             "excluded_claim_reasons": initial_audit["excluded_candidates"], "roles": per_role,
             "scope": "Source-aligned affirmative object-existence subset; raw scene graphs and all images remain persisted"}
    audit["parser_integrity"] = {
        "records_with_rejected_triples": sum(bool(r.get("disco_graph", {}).get("rejected_triples")) for r in records),
        "rejected_triple_occurrences_across_rounds": sum(len(r.get("disco_graph", {}).get("rejected_triples", [])) for r in records),
        "note": "Empty generated triple slots are retained in raw round history and excluded from claims; they are never visual negatives."
    }
    payload = rows_bytes(cases)
    audit["cases_sha256"] = hashlib.sha256(payload).hexdigest()
    immutable_bytes(details / "cases.jsonl", payload)
    immutable_bytes(details / "case_preparation.json", (json.dumps(audit, indent=2) + "\n").encode())
    return audit


def validate_scores(details, fgclip_root):
    from mvpg_run.claim_feedback.diagnostics import load_cases
    source = load_cases(details / "cases.jsonl")
    scored = load_cases(details / "scored.jsonl")
    if len(source) != len(scored):
        raise ValueError("Existing score file is incomplete")
    fingerprint = sha(details / "cases.jsonl")
    model_root = Path(fgclip_root).resolve()
    expected_model = {"root": str(model_root), "score_mode": "global", "text_form": "raw", "walk_type": "short",
        "weights_sha256": sha(model_root / "model.safetensors"), "config_sha256": sha(model_root / "config.json"),
        "tokenizer_sha256": sha(model_root / "tokenizer.model")}
    for original, row in zip(source, scored):
        if row["case_id"] != original["case_id"]:
            raise ValueError("Score file order or case identity differs")
        if row.get("score_provenance", {}).get("global_raw", {}).get("input_sha256") != fingerprint:
            raise ValueError("Existing score file belongs to different cases")
        if "global_raw" not in row.get("scores", {}):
            raise ValueError("Existing score file is missing global_raw")
        provenance = row.get("score_provenance", {}).get("global_raw", {})
        if any(provenance.get(k) != v for k, v in expected_model.items()):
            raise ValueError("Existing scores use different FG-CLIP weights or scoring configuration")
        if any(row.get(k) != v for k, v in original.items() if k not in ("scores", "score_provenance")):
            raise ValueError("Existing scores changed frozen case fields")
    return len(scored)


def join(root):
    from mvpg_run.claim_feedback.diagnostics import JUDGE_LABELS, load_cases, validate_cases
    details = root / "details"
    cases = load_cases(details / "cases.jsonl")
    judged = load_cases(details / "judged.jsonl")
    scored = load_cases(details / "scored.jsonl")
    validate_scores(details, read_json(details / "plan.json")["fgclip_root"])
    by_judge, by_score = ({r["case_id"]: r for r in rows} for rows in (judged, scored))
    expected = {r["case_id"] for r in cases}
    if set(by_judge) != expected or set(by_score) != expected:
        raise ValueError("Judge/score case IDs differ from the complete corpus")
    result = []
    for source in cases:
        j, s = by_judge[source["case_id"]], by_score[source["case_id"]]
        for key, value in source.items():
            if key not in ("scores", "score_provenance", "judge_label") and (j.get(key) != value or s.get(key) != value):
                raise ValueError(f"Judge/score changed the frozen case field {key}")
        if j.get("judge_label") not in JUDGE_LABELS or j.get("judge_error"):
            raise ValueError("Unfinished or failed visual judgments cannot be excluded or relabeled")
        row = copy.deepcopy(j)
        row["scores"] = s["scores"]
        row["score_provenance"] = s["score_provenance"]
        result.append(row)
    validate_cases(result)
    payload = rows_bytes(result)
    immutable_bytes(details / "joined.jsonl", payload)
    return {"cases": len(result), "sha256": hashlib.sha256(payload).hexdigest(),
            "roles": dict(Counter(r["analysis_role"] for r in result)),
            "labels": dict(Counter(r["judge_label"] for r in result))}


class Runner:
    def __init__(self, root, plan):
        self.root, self.details, self.plan = root, root / "details", plan
        self.logs = self.details / "logs"
        self.logs.mkdir(exist_ok=True)
        self.active = {}
        manifest = {"schema": "grounding_calibration_orchestration_v1", "plan_sha256": sha(self.details / "plan.json"),
                    "input_sha256": sha(self.details / "inputs.jsonl")}
        immutable_bytes(self.details / "run_manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        self.state = {"status": "running", "stage": "initializing", "pid": os.getpid(),
                      "updated_utc": now(), "stages": {}, "plan_sha256": manifest["plan_sha256"]}
        self.final_published = (root / "results.json").exists() and read_json(root / "results.json").get("schema") == "grounding_calibration_results_v1"
        self.save()

    def save(self):
        self.state["updated_utc"] = now()
        write_json(self.details / "status.json", self.state)
        # The analysis subprocess publishes while await_jobs is still polling.
        # Detect that publication before writing another progress summary.
        published = self.root / "results.json"
        if published.exists():
            self.final_published |= read_json(published).get("schema") == "grounding_calibration_results_v1"
        if not self.final_published and self.state["stage"] != "analysis":
            summary = {k: self.state[k] for k in ("status", "stage", "updated_utc")}
            progress = {}
            for name, path in (("generation", self.details / "generation/status.json"),
                               ("decomposition", self.details / "decomposition/status.json"),
                               ("judge", self.details / "judged.jsonl.status.json")):
                if path.exists():
                    progress[name] = read_json(path)
            summary.update(scope="Full original-image corpus; visual judge is a silver reference, not human GT",
                           details="details/", stages=self.state["stages"], progress=progress, thresholds=None)
            write_json(self.root / "results.json", summary)
            lines = []
            for name, label in (("generation", "图片描述"), ("decomposition", "声明分解")):
                if name in progress:
                    item = progress[name]
                    lines.append(f"{label}：{item['completed_records']}/{item['total']}。")
            if "judge" in progress:
                item = progress["judge"]
                lines.append(f"视觉判别：{item['cases_judged']}/{item['cases_total']}条声明。")
            atomic_bytes(self.root / "results.md", ("# 全量视觉接地性校准\n\n"
                f"状态：{self.state['status']}；当前阶段：{self.state['stage']}。\n\n"
                + "\n\n".join(lines) + "\n\n"
                "图片描述、原始图结构、声明、裁判记录与FG-CLIP分数均保存在details/。"
                "不使用旧阈值预筛选。最终阈值仅用独立开发图像选择，测试图像用于验证。\n").encode())

    def command(self, name, args):
        self.state["stage"] = name
        stage = self.state["stages"].setdefault(name, {})
        log = (self.logs / (name + ".log")).open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update(PYTHONPATH=str(REPO) + os.pathsep + env.get("PYTHONPATH", ""), OMP_NUM_THREADS="8",
                   TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        process = subprocess.Popen([sys.executable, "-u", *map(str, args)], cwd=REPO, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        stage.update(status="running", pid=process.pid, started_utc=now(), log=str(self.logs / (name + ".log")))
        self.active[name] = (process, log)
        self.save()
        return process, log

    def await_jobs(self, jobs):
        pending = dict(jobs)
        failed = []
        while pending:
            for name, (process, log) in list(pending.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    self.state["stages"][name].update(status="complete" if code == 0 else "failed",
                                                     exit_code=code, finished_utc=now())
                    if code:
                        failed.append(name)
                    self.active.pop(name, None)
                    del pending[name]
            self.save()
            if pending:
                time.sleep(10)
        if failed:
            raise RuntimeError("Stages failed: " + ", ".join(failed) + "; outputs are retained for resume")

    def stop_owned_jobs(self):
        for process, _ in self.active.values():
            if process.poll() is None:
                process.terminate()
        for name, (process, log) in self.active.items():
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            log.close()
            self.state["stages"][name].update(status="interrupted", exit_code=code, finished_utc=now())
        self.active.clear()

    def corpus_stage(self, name, args):
        path = self.details / name
        # A separately launched generation stage may already own the lock.
        while (path / ".run.lock").exists():
            try:
                with stage_lock(path):
                    break
            except ValueError:
                self.state["stage"] = "waiting_for_" + name
                self.save(); time.sleep(15)
        if (path / "manifest.json").exists():
            args.append("--resume")
        self.await_jobs({name: self.command(name, args)})


def run(root):
    plan = read_json(root / "details/plan.json")
    details = root / "details"
    frozen_groups(read_rows(details / "inputs.jsonl"))
    with stage_lock(details):
        runner = Runner(root, plan)
        try:
            shared = ["--device", "cuda", "--seed", str(plan.get("seed", 42)), "--image-root", plan.get("image_root", ".")]
            runner.corpus_stage("generation", ["run/build_grounding_corpus.py", "generate", "--input", details / "inputs.jsonl",
                "--output-dir", details / "generation", "--model-name", plan["model_name"],
                "--batch-size", str(plan.get("batch_size", 8)), "--max-new-tokens", str(plan.get("max_new_tokens", 192)),
                "--temperature", str(plan.get("temperature", .7)), "--top-p", str(plan.get("top_p", .9)), *shared])
            runner.corpus_stage("decomposition", ["run/build_grounding_corpus.py", "decompose", "--input", details / "generation/responses.jsonl",
                "--output-dir", details / "decomposition", "--disco-root", plan["disco_root"],
                "--seed-model", plan["seed_model"], "--refiner-model", plan["refiner_model"], "--rounds", "3", *shared])
            runner.state["stage"] = "prepare_claims"; runner.save()
            runner.state["stages"]["prepare"] = {"status": "complete", **prepare(root, plan)}
            runner.save()
            judged = details / "judged.jsonl"
            judge_args = ["-m", "mvpg_run.claim_feedback.grounded_judge", "--input", details / "cases.jsonl",
                "--output", judged, "--model", plan["judge_model"], "--workers", str(plan.get("judge_workers", 4)),
                "--image-root", plan.get("image_root", "."), "--image-detail", "original"]
            if Path(str(judged) + ".manifest.json").exists():
                judge_args.append("--resume")
            jobs = {}
            if (details / "scored.jsonl").exists():
                runner.state["stages"]["score"] = {"status": "complete", "reused_cases": validate_scores(details, plan["fgclip_root"])}
            else:
                score_args = ["-m", "mvpg_run.claim_feedback.diagnostics", "score", "--input", details / "cases.jsonl",
                    "--output", details / "scored.jsonl", "--fgclip-root", plan["fgclip_root"], "--device", "cuda",
                    "--image-root", plan.get("image_root", "."), "--score-mode", "global", "--text-form", "raw"]
                jobs["score"] = runner.command("score", score_args)
            jobs["judge"] = runner.command("judge", judge_args)
            runner.state["stage"] = "judge_and_score"; runner.save(); runner.await_jobs(jobs)
            runner.state["stage"] = "join"; runner.save()
            runner.state["stages"]["join"] = {"status": "complete", **join(root)}
            runner.await_jobs({"analysis": runner.command("analysis", ["run/analyze_grounding_calibration.py", "--input", details / "joined.jsonl",
                "--output-dir", root, "--score-key", plan.get("score_key", "global_raw"),
                "--target-precision", str(plan.get("target_precision", .95)),
                "--min-selected", str(plan.get("min_selected", 30)),
                "--bootstrap-samples", str(plan.get("bootstrap_samples", 500)),
                "--seed", str(plan.get("seed", 42)), "--plots"])})
            result = read_json(root / "results.json")
            if result.get("schema") != "grounding_calibration_results_v1" or result.get("status") != "complete":
                raise ValueError("Final analysis has not published complete formal and retrospective results")
            runner.final_published = True
            runner.state.update(status="complete", stage="complete", finished_utc=now())
            runner.save()
            return runner.state
        except BaseException as exc:
            runner.stop_owned_jobs()
            runner.state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                error=f"{type(exc).__name__}: {exc}")
            # If analysis published any final artifact, never replace it with
            # a generic progress page, including a partially complete report.
            if (root / "results.json").exists():
                runner.final_published |= read_json(root / "results.json").get("schema") == "grounding_calibration_results_v1"
            runner.save()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Pipeline termination requested")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        result = run(args.run.expanduser().resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Grounding calibration: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
