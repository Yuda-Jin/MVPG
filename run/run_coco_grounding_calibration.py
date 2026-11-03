"""Resume a frozen COCO sample through captioning, DiscoSG, DINO and visual judgment.

Grounding DINO is the primary score; FG-CLIP is an automatic same-case baseline.
Neither detector thresholds nor previous weak labels prefilter the claim corpus.
The independently image-reading judge provides silver, not human, reference labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from run.build_grounding_corpus import atomic_bytes, read_json, sha, stage_lock, write_json
from run.run_grounding_calibration import (Runner, frozen_groups, immutable_bytes,
                                           prepare, read_rows, rows_bytes)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_environment(path):
    """Read simple .env assignments without a shell, interpolation or secret output."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return
    for number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = text.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        key, separator, value = text.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid environment assignment on line {number}")
        if value[:1] in ("'", '"'):
            quote = value[0]
            closing = value.find(quote, 1)
            trailing = value[closing + 1:].strip() if closing >= 0 else ""
            if closing < 0 or trailing and not trailing.startswith("#"):
                raise ValueError(f"Invalid quoted environment value on line {number}")
            value = value[1:closing]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        os.environ.setdefault(key, value)


DEFAULTS = {
    "model_name": "/root/autodl-tmp/cache/llava-v1.5-7b",
    "disco_root": "/root/autodl-tmp/cache/discosg/source",
    "seed_model": "/root/autodl-tmp/cache/discosg/seed",
    "refiner_model": "/root/autodl-tmp/cache/discosg/refiner-transformers434",
    "dino_root": "/root/autodl-tmp/cache/grounding-dino-tiny",
    "score_python": "/root/autodl-tmp/envs/grounding-dino/bin/python",
    "fgclip_root": "/root/autodl-tmp/cache/fgclip2-base-patch16",
    "image_root": ".", "device": "cuda", "seed": 42, "expected_images": 1000,
    "batch_size": 8, "max_new_tokens": 192, "temperature": .7, "top_p": .9,
    "parser_batch_size": 4, "dino_batch_size": 4, "judge_workers": 4,
    "judge_timeout": 90., "judge_max_retries": 2, "judge_max_passes": 3,
    "judge_max_output_tokens": 4096, "target_precision": .95,
    "min_selected": 30, "bootstrap_samples": 500,
}
PATH_FIELDS = {"model_name", "disco_root", "seed_model", "refiner_model", "dino_root",
               "score_python", "fgclip_root", "image_root"}


def normalize_option(key, value):
    return str(Path(value).expanduser().resolve()) if key in PATH_FIELDS else value


def validate_options(plan):
    for key in ("expected_images", "batch_size", "max_new_tokens", "parser_batch_size", "dino_batch_size",
                "judge_workers", "judge_max_output_tokens", "judge_max_passes", "min_selected", "bootstrap_samples"):
        if plan[key] < 1:
            raise ValueError(f"Frozen option must be positive: {key}")
    if plan["judge_max_retries"] < 0:
        raise ValueError("judge-max-retries cannot be negative")
    if not math.isfinite(plan["judge_timeout"]) or plan["judge_timeout"] <= 0:
        raise ValueError("judge-timeout must be positive and finite")
    if not math.isfinite(plan["temperature"]) or plan["temperature"] < 0:
        raise ValueError("temperature must be nonnegative and finite")
    if not math.isfinite(plan["top_p"]) or not 0 < plan["top_p"] <= 1:
        raise ValueError("top-p must be in (0,1]")
    if not math.isfinite(plan["target_precision"]) or not 0 < plan["target_precision"] <= 1:
        raise ValueError("target-precision must be in (0,1]")


def validate_input(records, expected_images, image_root="."):
    if len(records) != expected_images:
        raise ValueError(f"Expected exactly {expected_images} sampled images; received {len(records)}")
    mapping = frozen_groups(records)
    if any(item["analysis_role"] != "formal" for item in mapping.values()):
        raise ValueError("This new COCO experiment only accepts formal sampled images")
    if {item["split"] for item in mapping.values()} != {"dev", "test"}:
        raise ValueError("Frozen COCO input needs nonempty dev and test image groups")
    for record in records:
        path = Path(record["image"])
        path = path if path.is_absolute() else Path(image_root) / path
        if not path.is_file():
            raise FileNotFoundError(path)
        if mapping[record["id"]]["image_group"] != "sha256:" + sha(path):
            raise ValueError(f"Sampled image differs from frozen image group: {record['id']}")
        if record.get("response") or record.get("claims"):
            raise ValueError("COCO sample input must precede generation and labels")
    return {"images": len(records), "groups": len({m['image_group'] for m in mapping.values()}),
            "split_images": dict(Counter(m["split"] for m in mapping.values())),
            "reference_scope": "Object-existence silver labels; no preference pairs or training updates"}


def initialize(args):
    root = args.run.expanduser().resolve()
    details = root / "details"
    details.mkdir(parents=True, exist_ok=True)
    with stage_lock(details):
        plan_path = details / "plan.json"
        if plan_path.exists():
            plan = read_json(plan_path)
            if plan.get("schema") != "coco_grounding_dino_calibration_plan_v1":
                raise ValueError("Existing output belongs to a different experiment")
            if args.input is not None and sha(args.input) != plan["input_sha256"]:
                raise ValueError("Requested sample differs from the frozen experiment input")
            for key in DEFAULTS:
                override = getattr(args, key)
                if override is not None and normalize_option(key, override) != plan[key]:
                    raise ValueError(f"Resume changes frozen option: {key}")
            if args.judge_model is not None and args.judge_model != plan["judge_model"]:
                raise ValueError("Resume changes frozen judge model")
            if sha(details / "inputs.jsonl") != plan["input_sha256"]:
                raise ValueError("Frozen experiment input changed")
        else:
            source = (args.input or details / "sample/images.jsonl").expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            options = {key: normalize_option(key, getattr(args, key) if getattr(args, key) is not None else value)
                       for key, value in DEFAULTS.items()}
            judge = args.judge_model or os.getenv("GPT_MODEL") or os.getenv("JUDGE_MODEL")
            if not judge or not judge.strip():
                raise ValueError("Set GPT_MODEL or pass --judge-model")
            plan = {"schema": "coco_grounding_dino_calibration_plan_v1", **options,
                    "judge_model": judge.strip(), "input_sha256": sha(source),
                    "score_key": "dino_max", "score_text_form": "raw_object_phrase",
                    "decomposition_rounds": 3, "judge_id_encoding": "request_alias",
                    "sampling_scope": "1000 random images from the user's locally available COCO validation subset",
                    "split_protocol": "Frozen content-level dev/test before generation; no retrospective images",
                    "usage": "Calibration only; do not use these validation images as model-training data",
                    "created_utc": now()}
            validate_options(plan)
            validate_input(read_rows(source), plan["expected_images"], plan["image_root"])
            immutable_bytes(details / "inputs.jsonl", source.read_bytes())
            immutable_bytes(plan_path, (json.dumps(plan, ensure_ascii=False, indent=2) + "\n").encode())
        validate_options(plan)
        audit = validate_input(read_rows(details / "inputs.jsonl"), plan["expected_images"], plan["image_root"])
        immutable_bytes(details / "input_verification.json", (json.dumps(audit, indent=2) + "\n").encode())
    return root, plan


class CocoRunner(Runner):
    def save(self):
        self.state["updated_utc"] = now()
        write_json(self.details / "status.json", self.state)
        published = self.root / "results.json"
        if published.exists():
            self.final_published |= read_json(published).get("schema") == "grounding_calibration_results_v1"
        if self.final_published or self.state["stage"] == "analysis":
            return
        progress = {}
        for name, path in (("generation", self.details / "generation/status.json"),
                           ("decomposition", self.details / "decomposition/status.json"),
                           ("score", Path(str(self.details / "scored.jsonl") + ".journal") / "status.json"),
                           ("judge", Path(str(self.details / "judged.jsonl") + ".status.json"))):
            if path.exists():
                progress[name] = read_json(path)
        summary = {k: self.state[k] for k in ("status", "stage", "updated_utc")}
        summary.update(score_key="dino_max", scope="Frozen COCO calibration; reference labels are silver, not human GT",
                       details="details/", progress=progress, stages=self.state["stages"], thresholds=None)
        write_json(self.root / "results.json", summary)
        lines = ["# COCO 1000 图声明接地性校准", "",
                 f"状态：{self.state['status']}；阶段：{self.state['stage']}。", "",
                 "自动流程：原始 LLaVA-7B 描述 → DiscoSG 三轮拆分 → Grounding DINO 分数与独立视觉裁判 → 阈值校准 → 同批 FG-CLIP 基线。", "",
                 "本轮只校准，不进行模型训练。开发/测试图片已提前固定，不使用旧分数阈值预筛。", ""]
        for name, item in progress.items():
            completed = item.get("completed_records", item.get("cases_judged", item.get("completed_cases", "—")))
            total = item.get("total", item.get("cases_total", item.get("total_cases", "—")))
            lines += [f"- {name}：{item.get('status', 'running')}，{completed}/{total}。"]
        lines += ["", "全部声明、区域证据、裁判记录和日志保存在 details/；后续步骤自动执行。", ""]
        atomic_bytes(self.root / "results.md", "\n".join(lines).encode())

    def command(self, name, args, *, python=None):
        if python is None:
            return super().command(name, args)
        self.state["stage"] = name
        stage = self.state["stages"].setdefault(name, {})
        log_path = self.logs / (name + ".log")
        log = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update(PYTHONPATH=str(REPO) + os.pathsep + env.get("PYTHONPATH", ""),
                   OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        try:
            process = subprocess.Popen([str(python), "-u", *map(str, args)], cwd=REPO, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            raise
        stage.update(status="running", pid=process.pid, started_utc=now(), log=str(log_path),
                     interpreter=str(python))
        self.active[name] = (process, log)
        self.save()
        return process, log


def validate_scored(cases, rows, *, score_key):
    from mvpg_run.claim_feedback.diagnostics import validate_cases
    validate_cases(rows)
    if [r["case_id"] for r in rows] != [r["case_id"] for r in cases]:
        raise ValueError("Scores must cover every frozen case in its original order")
    for source, row in zip(cases, rows):
        if any(row.get(key) != value for key, value in source.items()
               if key not in ("scores", "score_provenance")):
            raise ValueError("Scorer changed frozen source fields")
        value = row.get("scores", {}).get(score_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"A frozen case lacks a finite {score_key} score")
        if not row.get("score_provenance", {}).get(score_key):
            raise ValueError("A scored case lacks scoring provenance")


def join_scored(root, *, scored_name="scored.jsonl", output_name="joined.jsonl", score_key="dino_max"):
    from mvpg_run.claim_feedback.diagnostics import JUDGE_LABELS, load_cases, validate_cases
    details = root / "details"
    cases = load_cases(details / "cases.jsonl")
    judged = load_cases(details / "judged.jsonl")
    scored = load_cases(details / scored_name)
    validate_scored(cases, scored, score_key=score_key)
    expected = {r["case_id"] for r in cases}
    by_judge = {r["case_id"]: r for r in judged}
    if set(by_judge) != expected or len(judged) != len(cases):
        raise ValueError("Visual judgments do not cover the complete frozen claim set")
    result = []
    for source, score in zip(cases, scored):
        judgment = by_judge[source["case_id"]]
        if any(judgment.get(key) != value for key, value in source.items()
               if key not in ("scores", "score_provenance", "judge_label")):
            raise ValueError("Visual judge changed frozen case fields")
        if judgment.get("judge_label") not in JUDGE_LABELS or judgment.get("judge_error"):
            raise ValueError("Failed or unfinished visual judgments cannot be silently removed")
        row = copy.deepcopy(judgment)
        # Keep detector boxes/provenance as well as scores, without allowing any
        # scorer field to overwrite the independently obtained judgment.
        for key, value in score.items():
            if key in ("scores", "score_provenance") or key not in source and key != "judge" and not key.startswith("judge_"):
                row[key] = copy.deepcopy(value)
        result.append(row)
    validate_cases(result)
    immutable_bytes(details / output_name, rows_bytes(result))
    return {"cases": len(result), "sha256": sha(details / output_name),
            "labels": dict(Counter(r["judge_label"] for r in result))}


def run_judge(runner):
    plan, details = runner.plan, runner.details
    output = details / "judged.jsonl"
    base_args = ["-m", "mvpg_run.claim_feedback.grounded_judge", "--input", details / "cases.jsonl",
                 "--output", output, "--model", plan["judge_model"], "--workers", str(plan["judge_workers"]),
                 "--timeout", str(plan["judge_timeout"]), "--max-retries", str(plan["judge_max_retries"]),
                 "--max-output-tokens", str(plan["judge_max_output_tokens"]),
                 "--image-root", plan["image_root"], "--image-detail", "original", "--id-encoding", "request_alias"]
    for attempt in range(plan["judge_max_passes"]):
        args = list(base_args)
        if Path(str(output) + ".manifest.json").exists():
            args.append("--resume")
        name = "judge" if attempt == 0 else f"judge_retry_{attempt}"
        try:
            runner.await_jobs({name: runner.command(name, args)})
        except RuntimeError:
            status = Path(str(output) + ".status.json")
            journal = Path(str(output) + ".images.progress.jsonl")
            if not status.exists() or not journal.exists():
                raise
            latest = {}
            for row in read_rows(journal):
                latest[row["image_key"]] = row
            if any(r.get("judge_fatal_error") for r in latest.values()):
                raise RuntimeError("Visual judge reported a fatal error; inspect its retained journal")
            if attempt + 1 == plan["judge_max_passes"]:
                raise RuntimeError("Visual judge exhausted bounded retries; resume retained progress after resolving the error")
            runner.state["stages"][name]["will_retry_failed_images_only"] = True
            runner.save()
            time.sleep(min(30, 5 * (attempt + 1)))
            continue
        status = read_json(str(output) + ".status.json")
        if status.get("status") != "complete" or status.get("cases_judged") != status.get("cases_total"):
            raise ValueError("Visual judge exited successfully without complete reference labels")
        runner.state["stages"]["judge"] = {"status": "complete", "passes_this_run": attempt + 1,
                                             "cases": status["cases_total"], "id_encoding": "request_alias"}
        runner.save()
        return


def analysis_args(details, root, plan, *, input_name="joined.jsonl", score_key="dino_max"):
    return ["run/analyze_grounding_calibration.py", "--input", details / input_name, "--output-dir", root,
            "--score-key", score_key, "--target-precision", str(plan["target_precision"]),
            "--min-selected", str(plan["min_selected"]), "--bootstrap-samples", str(plan["bootstrap_samples"]),
            "--seed", str(plan["seed"]), "--plots"]


def run(root, plan):
    details = root / "details"
    with stage_lock(details):
        runner = CocoRunner(root, plan)
        try:
            shared = ["--device", plan["device"], "--seed", str(plan["seed"]), "--image-root", plan["image_root"]]
            runner.corpus_stage("generation", ["run/build_grounding_corpus.py", "generate", "--input", details / "inputs.jsonl",
                "--output-dir", details / "generation", "--model-name", plan["model_name"], "--precision", "bf16",
                "--batch-size", str(plan["batch_size"]), "--max-new-tokens", str(plan["max_new_tokens"]),
                "--temperature", str(plan["temperature"]), "--top-p", str(plan["top_p"]), *shared])
            runner.corpus_stage("decomposition", ["run/build_grounding_corpus.py", "decompose", "--input", details / "generation/responses.jsonl",
                "--output-dir", details / "decomposition", "--disco-root", plan["disco_root"],
                "--seed-model", plan["seed_model"], "--refiner-model", plan["refiner_model"], "--rounds", "3",
                "--parser-batch-size", str(plan["parser_batch_size"]), *shared])
            runner.state["stage"] = "prepare_claims"; runner.save()
            runner.state["stages"]["prepare"] = {"status": "complete", **prepare(root, plan)}
            runner.save()
            score_args = ["run/score_grounding_dino.py", "--input", details / "cases.jsonl", "--output", details / "scored.jsonl",
                          "--model-root", plan["dino_root"], "--device", plan["device"], "--image-root", plan["image_root"],
                          "--batch-size", str(plan["dino_batch_size"])]
            if (Path(str(details / "scored.jsonl") + ".journal") / "manifest.json").exists():
                score_args.append("--resume")
            # The scorer runs in its own environment; mvpg's LLaVA/DiscoSG
            # dependencies are never upgraded to accommodate a detector.
            runner.await_jobs({"score": runner.command("score", score_args, python=plan["score_python"])})
            run_judge(runner)
            runner.state["stage"] = "join"; runner.save()
            runner.state["stages"]["join"] = {"status": "complete", **join_scored(root)}
            runner.await_jobs({"analysis": runner.command("analysis", analysis_args(details, root, plan))})
            main = read_json(root / "results.json")
            if main.get("schema") != "grounding_calibration_results_v1" or main.get("status") != "complete":
                raise ValueError("DINO analysis did not publish a complete result")
            runner.final_published = True
            baseline = details / "fgclip_baseline"
            baseline.mkdir(exist_ok=True)
            from run.run_grounding_calibration import validate_scores as validate_fgclip_scores
            # The old FG-CLIP validator uses the conventional scored.jsonl name.
            # Score into the baseline directory alongside a frozen cases copy.
            immutable_bytes(baseline / "cases.jsonl", (details / "cases.jsonl").read_bytes())
            if (baseline / "scored.jsonl").exists():
                runner.state["stages"]["fgclip_score"] = {"status": "complete", "reused_cases": validate_fgclip_scores(baseline, plan["fgclip_root"])}
            else:
                runner.await_jobs({"fgclip_score": runner.command("fgclip_score", ["-m", "mvpg_run.claim_feedback.diagnostics", "score",
                    "--input", baseline / "cases.jsonl", "--output", baseline / "scored.jsonl", "--fgclip-root", plan["fgclip_root"],
                    "--device", plan["device"], "--image-root", plan["image_root"], "--score-mode", "global", "--text-form", "raw"])})
                validate_fgclip_scores(baseline, plan["fgclip_root"])
            immutable_bytes(details / "fgclip_scored.jsonl", (baseline / "scored.jsonl").read_bytes())
            runner.state["stages"]["fgclip_join"] = {"status": "complete", **join_scored(root,
                scored_name="fgclip_scored.jsonl", output_name="fgclip_joined.jsonl", score_key="global_raw")}
            runner.await_jobs({"fgclip_analysis": runner.command("fgclip_analysis", analysis_args(details, baseline, plan,
                input_name="fgclip_joined.jsonl", score_key="global_raw"))})
            base = read_json(baseline / "results.json")
            if base.get("status") != "complete":
                raise ValueError("Same-case FG-CLIP baseline analysis is incomplete")
            main.update(baseline={"status": "complete", "score_key": "global_raw", "results": "details/fgclip_baseline/results.json",
                                  "same_cases_and_judge": True},
                        sample={"images": plan["expected_images"], "input_sha256": plan["input_sha256"], "usage": plan["usage"]})
            write_json(root / "results.json", main)
            text = (root / "results.md").read_text(encoding="utf-8")
            if "同批 FG-CLIP 基线" not in text:
                atomic_bytes(root / "results.md", (text.rstrip() + "\n\n同批 FG-CLIP 基线：[结果](details/fgclip_baseline/results.md)。"
                    "使用相同声明、相同裁判银标和相同冻结划分；图片仅用于校准，未训练模型。\n").encode())
            runner.state.update(status="complete", stage="complete", finished_utc=now())
            runner.save()
            return runner.state
        except BaseException as exc:
            runner.stop_owned_jobs()
            runner.state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                error=f"{type(exc).__name__}: {exc}")
            runner.save()
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="Defaults to RUN/details/sample/images.jsonl on first launch")
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--judge-model")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze and verify sample without starting inference")
    for key, value in DEFAULTS.items():
        names = ["--" + key.replace("_", "-")]
        if key == "score_python":
            names.append("--dino-python")
        parser.add_argument(*names, dest=key, type=type(value), default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Pipeline termination requested")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        load_environment(args.env_file)
        if not os.getenv("OPENAI_API_KEY", "").strip() and not args.prepare_only:
            raise ValueError("OPENAI_API_KEY is required for the independent visual-judge stage")
        root, plan = initialize(args)
        if args.prepare_only:
            result = {"status": "prepared", "run": str(root), "images": plan["expected_images"], "score_key": "dino_max"}
        else:
            result = run(root, plan)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(2, f"COCO grounding calibration: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
