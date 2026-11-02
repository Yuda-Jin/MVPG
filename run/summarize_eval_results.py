"""Validate full benchmark artifacts and summarize independent evaluation stages.

The original --log interface remains supported. The runner uses --logs-dir and
--status so a failed or skipped stage never borrows a score from another stage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

POPE_METRICS = ("accuracy", "precision", "recall", "f1", "yes_ratio")
CSV_COLUMNS = [
    "model_suffix", "mmhal_avg_score", "mmhal_hallucination_rate", "llava_bench_gpt_score",
    *[f"pope_{category}_{metric}" for category in ("random", "popular", "adv") for metric in POPE_METRICS],
    "amber_chair", "amber_cover", "amber_hal", "amber_cog",
    *[f"amber_{section}_{metric}" for section in ("discriminative", "existence", "attribute", "state", "number", "action", "relation")
      for metric in ("accuracy", "precision", "recall", "f1")],
    "amber_yes_no_parsing", "amber_invalid_discriminative_responses",
    *[f"object_hal_{mode}_{metric}" for mode in ("local", "gpt") for metric in ("chairs", "chairi", "chairs_refine")],
    "object_hal_chairs", "object_hal_chairi", "object_hal_chairs_refine", "object_hal_scoring",
    "all_benchmarks_scored",
]


def _read(path: str | Path, array=False):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream) if array else [json.loads(line) for line in stream if line.strip()]


def validate_artifact(kind: str, path: str, source: str):
    """Require exact order/IDs as several retained scorers align by position."""
    rows = _read(path, array=kind == "mmhal")
    questions = _read(source, array=kind in ("mmhal", "amber"))
    expected = {"mmhal": 96, "qa": 90, "qa_review": 90, "pope": 3000,
                "amber": 15220, "object": 300}[kind]
    if len(rows) != expected or len(questions) != expected:
        raise ValueError(f"{kind}: expected {expected} answers and source questions; got {len(rows)}, {len(questions)}")
    if kind in ("qa", "qa_review", "pope", "amber"):
        key = "id" if kind == "amber" else "question_id"
        actual_ids = [row[key] for row in rows]
        expected_ids = [row[key] for row in questions]
        if len(set(actual_ids)) != expected or actual_ids != expected_ids:
            raise ValueError(f"{kind}: duplicate, missing or out-of-order IDs")
    if kind == "mmhal":
        for row, question in zip(rows, questions):
            if any(row[key] != question[key] for key in ("question", "image_src", "gt_answer")):
                raise ValueError("MMHal source mismatch")
    elif kind == "object":
        if any(row["image_id"] != question["image_id"] or row["prompt"] != question["question"]
               for row, question in zip(rows, questions)):
            raise ValueError("Object HalBench source mismatch")
    if kind == "qa_review":
        for row in rows:
            pair = row.get("tuple")
            if not isinstance(pair, list) or len(pair) != 2 or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 10
                    for value in pair):
                raise ValueError("invalid qa90 judge score")
    else:
        field = {"mmhal": "model_answer", "amber": "response"}.get(kind, "text")
        if any(not isinstance(row.get(field), str) for row in rows):
            raise ValueError(f"{kind}: missing text response")
    return len(rows)


def read_status(path):
    stages = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            stages[row["stage"]] = row
    return stages


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(path, model, adapter, judge, sources):
    adapter = Path(adapter).resolve()
    names = ["adapter_config.json"]
    weights = next((name for name in ("adapter_model.safetensors", "adapter_model.bin")
                    if (adapter / name).is_file()), None)
    if not weights or not (adapter / names[0]).is_file():
        raise ValueError(f"incomplete adapter: {adapter}")
    names.append(weights)
    manifest = {
        "model": str(Path(model).resolve()), "adapter": str(adapter), "judge_model": judge,
        "judge_endpoint": os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL"),
        "judge_thinking": os.getenv("JUDGE_THINKING"),
        "batch_size": os.getenv("EVAL_BATCH_SIZE", "1"),
        "adapter_sha256": {name: _hash(adapter / name) for name in names},
        "sources_sha256": {str(Path(source).resolve()): _hash(source) for source in sources},
    }
    path = Path(path)
    if path.exists():
        if _read(path, array=True) != manifest:
            raise ValueError("Evaluation manifest differs (adapter, judge or question data). Use a new MODEL_SUFFIX/EVAL_ROOT.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_text(text):
    result = {}
    patterns = {
        "mmhal_avg_score": r"Average score:\s*([\d.]+)",
        "mmhal_hallucination_rate": r"Hallucination rate:\s*([\d.]+)",
        "llava_bench_gpt_score": r"^all\s+([\d.]+)\s*$",
        "pope_adv_accuracy": r"^Accuracy:\s*([\d.]+)\s*$",
        "pope_adv_precision": r"^Precision:\s*([\d.]+)\s*$",
        "pope_adv_recall": r"^Recall:\s*([\d.]+)\s*$",
        "pope_adv_f1": r"^F1 score:\s*([\d.]+)\s*$",
        "pope_adv_yes_ratio": r"^Yes ratio:\s*([\d.]+)\s*$",
        "amber_chair": r"^CHAIR:\s*([\d.]+)\s*$",
        "amber_cover": r"^Cover:\s*([\d.]+)\s*$",
        "amber_hal": r"^Hal:\s*([\d.]+)\s*$",
        "amber_cog": r"^Cog:\s*([\d.]+)\s*$",
        "object_hal_chairs": r"^CHAIRs:\s*([\d.]+)\s*$",
        "object_hal_chairi": r"^CHAIRi:\s*([\d.]+)\s*$",
        "object_hal_chairs_refine": r"^CHAIRs_refine:\s*([\d.]+)\s*$",
    }
    for key, pattern in patterns.items():
        values = re.findall(pattern, text, re.M)
        if values:
            result[key] = float(values[-1])
    return result


def parse_log(path):
    return parse_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def parse_amber_sections(text):
    result, section = {}, None
    headings = {"Descriminative Task:": "discriminative", "Discriminative Task:": "discriminative",
                "Exsitence:": "existence", "Existence:": "existence", "Attribute:": "attribute",
                "State:": "state", "Number:": "number", "Action:": "action", "Relation:": "relation"}
    for line in text.splitlines():
        line = line.strip()
        if line in headings:
            section = headings[line]
        match = re.fullmatch(r"(Accuracy|Precision|Recall|F1):\s*([\d.]+)", line)
        if section and match:
            result[f"amber_{section}_{match[1].lower()}"] = float(match[2])
        if line.startswith("Yes/No parsing: "):
            result["amber_yes_no_parsing"] = line.split(": ", 1)[1]
        if line.startswith("Invalid discriminative responses: "):
            result["amber_invalid_discriminative_responses"] = int(line.split(": ", 1)[1])
    return result


def validate_score(stage, path):
    parsed = parse_log(path)
    if stage == "llava_review":
        return Path(path).is_file()  # The caller separately validates all 90 review rows.
    if stage == "mmhal_score":
        required = ("mmhal_avg_score", "mmhal_hallucination_rate")
    elif stage == "llava_score":
        required = ("llava_bench_gpt_score",)
    elif stage.startswith("pope_"):
        required = tuple(f"pope_adv_{metric}" for metric in POPE_METRICS)
    elif stage == "amber_score":
        parsed.update(parse_amber_sections(Path(path).read_text(encoding="utf-8", errors="replace")))
        required = ("amber_chair", "amber_cover", "amber_hal", "amber_cog", *(
            f"amber_{section}_{metric}" for section in ("discriminative", "existence", "attribute", "relation")
            for metric in ("accuracy", "precision", "recall", "f1")))
    elif stage.startswith("object_score_"):
        required = tuple(f"object_hal_{metric}" for metric in ("chairs", "chairi", "chairs_refine"))
    else:
        raise ValueError(f"unknown score stage: {stage}")
    return all(key in parsed and math.isfinite(parsed[key]) for key in required)


def collect_stages(log_dir, status_path):
    stages, result = read_status(status_path), {}
    def complete(name):
        return stages.get(name, {}).get("status") == "completed"
    def text(name):
        return (Path(log_dir) / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    for name, prefix in (("mmhal_score", "mmhal_"), ("llava_score", "llava_")):
        if complete(name):
            result.update({key: value for key, value in parse_text(text(name)).items() if key.startswith(prefix)})
    for category in ("random", "popular", "adversarial"):
        name = f"pope_{category}_score"
        if complete(name):
            for metric in POPE_METRICS:
                key = f"pope_adv_{metric}"
                parsed = parse_text(text(name))
                if key in parsed:
                    short = "adv" if category == "adversarial" else category
                    result[f"pope_{short}_{metric}"] = parsed[key]
    if complete("amber_score"):
        amber_text = text("amber_score")
        result.update({key: value for key, value in parse_text(amber_text).items() if key.startswith("amber_")})
        result.update(parse_amber_sections(amber_text))
    for mode in ("local", "gpt"):
        name = f"object_score_{mode}"
        if complete(name):
            parsed = parse_text(text(name))
            for metric in ("chairs", "chairi", "chairs_refine"):
                key = f"object_hal_{metric}"
                if key in parsed:
                    result[f"object_hal_{mode}_{metric}"] = parsed[key]
                    result[key] = parsed[key]
            result["object_hal_scoring"] = mode
    required = ["mmhal_score", "llava_score", "pope_random_score", "pope_popular_score",
                "pope_adversarial_score", "amber_score", "object_score_gpt"]
    result["all_benchmarks_scored"] = all(complete(name) for name in required)
    return result, stages


def upsert_csv(out_root, model_suffix, scores):
    path = Path(out_root) / "summary.csv"
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row.get("model_suffix") != model_suffix]
    rows.append({"model_suffix": model_suffix, **scores})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", help="Legacy combined log (single POPE category only)")
    parser.add_argument("--logs-dir")
    parser.add_argument("--status")
    parser.add_argument("--out-root")
    parser.add_argument("--model-suffix")
    parser.add_argument("--validate", nargs=3, metavar=("KIND", "ARTIFACT", "SOURCE"))
    parser.add_argument("--validate-score", nargs=2, metavar=("STAGE", "LOG"))
    parser.add_argument("--stage-completed", nargs=2, metavar=("STATUS", "STAGE"))
    parser.add_argument("--prepare-run")
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--judge-model")
    parser.add_argument("--sources", nargs="+")
    args = parser.parse_args()
    if args.validate:
        try:
            print(validate_artifact(*args.validate))
            return 0
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(str(error), file=sys.stderr)
            return 1
    if args.validate_score:
        try:
            return 0 if validate_score(*args.validate_score) else 1
        except (OSError, ValueError):
            return 1
    if args.stage_completed:
        try:
            path, stage = args.stage_completed
            return 0 if read_status(path).get(stage, {}).get("status") == "completed" else 1
        except OSError:
            return 1
    if args.prepare_run:
        if not all((args.model, args.adapter, args.judge_model, args.sources)):
            parser.error("--prepare-run needs --model, --adapter, --judge-model and --sources")
        try:
            prepare_run(args.prepare_run, args.model, args.adapter, args.judge_model, args.sources)
            return 0
        except (OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
    if not args.out_root or not args.model_suffix:
        parser.error("summarization needs --out-root and --model-suffix")
    stages = {}
    if args.logs_dir and args.status:
        scores, stages = collect_stages(args.logs_dir, args.status)
    elif args.log:
        scores = parse_log(args.log)
    else:
        parser.error("supply --log or both --logs-dir and --status")
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    path = Path(args.out_root) / f"summary_{args.model_suffix}.json"
    path.write_text(json.dumps({"model_suffix": args.model_suffix, **scores, "stages": stages}, indent=2) + "\n", encoding="utf-8")
    csv_path = upsert_csv(args.out_root, args.model_suffix, scores)
    for key in CSV_COLUMNS[1:]:
        if key in scores:
            print(f"{key}: {scores[key]}")
    for stage, row in stages.items():
        if row["status"] != "completed":
            print(f"{stage}: {row['status']} ({row['message']})")
    print(f"Summary JSON: {path}")
    print(f"Summary CSV: {csv_path}")
    return 1 if any(row["status"] in ("failed", "blocked") for row in stages.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
