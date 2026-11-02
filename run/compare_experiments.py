"""Maintain sourced paper results and local experiments without mixing protocols.

The editable reference data is versioned under run/reference_results. Generated
registry.json and comparison.md live under output/experiment_comparison.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

SCHEMA_VERSION = 1
DEFAULT_REFERENCE = Path(__file__).resolve().parent / "reference_results/opa_dpo.json"
GENERATIVE_COLUMNS = (
    ("object_hal_chairs", "ObjectHal CHAIRs↓"),
    ("object_hal_chairi", "ObjectHal CHAIRi↓"),
    ("mmhal_avg_score", "MMHal Score↑"),
    ("mmhal_hallucination_rate", "MMHal Hal↓ (%)"),
    ("amber_chair", "AMBER CHAIR↓"),
    ("amber_cover", "Cover↑"), ("amber_hal", "Hal↓"), ("amber_cog", "Cog↓"),
)
DISCRIMINATIVE_COLUMNS = (
    ("pope_adv_accuracy", "POPE adv Acc↑ (%)"),
    ("pope_adv_precision", "Precision↑ (%)"),
    ("pope_adv_f1", "F1↑ (%)"),
    ("pope_random_f1", "random F1↑ (%)"),
    ("pope_popular_f1", "popular F1↑ (%)"),
    ("amber_discriminative_f1", "AMBER 判别 F1↑"),
    ("llava_bench_gpt_score", "qa90 相对分数↑"),
)
REQUIRED_SCORE_STAGES = {"mmhal_score", "llava_score", "pope_random_score",
    "pope_popular_score", "pope_adversarial_score", "amber_score", "object_score_gpt"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_snapshot(path):
    """Bind parsed progress and its digest to one atomic manifest snapshot."""
    payload = Path(path).read_bytes()
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def registry_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".registry.lock"
    deadline = time.monotonic() + 10
    while True:
        try:
            handle = path.open("x", encoding="utf-8")
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Comparison registry is locked: {path}. Check the previous writer before removing a stale lock.")
            time.sleep(0.1)
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def validate_row(row):
    if not isinstance(row, dict):
        raise ValueError("comparison rows must be JSON objects")
    for key in ("run_id", "method", "model", "judge", "status"):
        row[key] = _nonempty(row.get(key), key)
    if row.get("source_type") not in ("paper_reported", "local_run"):
        raise ValueError("source_type must be paper_reported or local_run")
    steps = row.get("steps")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 0):
        raise ValueError("steps must be null or a nonnegative successful-update count")
    if not isinstance(row.get("metrics"), dict):
        raise ValueError("metrics must be an object")
    for key, value in row["metrics"].items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"metric {key} must be a finite number or null")
        if value is not None and (key.startswith("pope_") or key == "mmhal_hallucination_rate") and not 0 <= value <= 1:
            raise ValueError(f"{key} uses a 0..1 ratio, not a percentage")
    if not isinstance(row.get("protocol", {}), dict) or not isinstance(row.get("source", {}), dict):
        raise ValueError("protocol and source must be objects")
    if row.get("protocol", {}).get("evaluation_status") == "not_requested":
        if row["metrics"] or row["judge"] != "not_evaluated":
            raise ValueError("Training-only rows must have empty metrics and judge not_evaluated")
        if row["status"] not in ("training_running", "trained_not_evaluated", "failed", "incomplete"):
            raise ValueError("Invalid training-only status")
    return row


def upsert(rows, incoming, *, allow_incomplete_update=False):
    """Never silently relabel an experiment or downgrade a complete result."""
    incoming = validate_row(incoming)
    result = list(rows)
    for index, old in enumerate(result):
        if old["run_id"] != incoming["run_id"]:
            continue
        training_only = incoming.get("protocol", {}).get("evaluation_status") == "not_requested"
        was_training_only = old.get("protocol", {}).get("evaluation_status") == "not_requested"
        if training_only != was_training_only:
            raise ValueError("Training-only and evaluated records require a new run_id")
        for key in ("source_type", "method", "model", "judge"):
            if old[key] != incoming[key]:
                raise ValueError(f"run_id {incoming['run_id']!r} already has a different {key}; use a new run_id")
        for key in ("judge_thinking", "judge_endpoint", "batch_size", "sources_sha256", "adapter_sha256",
                    "training_config_sha256"):
            previous = old.get("protocol", {}).get(key)
            current = incoming.get("protocol", {}).get(key)
            if previous is not None and current is not None and previous != current:
                raise ValueError(f"run_id {incoming['run_id']!r} changed protocol {key}; use a new run_id")
        if training_only:
            if old.get("source", {}).get("training_manifest") != incoming.get("source", {}).get("training_manifest"):
                raise ValueError("Training manifest changed; use a new run_id")
            old_counts = old.get("protocol", {}).get("training_counts", {})
            new_counts = incoming.get("protocol", {}).get("training_counts", {})
            if any(new_counts.get(key, -1) < old_counts.get(key, 0) for key in ("attempted", "updated", "skipped")):
                raise ValueError("Refusing to decrease recorded training progress")
            if old["status"] == "trained_not_evaluated" and incoming["status"] != old["status"]:
                raise ValueError("Completed training cannot revert to an unfinished status; use a new run_id")
        if old["status"] == "complete" and incoming["status"] != "complete" and not allow_incomplete_update:
            raise ValueError("Refusing to replace a complete run with an incomplete result; use a new run_id or --allow-incomplete-update")
        # Keep earlier snapshots if a completed/partial result is deliberately updated.
        old_payload = {key: value for key, value in old.items() if key != "history"}
        new_payload = {key: value for key, value in incoming.items() if key != "history"}
        history = list(old.get("history", []))
        if old_payload != new_payload:
            history.append(old_payload)
        if history:
            incoming = dict(incoming, history=history)
        result[index] = incoming
        return result
    result.append(incoming)
    return result


def _safe_endpoint(value):
    if not value:
        return None
    parts = urlsplit(value)
    # Endpoint identity is useful; credentials and URL queries are not result data.
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def training_row(training_manifest, *, run_id, method, model, steps=None,
                 training_data=None, feedback_source=None):
    """Register training progress without presenting it as benchmark evidence."""
    path = Path(training_manifest).resolve()
    manifest, digest = read_json_snapshot(path)
    if not isinstance(manifest, dict):
        raise ValueError("training manifest must be an object")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("training manifest needs actual successful-update counts")
    for key in ("attempted", "updated", "skipped"):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"training count {key} must be a nonnegative integer")
    if counts["attempted"] != counts["updated"] + counts["skipped"]:
        raise ValueError("training attempted count must equal updates plus skips")
    actual = counts["updated"]
    if steps is not None and steps != actual:
        raise ValueError("steps must equal actual successful updates in training manifest")
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("training manifest needs settings")
    target = settings.get("max_updates")
    if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target < 0):
        raise ValueError("max_updates must be null or a nonnegative integer")
    raw_status = manifest.get("status")
    if raw_status == "running":
        status = "training_running"
    elif raw_status == "failed":
        status = "failed"
    elif raw_status == "complete" and manifest.get("target_updates_reached") is not False and (target is None or actual >= target):
        status = "trained_not_evaluated"
    else:
        # A checkpoint alone is not evidence that a whole training job finished.
        status = "incomplete"
    pool = manifest.get("pool") if isinstance(manifest.get("pool"), dict) else {}
    pool_sha = manifest.get("pool_sha256", pool.get("sha256"))
    pool_records = manifest.get("pool_records", pool.get("records"))
    identity = {"settings": settings, "pool_sha256": pool_sha, "pool_records": pool_records,
                "backend": manifest.get("backend"), "signature": manifest.get("signature"),
                "resume": manifest.get("resume", manifest.get("resume_info")),
                "counts_at_start": manifest.get("counts_at_start"),
                "started_utc": manifest.get("started_utc"),
                "protocol": manifest.get("protocol"),
                "checkpoint_step_unit": manifest.get("checkpoint_step_unit")}
    config_digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    if method == "claim_feedback" and settings.get("mode"):
        method += "/" + str(settings["mode"])
    protocol = {"evaluation_status": "not_requested", "directly_comparable_to_paper": False,
        "training_status": raw_status, "training_target_updates": target,
        "training_settings": settings, "training_counts": counts,
        "training_pool_sha256": pool_sha, "training_pool_records": pool_records,
        "training_config_sha256": config_digest,
        "comparability_note": "Training progress only; benchmark metrics have not been measured for this run."}
    if training_data:
        protocol["training_data"] = training_data
    if feedback_source:
        protocol["training_feedback_reported"] = feedback_source
    return validate_row({"run_id": run_id, "source_type": "local_run", "method": method,
        "model": model, "steps": actual, "judge": "not_evaluated", "status": status,
        "metrics": {}, "protocol": protocol,
        "source": {"training_manifest": str(path), "training_manifest_sha256": digest}})


def local_row(summary_path, *, run_id, method, model, judge, steps=None,
              training_manifest=None, evaluation_manifest=None, mmhal_reviews=None,
              training_data=None, feedback_source=None):
    summary_path = Path(summary_path).resolve()
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("evaluation summary must be an object")
    metrics = {key: value for key, value in summary.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)}
    stages = summary.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("summary stages must be an object")
    complete = summary.get("all_benchmarks_scored") is True and REQUIRED_SCORE_STAGES.issubset(stages) and all(
        stage.get("status") == "completed" for stage in stages.values())
    protocol = {"directly_comparable_to_paper": False,
                "amber_yes_no_parsing": summary.get("amber_yes_no_parsing"),
                "object_hal_scoring": summary.get("object_hal_scoring"),
                "comparability_note": "Paper results are references, not this project's reproduction; check judges, datasets, extraction and inference settings before comparison."}
    source = {"summary": str(summary_path), "summary_sha256": file_hash(summary_path)}
    if training_data:
        protocol["training_data"] = training_data
    if feedback_source:
        protocol["training_feedback_reported"] = feedback_source
    if evaluation_manifest is None:
        suffix = summary.get("model_suffix", run_id)
        candidate = summary_path.parent / "logs" / suffix / "manifest.json"
        evaluation_manifest = candidate if candidate.is_file() else None
    if evaluation_manifest is not None:
        manifest = read_json(evaluation_manifest)
        if manifest.get("judge_model") != judge:
            raise ValueError("judge differs from the evaluation manifest")
        for key in ("batch_size", "judge_thinking", "sources_sha256", "adapter_sha256"):
            protocol[key] = manifest.get(key)
        protocol["judge_endpoint"] = _safe_endpoint(manifest.get("judge_endpoint"))
        protocol["model_path"] = manifest.get("model")
        source.update(evaluation_manifest=str(Path(evaluation_manifest).resolve()),
                      evaluation_manifest_sha256=file_hash(evaluation_manifest))
    if training_manifest is not None:
        manifest, manifest_digest = read_json_snapshot(training_manifest)
        actual = manifest.get("counts", {}).get("updated")
        if actual is not None:
            if steps is not None and steps != actual:
                raise ValueError("steps must equal actual successful updates in training manifest")
            steps = actual
        protocol["training_status"] = manifest.get("status")
        protocol["training_pool_sha256"] = manifest.get("pool_sha256")
        protocol["training_settings"] = manifest.get("settings")
        protocol["training_counts"] = manifest.get("counts")
        protocol["training_pool_records"] = manifest.get("pool_records")
        mode = (manifest.get("settings") or {}).get("mode")
        if method == "claim_feedback" and mode:
            method += "/" + str(mode)
        source.update(training_manifest=str(Path(training_manifest).resolve()),
                      training_manifest_sha256=manifest_digest)
    # Prefer original 96 judgments over two-decimal log output when available.
    if mmhal_reviews is None and summary.get("model_suffix"):
        candidate = summary_path.parent / "mmhal_bench/review" / f"review-file-{summary['model_suffix']}.jsonl"
        mmhal_reviews = candidate if candidate.is_file() else None
    if mmhal_reviews is not None:
        reviews = read_json(mmhal_reviews)
        if not isinstance(reviews, list) or len(reviews) != 96:
            raise ValueError("exact MMHal metrics need all 96 ordered reviews")
        ratings = []
        for review in reviews:
            found = {int(x) for x in re.findall(r"\brating\s*:\s*([0-6])\b", review, re.I)} if isinstance(review, str) else set()
            if len(found) != 1:
                raise ValueError("MMHal review has no unambiguous rating")
            ratings.append(found.pop())
        if metrics.get("mmhal_avg_score") is not None:
            metrics.update(mmhal_avg_score=sum(ratings) / 96,
                           mmhal_hallucination_rate=sum(value < 3 for value in ratings) / 96)
            protocol["mmhal_metrics_precision"] = "recomputed from 96 original reviews"
            source.update(mmhal_reviews=str(Path(mmhal_reviews).resolve()),
                          mmhal_reviews_sha256=file_hash(mmhal_reviews))
    return validate_row({"run_id": run_id, "source_type": "local_run", "method": method,
        "model": model, "steps": steps, "judge": judge,
        "status": "complete" if complete else "incomplete", "metrics": metrics,
        "protocol": protocol, "source": source})


def display_metric(row, key):
    value = row["metrics"].get(key)
    if value is None:
        return "-"
    if key.startswith("pope_") or key == "mmhal_hallucination_rate":
        value *= 100
    return f"{value:.2f}"


def _cell(value):
    return str(value if value is not None else "-").replace("|", "\\|").replace("\n", " ")


def _table(rows, columns):
    lines = ["| 实验 / 方法 | " + " | ".join(label for _, label in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    for row in rows:
        lines.append("| " + _cell(row.get("_table_label", row["run_id"] + " / " + row["method"])) + " | "
                     + " | ".join(display_metric(row, key) for key, _ in columns) + " |")
    return lines


def render_markdown(registry):
    lines = ["# 实验对照表", "", "由 `run/compare_experiments.py` 维护；完整原始值、来源路径、SHA256 和设置保存在同目录 `registry.json`。",
        "", "论文原报与本项目实测分开列出，论文行不代表本项目复现。裁判、问题集、对象提取、精度或批量设置不同的结果不能直接比较。",
        "", "`-` 表示未报告/未评测，不是 0。除 MMHal Score（0–6）和 qa90 相对分数外，表中数值为百分数；JSON 内的 POPE 指标和 MMHal 幻觉率仍为 0–1 比例。",
        "", "训练反馈来源与评测裁判是两个字段。ObjectHal 沿用每行原设置，不能把本地提取和大模型辅助提取混为同一协议。"]
    counters = {"paper_reported": 0, "local_run": 0}
    for kind, title in (("paper_reported", "论文原报（未复现）"), ("local_run", "本项目实测")):
        rows = []
        for original in registry["rows"]:
            if original["source_type"] != kind:
                continue
            row = dict(original)
            counters[kind] += 1
            ref = ("P" if kind == "paper_reported" else "L") + str(counters[kind])
            marker = row.get("protocol", {}).get("provenance_marker", "")
            marker = "" if marker == "own" else marker
            row["_ref"] = ref
            row["_table_label"] = f"{row['method']}{marker} [{ref}]"
            if kind == "local_run" and row.get("steps") is not None:
                row["_table_label"] = f"{row['method']} · {row['steps']}步 [{ref}]"
            rows.append(row)
        lines += ["", "## " + title, ""]
        if not rows:
            lines += ["尚无已核对记录。"]
            continue
        if kind == "paper_reported":
            source = rows[0].get("source", {})
            lines += [f"来源：Yang 等论文表 {source.get('table', '-')}，PDF 第 {source.get('pdf_page', '-')} 页（印刷页 {source.get('printed_page', '-')}）。",
                "", "§ 为作者对可用检查点的复测；† / ‡ / ⋆ 分别引用原论文 [19] / [17] / [27] 的数字。OPA / OPA-DPO 为该论文自己的实验，均不是本项目复现。",
                "", "论文未给出 MMHal GPT-4 的具体快照；ObjectHal 的具体对象提取/裁判模型未说明。表 2 只报告 AMBER 生成式和 POPE adversarial 的 Accuracy / Precision，没有 POPE F1、random/popular、AMBER 判别式或 qa90。"]
            recipes = [r.get("protocol", {}).get("training_recipe_reported", {}) for r in rows]
            opa = next((x["OPA"] for x in recipes if "OPA" in x), None)
            dpo = next((x["DPO"] for x in recipes if "DPO" in x), None)
            if opa:
                lines += ["", f"论文 OPA 设置：{opa['epochs']} epochs、全局 batch {opa['global_batch_size']}、lr {opa['learning_rate']:g}、{opa['lr_schedule']}、LoRA r={opa['lora_rank']} / alpha={opa['lora_alpha']}。"]
            if dpo:
                lines += [f"DPO 阶段：{dpo['epochs']} epochs、全局 batch {dpo['global_batch_size']}、lr {dpo['learning_rate']:g}、beta={dpo['beta']:g}、gamma_IF={dpo['gamma_IF']:g}、gamma_Anc={dpo['gamma_Anc']:g}、delta={dpo['delta']:g}、图像 mask {dpo['image_mask_ratio']*100:g}%。论文未报告精确优化器步数，不按 epoch 推算。"]
        for model in dict.fromkeys(row["model"] for row in rows):
            group = [row for row in rows if row["model"] == model]
            lines += ["", "### " + _cell(model), "", "| 方法 / ID | 训练数据预算 | 训练反馈 | 实际更新步数 | 状态 |",
                      "|---|---|---|---:|---|"]
            for row in group:
                protocol = row.get("protocol", {})
                budget = None
                if protocol.get("training_samples_reported") is not None:
                    budget = f"{protocol['training_samples_reported']:,}（原报）"
                elif protocol.get("training_pool_records") is not None:
                    budget = f"回答池 {protocol['training_pool_records']}"
                status = "论文原报" if kind == "paper_reported" else {
                    "complete": "已完成", "training_running": "训练中·未评测",
                    "trained_not_evaluated": "训练完成·未评测", "failed": "训练失败·未评测",
                }.get(row["status"], "未完成")
                values = [row["_table_label"], budget, protocol.get("training_feedback_reported"), row.get("steps"), status]
                lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
            lines += ["", "生成式幻觉指标：", "", *_table(group, GENERATIVE_COLUMNS),
                      "", "判别与一般能力指标：", "", *_table(group, DISCRIMINATIVE_COLUMNS), "", "来源与设置：", ""]
            for row in group:
                source, protocol = row.get("source", {}), row.get("protocol", {})
                if kind == "paper_reported":
                    note = protocol.get("provenance_note", "论文报告")
                    lines.append(f"- [{row['_ref']}] {_cell(row['method'])}：表 {source.get('table', '-')}，PDF 第 {source.get('pdf_page', '-')} 页；{_cell(note)}。裁判：{_cell(row['judge'])}。")
                else:
                    training_only = protocol.get("evaluation_status") == "not_requested"
                    lines += [f"- [{row['_ref']}] 实验 ID：`{_cell(row['run_id'])}`；裁判：`{_cell(row['judge'])}`。"]
                    if training_only:
                        lines.append(f"  数据：{_cell(protocol.get('training_data'))}。本次仅训练，尚未进行 benchmark 评测；目标累计更新：{_cell(protocol.get('training_target_updates'))}。")
                    else:
                        lines.append(f"  数据：{_cell(protocol.get('training_data'))}。评测 batch 设置：{_cell(protocol.get('batch_size'))}；thinking：{_cell(protocol.get('judge_thinking'))}；AMBER 解析：{_cell(protocol.get('amber_yes_no_parsing'))}；ObjectHal 提取：{_cell(protocol.get('object_hal_scoring'))}。")
                    counts = protocol.get("training_counts") or {}
                    if counts:
                        lines.append(f"  训练计数：尝试 {counts.get('attempted', '-')}、实际更新 {counts.get('updated', '-')}、跳过 {counts.get('skipped', '-')}。")
                    settings = protocol.get("training_settings") or {}
                    if settings:
                        lines.append(f"  训练设置：mode={_cell(settings.get('mode'))}，lr={_cell(settings.get('lr'))}，更新范数上限={_cell(settings.get('max_update_norm'))}，seed={_cell(settings.get('seed'))}。")
                    evidence = source.get("training_manifest") if training_only else source.get("summary")
                    lines.append(f"  证据：`{_cell(Path(evidence or '-').name)}`；完整来源与设置见 registry.json 的 `{_cell(row['run_id'])}` 记录。")
    return "\n".join(lines) + "\n"


def update_registry(output_dir, incoming=None, *, reference_file=DEFAULT_REFERENCE,
                    allow_incomplete_update=False):
    directory = Path(output_dir)
    with registry_lock(directory):
        path = directory / "registry.json"
        registry = read_json(path) if path.exists() else {"schema_version": SCHEMA_VERSION, "rows": []}
        if registry.get("schema_version") != SCHEMA_VERSION or not isinstance(registry.get("rows"), list):
            raise ValueError("unsupported comparison registry schema")
        rows = [validate_row(row) for row in registry["rows"]]
        if len({row["run_id"] for row in rows}) != len(rows):
            raise ValueError("duplicate run_id in comparison registry")
        if reference_file is not None:
            reference = read_json(reference_file)
            if reference.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported reference schema")
            for row in reference.get("rows", []):
                if row.get("source_type") != "paper_reported":
                    raise ValueError("reference rows must be paper_reported")
                rows = upsert(rows, row)
            registry["reference_source"] = {"path": str(Path(reference_file).resolve()),
                                            "sha256": file_hash(reference_file)}
        if incoming is not None:
            rows = upsert(rows, incoming, allow_incomplete_update=allow_incomplete_update)
        registry["rows"] = rows
        registry["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_text(path, json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_text(directory / "comparison.md", render_markdown(registry))
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary")
    mode.add_argument("--training-only", action="store_true", help="Record actual training progress with no benchmark scores")
    parser.add_argument("--run-id")
    parser.add_argument("--method")
    parser.add_argument("--model")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--judge")
    parser.add_argument("--training-manifest")
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--mmhal-reviews")
    parser.add_argument("--training-data", help="Descriptive dataset/protocol label; does not change training")
    parser.add_argument("--feedback-source", help="Training feedback source, separate from the evaluation judge")
    parser.add_argument("--reference-file", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--output-dir", default="output/experiment_comparison")
    parser.add_argument("--allow-incomplete-update", action="store_true")
    args = parser.parse_args()
    if args.summary and not all((args.run_id, args.method, args.model, args.judge)):
        parser.error("--summary requires --run-id, --method, --model and --judge")
    if args.training_only:
        if not all((args.run_id, args.method, args.model, args.training_manifest)):
            parser.error("--training-only requires --run-id, --method, --model and --training-manifest")
        if args.judge is not None or args.evaluation_manifest is not None or args.mmhal_reviews is not None:
            parser.error("--training-only does not accept evaluation or judge arguments")
    try:
        if args.training_only:
            incoming = training_row(args.training_manifest, run_id=args.run_id, method=args.method,
                model=args.model, steps=args.steps, training_data=args.training_data,
                feedback_source=args.feedback_source)
        else:
            incoming = local_row(args.summary, run_id=args.run_id, method=args.method,
                model=args.model, judge=args.judge, steps=args.steps,
                training_manifest=args.training_manifest, evaluation_manifest=args.evaluation_manifest,
                mmhal_reviews=args.mmhal_reviews, training_data=args.training_data,
                feedback_source=args.feedback_source) if args.summary else None
        registry = update_registry(args.output_dir, incoming, reference_file=args.reference_file,
                                   allow_incomplete_update=args.allow_incomplete_update)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Comparison update failed: {error}", file=sys.stderr)
        return 1
    print(f"Comparison rows: {len(registry['rows'])}; table: {Path(args.output_dir) / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
