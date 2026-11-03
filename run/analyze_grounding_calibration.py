"""Select FG-CLIP thresholds on formal dev images and evaluate frozen test images.

The reference is an image-reading model's silver labels, not human ground truth.
Only explicitly marked formal cases enter selection; retrospective cases remain
in corpus accounting. No model inference or external requests happen here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mvpg_run.claim_feedback.diagnostic_metrics import analyze_cases

LABELS = ("supported", "unsupported", "uncertain")
ROLES = ("formal", "retrospective")


def require(value, message):
    if not value:
        raise ValueError(message)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                prefix="." + path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_cases(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def counts(rows, score_key):
    labels = Counter(row.get("judge_label") for row in rows)
    return {
        "n_cases": len(rows), "n_images": len({row["image_group"] for row in rows}),
        **{label: labels[label] for label in LABELS},
        "unlabeled_or_invalid": sum(value for key, value in labels.items() if key not in LABELS),
        "judge_error_cases": sum(bool(row.get("judge_error")) for row in rows),
        "scored_cases": sum(isinstance(row.get("scores", {}).get(score_key), (int, float))
            and not isinstance(row["scores"][score_key], bool)
            and math.isfinite(row["scores"][score_key]) for row in rows),
    }


def formal_rows(cases, score_key):
    """Validate accounting and scope before deriving any threshold or output."""
    require(isinstance(score_key, str) and score_key, "A score key is required")
    require(cases, "No input cases")
    seen, image_roles = set(), {}
    for row in cases:
        require(isinstance(row, dict), "Cases must be JSON objects")
        for key in ("case_id", "record_id", "image_group"):
            require(isinstance(row.get(key), str) and row[key].strip(), f"Missing case field: {key}")
        require(row["case_id"] not in seen, f"Duplicate case_id: {row['case_id']}")
        seen.add(row["case_id"])
        role = row.get("analysis_role")
        require(role in ROLES, "Every case needs analysis_role=formal or retrospective")
        previous = image_roles.setdefault(row["image_group"], role)
        require(previous == role, "An image group crosses formal and retrospective roles")
        require(isinstance(row.get("scores"), dict), "Every case needs a scores object")
        if role == "formal":
            require(row.get("judge_label") in LABELS and not row.get("judge_error"),
                    f"Formal case lacks a successful three-state judgment: {row['case_id']}")
            value = row["scores"].get(score_key)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                    f"Formal case lacks a finite {score_key} score: {row['case_id']}")
    # analyze_cases subsequently enforces dev/test image separation, record
    # identity, and nonempty dev/test sets. Retrospective rows are never passed.
    return [row for row in cases if row["analysis_role"] == "formal"]


def _class_counts(rows):
    return Counter(row["judge_label"] for row in rows)


def _missing_class(labels):
    missing = [label for label in ("supported", "unsupported") if not labels[label]]
    return "missing_" + "_and_".join(missing) if missing else None


def select_negative_f1(dev, score_key):
    """O(n log n) exact tie-group selection, with no access to test cases.

    F1 and precision use determinate labels only. Final ties prefer fewer total
    selected cases, including uncertain cases. Fractions avoid floating-point
    tie-breaking artifacts. All equal scores enter or leave as a group.
    """
    labels = _class_counts(dev)
    reason = _missing_class(labels)
    selection = {"split": "dev", "criterion": "maximum_unsupported_F1_on_determinate_labels",
        "tie_break": "higher determinate precision, then fewer all-case selections",
        "score_orientation": "lower_is_more_unsupported", "comparison": "score <= threshold",
        "n_cases": len(dev), "n_determinate": labels["supported"] + labels["unsupported"],
        "n_uncertain": labels["uncertain"], "undefined_reason": reason}
    if reason:
        return {"threshold": None, "selection": selection}
    groups = defaultdict(Counter)
    for row in dev:
        groups[row["scores"][score_key]][row["judge_label"]] += 1
    tp = fp = selected_all = 0
    best, best_utility = None, None
    for threshold in sorted(groups):
        group = groups[threshold]
        tp += group["unsupported"]
        fp += group["supported"]
        selected_all += sum(group.values())
        decided = tp + fp
        f1 = Fraction(2 * tp, decided + labels["unsupported"])
        precision = Fraction(tp, decided) if decided else Fraction(-1)
        utility = (f1, precision, -selected_all)
        if best_utility is None or utility > best_utility:
            best_utility, best = utility, threshold
    return {"threshold": float(best), "selection": selection}


def negative_metrics(rows, score_key, threshold):
    labels = _class_counts(rows)
    selected = [] if threshold is None else [r for r in rows if r["scores"][score_key] <= threshold]
    chosen = _class_counts(selected)
    tp, fp = chosen["unsupported"], chosen["supported"]
    fn, tn = labels["unsupported"] - tp, labels["supported"] - fp
    reason = "policy_disabled" if threshold is None else _missing_class(labels)
    decided_selected = tp + fp
    return {
        "threshold": threshold, "n_cases": len(rows),
        "n_images": len({row["image_group"] for row in rows}),
        "n_determinate": labels["supported"] + labels["unsupported"],
        "n_supported": labels["supported"], "n_unsupported": labels["unsupported"],
        "n_uncertain": labels["uncertain"],
        "confusion_determinate_only": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_selected": len(selected), "n_selected_determinate": decided_selected,
        "n_selected_uncertain": chosen["uncertain"],
        "selected_uncertain_rate": chosen["uncertain"] / len(selected) if selected else None,
        "uncertain_selection_rate": chosen["uncertain"] / labels["uncertain"] if labels["uncertain"] else None,
        "all_case_selection_rate": len(selected) / len(rows) if rows else None,
        "precision_on_decided": tp / decided_selected if decided_selected else None,
        "conservative_precision": tp / len(selected) if selected else None,
        "recall_unsupported": tp / labels["unsupported"] if labels["unsupported"] else None,
        "f1_unsupported": None if reason else 2 * tp / (2 * tp + fp + fn),
        "f1_undefined_reason": reason,
    }


def build_report(cases, *, score_key="global_raw", target_precision=.95, min_selected=30,
                 bootstrap_samples=500, seed=42, current_negative=None,
                 current_positive=None):
    formal = formal_rows(cases, score_key)
    report = analyze_cases(formal, score_key=score_key, target_precision=target_precision,
        min_selected=min_selected, bootstrap_samples=bootstrap_samples, seed=seed,
        current_negative_threshold=current_negative, current_positive_threshold=current_positive)
    dev = [row for row in formal if row["split"] == "dev"]
    test = [row for row in formal if row["split"] == "test"]
    best = select_negative_f1(dev, score_key)
    best.update(dev=negative_metrics(dev, score_key, best["threshold"]),
                test=negative_metrics(test, score_key, best["threshold"]))
    corpus = {"all": counts(cases, score_key),
              **{role: counts([c for c in cases if c["analysis_role"] == role], score_key) for role in ROLES}}
    return {"schema": "grounding_calibration_analysis_v1", "reference": "visual_judge_silver_labels",
            "scope": "formal dev/test image groups only; retrospective cases excluded from all threshold selection and test metrics",
            "corpus": corpus, "best_f1_negative": best, "high_precision_analysis": report,
            "definitions": {
                "f1": "Unsupported is the positive class; score <= threshold. Uncertain cases are excluded from F1/confusion, never counted as truth-negative or false-positive.",
                "conservative_precision": "Correct selected labels divided by all selected cases, including unconfirmed uncertain labels.",
                "missing_class": "A dev split missing either determinate class yields no F1 threshold; a test split missing either class yields null test F1 without changing the dev-selected threshold.",
                "high_precision": "Dev-selected two-tail policies are maximum-coverage working points under their reported precision criteria; a tail may be disabled (null).",
            }, "limitations": [
                "Visual-judge reference labels are silver labels, not human ground truth.",
                "Thresholds and score variants must not be chosen after viewing test results.",
                "Uncertainty, false labels from the judge and model-version drift can limit conclusions.",
                "Image grouping/duplicate exclusion is an upstream responsibility; this analysis enforces the supplied image groups.",
            ]}


def _number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def _threshold(value):
    # Keep the selected observed score exactly usable for <=/>= comparisons.
    return "—" if value is None else repr(float(value))


def render_summary(result, report):
    corpus, hp, f1 = report["corpus"], report["high_precision_analysis"], report["best_f1_negative"]
    lines = ["# FG-CLIP声明接地性校准", "",
        "参考标签来自实际看图的外部模型，属于裁判银标，不等同于人工真值。阈值只在formal开发图片选择，测试图片只用于冻结后的评价。", "",
        f"状态：{result['status']}；输入候选涉及{corpus['all']['n_images']}图、{corpus['all']['n_cases']}例；"
        f"formal {corpus['formal']['n_images']}图、{corpus['formal']['n_cases']}例；"
        f"retrospective {corpus['retrospective']['n_images']}图、{corpus['retrospective']['n_cases']}例，未参与阈值选择。", "",
        f"当前分数条件：`{hp['score_key']}`，高分表示更受支持。", "",
        "| 选择规则 | 负阈值 | 正阈值 | 冻结测试结果 |", "|---|---:|---:|---|"]
    for key, label in (("strict_policy", "高精度严格点"), ("exploratory_policy", "高精度探索点"), ("current_policy", "原阈值")):
        policy = hp.get(key)
        if policy is None:
            continue
        test = policy["test"]
        lines.append(f"| {label} | {_threshold(policy['thresholds']['negative'])} | {_threshold(policy['thresholds']['positive'])} | "
                     f"负反馈保守精度 {_number(test['negative']['conservative_precision'])}；总覆盖率 {_number(test['coverage'])} |")
    lines.append(f"| 开发集最大F1点 | {_threshold(f1['threshold'])} | 不适用 | "
                 f"测试F1 {_number(f1['test']['f1_unsupported'])}；确定标签精度 {_number(f1['test']['precision_on_decided'])} |")
    lines += ["", "— 表示没有可用阈值或指标未定义，详细原因见results.json与明细。严格点找不到可靠负阈值时，不强行提高阈值凑负样本。", "",
              "| formal分割 | supported | unsupported | uncertain | AUROC | AP（不支持） |", "|---|---:|---:|---:|---:|---:|"]
    for split in ("dev", "test"):
        item = hp["splits"][split]
        c, ranking = item["counts"], item["ranking"]
        lines.append(f"| {split} | {c['supported']} | {c['unsupported']} | {c['uncertain']} | "
                     f"{_number(ranking['auroc_supported'])} | {_number(ranking['ap_unsupported'])} |")
    lines += ["", "最大F1仅对裁判确定的支持/不支持标签计算，未决样本单独记录。"
              "高精度策略的保守精度则将选中的未决样本计入分母，避免靠删除难例虚增质量。", "",
              f"最大F1点在test选中未决{f1['test']['n_selected_uncertain']}例；选中未决比例 {_number(f1['test']['selected_uncertain_rate'])}。", "",
              "开发集Wilson筛选不是阈值搜索后的统计保证；冻结测试的按图bootstrap区间、各极性精度/召回/覆盖率见`details/analysis_report.json`。", "",
              "没有可用候选的图片以及原始候选到可对齐子集的覆盖率，应结合上游图片/解析清单查看；这里不将空图片静默当作已评估声明。", "",
              "原始图片、描述、声明、裁判记录和分数由实验流水线保存在details/。本程序只做离线分析，不改写输入标签。", ""]
    return "\n".join(lines)


def _curves(rows, score_key, *, include_uncertain=False, reverse=False, target="unsupported"):
    chosen = rows if include_uncertain else [r for r in rows if r["judge_label"] != "uncertain"]
    groups = defaultdict(Counter)
    for row in chosen:
        groups[row["scores"][score_key]][row["judge_label"]] += 1
    total_target = sum(r["judge_label"] == target for r in chosen)
    total_other = sum(r["judge_label"] != target and r["judge_label"] != "uncertain" for r in chosen)
    selected = correct = wrong = 0
    points = []
    for threshold in sorted(groups, reverse=reverse):
        group = groups[threshold]
        selected += sum(group.values())
        correct += group[target]
        wrong += group["supported" if target == "unsupported" else "unsupported"]
        points.append({"threshold": threshold, "coverage": selected / len(chosen),
                       "precision": correct / selected,
                       "recall": correct / total_target if total_target else None,
                       "fpr": wrong / total_other if total_other else None})
    return points


def plot_report(formal, score_key, report, directory):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib/numpy is not installed", "files": []}
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    files = []
    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=180, facecolor="white")
            files.append(str(path.relative_to(directory.parent.parent)))
        plt.close(fig)
    colors = {"supported": "#2874a6", "unsupported": "#c45a34", "uncertain": "#87939d"}
    values = [r["scores"][score_key] for r in formal]
    low, high = min(values), max(values)
    if low == high:
        low, high = low - .001, high + .001
    bins = np.linspace(low, high, 41)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ("dev", "test")):
        selected = [r for r in formal if r["split"] == split]
        for label in LABELS:
            scores = [r["scores"][score_key] for r in selected if r["judge_label"] == label]
            if scores:
                ax.hist(scores, bins=bins, density=True, histtype="step", linewidth=1.7,
                        color=colors[label], label=f"{label} (n={len(scores)})")
        ax.set(title=f"Formal {split}: silver-label score density", xlabel=score_key, ylabel="Within-label density")
        ax.legend(frameon=False)
    save(fig, "score_distributions")
    test = [r for r in formal if r["split"] == "test"]
    points = _curves(test, score_key)
    ranking = report["high_precision_analysis"]["splits"]["test"]["ranking"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if ranking["auroc_supported"] is not None:
        axes[0].plot([0] + [p["fpr"] for p in points], [0] + [p["recall"] for p in points],
                     label=f"AUROC = {ranking['auroc_supported']:.3f}")
        axes[0].plot([0, 1], [0, 1], ":", color="#87939d")
        axes[0].legend(frameon=False)
    else:
        axes[0].text(.5, .5, "Undefined: one determinate class missing", ha="center", transform=axes[0].transAxes)
    if ranking["supported"] and ranking["unsupported"]:
        axes[1].step([0] + [p["recall"] for p in points], [1] + [p["precision"] for p in points],
                     where="pre", label=f"AP = {ranking['ap_unsupported']:.3f}")
        axes[1].axhline(ranking["unsupported_prevalence"], linestyle=":", color="#87939d", label="Unsupported prevalence")
        axes[1].legend(frameon=False)
    else:
        axes[1].text(.5, .5, "Undefined: one determinate class missing", ha="center", transform=axes[1].transAxes)
    axes[0].set(title="Frozen test ROC (unsupported is positive)", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(title="Frozen test PR (determinate labels only)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    save(fig, "test_roc_pr")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target, reverse, label in (("unsupported", False, "Negative tail"), ("supported", True, "Positive tail")):
        points = _curves(test, score_key, include_uncertain=True, reverse=reverse, target=target)
        ax.plot([p["coverage"] for p in points], [p["precision"] for p in points], label=label)
    ax.axhline(report["high_precision_analysis"]["strict_policy"]["selection"]["target_precision"],
               color="#87939d", linestyle=":", label="Dev selection precision target")
    ax.set(title="Frozen test precision / coverage", xlabel="Coverage of all test cases", ylabel="Conservative precision (uncertain included)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(frameon=False)
    save(fig, "test_precision_coverage")
    return {"status": "complete", "files": files,
            "note": "Plots describe frozen dev/test data; they do not refit thresholds on test."}


def analyze_file(input_path, output_dir, *, plots=False, **options):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    payload = input_path.read_bytes()
    cases = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    report = build_report(cases, **options)
    report.update(input_path=str(input_path), input_sha256=hashlib.sha256(payload).hexdigest(),
                  analyzed_utc=datetime.now(timezone.utc).isoformat())
    score_key = options.get("score_key", "global_raw")
    formal = [c for c in cases if c["analysis_role"] == "formal"]
    plot_info = plot_report(formal, score_key, report, output_dir / "details/figures") if plots else {"status": "not_requested", "files": []}
    hp = report["high_precision_analysis"]
    whole = report["corpus"]["all"]
    status = ("complete" if not whole["unlabeled_or_invalid"] and not whole["judge_error_cases"]
              and whole["scored_cases"] == whole["n_cases"] else "formal_complete_retrospective_incomplete")
    result = {"schema": "grounding_calibration_results_v1", "status": status,
              "reference": "visual_judge_silver_labels", "score_key": score_key,
              "input_sha256": report["input_sha256"], "corpus": report["corpus"],
              "thresholds": {"best_f1_negative": report["best_f1_negative"]["threshold"],
                  "strict": hp["strict_policy"]["thresholds"], "exploratory": hp["exploratory_policy"]["thresholds"]},
              "best_f1_negative": report["best_f1_negative"],
              "strict_policy": hp["strict_policy"], "exploratory_policy": hp["exploratory_policy"],
              "current_policy": hp.get("current_policy"), "formal_splits": hp["splits"],
              "details": "details/analysis_report.json", "plots": plot_info}
    report["plots"] = plot_info
    # Invalid formal input or calibration errors above leave existing progress
    # result files intact. Publish the complete analysis before the short view.
    atomic_json(output_dir / "details/analysis_report.json", report)
    atomic_text(output_dir / "results.md", render_summary(result, report))
    atomic_json(output_dir / "results.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-key", default="global_raw")
    parser.add_argument("--target-precision", type=float, default=.95)
    parser.add_argument("--min-selected", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--current-negative-threshold", type=float)
    parser.add_argument("--current-positive-threshold", type=float)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = analyze_file(args.input, args.output_dir, plots=args.plots, score_key=args.score_key,
            target_precision=args.target_precision, min_selected=args.min_selected,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
            current_negative=args.current_negative_threshold, current_positive=args.current_positive_threshold)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Grounding analysis: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "results": str(args.output_dir / "results.md"),
                      "thresholds": result["thresholds"]}, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
