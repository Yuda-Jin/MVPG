# 从 eval_all_metrics_mvpg.sh 的全程运行日志中解析各基准分数，
# 汇总为 summary_${MODEL_SUFFIX}.json + summary.csv（append，按 model_suffix 去重，
# 便于多个 ckpt 的横向对比）。
#
# 解析依赖各评测脚本的固定打印格式：
#   MMHal  summarize_gpt_mmhal.py   : "Average score: x.xx" / "Hallucination rate: x.xx"
#   qa90   summarize_gpt_review.py  : "all xx.x"（sorted 后 all 为总体分）
#   POPE   summarize_eval_pope.py   : "Accuracy: 0.xxx" / "Precision: ..." / "F1 score: ..." / "Yes ratio: ..."
#   AMBER  AMBER_eval.py ('g')      : "CHAIR:\t 45.2" / "Cover:" / "Hal:" / "Cog:"
#   ObjHal eval_gpt_obj_halbench.py : "CHAIRs: 42.13" / "CHAIRi: ..." / "CHAIRs_refine: ..."

import argparse
import csv
import json
import os
import re

CSV_COLUMNS = [
    "model_suffix",
    "mmhal_avg_score", "mmhal_hallucination_rate",
    "llava_bench_gpt_score",
    "pope_adv_accuracy", "pope_adv_precision", "pope_adv_recall",
    "pope_adv_f1", "pope_adv_yes_ratio",
    "amber_chair", "amber_cover", "amber_hal", "amber_cog",
    "object_hal_chairs", "object_hal_chairi", "object_hal_chairs_refine",
]


def _num(x):
    return float(x)


def parse_log(log_path):
    text = open(log_path, "r", encoding="utf-8", errors="ignore").read()
    result = {}

    # MMHal
    m = re.findall(r"Average score:\s*([\d.]+)", text)
    r = re.findall(r"Hallucination rate:\s*([\d.]+)", text)
    if m and r:
        result["mmhal_avg_score"] = _num(m[-1])
        result["mmhal_hallucination_rate"] = _num(r[-1])

    # LLaVA-Bench qa90（"all xx.x" 为总体 GPT 分）
    m = re.findall(r"^all\s+([\d.]+)\s*$", text, re.M)
    if m:
        result["llava_bench_gpt_score"] = _num(m[-1])

    # POPE adversarial
    m = re.findall(r"^Accuracy:\s*([\d.]+)\s*$", text, re.M)
    r = re.findall(r"^Precision:\s*([\d.]+)\s*$", text, re.M)
    if m and r:
        result["pope_adv_accuracy"] = _num(m[-1])
        result["pope_adv_precision"] = _num(r[-1])
        for key, pat in [
            ("pope_adv_recall", r"^Recall:\s*([\d.]+)\s*$"),
            ("pope_adv_f1", r"^F1 score:\s*([\d.]+)\s*$"),
            ("pope_adv_yes_ratio", r"^Yes ratio:\s*([\d.]+)\s*$"),
        ]:
            mm = re.findall(pat, text, re.M)
            if mm:
                result[key] = _num(mm[-1])

    # AMBER 生成式任务（tab 分隔冒号格式）
    m = re.findall(r"^CHAIR:\s*([\d.]+)\s*$", text, re.M)
    if m:
        result["amber_chair"] = _num(m[-1])
        for key, pat in [
            ("amber_cover", r"^Cover:\s*([\d.]+)\s*$"),
            ("amber_hal", r"^Hal:\s*([\d.]+)\s*$"),
            ("amber_cog", r"^Cog:\s*([\d.]+)\s*$"),
        ]:
            mm = re.findall(pat, text, re.M)
            if mm:
                result[key] = _num(mm[-1])

    # Object-Hal（取最后一次出现，避开中途的表头行）
    m = re.findall(r"^CHAIRs:\s*([\d.]+)\s*$", text, re.M)
    if m:
        result["object_hal_chairs"] = _num(m[-1])
    for key, pat in [
        ("object_hal_chairi", r"^CHAIRi:\s*([\d.]+)\s*$"),
        ("object_hal_chairs_refine", r"^CHAIRs_refine:\s*([\d.]+)\s*$"),
    ]:
        mm = re.findall(pat, text, re.M)
        if mm:
            result[key] = _num(mm[-1])

    return result


def upsert_csv(out_root, model_suffix, row):
    """summary.csv 按行追加；同 model_suffix 重跑时覆盖旧行（多 ckpt 对比友好）。"""
    csv_path = os.path.join(out_root, "summary.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("model_suffix") != model_suffix]

    rows.append({k: row.get(k, "") for k in CSV_COLUMNS})
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="eval_all_metrics_mvpg.sh 的全程日志")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--model-suffix", type=str, required=True)
    args = parser.parse_args()

    scores = parse_log(args.log)
    if not scores:
        print(f"[summary] 未从 {args.log} 解析到任何分数（评测可能未跑完）")

    summary_path = os.path.join(args.out_root, f"summary_{args.model_suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model_suffix": args.model_suffix, **scores}, f, indent=2, ensure_ascii=False)

    csv_path = upsert_csv(args.out_root, args.model_suffix, scores)

    print("==" * 30)
    print("Evaluation summary")
    print("==" * 30)
    for k in CSV_COLUMNS[1:]:
        if k in scores:
            print(f"{k}: {scores[k]}")
    print(f"[summary] JSON : {summary_path}")
    print(f"[summary] CSV  : {csv_path}")
