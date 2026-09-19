"""小实验：用「整条 caption 的最大视觉支持」当 baseline，
以子句级优势值 adv = s_box(子句) − s_box(整条 caption) 判断该子句是否为幻觉。

- 只在**主图**上做（不使用子图），图片为 `<data>/<gid>/<gid>.jpg`。
- 子句切分沿用联合 `[.,;]` 并过滤 <3 词的碎片（与 verify_fgclip_split_compare.py 一致）。
- 全文本 = 该 caption 的子句用空格拼接（保证子句集合是全文本的一个划分，避免标点干扰）。
- 视觉支持沿用 box 头 × dense patch 的 max_patch_sim（raw 余弦，无 softmax）。

对照臂：句内均值 baseline（advm = s_clause − 同 caption 子句均值），
用来直接对比两种「零点」定义在判断幻觉这件事上的准确率。

数据：
- 负样本（含幻觉）：`<data>/neg_captions.json`，{gid: [caption, ...]}，幻觉子句由 HALL_CORES 核心词定位。
- 真实样本（无幻觉）：`<data>/model_captions.json` 的 `<real_key>` 主图行，
  用来测「真实子句被判为幻觉」的假阳性率。

用法：
    /root/miniconda3/envs/mvpg/bin/python data/verify_fgclip_clause_fulltext_adv.py
产物：
    /root/MVPG/fgclip_clause_fulltext_adv_results.json
    /root/MVPG/fgclip_clause_fulltext_adv_report.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/MVPG")
sys.path.insert(0, "/root/MVPG/data")

import torch  # noqa: E402

from mvpg_run.fgclip_local import load_fgclip  # noqa: E402
from verify_fgclip_split_compare import (  # noqa: E402
    HALL_CORES,
    dense_features,
    encode_text_walk,
    max_patch_sim,
    split_clauses,
)

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT = r"[.,;]"

# 阈值扫描：pred = adv < tau
TAUS = [-0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0,
        0.005, 0.01, 0.015, 0.02, 0.03, 0.04]


def main_img_path(root, gid):
    d = Path(root) / gid
    if not d.is_dir():
        return None
    for s in IMG_SUFFIXES:
        p = d / (gid + s)
        if p.is_file():
            return str(p)
    return None


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0


def cohens_d(a, b):
    """d = (mean(b) - mean(a)) / pooled_sd。传入 (幻觉, 真实) 时 d>0 表示真实更高。"""
    sa, sb = std(a), std(b)
    sp = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    return (mean(b) - mean(a)) / sp if sp > 0 else 0.0


def auc_pos_higher(pos, neg):
    """AUC = P(pos > neg)，带并列平均秩（Mann-Whitney）。"""
    if not pos or not neg:
        return float("nan")
    vals = sorted(pos + neg)
    n = len(vals)
    ranks, i = {}, 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1] == vals[i]:
            j += 1
        ranks[vals[i]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = sum(ranks[v] for v in pos) - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def judge(rows, key, tau):
    """pred = (rows[key] < tau) 判为幻觉；幻觉为正类。"""
    tp = fp = tn = fn = 0
    for r in rows:
        pred = r[key] < tau
        if r["is_hall"]:
            tp += 1 if pred else 0
            fn += 0 if pred else 1
        else:
            fp += 1 if pred else 0
            tn += 0 if pred else 1
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tau": tau, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc": (tp + tn) / n if n else 0.0,
        "prec": prec, "rec": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "suppress_acc": rec,        # 幻觉子句被判为负（正确抑制）的比例
        "encourage_acc": spec,      # 真实子句未被抑制（正确不判）的比例
    }


def describe(rows, key):
    """对某个优势值信号做区分度统计 + 各阈值判定准确率。"""
    hall = [r[key] for r in rows if r["is_hall"]]
    real = [r[key] for r in rows if not r["is_hall"]]
    # 组级 margin：(真实均值 − 幻觉均值)，>0 表示方向正确
    groups = {}
    for r in rows:
        groups.setdefault((r["gid"], r["cap_idx"]), []).append(r)
    margins = []
    for gr in groups.values():
        mh = [x[key] for x in gr if x["is_hall"]]
        mr = [x[key] for x in gr if not x["is_hall"]]
        if mh and mr:
            margins.append(mean(mr) - mean(mh))
    sweep = [judge(rows, key, t) for t in TAUS]
    at0 = judge(rows, key, 0.0)
    best_f1 = max(sweep, key=lambda d: d["f1"])
    best_acc = max(sweep, key=lambda d: d["acc"])
    return {
        "n_hall": len(hall), "n_real": len(real),
        "hall_mean": mean(hall), "real_mean": mean(real),
        "cohens_d": cohens_d(hall, real),
        "auc": auc_pos_higher([-r[key] for r in rows if r["is_hall"]],
                              [-r[key] for r in rows if not r["is_hall"]]),
        "at_tau0": at0,
        "best_f1": best_f1, "best_acc": best_acc,
        "sweep": sweep,
        "margin_mean": mean(margins),
        "margin_gt0": sum(1 for m in margins if m > 0),
        "n_margin": len(margins),
    }


def top1_localization(rows, key):
    """含幻觉子句的 caption 里，优势值最低的子句是否为幻觉子句。"""
    groups = {}
    for r in rows:
        groups.setdefault((r["gid"], r["cap_idx"]), []).append(r)
    n_cap = n_hit = 0
    for gr in groups.values():
        if not any(x["is_hall"] for x in gr):
            continue
        n_cap += 1
        if min(gr, key=lambda x: x[key])["is_hall"]:
            n_hit += 1
    return {"n_cap": n_cap, "n_hit": n_hit, "acc": n_hit / n_cap if n_cap else 0.0}


def fulltext_diag(rows):
    """检查 s_full 与子句得分的相对位置（若 s_full 系统性不低于所有子句，adv 会全为正）。"""
    groups = {}
    for r in rows:
        groups.setdefault((r["gid"], r["cap_idx"]), []).append(r)
    ge_max, d_max, d_mean = [], [], []
    for gr in groups.values():
        sf = gr[0]["s_full_box"]
        sb = [x["s_box"] for x in gr]
        ge_max.append(1.0 if sf >= max(sb) else 0.0)
        d_max.append(sf - max(sb))
        d_mean.append(sf - mean(sb))
    return {
        "n_group": len(groups),
        "s_full_mean": mean([gr[0]["s_full_box"] for gr in groups.values()]),
        "s_full_ge_clause_max_rate": mean(ge_max),
        "s_full_minus_clause_max_mean": mean(d_max),
        "s_full_minus_clause_mean_mean": mean(d_mean),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/SA1B_5_threshold")
    ap.add_argument("--fgclip-root", default="/root/autodl-tmp/cache/fgclip2-base-patch16")
    ap.add_argument("--real-key", default="llava-v1.5-7b",
                    help="model_captions.json 中用作「无幻觉对照组」的模型名")
    ap.add_argument("--no-real", action="store_true", help="跳过真实 caption 对照组")
    ap.add_argument("--device", default=None, help="默认自动选择 cuda/cpu")
    ap.add_argument("--out-json", default="/root/MVPG/fgclip_clause_fulltext_adv_results.json")
    ap.add_argument("--out-md", default="/root/MVPG/fgclip_clause_fulltext_adv_report.md")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp] device={device}")
    t0 = time.time()
    fg = load_fgclip(args.fgclip_root, device)
    neg = json.load(open(Path(args.data) / "neg_captions.json"))
    print(f"[exp] FG-CLIP loaded in {time.time() - t0:.1f}s")

    rows, real_rows = [], []
    n_cap_used = n_cap_skip = 0

    def score_group(dense, valid, clauses, gid, cap_idx, cores):
        """返回该 caption 下每个子句的一行记录。"""
        full = " ".join(clauses)
        s_box = max_patch_sim(dense, valid, encode_text_walk(fg, clauses, "box")).tolist()
        s_short = max_patch_sim(dense, valid, encode_text_walk(fg, clauses, "short")).tolist()
        s_full_box = float(max_patch_sim(dense, valid, encode_text_walk(fg, [full], "box"))[0])
        s_full_short = float(max_patch_sim(dense, valid, encode_text_walk(fg, [full], "short"))[0])
        m_box, m_short = mean(s_box), mean(s_short)
        out = []
        for cl, sb, ss in zip(clauses, s_box, s_short):
            out.append({
                "gid": gid, "cap_idx": cap_idx, "clause": cl.strip(),
                "is_hall": bool(cores) and any(c in cl.lower() for c in cores),
                "s_box": sb, "s_short": ss,
                "s_full_box": s_full_box, "s_full_short": s_full_short,
                # 方案：减去整条 caption 的最大视觉支持
                "adv_box": sb - s_full_box, "adv_short": ss - s_full_short,
                # 对照：减去同 caption 子句均值
                "advm_box": sb - m_box, "advm_short": ss - m_short,
            })
        return out

    for gid, cores in HALL_CORES.items():
        img = main_img_path(args.data, gid)
        if img is None:
            print(f"[exp] skip {gid}: 无主图")
            continue
        dense, valid = dense_features(fg, img)
        for cap_idx, cap in enumerate(neg.get(gid, [])):
            clauses = split_clauses(cap, SPLIT)
            if len(clauses) < 2:
                n_cap_skip += 1
                continue
            rows.extend(score_group(dense, valid, clauses, gid, cap_idx, cores))
            n_cap_used += 1

    # ---- 真实 caption 对照组（无幻觉，全部子句 is_hall=False）----
    if not args.no_real:
        caps = json.load(open(Path(args.data) / "model_captions.json"))
        for gid in HALL_CORES:
            cap = (caps.get(gid, {}).get(args.real_key) or {}).get(gid)
            img = main_img_path(args.data, gid)
            if not cap or img is None:
                continue
            clauses = split_clauses(cap, SPLIT)
            if len(clauses) < 2:
                continue
            dense, valid = dense_features(fg, img)
            real_rows.extend(score_group(dense, valid, clauses, gid, -1, None))

    print(f"[exp] 负样本 caption {n_cap_used} 条（跳过 {n_cap_skip}）→ 子句 {len(rows)}；"
          f"真实 caption 子句 {len(real_rows)}")

    summary = {
        "n_cap": n_cap_used, "n_clause": len(rows),
        "n_clause_real_caption": len(real_rows),
        "adv_box": describe(rows, "adv_box"),
        "adv_short": describe(rows, "adv_short"),
        "advm_box": describe(rows, "advm_box"),
        "advm_short": describe(rows, "advm_short"),
        "top1_adv_box": top1_localization(rows, "adv_box"),
        "top1_advm_box": top1_localization(rows, "advm_box"),
        "fulltext_diag": fulltext_diag(rows),
    }
    if real_rows:
        summary["real_caption_false_suppress"] = {
            key: {
                "n_clause": len(real_rows),
                "fp_rate_at_tau0": sum(1 for r in real_rows if r[key] < 0.0) / len(real_rows),
                "mean": mean([r[key] for r in real_rows]),
                "neg_rate": sum(1 for r in real_rows if r[key] < 0.0) / len(real_rows),
            }
            for key in ["adv_box", "adv_short", "advm_box", "advm_short"]
        }

    json.dump({"summary": summary, "rows": rows, "real_rows": real_rows},
              open(args.out_json, "w"), ensure_ascii=False, indent=1)

    # ---------------- 报告 ----------------
    s = summary
    ab, as_ = s["adv_box"], s["adv_short"]
    mb, ms = s["advm_box"], s["advm_short"]
    L = []
    L.append("# 子句优势值（减去整条 caption 最大视觉支持）判断幻觉\n")
    L.append(f"- 数据：`{args.data}` 负样本 caption，**{s['n_cap']} 条 / {s['n_clause']} 子句**"
             f"（**仅主图**）；真实 caption 对照 {s['n_clause_real_caption']} 子句\n")
    L.append("- 信号：联合 `[.,;]` 切子句 → box 头 × dense patch max_patch_sim（raw）\n")
    L.append("- 优势值：`adv = s_box(子句) − s_box(整条 caption)`；"
             "`advm = s_box(子句) − 同 caption 子句均值`（对照）\n")
    L.append("- 判定：`adv < tau` 判为幻觉（tau=0 为默认）\n")

    L.append("\n## 1. 区分度\n")
    L.append("| 信号 | 幻觉均值 | 真实均值 | Cohen's d | AUC | 组级 margin>0 |")
    L.append("|---|---|---|---|---|---|")
    for name, m in [("adv (box)", ab), ("adv (short)", as_),
                    ("advm (box)", mb), ("advm (short)", ms)]:
        L.append(f"| {name} | {m['hall_mean']:+.4f} | {m['real_mean']:+.4f} | "
                 f"**{m['cohens_d']:+.2f}** | {m['auc']:.3f} | {m['margin_gt0']}/{m['n_margin']} |")

    L.append("\n## 2. 判断准确率（tau=0）\n")
    L.append("| 信号 | 整体准确率 | 抑制正确率(幻觉) | 鼓励正确率(真实) | 精确率 | 召回 | F1 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, m in [("adv (box)", ab), ("adv (short)", as_),
                    ("advm (box)", mb), ("advm (short)", ms)]:
        d = m["at_tau0"]
        L.append(f"| {name} | **{d['acc']*100:.1f}%** | {d['suppress_acc']*100:.1f}% | "
                 f"{d['encourage_acc']*100:.1f}% | {d['prec']*100:.1f}% | {d['rec']*100:.1f}% | "
                 f"{d['f1']*100:.1f}% |")

    L.append("\n## 3. 阈值扫描（最佳可达准确率，用于区分「零点偏移」与「信号弱」）\n")
    L.append("| 信号 | 最优准确率 | 对应 tau | 最优 F1 | 对应 tau |")
    L.append("|---|---|---|---|---|")
    for name, m in [("adv (box)", ab), ("adv (short)", as_),
                    ("advm (box)", mb), ("advm (short)", ms)]:
        L.append(f"| {name} | **{m['best_acc']['acc']*100:.1f}%** | {m['best_acc']['tau']:+.3f} | "
                 f"{m['best_f1']['f1']*100:.1f}% | {m['best_f1']['tau']:+.3f} |")
    L.append("\n### adv (box) 逐阈值明细\n")
    L.append("| tau | 准确率 | 精确率 | 召回 | F1 |")
    L.append("|---|---|---|---|---|")
    for d in ab["sweep"]:
        L.append(f"| {d['tau']:+.3f} | {d['acc']*100:.1f}% | {d['prec']*100:.1f}% | "
                 f"{d['rec']*100:.1f}% | {d['f1']*100:.1f}% |")

    t1 = s["top1_adv_box"]
    t1m = s["top1_advm_box"]
    L.append("\n## 4. 定位能力：优势值最低的子句是否为幻觉子句\n")
    L.append(f"- adv（全文本 baseline）：**{t1['acc']*100:.1f}%**（{t1['n_hit']}/{t1['n_cap']}）\n")
    L.append(f"- advm（句内均值 baseline）：{t1m['acc']*100:.1f}%（{t1m['n_hit']}/{t1m['n_cap']}）\n")

    fd = s["fulltext_diag"]
    L.append("\n## 5. 全文本基线诊断\n")
    L.append(f"- s_full 均值 {fd['s_full_mean']:.4f}；"
             f"s_full ≥ 该 caption 子句最大值的比例：**{fd['s_full_ge_clause_max_rate']*100:.1f}%**\n")
    L.append(f"- s_full − max(子句) 均值：{fd['s_full_minus_clause_max_mean']:+.4f}；"
             f"s_full − mean(子句) 均值：{fd['s_full_minus_clause_mean_mean']:+.4f}\n")
    if fd["s_full_ge_clause_max_rate"] > 0.9:
        L.append("- 警告：全文本得分几乎总是最高，说明长文本并没有被稀释成「平均值」，"
                 "adv 会系统性偏负，判定会偏向「全是幻觉」。\n")

    if "real_caption_false_suppress" in s:
        L.append("\n## 6. 真实 caption（无幻觉）上的假阳性率\n")
        L.append("| 信号 | 被判为幻觉的子句比例（tau=0） | 优势值均值 |")
        L.append("|---|---|---|")
        for key, label in [("adv_box", "adv (box)"), ("adv_short", "adv (short)"),
                           ("advm_box", "advm (box)"), ("advm_short", "advm (short)")]:
            d = s["real_caption_false_suppress"][key]
            L.append(f"| {label} | **{d['fp_rate_at_tau0']*100:.1f}%** | {d['mean']:+.4f} |")

    out = "\n".join(L) + "\n"
    Path(args.out_md).write_text(out, encoding="utf-8")
    print(out)
    print(f"[exp] 写出 {args.out_json} 与 {args.out_md}")


if __name__ == "__main__":
    main()
