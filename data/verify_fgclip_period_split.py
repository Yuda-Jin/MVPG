"""全量评测「只按句号分割」下，最大视觉支持（max_patch_sim）对幻觉子句的判别力。

与 `clause_visual_support.py`（`.,;` 联合切分）对照，回答：
    split 收紧到**只用句号**（一条 caption -> 若干完整句子）后，
    子句的「最大视觉支持」s_short / s_box 还能不能把幻觉句和真实句分开？

GT 来源与 `clause_visual_support.py` 完全一致：`neg_clause_labels.jsonl` 里由独立 VLM
复核通道找回的幻觉短语（`absent`，按 caption 记录 phrase），这里把短语**落到句号句**上
（normalize 后子串包含即判 hall=1），因此标签与 FG-CLIP 打分独立、无循环论证。

打分口径：句向量 × 主图全部 patch 向量的 masked max 余弦（= 「最大视觉支持」），
walk=short / box 两路，与既有报告可比。

用法：
    PY=/root/miniconda3/envs/mvpg/bin/python
    $PY data/verify_fgclip_period_split.py --limit 400          # 先小样本
    $PY data/verify_fgclip_period_split.py                      # 全量（断点续传）
    $PY data/verify_fgclip_period_split.py --stage analyze      # 只看报告

产物：
    /root/MVPG/period_clause_scores.jsonl
    /root/MVPG/period_clause_report.md
    /root/MVPG/period_clause_hist.png
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/MVPG")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from calibrate_paired_ref import auc_score, quantile  # noqa: E402
from clause_visual_support import (  # noqa: E402
    average_precision,
    cohens_d,
    is_cont_fragment,
    load_done_gids,
    main_image_path,
    pick_f1_threshold,
    sim_columns,
)
from mvpg_run.fgclip_local import load_fgclip  # noqa: E402

FGCLIP_ROOT = "/root/autodl-tmp/cache/fgclip2-base-patch16"
DATA = "/root/autodl-tmp/data/SA1B_5_threshold"


def normalize(text):
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def period_sentences(caption, min_words=3):
    """只按句号切分：`". "` 处断开，保留句尾句号；过滤 < min_words 词的空句/碎片。"""
    out = []
    for p in re.split(r"(?<=\.)\s+", (caption or "").strip()):
        p = p.strip()
        if p and len(p.split()) >= min_words:
            out.append(p)
    return out


def label_sentences(sents, phrases):
    """短语 normalize 后落到句子上（子串包含即 hall）。返回 labels(list[int])。"""
    pn = [normalize(p) for p in phrases if p]
    labels = []
    for s in sents:
        ns = normalize(s)
        labels.append(1 if any(p and p in ns for p in pn) else 0)
    return labels


def load_gt(labels_path, limit=0):
    """neg_clause_labels.jsonl -> {gid: {cap: [phrase,...]}}（去重、跳过 error 行）。"""
    recs = {}
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r or "absent" not in r:
                continue
            per_cap = defaultdict(list)
            for a in r["absent"]:
                if a.get("phrase"):
                    per_cap[a["cap"]].append(a["phrase"])
            recs[r["gid"]] = dict(per_cap)
    gids = sorted(recs)
    return {g: recs[g] for g in (gids[:limit] if limit else gids)}


# ---------------------------------------------------------------- score

def stage_score(args):
    fg = load_fgclip(args.fgclip_root, args.device)
    with open(f"{args.image_dir}/neg_captions.json", encoding="utf-8") as f:
        caps = json.load(f)
    gt = load_gt(args.labels, args.limit)
    done = load_done_gids(args.out)
    todo = [g for g in gt if g in caps and g not in done]
    print(f"GT {len(gt)} 张，已完成 {len(done)}，本次待处理 {len(todo)} 张")
    if not todo:
        return

    walks = args.walks
    t0 = time.time()
    n_rows = n_img = n_fail = n_drop = 0
    with open(args.out, "a", encoding="utf-8") as fo:
        for n, gid in enumerate(todo, 1):
            img = main_image_path(args.image_dir, gid)
            if img is None:
                n_fail += 1
                continue
            dense, valid = fg.image_dense_features([img], batch_size=1)
            dense, valid = dense[0], valid[0]
            wrote = False
            for ci, cap in enumerate(caps[gid]):
                sents = period_sentences(cap, args.min_words)
                if not sents:
                    continue
                labels = label_sentences(sents, gt[gid].get(ci, []))
                n_drop += sum(1 for l in labels if l) == 0
                for walk in walks:
                    feat = fg.encode_texts_walk(sents, walk_type=walk, batch_size=64)
                    sim = sim_columns(dense, valid, feat)      # (P, N)
                    vals = sim.max(dim=0).values.tolist()
                    for j, (s, lab, v) in enumerate(zip(sents, labels, vals)):
                        rec = {"gid": gid, "cap": ci, "idx": j, "label": int(lab),
                               "text": s, "n_words": len(s.split()),
                               "walk": walk, "s": float(v)}
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_rows += 1
                        wrote = True
            fo.flush()
            if wrote:
                n_img += 1
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] 句 {n_rows} fail={n_fail} "
                      f"用时 {el:.0f}s ({n / max(el, 1e-9):.2f} 张/s)")

    print(f"打分完成: 写出 {n_rows} 条句子 / {n_img} 图 -> {args.out}（缺幻觉句的 caption 丢弃 {n_drop}）")


# ---------------------------------------------------------------- baseline

def comma_baseline(scores_path, drop_fragments, gids=None):
    """从既有 `.,;` 联合切分打分里取 s_short/s_box 的 AUC，作直接对照。

    gids 非空时只统计同批图，保证与句号-only 是同一批数据的公平对照。
    """
    if not scores_path.is_file():
        return None
    hall, real = {"s_short": [], "s_box": []}, {"s_short": [], "s_box": []}
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if gids is not None and r.get("gid") not in gids:
                continue
            if drop_fragments and is_cont_fragment(r.get("text", "")):
                continue
            for k in ("s_short", "s_box"):
                if k in r:
                    (hall if r["label"] == 1 else real)[k].append(r[k])
    out = {}
    for k in ("s_short", "s_box"):
        out[k] = {"auc": auc_score(real[k], hall[k]), "n_hall": len(hall[k]),
                  "n_real": len(real[k])}
    return out


# ---------------------------------------------------------------- analyze

def fmt(x, nd=4):
    return "nan" if x != x else f"{x:.{nd}f}"


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def top1_loc(rows):
    """每条 caption 内最大视觉支持最低的句子是否 = 幻觉句。"""
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[(r["gid"], r["cap"])].append(r)
    hit = tot = 0
    for _, rs in by_cap.items():
        if not any(r["label"] == 1 for r in rs):
            continue
        tot += 1
        if min(rs, key=lambda r: r["s"])["label"] == 1:
            hit += 1
    return hit, tot


def stage_analyze(args):
    all_rows = read_rows(args.out)
    rows_short = [r for r in all_rows if r["walk"] == "short"]
    rows_box = [r for r in all_rows if r["walk"] == "box"]
    n_gid = len({r["gid"] for r in all_rows})
    gids = {r["gid"] for r in all_rows}
    print(f"句子 {len(all_rows)}（short {len(rows_short)} / box {len(rows_box)}），图 {n_gid} 张")

    L = []
    L.append("# 句号-only 子句切分：「最大视觉支持」的幻觉判别力\n")
    L.append(f"- 数据：{n_gid} 张主图 / 句号句 {len(rows_short)} 条（walk=short）")
    L.append("- 切分：**只用句号**（`\". \"` 处断开，一条 caption → 若干完整句子），"
             "不再按逗号称切，故天然没有逗号碎片（F2/F5）。")
    L.append("- 打分：「最大视觉支持」= 句向量 × 主图全部 patch 的 masked max 余弦。")
    L.append("- GT：`neg_clause_labels.jsonl` 的 VLM 复核幻觉短语落到句号句上"
             "（normalize 后子串包含 → label=1）。")
    L.append("- 方向：支持度**越低**越像幻觉，`score <= t` 判幻觉。\n")

    # ---- 全局分布 ----
    L.append("## 1. 分布（幻觉句 vs 真实句）\n")
    L.append("| 指标 | 幻觉 mean±std | 真实 mean±std | 幻觉 median | 真实 median | AUC | Cohen's d |")
    L.append("|---|---|---|---|---|---|---|")
    stats = {}
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        hall = [r["s"] for r in rows if r["label"] == 1]
        real = [r["s"] for r in rows if r["label"] == 0]
        ma, mr = sum(hall) / len(hall), sum(real) / len(real)
        sa = (sum((x - ma) ** 2 for x in hall) / (len(hall) - 1)) ** 0.5
        sr = (sum((x - mr) ** 2 for x in real) / (len(real) - 1)) ** 0.5
        auc = auc_score(real, hall)
        d = cohens_d(real, hall)
        stats[walk] = {"auc": auc, "d": d, "hall_mean": ma, "real_mean": mr}
        L.append(f"| s_{walk} | {fmt(ma)} ± {fmt(sa)} | {fmt(mr)} ± {fmt(sr)} | "
                 f"{fmt(quantile(hall, 0.5))} | {fmt(quantile(real, 0.5))} | "
                 f"{fmt(auc, 3)} | {fmt(d, 3)} |")
    L.append("")

    # ---- top-1 定位 ----
    L.append("## 2. 句内 top-1 定位（每条 caption 最低支持度句是否=幻觉句）\n")
    L.append("| 指标 | 命中 | 命中率 |")
    L.append("|---|---|---|")
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        h, t = top1_loc(rows)
        L.append(f"| s_{walk} | {h}/{t} | {fmt(h / t if t else float('nan'), 3)} |")
    L.append("")

    # ---- 词数分桶 ----
    L.append("## 3. 按句长分桶的可分性\n")
    L.append("| 词数桶 | 句数 | 幻觉数 | AUC(s_short) | AUC(s_box) |")
    L.append("|---|---|---|---|---|")
    for lo, hi in [(3, 8), (8, 14), (14, 22), (22, 10 ** 9)]:
        sel = [r for r in rows_short if lo <= r["n_words"] < hi]
        sh = [r["s"] for r in sel if r["label"] == 1]
        sr = [r["s"] for r in sel if r["label"] == 0]
        bsel = [r for r in rows_box if lo <= r["n_words"] < hi]
        bh = [r["s"] for r in bsel if r["label"] == 1]
        br = [r["s"] for r in bsel if r["label"] == 0]
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        L.append(f"| {lab} | {len(sel)} | {len(sh)} | {fmt(auc_score(sr, sh), 3)} | "
                 f"{fmt(auc_score(br, bh), 3)} |")
    L.append("")

    # ---- F1 工作点 ----
    L.append("## 4. F1 最优阈值（可直接用的分割点）\n")
    L.append("| 指标 | 阈值 t | 精度 | 召回 | F1 | 基率(幻觉占比) |")
    L.append("|---|---|---|---|---|---|")
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        hall = [r["s"] for r in rows if r["label"] == 1]
        real = [r["s"] for r in rows if r["label"] == 0]
        t, p, r_, f1 = pick_f1_threshold(hall, real)
        base = len(hall) / max(len(hall) + len(real), 1)
        L.append(f"| s_{walk} | {fmt(t)} | {fmt(p, 3)} | {fmt(r_, 3)} | {fmt(f1, 3)} | "
                 f"{fmt(base, 3)} |")
    L.append("")

    # ---- 量化：句内真实内容对幻觉的「稀释」 ----
    L.append("## 5. 与「逗号/联合切分」基线对照\n")
    L.append("同一批 GT，只改切分口径：")
    L.append("- **句号-only**（本实验）：一条 caption 切成若干完整句子，幻觉短语所在句同时"
             "含大量真实内容 → 句子整体仍能匹配到真实 patch，最大视觉支持被稀释。")
    L.append("- **`.,;` 联合切分**（`clause_visual_support.py`）：把句子再按逗号切碎，"
             "幻觉短语单独成短片段，稀释最轻，但会引入逗号碎片污染。\n")
    base_all = comma_baseline(Path(args.comma_scores), False, gids)
    base_filt = comma_baseline(Path(args.comma_scores), True, gids)
    L.append("| 切分口径 | s_short AUC | s_box AUC |")
    L.append("|---|---|---|")
    L.append(f"| 句号-only（本实验） | {fmt(stats['short']['auc'], 3)} | "
             f"{fmt(stats['box']['auc'], 3)} |")
    if base_all:
        L.append(f"| `.,;` 联合（同批图，无过滤） | {fmt(base_all['s_short']['auc'], 3)} | "
                 f"{fmt(base_all['s_box']['auc'], 3)} |")
    if base_filt:
        L.append(f"| `.,;` 联合（同批图，剔除 F2 续接片） | {fmt(base_filt['s_short']['auc'], 3)} | "
                 f"{fmt(base_filt['s_box']['auc'], 3)} |")
    L.append("")

    report = "\n".join(L) + "\n"
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)

    # ---- 直方图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for ax, (walk, rows) in zip(axes, (("short", rows_short), ("box", rows_box))):
            hall = [r["s"] for r in rows if r["label"] == 1]
            real = [r["s"] for r in rows if r["label"] == 0]
            lo, hi = quantile(real + hall, 0.005), quantile(real + hall, 0.995)
            bins = [lo + (hi - lo) * i / 60 for i in range(61)]
            ax.hist(real, bins=bins, density=True, alpha=0.55, color="#4c72b0",
                    label=f"real n={len(real)}")
            ax.hist(hall, bins=bins, density=True, alpha=0.55, color="#c44e52",
                    label=f"hallucinated n={len(hall)}")
            ax.set_title(f"s_{walk} (period split)  AUC={auc_score(real, hall):.3f}")
            ax.set_xlabel("max visual support")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"分布图已保存: {args.plot}")
    except Exception as e:
        print(f"绘图跳过: {e}")


def main():
    ap = argparse.ArgumentParser(description="句号-only 子句切分的幻觉判别力评测")
    ap.add_argument("--stage", choices=["score", "analyze", "all"], default="all")
    ap.add_argument("--image-dir", default=DATA)
    ap.add_argument("--labels", default="/root/MVPG/neg_clause_labels.jsonl")
    ap.add_argument("--out", default="/root/MVPG/period_clause_scores.jsonl")
    ap.add_argument("--report", default="/root/MVPG/period_clause_report.md")
    ap.add_argument("--plot", default="/root/MVPG/period_clause_hist.png")
    ap.add_argument("--comma-scores", default="/root/MVPG/clause_support_scores.jsonl")
    ap.add_argument("--fgclip-root", default=FGCLIP_ROOT)
    ap.add_argument("--walks", default="short,box")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.walks = [w for w in args.walks.split(",") if w]
    if args.stage in ("score", "all"):
        stage_score(args)
    if args.stage in ("analyze", "all"):
        stage_analyze(args)


if __name__ == "__main__":
    main()
