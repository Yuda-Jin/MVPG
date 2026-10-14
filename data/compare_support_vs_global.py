"""对比「最大视觉支持」与「全图特征」两种句子级打分方案的幻觉判别力。

- 方案 A（现状）max visual support：句向量 × 主图全部 patch 的 masked max 余弦。
- 方案 B 全图特征：句向量 × 主图全局特征（`get_image_features` 的 pooler 输出）一次余弦。
- 方案 C（附带）全局特征变体：句向量 × 主图有效 patch 的 masked mean 余弦。

三者在同一 FG-CLIP2 投影空间、同一 L2 归一化口径下逐句计算，用同一 GT 评测：
AUC / Cohen's d / top-1 定位 / F1 最优阈值 / 按句长分桶，直接回答「哪种方案判别最好」。

GT 与切分口径完全复用 `verify_fgclip_period_split.py`（句号-only 切分 + VLM 复核短语落句），
故结果与 `period_clause_report.md` 可直接对照。

用法：
    PY=/root/miniconda3/envs/mvpg/bin/python
    $PY data/compare_support_vs_global.py --limit 300     # 小样本
    $PY data/compare_support_vs_global.py                 # 全量（按 gid 断点续传）
    $PY data/compare_support_vs_global.py --stage analyze # 只看报告

产物：
    /root/MVPG/support_vs_global_scores.jsonl
    /root/MVPG/support_vs_global_report.md
    /root/MVPG/support_vs_global_hist.png
"""

import argparse
import json
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
    cohens_d,
    load_done_gids,
    main_image_path,
    pick_f1_threshold,
    sim_columns,
)
from verify_fgclip_period_split import (  # noqa: E402
    DATA,
    FGCLIP_ROOT,
    label_sentences,
    load_gt,
    period_sentences,
)
from mvpg_run.fgclip_local import load_fgclip  # noqa: E402

METRICS = ("s_max", "s_glob", "s_mean")  # A / B / C
LABELS = {"s_max": "最大视觉支持(A: max patch)", "s_glob": "全图特征(B: pooler 全局)",
          "s_mean": "全局变体(C: masked mean patch)"}


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
    n_rows = n_img = n_fail = 0
    with open(args.out, "a", encoding="utf-8") as fo:
        for n, gid in enumerate(todo, 1):
            img = main_image_path(args.image_dir, gid)
            if img is None:
                n_fail += 1
                continue
            dense, valid = fg.image_dense_features([img], batch_size=1)
            dense, valid = dense[0], valid[0]                      # (P,H),(P,)
            glob = fg.encode_image_paths([img], batch_size=1)[0]   # (H,) 已归一化
            dn = F.normalize(dense, p=2, dim=-1)                   # 归一化 patch
            mean_vec = dn[valid].mean(dim=0)
            mean_vec = F.normalize(mean_vec, p=2, dim=-1)          # 归一化 masked mean

            wrote = False
            for ci, cap in enumerate(caps[gid]):
                sents = period_sentences(cap, args.min_words)
                if not sents:
                    continue
                labels = label_sentences(sents, gt[gid].get(ci, []))
                for walk in walks:
                    feat = fg.encode_texts_walk(sents, walk_type=walk, batch_size=64)
                    tn = F.normalize(feat, p=2, dim=-1)            # (N,H)
                    sim = sim_columns(dense, valid, feat)          # (P,N)
                    s_max = sim.max(dim=0).values.tolist()
                    s_glob = (tn @ glob).tolist()
                    s_mean = (tn @ mean_vec).tolist()
                    for j, (s, lab) in enumerate(zip(sents, labels)):
                        fo.write(json.dumps({
                            "gid": gid, "cap": ci, "idx": j, "label": int(lab),
                            "text": s, "n_words": len(s.split()), "walk": walk,
                            "s_max": float(s_max[j]), "s_glob": float(s_glob[j]),
                            "s_mean": float(s_mean[j]),
                        }, ensure_ascii=False) + "\n")
                        n_rows += 1
                        wrote = True
            fo.flush()
            if wrote:
                n_img += 1
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] 句 {n_rows} fail={n_fail} "
                      f"用时 {el:.0f}s ({n / max(el, 1e-9):.2f} 张/s)")

    print(f"打分完成: 写出 {n_rows} 条句子 / {n_img} 图 -> {args.out}")


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


def top1_loc(rows, metric):
    """每条 caption 内该指标最低的句子是否=幻觉句（越低越像幻觉）。"""
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[(r["gid"], r["cap"])].append(r)
    hit = tot = 0
    for _, rs in by_cap.items():
        if not any(r["label"] == 1 for r in rs):
            continue
        tot += 1
        if min(rs, key=lambda r: r[metric])["label"] == 1:
            hit += 1
    return hit, tot


def stage_analyze(args):
    all_rows = read_rows(args.out)
    rows_short = [r for r in all_rows if r["walk"] == "short"]
    rows_box = [r for r in all_rows if r["walk"] == "box"]
    n_gid = len({r["gid"] for r in all_rows})
    print(f"句子 {len(all_rows)}（short {len(rows_short)} / box {len(rows_box)}），图 {n_gid} 张")

    L = []
    L.append("# 句子级幻觉判别：最大视觉支持 vs 全图特征\n")
    L.append(f"- 数据：{n_gid} 张主图 / 句号句 {len(rows_short)} 条（walk=short），"
             f"{len(rows_box)} 条（walk=box）")
    L.append("- 切分：只用句号（与 `period_clause_report.md` 一致）。")
    L.append("- GT：`neg_clause_labels.jsonl` 的 VLM 复核幻觉短语落到句号句上。")
    L.append("- 指标：")
    L.append("  - **s_max（方案A，现状）**：句向量 × 主图全部 patch 的 masked max 余弦。")
    L.append("  - **s_glob（方案B）**：句向量 × 主图全局特征（pooler）一次余弦。")
    L.append("  - **s_mean（方案C）**：句向量 × 主图有效 patch 的 masked mean 余弦。")
    L.append("- 方向：三者都是**越低越像幻觉**。\n")

    # ---- 全局分布 ----
    L.append("## 1. 分布与可分性（幻觉句 vs 真实句）\n")
    L.append("| walk | 方案 | 幻觉 mean±std | 真实 mean±std | AUC | Cohen's d | ΔAUC(相对A) |")
    L.append("|---|---|---|---|---|---|---|")
    auc_ref = {}
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        base = None
        for m in METRICS:
            hall = [r[m] for r in rows if r["label"] == 1]
            real = [r[m] for r in rows if r["label"] == 0]
            ma = sum(hall) / len(hall)
            mr = sum(real) / len(real)
            sa = (sum((x - ma) ** 2 for x in hall) / (len(hall) - 1)) ** 0.5
            sr = (sum((x - mr) ** 2 for x in real) / (len(real) - 1)) ** 0.5
            auc = auc_score(real, hall)
            d = cohens_d(real, hall)
            if base is None:
                base = auc
                auc_ref[walk] = auc
                delta = "—"
            else:
                delta = f"{auc - base:+.3f}"
            L.append(f"| {walk} | {LABELS[m]} | {fmt(ma)} ± {fmt(sa)} | {fmt(mr)} ± {fmt(sr)} | "
                     f"{fmt(auc, 3)} | {fmt(d, 3)} | {delta} |")
    L.append("")

    # ---- top-1 定位 ----
    L.append("## 2. 句内 top-1 定位（每条 caption 该指标最低句是否=幻觉句）\n")
    L.append("| walk | 方案 | 命中 | 命中率 |")
    L.append("|---|---|---|---|")
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        for m in METRICS:
            h, t = top1_loc(rows, m)
            L.append(f"| {walk} | {LABELS[m]} | {h}/{t} | {fmt(h / t if t else float('nan'), 3)} |")
    L.append("")

    # ---- 词数分桶 ----
    L.append("## 3. 按句长分桶的可分性（AUC）\n")
    L.append("| walk | 词数桶 | 句数 | 幻觉数 | s_max(A) | s_glob(B) | s_mean(C) |")
    L.append("|---|---|---|---|---|---|---|")
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        for lo, hi in [(3, 8), (8, 14), (14, 22), (22, 10 ** 9)]:
            sel = [r for r in rows if lo <= r["n_words"] < hi]
            nh = sum(1 for r in sel if r["label"] == 1)
            cells = []
            for m in METRICS:
                sh = [r[m] for r in sel if r["label"] == 1]
                sr = [r[m] for r in sel if r["label"] == 0]
                cells.append(fmt(auc_score(sr, sh), 3) if sh and sr else "nan")
            lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
            L.append(f"| {walk} | {lab} | {len(sel)} | {nh} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    # ---- F1 最优阈值 ----
    L.append("## 4. F1 最优阈值\n")
    L.append("| walk | 方案 | 阈值 t | 精度 | 召回 | F1 |")
    L.append("|---|---|---|---|---|---|")
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        for m in METRICS:
            hall = [r[m] for r in rows if r["label"] == 1]
            real = [r[m] for r in rows if r["label"] == 0]
            t, p, r_, f1 = pick_f1_threshold(hall, real)
            L.append(f"| {walk} | {LABELS[m]} | {fmt(t)} | {fmt(p, 3)} | {fmt(r_, 3)} | {fmt(f1, 3)} |")
    L.append("")

    # ---- 结论 ----
    L.append("## 5. 结论\n")
    best = {}
    for walk, rows in (("short", rows_short), ("box", rows_box)):
        aucs = {}
        for m in METRICS:
            hall = [r[m] for r in rows if r["label"] == 1]
            real = [r[m] for r in rows if r["label"] == 0]
            aucs[m] = auc_score(real, hall)
        w = max(aucs, key=aucs.get)
        best[walk] = (w, aucs)
        L.append(f"- walk={walk}：最好为 **{LABELS[w]}**，AUC={fmt(aucs[w], 3)}；"
                 f"（A={fmt(aucs['s_max'], 3)}, B={fmt(aucs['s_glob'], 3)}, "
                 f"C={fmt(aucs['s_mean'], 3)}）")
    L.append("")

    report = "\n".join(L) + "\n"
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)

    # ---- 直方图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ri, (walk, rows) in enumerate((("short", rows_short), ("box", rows_box))):
            for ci, m in enumerate(METRICS):
                ax = axes[ri][ci]
                hall = [r[m] for r in rows if r["label"] == 1]
                real = [r[m] for r in rows if r["label"] == 0]
                lo, hi = quantile(real + hall, 0.005), quantile(real + hall, 0.995)
                bins = [lo + (hi - lo) * i / 60 for i in range(61)]
                ax.hist(real, bins=bins, density=True, alpha=0.55, color="#4c72b0",
                        label=f"real n={len(real)}")
                ax.hist(hall, bins=bins, density=True, alpha=0.55, color="#c44e52",
                        label=f"hall n={len(hall)}")
                ax.set_title(f"{walk} · {LABELS[m]}\nAUC={auc_score(real, hall):.3f}", fontsize=9)
                ax.set_xlabel("score")
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"分布图已保存: {args.plot}")
    except Exception as e:
        print(f"绘图跳过: {e}")


def main():
    ap = argparse.ArgumentParser(description="最大视觉支持 vs 全图特征的句子级判别力对比")
    ap.add_argument("--stage", choices=["score", "analyze", "all"], default="all")
    ap.add_argument("--image-dir", default=DATA)
    ap.add_argument("--labels", default="/root/MVPG/neg_clause_labels.jsonl")
    ap.add_argument("--out", default="/root/MVPG/support_vs_global_scores.jsonl")
    ap.add_argument("--report", default="/root/MVPG/support_vs_global_report.md")
    ap.add_argument("--plot", default="/root/MVPG/support_vs_global_hist.png")
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
