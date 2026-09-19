#!/usr/bin/env python3
"""复刻 OHD-Caps（Liu et al., EMNLP 2024）Table 1 的多选排名评测，
量化「**冻结的** FG-CLIP2」在幻觉判别上的先天能力。

背景：该论文指出 CLIP 的预训练负样本来自 batch 内其他图的 caption，
区分过于容易，模型从不需要分辨物体级的细微错误 —— 因此**冻结的 CLIP 本身
并不具备幻觉判别能力**（论文 Table 1：CLIP ViT-L/14 在 28 选 1 上只有 19.0%）。

本脚本用本项目的子句级标注数据把这个结论搬到自己的场景上，回答：
「不是阈值没调好，而是模型压根没学过」这个判断是否成立。

两个任务（均为多选排名正确率，可对照随机基线）：

  rank —— **子句级多选**（免推理，直接读 clause_support_scores.jsonl）
      每条 caption 有 K 个子句、恰 1 个是幻觉子句。
      * argmin_acc : 幻觉子句的视觉支持 s 排最后（最低）的比例     随机 = 1/K
      * pairwise   : 任取一个 (幻觉, 真实) 对，s(真实) > s(幻觉) 的比例  随机 = 0.5
      信号：s_box / s_short（原图）与 s_swap_box / s_swap_short（换图基线）。
      换图那两列是关键对照：若换图后准确率不降，说明靠的是文本先验而非视觉。

  cap —— **整句级反事实多选**（需 GPU）
      对一条 caption 做「留一子句」构造 K 个候选：每个候选 = 原 caption 删掉第 i 个子句。
      恰有一个候选删掉的是幻觉子句（该候选才是与图相符的忠实描述）。
      问：模型能否把这个忠实候选排到第一？
      * K 个候选**长度完全对齐**（都是 K−1 个子句），唯一变量是「少的是哪个子句」，
        因此排除了长度/流畅度等混淆。
      * 随机基线 = 1/K。
      同时用「换一张无关图」再算一遍作对照。

用法：
    PY=/root/miniconda3/envs/mvpg/bin/python
    $PY data/verify_fgclip_ohdcaps.py --stage rank
    $PY data/verify_fgclip_ohdcaps.py --stage cap --device cuda

产物：
    /root/MVPG/fgclip_ohdcaps_rank.json     （rank 阶段明细+汇总）
    /root/MVPG/fgclip_ohdcaps_cap.jsonl     （cap 阶段逐 gid 明细，可断点续传）
    /root/MVPG/fgclip_ohdcaps_report.md     （两阶段汇总报告）
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/MVPG")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from clause_visual_support import (  # noqa: E402
    is_cont_fragment, load_labels, main_image_path, sim_columns,
)
from mvpg_run.fgclip_local import FGChip, load_fgclip  # noqa: E402

DATA_DIR = "/root/autodl-tmp/data/SA1B_5_threshold"
SCORES = "/root/MVPG/clause_support_scores.jsonl"
OUT_RANK = "/root/MVPG/fgclip_ohdcaps_rank.json"
OUT_CAP = "/root/MVPG/fgclip_ohdcaps_cap.jsonl"
OUT_MD = "/root/MVPG/fgclip_ohdcaps_report.md"

RANK_SIGNALS = ["s_box", "s_short", "s_swap_box", "s_swap_short",
                "d_box", "d_short"]          # d_* = s_* − s_swap_*（扣掉文本先验的净视觉证据）
CAP_SIGNALS = ["global_short", "global_long", "dense_short", "dense_long"]


# ============================================================ rank：子句级多选

def load_groups(path):
    """clause_support_scores.jsonl -> {(gid,cap): [row 按 idx 排序]}。"""
    by = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "s_short" in r:
                r["d_box"] = r["s_box"] - r.get("s_swap_box", 0.0)
                r["d_short"] = r["s_short"] - r.get("s_swap_short", 0.0)
            by[(r["gid"], r["cap"])][r["idx"]] = r
    return {k: [v[i] for i in sorted(v)] for k, v in by.items()}


def rank_metrics(rows, signal, frag_fn=None):
    """对一个 signal 统计 argmin_acc / pairwise_acc。

    frag_fn(row) 为真时该子句被排除出候选集；若幻觉子句本身被排除则该组跳过。
    返回 (n_caption, n_pair, argmin_hits, pair_hits, sum_inv_k)
    """
    n_cap = n_pair = hit_min = hit_pair = 0
    sum_inv_k = 0.0
    for rws in rows:
        cand = rws if frag_fn is None else [r for r in rws if not frag_fn(r)]
        hall = [r for r in cand if r["label"] == 1]
        real = [r for r in cand if r["label"] == 0]
        if len(cand) < 2 or len(hall) != 1 or not real:
            continue
        if any(signal not in r for r in cand):
            continue
        n_cap += 1
        sum_inv_k += 1.0 / len(cand)
        h = hall[0]
        if all(r[signal] > h[signal] for r in real):
            hit_min += 1
        for r in real:
            n_pair += 1
            if r[signal] > h[signal]:
                hit_pair += 1
    return n_cap, n_pair, hit_min, hit_pair, sum_inv_k


def stage_rank(args):
    t0 = time.time()
    rows = list(load_groups(args.scores).values())
    lines = []
    data = {}
    for sig in RANK_SIGNALS:
        for tag, fn in (("all", None), ("dropF2", lambda r: is_cont_fragment(r["text"]))):
            n_cap, n_pair, hm, hp, sik = rank_metrics(rows, sig, fn)
            if not n_cap:
                continue
            data[f"{sig}|{tag}"] = {
                "n_caption": n_cap, "n_pair": n_pair,
                "argmin_acc": hm / n_cap, "argmin_base": sik / n_cap,
                "pairwise_acc": hp / n_pair,
            }
            lines.append((sig, tag, data[f"{sig}|{tag}"]))
    json.dump(data, open(OUT_RANK, "w"), ensure_ascii=False, indent=2)
    print(f"[rank] {len(rows)} 组 caption，耗时 {time.time()-t0:.1f}s -> {OUT_RANK}")
    return lines


# ============================================================ cap：整句级反事实多选

def enc_texts(fg, texts, walk, chunk=2000):
    """分块编码文本（walk_type 决定用哪个头），返回 (N,D) float32。"""
    out = []
    for i in range(0, len(texts), chunk):
        out.append(fg.encode_texts_walk(texts[i:i + chunk], walk_type=walk, batch_size=32))
    return torch.cat(out, dim=0)


def stage_cap(args):
    t0 = time.time()
    recs = load_labels(args.labels, args.limit)
    gids = [r["gid"] for r in recs]
    img_paths = {g: main_image_path(args.image_dir, g) for g in gids}
    recs = [r for r in recs if img_paths[r["gid"]]]
    gids = [r["gid"] for r in recs]
    print(f"[cap] {len(recs)} 张图（有主图）")

    done = set()
    if os.path.isfile(OUT_CAP) and args.resume:
        for line in open(OUT_CAP, encoding="utf-8"):
            try:
                done.add(json.loads(line)["gid"])
            except Exception:
                pass
        print(f"[cap] 已完成 {len(done)}，续跑剩余")

    fg = load_fgclip(args.fgclip_root, args.device)
    print(f"[cap] FG-CLIP2 loaded {time.time()-t0:.1f}s")

    # 换图对照：把 i 号图的文本打到 (i + n//2) 号图上
    swap_idx = {g: gids[(i + len(gids) // 2) % len(gids)] for i, g in enumerate(gids)}
    swap_cache = {}

    fout = open(OUT_CAP, "a", encoding="utf-8")
    n_img = 0
    for rec in recs:
        gid = rec["gid"]
        if gid in done:
            continue
        p = img_paths[gid]
        sp = img_paths.get(swap_idx[gid]) or p

        # ---- 构造「留一子句」候选 ----
        items = []            # (cap, K, hall_idx, [cand_text, ...])
        for ci, (cls, labs) in enumerate(zip(rec["clauses"], rec["labels"])):
            if sum(labs) != 1 or len(cls) < 2:
                continue
            hi = labs.index(1)
            cands = [" ".join(c for j, c in enumerate(cls) if j != i) for i in range(len(cls))]
            items.append((ci, len(cls), hi, cands))
        if not items:
            continue

        texts, meta = [], []
        for ci, K, hi, cands in items:
            for i, t in enumerate(cands):
                texts.append(t)
                meta.append((ci, K, hi, i))

        # ---- 图像特征 ----
        gimg = fg.encode_image_paths([p, sp], batch_size=2)     # (2,D)
        dense, valid = fg.image_dense_features([p], batch_size=1)
        dense, valid = dense[0], valid[0]

        sims = {}
        for walk in ("short", "long"):
            tf = enc_texts(fg, texts, walk)
            tn = F.normalize(tf, p=2, dim=-1)
            sims[f"global_{walk}"] = (gimg[0] @ tn.T).tolist()
            sims[f"dense_{walk}"] = sim_columns(dense, valid, tf).max(dim=0).values.tolist()
        # 换图对照（只对 global 特征做，省显存）
        ts = F.normalize(enc_texts(fg, texts, "short"), p=2, dim=-1)
        sims["swap_global_short"] = (gimg[1] @ ts.T).tolist()

        # ---- 按 caption 汇总排名 ----
        rows_out = []
        by_cap = defaultdict(list)
        for i, (ci, K, hi, j) in enumerate(meta):
            by_cap[ci].append((j, hi, K, i))
        for ci, ent in by_cap.items():
            row = {"gid": gid, "cap": ci, "K": ent[0][2]}
            for sig in list(sims) + ["swap_global_short"]:
                sv = sims[sig]
                order = sorted(range(len(ent)), key=lambda t: -sv[ent[t][3]])
                rank = [ent[t][0] for t in order].index(ent[0][1])   # 忠实候选的名次(0=第一)
                row[sig] = 1.0 / (rank + 1)                           # 用倒数名次，便于聚合
                row[sig + "_top1"] = 1.0 if rank == 0 else 0.0
            rows_out.append(row)
        fout.write(json.dumps({"gid": gid, "rows": rows_out}, ensure_ascii=False) + "\n")
        fout.flush()
        n_img += 1
        if n_img % 50 == 0:
            print(f"[cap] {n_img} 张，{time.time()-t0:.0f}s")
    fout.close()
    print(f"[cap] 完成 {n_img} 张，耗时 {(time.time()-t0)/60:.1f} min -> {OUT_CAP}")


# ============================================================ 报告

def summarize_cap(path):
    agg = defaultdict(lambda: {"n": 0, "top1": 0.0, "rr": 0.0, "base": 0.0})
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        for r in d["rows"]:
            for sig in CAP_SIGNALS + ["swap_global_short"]:
                if sig not in r:
                    continue
                a = agg[sig]
                a["n"] += 1
                a["top1"] += r[sig + "_top1"]
                a["rr"] += r[sig]
                a["base"] += 1.0 / r["K"]
    return {k: {"n": v["n"], "top1_acc": v["top1"] / v["n"],
                "mrr": v["rr"] / v["n"], "top1_base": v["base"] / v["n"]}
            for k, v in agg.items() if v["n"]}


def write_report(rank_rows, cap_sum):
    L = ["# 冻结 FG-CLIP2 幻觉判别能力评测（复刻 OHD-Caps Table 1）\n",
         "- 论文：Liu et al., EMNLP 2024, *Investigating and Mitigating Object Hallucinations "
         "in Pretrained Vision-Language (CLIP) Models*",
         "- 目的：判定「固定阈值调不好」是**方法问题**还是**模型先天不具备该能力**。\n"]

    L.append("## 1. 子句级多选（每条 caption 内挑出幻觉子句）\n")
    L.append("K 为候选子句数（随机基线 1/K ≈ 0.25）；`pairwise` 为任取一个"
             "（幻觉, 真实）对时 `s(真实) > s(幻觉)` 的比例（随机 0.5）。\n")
    L.append("`s_swap_*` 列是**换图基线**：同一子句打到无关图上的同一指标 —— "
             "若它和原图口径的准确率接近，说明判对靠的是**文本先验而非图像证据**。\n")
    L.append("`d_*` 为**差分信号** `s − s_swap`，即扣掉文本先验后剩下的净视觉证据。\n")
    L.append("| 信号 | 候选集 | n_caption | n_pair | argmin 准确率 | 随机 1/K | pairwise 准确率 |")
    L.append("|---|---|---|---|---|---|---|")
    for sig, tag, d in rank_rows:
        L.append(f"| {sig} | {tag} | {d['n_caption']} | {d['n_pair']} | "
                 f"{d['argmin_acc']:.3f} | {d['argmin_base']:.3f} | {d['pairwise_acc']:.3f} |")

    if cap_sum:
        L.append("\n## 2. 整句级反事实多选（留一子句，长度完全对齐）\n")
        L.append("对每条 caption 构造 K 个候选（各删掉 1 个子句），恰有一个删掉的是幻觉子句；"
                 "问模型能否把它排第一。候选之间**长度一致**，唯一变量是「少的是哪个子句」。\n")
        L.append("| 信号 | n | top-1 准确率 | 随机 1/K | MRR |")
        L.append("|---|---|---|---|---|")
        for sig in CAP_SIGNALS + ["swap_global_short"]:
            if sig not in cap_sum:
                continue
            d = cap_sum[sig]
            L.append(f"| {sig} | {d['n']} | {d['top1_acc']:.3f} | "
                     f"{d['top1_base']:.3f} | {d['mrr']:.3f} |")

        # ---------------- 结论（数据驱动） ----------------
        g = cap_sum.get("global_short", {})
        dn = cap_sum.get("dense_short", {})
        sw = cap_sum.get("swap_global_short", {})
        r = {f"{s}|all": v for s, t, v in rank_rows for _ in [0] if t == "all"}
        L.append("\n## 3. 诊断与结论\n")
        L.append("**文本编码口径**：严格对齐官方 README —— tokenizer 只产出 `input_ids`"
                 "（官方 `model_input_names=['input_ids']`），`get_text_features` **不传 `attention_mask`**"
                 "（README 示例即 `get_text_features(**caption_input)`）。FG-CLIP2 文本塔无因果掩码、"
                 "池化取最后一个 token（右侧 padding 下恒为 pad 槽，训练时即充当聚合 query），"
                 "多传 mask 去屏蔽 pad key 属训练分布外用法，会显著改变相似度排序。\n")
        L.append("**`*_long` 仅供参考**：`walk_type=long` 走 196 长的 longtext 位置编码分支，"
                 "而本仓库 tokenizer 采用右侧 padding、池化取最后一个 token，长文本特征更易塌缩，"
                 "数值稳定性不如 short；官方 README 对短 caption 亦推荐 `short`。"
                 "故主口径取 `*_short`，`*_long` 仅列作稳健性对照。\n")
        if "s_short|all" in r and "s_swap_short|all" in r:
            a = r["s_short|all"]["pairwise_acc"]
            s = r["s_swap_short|all"]["pairwise_acc"]
            dd = r.get("d_short|all", {}).get("pairwise_acc", float("nan"))
            am = r["s_short|all"]["argmin_acc"]
            sm = r["s_swap_short|all"]["argmin_acc"]
            L.append(
                f"1. **子句级：原图视觉证据可用，但夹带明显文本先验。** pairwise 上原图 {a:.3f}、"
                f"换一张无关图 {s:.3f}（随机 0.5）——换图复现了原图 {100*s/a:.0f}% 的成绩；"
                f"扣掉换图基线后的差分信号 d_short 为 {dd:.3f}（仍明显高于随机 0.5）。"
                f"argmin 上：原图 {am:.3f} vs 换图 {sm:.3f}"
                f"（随机 {r['s_short|all']['argmin_base']:.3f}）。"
                f"即「幻觉子句含罕见物体 → 与任何图相似度都偏低」的纯语言先验贡献很大，"
                f"但原图信息并未被它完全解释。\n")
        if g and dn:
            base = g["top1_base"]
            d_acc = g["top1_acc"] - base
            word = ("基本等于随机" if d_acc <= 0.02 else
                    "仅略高于随机" if d_acc <= 0.08 else "明显高于随机")
            L.append(
                f"2. **整句级（反事实多选，候选长度对齐）：{word}。** "
                f"`global_short` top-1 = {g['top1_acc']:.3f}、`dense_short` = {dn['top1_acc']:.3f}，"
                f"随机基线 {base:.3f}（{d_acc:+.3f}）；MRR 分别 {g['mrr']:.3f} / {dn['mrr']:.3f}。\n")
        if sw:
            L.append(f"3. **换图对照**：`swap_global_short` top-1 = {sw['top1_acc']:.3f}，"
                     f"随机 {sw['top1_base']:.3f}，"
                     f"{'低于随机，符合「用错图就没信号」的预期' if sw['top1_acc'] <= sw['top1_base'] else '不低于随机'}。\n")
        # 总体判定以整句级反事实多选（真正的幻觉判别）为准，按实际增益分档
        if g:
            d_acc = g["top1_acc"] - g["top1_base"]
            if d_acc <= 0.02:
                verdict = ("**总体判定**：冻结 FG-CLIP2 在整句级反事实多选上**约等于随机**，"
                           "不具备可靠的物体级幻觉判别能力 —— 与 OHD-Caps 的核心结论一致，"
                           "「找一个科学的固定阈值」这条路在原模型上走不通，"
                           "要提升需像论文那样用反事实负样本 fine-tune。")
            elif d_acc <= 0.08:
                verdict = ("**总体判定**：冻结 FG-CLIP2 在整句级反事实多选上**仅略高于随机**，"
                           "单个判别信号尚不够可靠；可先试冻结特征上的线性探针/校准，"
                           "或按 OHD-Caps 用反事实负样本 fine-tune 放大该信号。")
            else:
                verdict = ("**总体判定**：冻结 FG-CLIP2 在整句级反事实多选上**明显高于随机**，"
                           "说明其视觉特征确实携带物体级判别信息。此前「基本等于随机」的结论"
                           "源自文本编码口径偏差（多传 attention_mask），修正后应重新评估"
                           "「固定阈值/校准」路线的可行性，并可继续用反事实负样本 fine-tune 进一步提升。")
            L.append("\n" + verdict + "\n")

    Path(OUT_MD).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"报告 -> {OUT_MD}")


def main():
    ap = argparse.ArgumentParser(description="复刻 OHD-Caps Table 1 的多选排名评测")
    ap.add_argument("--stage", choices=["rank", "cap", "report"], default="rank")
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--labels", default="/root/MVPG/neg_clause_labels.jsonl")
    ap.add_argument("--image-dir", default=DATA_DIR)
    ap.add_argument("--fgclip-root", default="/root/autodl-tmp/cache/fgclip2-base-patch16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    rank_rows = None
    if args.stage == "rank":
        rank_rows = stage_rank(args)
    elif args.stage == "cap":
        stage_cap(args)

    cap_sum = summarize_cap(OUT_CAP) if os.path.isfile(OUT_CAP) else {}
    if rank_rows is None and os.path.isfile(OUT_RANK):
        d = json.load(open(OUT_RANK))
        rank_rows = [(k.split("|")[0], k.split("|")[1], v) for k, v in d.items()]
    write_report(rank_rows or [], cap_sum)


if __name__ == "__main__":
    main()
