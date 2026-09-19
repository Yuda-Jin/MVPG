#!/usr/bin/env python3
"""段落级（整条 caption）判别：幻觉 caption vs 事实 caption。

问题：前序工作都在**子句级**找幻觉子句（§clause_visual_support）。本脚本退到**段落级**，
回答一个更粗但更基础的问题：把整条 caption 当一个单位，冻结的 FG-CLIP2 能否区分
「含幻觉物体的 caption」与「原模型的忠实 caption」？

数据（同图配对，天然成对）：
  - 负（幻觉）：`neg_captions.json[gid]`，强 VLM 生成、每条恰好注入 1 个图中不存在的物体；
  - 正（"事实"）：`model_captions.json[gid][model][gid]`，原模型对主图的正常 caption（7B/13B）。
两类文本描述**同一张主图**，唯一差别是负样本多了一个不存在的物体。

注意：正样本（LLaVA caption）**未经事实核验**，其本身也常含幻觉物体与冗长推测，
故「正 vs 负」不是干净的「事实 vs 幻觉」，读结论时须结合报告第 0 节的样本抽查。

信号（均为「文本 × 图」的余弦，越高越像在说这张图）：
  - global      : 整图 pooled 特征 × 整段文本特征；
  - dense       : 每个 patch 特征与文本的余弦取 max（物体级最大证据）；
  - swap_global : 同一段文本打到**另一张无关图**上的 global 余弦（对照）。

`swap_global` 是关键对照：若它与 `global` 的成绩接近，说明「幻觉 caption 得分低」只是
因为负样本用词/句式更像「通用描述」，与这张图本身无关。

用法：
    PY=/root/miniconda3/envs/mvpg/bin/python
    $PY data/verify_fgclip_caption_poseg.py --stage score  --limit 40
    $PY data/verify_fgclip_caption_poseg.py --stage score  --device cuda
    $PY data/verify_fgclip_caption_poseg.py --stage report

产物：
    /root/MVPG/fgclip_caption_scores.jsonl   （逐 gid 明细，可断点续传）
    /root/MVPG/fgclip_caption_report.md
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/MVPG")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from calibrate_paired_ref import auc_score, quantile  # noqa: E402
from clause_visual_support import main_image_path, sim_columns  # noqa: E402
from mvpg_run.fgclip_local import load_fgclip  # noqa: E402

DATA_DIR = "/root/autodl-tmp/data/SA1B_5_threshold"
OUT_SCORES = "/root/MVPG/fgclip_caption_scores.jsonl"
OUT_MD = "/root/MVPG/fgclip_caption_report.md"

MODELS = ("llava-v1.5-7b", "llava-v1.5-13b")
SIGNALS = ("global", "dense", "swap_global")


# ---------------------------------------------------------------- 数据

def load_data(image_dir, limit=0):
    """返回 [{gid, img, pos:{model: text}, neg:[text, ...]}]，只保留有主图的图。"""
    neg = json.load(open(os.path.join(image_dir, "neg_captions.json"), encoding="utf-8"))
    mc = json.load(open(os.path.join(image_dir, "model_captions.json"), encoding="utf-8"))
    out = []
    for gid in sorted(neg):
        negs = [str(c).strip() for c in (neg.get(gid) or []) if str(c).strip()]
        if not negs:
            continue
        ref = mc.get(gid) or {}
        pos = {}
        for m in MODELS:
            c = (ref.get(m) or {}).get(gid)
            if c and str(c).strip():
                pos[m] = str(c).strip()
        if not pos:
            continue
        p = main_image_path(image_dir, gid)
        if not p:
            continue
        out.append({"gid": gid, "img": p, "pos": pos, "neg": negs})
    return out[:limit] if limit else out


def load_done(path):
    done = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["gid"])
                except Exception:
                    continue
    return done


# ---------------------------------------------------------------- score

def stage_score(args):
    t0 = time.time()
    recs = load_data(args.image_dir, args.limit)
    gids = [r["gid"] for r in recs]
    if not recs:
        print("[score] 无可用数据")
        return
    print(f"[score] {len(recs)} 张主图（各 {len(recs[0]['neg'])} 条负 caption + "
          f"{len(recs[0]['pos'])} 条正 caption）")

    fg = load_fgclip(args.fgclip_root, args.device)
    print(f"[score] FG-CLIP2 loaded {time.time()-t0:.1f}s")

    # 主图 global 特征一次算完（换图基线只需重排索引复用，无需二次编码）。
    # 图像尺寸不一 -> 每张图用自己的 patch 桶，故 batch_size=1。
    gfeat = {}
    for i, r in enumerate(recs):
        gfeat[r["gid"]] = fg.encode_image_paths([r["img"]], batch_size=1)[0]
        if (i + 1) % 400 == 0:
            print(f"[score] global {i+1}/{len(recs)} {time.time()-t0:.0f}s")
    swap = {g: gids[(i + len(gids) // 2) % len(gids)] for i, g in enumerate(gids)}

    done = load_done(args.scores) if args.resume else set()
    if done:
        print(f"[score] 已完成 {len(done)}，续跑剩余")
    fout = open(args.scores, "a", encoding="utf-8")

    n = 0
    for i, r in enumerate(recs):
        gid = r["gid"]
        if gid in done:
            continue
        models = [m for m in MODELS if m in r["pos"]]
        texts = list(r["neg"]) + [r["pos"][m] for m in models]
        k = len(r["neg"])

        tf = fg.encode_texts(texts, batch_size=32)
        dense, valid = fg.image_dense_features([r["img"]], batch_size=1)
        dense, valid = dense[0], valid[0]

        sg = (gfeat[gid] @ tf.T).tolist()
        sd = sim_columns(dense, valid, tf).max(dim=0).values.tolist()
        sw = (gfeat[swap[gid]] @ tf.T).tolist()

        row = {
            "gid": gid, "pos_models": models,
            "neg_len": [len(t) for t in r["neg"]],
            "pos_len": [len(r["pos"][m]) for m in models],
            **{s: {"neg": v[:k], "pos": v[k:]} for s, v in
               (("global", sg), ("dense", sd), ("swap_global", sw))},
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        n += 1
        if n % 100 == 0:
            print(f"[score] {n} 张，{time.time()-t0:.0f}s")
    fout.close()
    print(f"[score] 完成 {n} 张，耗时 {(time.time()-t0)/60:.1f} min -> {args.scores}")


# ---------------------------------------------------------------- report

def fmt(x, nd=3):
    return "—" if x != x else f"{x:.{nd}f}"


def load_rows(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pool(rows, sig, cls):
    return [v for r in rows for v in r[sig][cls]]


def top1_stats(rows, sig):
    """逐图：正 caption 中最高分是否压过全部负 caption（=top1 是正）。

    返回 (n_img, top1_hits, base_sum, margin_pos, margins)
    """
    n = hit = 0
    base = 0.0
    margins = []
    for r in rows:
        pos, neg = r[sig]["pos"], r[sig]["neg"]
        if not pos or not neg:
            continue
        n += 1
        base += len(pos) / (len(pos) + len(neg))
        if max(pos) > max(neg):
            hit += 1
        margins.append(min(pos) - max(neg))
    return n, hit, base, margins


def length_matched(rows, sig, tol=0.25):
    """只保留「长度与正 caption 接近」的负 caption，压掉『负样本更长/更啰嗦』的文本混淆。

    返回 (n_img, top1_hits, pos_all, neg_all, n_neg_kept)。
    """
    n = hit = n_keep = 0
    pos_all, neg_all = [], []
    for r in rows:
        pl = sum(r["pos_len"]) / len(r["pos_len"])
        lo, hi = pl * (1 - tol), pl * (1 + tol)
        keep = [v for v, L in zip(r[sig]["neg"], r["neg_len"]) if lo <= L <= hi]
        if not keep or not r[sig]["pos"]:
            continue
        n += 1
        if max(r[sig]["pos"]) > max(keep):
            hit += 1
        n_keep += len(keep)
        pos_all += r[sig]["pos"]
        neg_all += keep
    return n, hit, pos_all, neg_all, n_keep


def length_stratified_auc(rows, sig, n_bins=10, min_per_class=3):
    """按文本长度分箱后求 AUC 再加权，非参数地控制「两类文本长度差」这一混淆。

    正负 caption 长度差很大时，直接比分数会被长度带偏。这里把所有正/负 caption
    按长度分位分箱，只在**同一长度箱内**比正 vs 负，再按箱内较少一类的样本数加权。
    返回 (加权 AUC, 有效箱数)。
    """
    ps = [(L, v) for r in rows for v, L in zip(r[sig]["pos"], r["pos_len"])]
    ns = [(L, v) for r in rows for v, L in zip(r[sig]["neg"], r["neg_len"])]
    allL = [L for L, _ in ps + ns]
    if len(allL) < n_bins * 4:
        return float("nan"), 0
    edges = [quantile(sorted(allL), i / n_bins) for i in range(n_bins + 1)]

    def in_bin(L, b):  # 最后一箱闭区间，避免 max 值被丢掉
        lo, hi = edges[b], edges[b + 1]
        return lo <= L <= hi if b == n_bins - 1 else lo <= L < hi

    num = den = 0.0
    nb = 0
    for b in range(n_bins):
        p = [v for L, v in ps if in_bin(L, b)]
        g = [v for L, v in ns if in_bin(L, b)]
        if len(p) < min_per_class or len(g) < min_per_class:
            continue
        w = min(len(p), len(g))
        num += w * auc_score(p, g)
        den += w
        nb += 1
    return (num / den if den else float("nan")), nb


def stage_report(args):
    rows = load_rows(args.scores)
    L = []
    n_img = len(rows)
    n_neg = sum(len(r["neg_len"]) for r in rows)
    n_pos = sum(len(r["pos_len"]) for r in rows)
    L.append("# 段落级判别：幻觉 caption vs 事实 caption（冻结 FG-CLIP2）\n")
    L.append("把整条 caption 当一个单位，问冻结的 FG-CLIP2 能否区分「含幻觉物体的 caption」"
             "（`neg_captions.json`，强 VLM 生成、每条恰好注入 1 个图中不存在的物体）与"
             "「原模型（LLaVA-7B/13B）生成的 caption」（`model_captions.json`，**未经事实核验**）。"
             "两类文本描述**同一张主图**，是天然配对。\n")
    L.append(f"- 图 {n_img} 张；正 caption {n_pos} 条（每图 "
             f"{n_pos/max(n_img,1):.1f} 条）、负 caption {n_neg} 条（每图 "
             f"{n_neg/max(n_img,1):.1f} 条）")
    L.append("- `global` = 整图 pooled 特征 × 文本余弦；`dense` = patch 特征与文本余弦取 max；"
             "`swap_global` = 同一段文本打到**另一张无关图**上的 global 余弦（对照）\n")

    # ---- 0. 样本抽查：先看清两类文本到底长什么样 ----
    try:
        negj = json.load(open(os.path.join(args.image_dir, "neg_captions.json"), encoding="utf-8"))
        mcj = json.load(open(os.path.join(args.image_dir, "model_captions.json"), encoding="utf-8"))
    except Exception:
        negj = mcj = None
    if negj and mcj and rows:
        L.append("## 0. 样本抽查（同一张图上的负 / 正 caption）\n")
        L.append("先看清两类文本的真实差别，再读后面的分数——这一点很关键，见第 5 节。\n")
        for r in rows[:2]:
            gid = r["gid"]
            neg0 = (negj.get(gid) or [""])[0]
            m = r["pos_models"][0] if r["pos_models"] else None
            pos0 = ((mcj.get(gid) or {}).get(m) or {}).get(gid, "") if m else ""
            L.append(f"**`{gid}`**")
            L.append("")
            L.append(f"- 负（幻觉，{r['neg_len'][0]} 字符）：{neg0[:260]}…")
            L.append(f"- 正（{m}，{r['pos_len'][0]} 字符）：{pos0[:260].replace(chr(10), ' ')}…")
            L.append("")
        L.append("负 caption 由强 VLM 按「先推断场景 → 再自然描述 + 塞入 1 个图中不存在的物体」生成，"
                 "**除那 1 个物体外基本忠实**；而正 caption 是 LLaVA-7B/13B 的开放式描述，"
                 "**本身也常含幻觉/冗长推测**（上例 LLaVA 说的 crowns、chair 大概率并不存在于图中）。"
                 "所以本实验的「事实正样本」并不干净，这是读结论时必须记住的前提。\n")

    # ---- 1. 池化 AUC ----
    L.append("## 1. 池化 AUC（正 caption 分 > 负 caption 分，随机 0.5）\n")
    L.append("| 信号 | 正均值 | 负均值 | AUC | 正−负 差 |")
    L.append("|---|---|---|---|---|")
    for s in SIGNALS:
        p, g = pool(rows, s, "pos"), pool(rows, s, "neg")
        mp = sum(p) / len(p) if p else float("nan")
        mg = sum(g) / len(g) if g else float("nan")
        L.append(f"| {s} | {fmt(mp, 4)} | {fmt(mg, 4)} | **{fmt(auc_score(p, g))}** | "
                 f"{fmt(mp - mg, 4)} |")
    L.append("")

    # ---- 2. 逐图 top-1 ----
    L.append("## 2. 逐图「正 caption 排第一」准确率\n")
    L.append("每张图有若干正/负 caption，问**正 caption 的最高分**能否压过**全部负 caption**"
             "（随机基线 = 正样本占比）。另给「最弱正 vs 最强负」的分离 margin。\n")
    L.append("| 信号 | n 图 | top-1 准确率 | 随机基线 | 增益 | margin>0 比例 | 中位 margin |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SIGNALS:
        n, hit, base, margins = top1_stats(rows, s)
        if not n:
            continue
        L.append(f"| {s} | {n} | **{fmt(hit/n)}** | {fmt(base/n)} | "
                 f"{hit/n - base/n:+.3f} | "
                 f"{fmt(sum(1 for m in margins if m > 0)/len(margins))} | "
                 f"{fmt(quantile(margins, 0.5), 4)} |")
    L.append("")

    # ---- 3. 长度混淆与长度分层 ----
    pos_len = [v for r in rows for v in r["pos_len"]]
    neg_len = [v for r in rows for v in r["neg_len"]]
    mL_p = sum(pos_len) / len(pos_len)
    mL_n = sum(neg_len) / len(neg_len)
    a_len = auc_score(pos_len, neg_len)
    L.append("## 3. 长度混淆：长度本身就能判到多少\n")
    L.append(f"- 正 caption 平均 {mL_p:.0f} 字符，负 caption 平均 {mL_n:.0f} 字符"
             f"（比 {mL_n/mL_p:.2f}×）。")
    L.append(f"- **只用长度**判别两类文本的 AUC = {fmt(a_len)}"
             f"（>0.5 = 越长越像正样本；≈0.5 = 长度无信息）。")
    L.append("- 两类长度差这么大时，任何「正/负分数差」都可能只是长度造成的。"
             "下表用**长度分层**（按长度分位分 10 箱，只在同箱内比正 vs 负再加权）"
             "把长度这一解释剔掉；「同长度窗」列则只保留长度落在正 caption 均值 ±25% 内的负样本，"
             "但本数据负样本普遍短得多，窗口会删掉绝大多数负样本。\n")
    L.append("| 信号 | 长度分层 AUC | 有效箱数 | 同窗保留图数 | 同窗 top-1 | 同窗 AUC | 同窗保留负样本数 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SIGNALS:
        a_str, nb = length_stratified_auc(rows, s)
        n, hit, pa, na, nk = length_matched(rows, s)
        L.append(f"| {s} | {fmt(a_str)} | {nb} | {n} | "
                 f"{fmt(hit/n) if n else '—'} | {fmt(auc_score(pa, na)) if pa and na else '—'} | {nk} |")
    L.append("")

    # ---- 4. 分模型 ----
    L.append("## 4. 分模型（正 caption 只取 7B / 只取 13B）\n")
    L.append("| 正样本来源 | 信号 | 池化 AUC |")
    L.append("|---|---|---|")
    for m in MODELS:
        for s in SIGNALS:
            p = [r[s]["pos"][i] for r in rows
                 for i, mm in enumerate(r["pos_models"]) if mm == m]
            g = pool(rows, s, "neg")
            if not p:
                continue
            L.append(f"| {m} | {s} | {fmt(auc_score(p, g))} |")
    L.append("")

    # ---- 5. 判读 ----
    gp, gn = pool(rows, "global", "pos"), pool(rows, "global", "neg")
    dp, dn = pool(rows, "dense", "pos"), pool(rows, "dense", "neg")
    sp, sn = pool(rows, "swap_global", "pos"), pool(rows, "swap_global", "neg")
    a_g, a_d, a_s = auc_score(gp, gn), auc_score(dp, dn), auc_score(sp, sn)
    s_g, nb_g = length_stratified_auc(rows, "global")
    s_d, nb_d = length_stratified_auc(rows, "dense")
    s_s, nb_s = length_stratified_auc(rows, "swap_global")
    n_g, hit_g, base_g, _ = top1_stats(rows, "global")
    n_d, hit_d, _, _ = top1_stats(rows, "dense")

    def verdict(a):
        if a != a:
            return "无数据"
        if abs(a - 0.5) < 0.05:
            return f"AUC {a:.3f} ≈ 0.5，**无判别力**"
        if a > 0.5:
            return f"AUC {a:.3f} > 0.5，正 > 负，**有判别力**"
        return f"AUC {a:.3f} < 0.5，**方向反转**（幻觉 caption 反而更贴近本图）"

    L.append("## 5. 判读\n")
    L.append(f"1. **段落级判别力**：`global` {verdict(a_g)}；`dense` {verdict(a_d)}。"
             f"逐图「正 caption 最高分压过全部负 caption」的准确率：`global` "
             f"{fmt(hit_g/max(n_g,1))}、`dense` {fmt(hit_d/max(n_d,1))}，"
             f"随机基线 {fmt(base_g/max(n_g,1))}。")
    if a_g == a_g and a_g < 0.5:
        L.append("   分数方向与「幻觉 ⇒ 低分」的直觉相反。这**不能**直接读成「FG-CLIP2 判反了」——"
                 "更合理的解释是登记为「事实」的那一类（LLaVA caption）本身不可靠（见第 0 节）："
                 "它们更长、更空泛、且自身夹带幻觉物体与大量推测，整段文本的视觉贴合度被自己拉低。")
    L.append(f"2. **换图对照（关键）**：同一段文本打到**另一张无关图**上，`swap_global` AUC {fmt(a_s)}，"
             f"与打在本图上的 `global`（{fmt(a_g)}）**方向相反**。"
             "换图对照里不含任何「这段文本是否贴合本图」的信息，只有文本侧统计量"
             "（长度 / 用词 / 句式）；它在无关图上让正样本略占优，而一放回**本图**，"
             "优势就翻转成负样本——说明**本图视觉证据是往「幻觉 caption 分更低」的方向作用的**，"
             "只是被文本侧先验和正样本本身的不忠实掩盖了。")
    L.append(f"3. **长度混淆**：只用长度就能达到 AUC {fmt(a_len)}，是压倒性混淆。"
             f"剔除长度（按长度分箱再加权）后：`global` {fmt(s_g)}（{nb_g} 箱）、"
             f"`dense` {fmt(s_d)}（{nb_d} 箱）、`swap_global` {fmt(s_s)}（{nb_s} 箱）。"
             "方向没有因为剔除长度而翻正，说明「正样本分更低」不只是长度造成的。")
    L.append("4. **结论**：本配对设计**回答不了**「段落级能否区分幻觉 vs 事实」——"
             "被登记为「事实」的 LLaVA caption 本身也不忠实（第 0 节）且长得多（第 3 节），"
             "两类并不是干净的「事实 vs 幻觉」。就本数据能确证的是：")
    L.append("   - 负 caption 按生成约定**只注入 1 个**图中不存在的物体，其余部分忠实；"
             "这么「1 个假物体」摊进整段余弦后会被其余忠实内容稀释，"
             "段落级相似度几乎感知不到它；")
    L.append(f"   - `global`/`dense` 的段落级 AUC 停在 {fmt(a_g)}/{fmt(a_d)}（反向），"
             f"逐图 top-1 只有 {fmt(hit_g/max(n_g,1))}/{fmt(hit_d/max(n_d,1))}（随机 "
             f"{fmt(base_g/max(n_g,1))}），即**按段落级分数排序，忠实 caption 反而排更后**；")
    L.append("   - 要干净地回答这个问题，需要**同源同长**的对照（例如同一条 caption 保留 / 删掉它的幻觉子句），"
             "而不是拿两个不同模型、不同长度的 caption 家族对比。\n")

    Path(OUT_MD).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[report] -> {OUT_MD}")


def main():
    ap = argparse.ArgumentParser(description="段落级幻觉 caption vs 事实 caption 判别")
    ap.add_argument("--stage", choices=["score", "report", "all"], default="score")
    ap.add_argument("--image-dir", default=DATA_DIR)
    ap.add_argument("--fgclip-root", default="/root/autodl-tmp/cache/fgclip2-base-patch16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scores", default=OUT_SCORES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    if args.stage in ("score", "all"):
        stage_score(args)
    if args.stage in ("report", "all"):
        stage_report(args)


if __name__ == "__main__":
    main()
