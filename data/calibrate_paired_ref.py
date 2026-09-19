"""方案 B（原模型配对参照差分）的离线标定脚本。

在动手改训练代码之前，先在**真实原模型 caption**（不是人工插入幻觉的 neg_captions）上
确定三个量：

  1. τ_match：`short` 头余弦在什么值以上，才算「两个子句在说同一件事」；
  2. δ：匹配到参照子句后做差分，|d| 小于多少算噪声（死区宽度）；
  3. 健康度：按 (τ_match, δ) 跑起来后的覆盖率（有参照的子句占比）、置零率、
     d 的符号与幅度分布。

信号定义（与训练保持一致）：
  - 打分：`box` 头文本特征 × dense patch 特征，masked max → s_box（raw，不归一化）；
  - 匹配：`short` 头文本特征余弦；
  - 差分：d = s_box(策略子句) − mean(s_box(匹配到的参照子句, cos >= τ_match))。

数据：image-dir 根级 model_captions.json，只取**主图行**（key == gid）。
默认 `--policy-key == --ref-key`（同基座自比）：参照池 = 同一条 caption 内除自己以外的
子句，用来标定 δ。也可以 `--policy-key` 换成别的来源（另一模型、或新采样的 caption json，
结构相同）做更现实的诊断。

重要提醒：同基座自比时两侧文本几乎重合，|d| 会偏小，得到的是 **δ 的下界**；真实训练里
策略与参照是两次独立生成，文本差异更大，δ 应取更保守（更大）的值。

用法（注意用 mvpg 环境）：
    /root/miniconda3/envs/mvpg/bin/python data/calibrate_paired_ref.py \
        --image-dir /root/autodl-tmp/data/SA1B_5_threshold \
        --limit 200

产物：
    /root/MVPG/paired_ref_calib_results.json   # 汇总 + 逐子句明细
    /root/MVPG/paired_ref_calib_report.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/MVPG")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from mvpg_run.fgclip_local import load_fgclip, preprocess_images  # noqa: E402

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLAUSE_SPLIT = re.compile(r"[.,;]")

# 判定「两个子句是否在说同一件事」用的停用词（仅用于构造标定标签，不参与信号）
STOP = set("""a an the of in on at to for with and or but is are was were be been being this that
these those it its their his her there here as by from into over under near next left right top
bottom front back side view image picture scene also which who whom whose what when where while
during after before between among some any few many most more less very much such other others
another one two three several can could may might will would shall should do does did done has
have had not no nor so than then them they you your we our us my me i he she him""".split())


def split_clauses(caption):
    """联合 `.,;` 切分并过滤 <3 词的碎片（与验证脚本 / 训练保持一致）。"""
    out = []
    for p in CLAUSE_SPLIT.split(caption or ""):
        p = p.strip()
        if len(p.split()) >= 3:
            out.append(p)
    return out


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * len(s))) - 1))
    return s[i]


def auc_score(pos, neg):
    """Mann-Whitney AUC（带并列平均秩）。"""
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
    rp = sum(ranks[v] for v in pos)
    u = rp - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def pick_threshold(pos, neg, target_precision):
    """在 short 余弦上取满足 precision>=target 的最低阈值（尽量保住召回）。"""
    cands = sorted(set(pos + neg), reverse=True)
    best = float("nan")
    for t in cands:
        tp = sum(1 for v in pos if v >= t)
        fp = sum(1 for v in neg if v >= t)
        if tp + fp == 0:
            continue
        if tp / (tp + fp) >= target_precision:
            best = t
        elif best == best:  # 一旦跌破目标精度就不再往下放
            break
    return best


def main_img_path(image_dir, gid):
    d = Path(image_dir) / gid
    if not d.is_dir():
        return None
    for s in IMG_SUFFIXES:
        p = d / (gid + s)
        if p.is_file():
            return str(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="/root/autodl-tmp/data/SA1B_5_threshold")
    ap.add_argument("--fgclip-root", default="/root/autodl-tmp/cache/fgclip2-base-patch16")
    ap.add_argument("--captions-json", default="model_captions.json")
    ap.add_argument("--ref-key", default="llava-v1.5-7b")
    ap.add_argument("--policy-key", default=None, help="默认与 --ref-key 相同（同基座自比）")
    ap.add_argument("--limit", type=int, default=200, help="<=0 表示全量")
    ap.add_argument("--topk", type=int, default=5, help="每个策略子句保留的候选参照数")
    ap.add_argument("--overlap-thr", type=float, default=0.4,
                    help="内容词 Jaccard 高于此值记为「同内容」对，用于标定 τ_match")
    ap.add_argument("--prec-target", type=float, default=0.9,
                    help="τ_match 取短余弦在该精度以上的最低阈值")
    ap.add_argument("--delta-q", type=float, default=0.9, help="δ 取 |d| 的该分位数")
    ap.add_argument("--out-json", default="/root/MVPG/paired_ref_calib_results.json")
    ap.add_argument("--out-md", default="/root/MVPG/paired_ref_calib_report.md")
    args = ap.parse_args()

    policy_key = args.policy_key or args.ref_key
    self_mode = policy_key == args.ref_key
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[calib] device={device} policy_key={policy_key} ref_key={args.ref_key} "
          f"self_mode={self_mode}")

    caps = json.load(open(Path(args.image_dir) / args.captions_json))
    gids = []
    for gid in sorted(caps):
        g = caps[gid]
        if not isinstance(g, dict):
            continue
        if (g.get(policy_key) or {}).get(gid) and (g.get(args.ref_key) or {}).get(gid):
            if main_img_path(args.image_dir, gid):
                gids.append(gid)
    if args.limit and args.limit > 0:
        gids = gids[:args.limit]
    print(f"[calib] 可用主图组数={len(gids)}（limit={args.limit}）")

    fg = load_fgclip(args.fgclip_root, device)

    pair_cos, pair_lab = [], []        # 匹配头标定：所有 (策略子句, 参照子句) 对
    rows = []                          # 逐策略子句明细
    n_skip = 0

    for n, gid in enumerate(gids):
        pol_cap = caps[gid][policy_key][gid]
        ref_cap = caps[gid][args.ref_key][gid]
        pol_cl = split_clauses(pol_cap)
        ref_cl = split_clauses(ref_cap)
        if len(pol_cl) < 2 or len(ref_cl) < 2:
            n_skip += 1
            continue

        im = Image.open(main_img_path(args.image_dir, gid)).convert("RGB")
        pv, mask, shapes = preprocess_images([im])
        pv, mask, shapes = pv.to(device), mask.to(device), shapes.to(device)
        with torch.no_grad():
            dense = fg.model.get_image_dense_feature(
                pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=shapes)
            box_pol = fg.encode_texts_walk(pol_cl, walk_type="box")
            box_ref = fg.encode_texts_walk(ref_cl, walk_type="box")
            short_pol = fg.encode_texts_walk(pol_cl, walk_type="short")
            short_ref = fg.encode_texts_walk(ref_cl, walk_type="short")

        valid = mask.bool()[0].to(device)
        dn = F.normalize(dense[0], p=2, dim=-1)                       # (P,H)

        def s_box(box_feat):
            sim = dn @ F.normalize(box_feat, p=2, dim=-1).T           # (P,N)
            sim = sim.masked_fill(~valid.unsqueeze(1), -1e9)
            return sim.max(dim=0).values.tolist()

        s_pol = s_box(box_pol)
        s_ref = s_box(box_ref)
        cos = (F.normalize(short_pol, p=2, dim=-1)
               @ F.normalize(short_ref, p=2, dim=-1).T).tolist()      # (Np,Nr)

        # 匹配头标定数据：全部对，标签由内容词 Jaccard 给出
        for i, pc in enumerate(pol_cl):
            pcw = content_words(pc)
            for j, rc in enumerate(ref_cl):
                if self_mode and i == j:
                    continue
                pair_cos.append(cos[i][j])
                pair_lab.append(1 if jaccard(pcw, content_words(rc)) >= args.overlap_thr else 0)

        # 逐子句明细：存 top-k 候选，τ_match 之后再筛选（避免 k 不足时把 -inf 平进来）
        for i, pc in enumerate(pol_cl):
            cand = []
            for j in sorted(range(len(ref_cl)), key=lambda x: -cos[i][x])[:args.topk]:
                if self_mode and i == j:
                    continue
                cand.append({"cos": cos[i][j], "s_ref": s_ref[j], "clause": ref_cl[j]})
            rows.append({"gid": gid, "clause": pc, "s_box": s_pol[i], "cands": cand})

        if (n + 1) % 20 == 0:
            print(f"[calib] {n + 1}/{len(gids)}")

    # ---------------- 1. τ_match ----------------
    pos = [c for c, l in zip(pair_cos, pair_lab) if l == 1]
    neg = [c for c, l in zip(pair_cos, pair_lab) if l == 0]
    tau = pick_threshold(pos, neg, args.prec_target)
    match_stats = {
        "n_pair": len(pair_cos), "n_pos": len(pos), "n_neg": len(neg),
        "pos_cos_mean": sum(pos) / len(pos) if pos else float("nan"),
        "neg_cos_mean": sum(neg) / len(neg) if neg else float("nan"),
        "auc": auc_score(pos, neg),
        "tau_match": tau, "prec_target": args.prec_target,
        "precision_at_tau": (sum(1 for v in pos if v >= tau) /
                             max(1, sum(1 for v in pos if v >= tau) + sum(1 for v in neg if v >= tau))),
        "recall_at_tau": (sum(1 for v in pos if v >= tau) / len(pos)) if pos else float("nan"),
    }

    # ---------------- 2. 覆盖率 + d 分布 ----------------
    def matched(row):
        return [c for c in row["cands"] if c["cos"] >= tau]

    cov_rows = [r for r in rows if matched(r)]
    coverage = len(cov_rows) / len(rows) if rows else 0.0
    d_all = [r["s_box"] - sum(c["s_ref"] for c in matched(r)) / len(matched(r)) for r in cov_rows]

    # ---------------- 3. δ 与健康度 ----------------
    delta = quantile([abs(x) for x in d_all], args.delta_q)
    nonzero = [x for x in d_all if abs(x) > delta]
    health = {
        "n_clause": len(rows), "n_covered": len(cov_rows), "coverage": coverage,
        "d_mean": sum(d_all) / len(d_all) if d_all else float("nan"),
        "d_std": (torch.tensor(d_all).std().item() if len(d_all) > 1 else float("nan")),
        "abs_d_p50": quantile([abs(x) for x in d_all], 0.5),
        "abs_d_p80": quantile([abs(x) for x in d_all], 0.8),
        "abs_d_p90": quantile([abs(x) for x in d_all], 0.9),
        "abs_d_p95": quantile([abs(x) for x in d_all], 0.95),
        "delta": delta, "delta_q": args.delta_q,
        "zero_rate": (len(d_all) - len(nonzero)) / len(d_all) if d_all else 0.0,
        "pos_rate": sum(1 for x in nonzero if x > 0) / len(nonzero) if nonzero else 0.0,
        "neg_rate": sum(1 for x in nonzero if x < 0) / len(nonzero) if nonzero else 0.0,
        "abs_d_nonzero_mean": (sum(abs(x) for x in nonzero) / len(nonzero)) if nonzero else 0.0,
    }

    summary = {
        "policy_key": policy_key, "ref_key": args.ref_key, "self_mode": self_mode,
        "n_gid": len(gids), "n_gid_skipped": n_skip, "topk": args.topk,
        "overlap_thr": args.overlap_thr,
        "match": match_stats, "health": health,
    }
    json.dump({"summary": summary, "rows": rows}, open(args.out_json, "w"),
              ensure_ascii=False, indent=1)

    m, h = match_stats, health
    L = []
    L.append("# 方案 B（配对参照差分）离线标定\n")
    L.append(f"- 策略 caption 来源：`{policy_key}`；参照来源：`{args.ref_key}`"
             f"（自比模式={self_mode}）\n")
    L.append(f"- 主图组数：{len(gids)}（跳过 {n_skip}）；策略子句数：{h['n_clause']}\n")
    L.append(f"- 标签：内容词 Jaccard >= {args.overlap_thr} 记为「同内容」\n")
    L.append("\n## 1. 匹配头标定（short 余弦）\n")
    L.append(f"- 同内容对 n={m['n_pos']}，余弦均值 **{m['pos_cos_mean']:.4f}**\n")
    L.append(f"- 异内容对 n={m['n_neg']}，余弦均值 **{m['neg_cos_mean']:.4f}**\n")
    L.append(f"- AUC = **{m['auc']:.4f}**\n")
    L.append(f"- 目标精度 {m['prec_target']:.2f} → **τ_match = {m['tau_match']:.4f}**"
             f"（精度 {m['precision_at_tau']:.3f} / 召回 {m['recall_at_tau']:.3f}）\n")
    L.append("\n## 2. 覆盖率与差分 d\n")
    L.append(f"- 覆盖率（有 cos>=τ 的参照）：**{h['coverage'] * 100:.1f}%**"
             f"（{h['n_covered']}/{h['n_clause']}）\n")
    L.append(f"- d 均值 {h['d_mean']:+.5f}，标准差 {h['d_std']:.5f}"
             f"（均值应≈0，明显偏离说明匹配有系统性偏置）\n")
    L.append(f"- |d| 分位：p50={h['abs_d_p50']:.4f} p80={h['abs_d_p80']:.4f} "
             f"p90={h['abs_d_p90']:.4f} p95={h['abs_d_p95']:.4f}\n")
    L.append("\n## 3. 死区与健康度\n")
    L.append(f"- **δ = {h['delta']:.4f}**（|d| 的 {h['delta_q']:.2f} 分位）\n")
    L.append(f"- 置零率（|d|<=δ）：**{h['zero_rate'] * 100:.1f}%**"
             f"（>90% 说明阈值过大，信号被吃掉）\n")
    L.append(f"- 非零子句中 d>0：{h['pos_rate'] * 100:.1f}%，d<0：{h['neg_rate'] * 100:.1f}%，"
             f"平均 |d|={h['abs_d_nonzero_mean']:.4f}\n")
    L.append("\n## 注意\n")
    L.append("- 自比模式下两侧文本几乎重合，|d| 偏小，此处 δ 是**下界**；"
             "真实训练中策略与参照独立生成，δ 应取更保守（更大）的值。\n")
    L.append("- 若 `d 均值` 明显不为 0，先查匹配头阈值，不要直接进训练。\n")
    L.append("- 覆盖率过低说明参照池（同基座主图 caption）与策略输出风格差异大，"
             "需要放宽 τ 或改匹配方式。\n")
    open(args.out_md, "w").write("".join(L))
    print("".join(L))
    print(f"[calib] 写出 {args.out_json} 与 {args.out_md}")


if __name__ == "__main__":
    main()
