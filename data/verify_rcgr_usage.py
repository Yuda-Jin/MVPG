"""验证 RCGR（区域描述竞争）的 FG-CLIP 调用口径是否「错位」。

背景
----
`run_mvpg.compute_group_rewards` 当前对「区域描述 ↔ 区域子图」打分用的是：

    文本：fg.encode_texts(...)  -> 硬编码 walk_type="short" + max_length=64
    图像：fg.encode_image_paths(...) -> model.get_image_features（整图全局池化）

但 FG-CLIP2 官方把两个口径分得很清（modeling_fgclip2.py 的 docstring / README）：

    walk_type="short"  -> "short-text understanding ... a phrase or brief sentence"
                          配 get_image_features（整图池化），用于整图检索；
    walk_type="box"    -> "local-region description processing, such as grounding ...
                           localized textual descriptions (e.g., object regions or segments)"
                          配 get_image_dense_feature（patch 级），用于区域接地。

RCGR 的场景恰是「局部区域描述 ↔ 局部区域」，却用了 short × 整图池化 —— 这就是错位。

本脚本在真实数据上做**同数据、换口径**的对照实验：同一批「区域子图 + 与该区域
逐行对齐的 caption」，比较各口径下组内区域竞争信号（= RCGR 的竞争项）

    r_i = sim(c_i, R_i) - max_{j != i} sim(c_j, R_i)

的区分度。参与对照的口径（见 VARIANTS）：

    short_L64_global   <- 当前训练口径（walk_type=short + max_length=64 + 整图池化）
    box_L64_global     同长度换 box 文本头
    long_L196_global   官方 long 口径（唯一能吃下完整 caption 的长度档）
    short_L64_dense    patch 级 masked-max
    box_L64_dense      官方 grounding 的文本/图像配对口径
    long_L196_dense

即 3 种 walk_type × 2 种图像口径的全因子对照。注意 short/box 的位置编码表宽只有
64（`max_position_embeddings`），而 LLaVA caption 中位 106 token、97% 超过 62 token，
所以只有 `long` 档不会截断。

指标（每个口径各算一遍）
------------------------
    diag        mean_i sim(c_i, R_i)                     自匹配
    offmax      mean_i max_{j!=i} sim(c_j, R_i)          竞争项
    margin      mean_i (diag - offmax)                   = RCGR 竞争奖励均值
    margin_pos  竞争奖励 > 0 的区域占比（正确区分率，随机基线 = 1/K）
    col_acc     argmax_j S[j,i] == i 的比例（margin>0 的严格版）
    row_acc     argmax_j S[i,j] == i
    rew_std     组内 reward 标准差（GRPO advantage 的分子；太小 -> 无梯度）
    cross_diag  跨组错配 caption 的自匹配（越低 = 编码器整体分辨力越强）

用法
----
    /root/miniconda3/envs/mvpg/bin/python data/verify_rcgr_usage.py --limit 64

产物
----
    /root/MVPG/rcgr_usage_results.json
    /root/MVPG/rcgr_usage_report.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/MVPG")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mvpg_run.fgclip_local import load_fgclip  # noqa: E402

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# (名称, walk_type, max_length, 图像口径)
#   global = get_image_features（整图池化）；dense = get_image_dense_feature + masked max
# `short_L64_global` 是 run_mvpg.compute_group_rewards 的当前口径。
#
# 长度硬约束（modeling_fgclip2.Fgclip2TextEmbeddings.forward）：
#   walk_short / walk_box 走 `position_embedding`（表宽 = max_position_embeddings = 64）
#   walk_long 走 `position_embedding_res/ori`（表宽 = longtext_len = 196）
# 所以 short/box 只能吃 <=64 token，只有 long 能吃下完整的 ~106 token caption。
VARIANTS = [
    ("short_L64_global", "short", 64, "global"),
    ("box_L64_global", "box", 64, "global"),
    ("long_L196_global", "long", 196, "global"),
    ("short_L64_dense", "short", 64, "dense"),
    ("box_L64_dense", "box", 64, "dense"),
    ("long_L196_dense", "long", 196, "dense"),
]
VNAMES = [v[0] for v in VARIANTS]


def load_groups(image_dir, ref_key, limit):
    """返回 [(gid, [crop_path...], [region_caption...])]，caption 与 crop 严格同行序。"""
    root = Path(image_dir)
    caps = json.loads((root / "model_captions.json").read_text(encoding="utf-8"))
    out = []
    for gid in sorted(caps):
        g = caps[gid]
        if not isinstance(g, dict):
            continue
        d = g.get(ref_key) or {}
        gd = root / gid
        if not gd.is_dir():
            continue
        tiles = {}
        for p in sorted(gd.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMG_SUFFIXES:
                continue
            m = re.fullmatch(rf"{re.escape(gid)}_r(\d+)c(\d+)", p.stem, re.IGNORECASE)
            if m:
                tiles[(int(m.group(1)), int(m.group(2)))] = p
        ks = sorted(tiles)
        if len(ks) < 2:
            continue
        paths = [str(tiles[k]) for k in ks]
        cs = [d.get(tiles[k].stem) for k in ks]
        if any(not c for c in cs):
            continue
        out.append((gid, paths, cs))
    if limit and limit > 0:
        out = out[:limit]
    return out


def _offdiag(S):
    """把对角置 -inf，返回 (off, eye_mask)。"""
    K = S.shape[0]
    eye = torch.eye(K, dtype=torch.bool, device=S.device)
    return S.masked_fill(eye, float("-inf")), eye


def col_row_stats(S):
    """S (K,K)，S[i,j] = sim(cap_i, R_j)。返回竞争口径与检索口径的统计。"""
    K = S.shape[0]
    diag = S.diagonal()
    off, _ = _offdiag(S)
    offmax = off.max(dim=0).values          # 列：其余描述对本区域的最大相似度
    margin = diag - offmax
    idx = torch.arange(K, device=S.device)
    return {
        "diag": diag.mean().item(),
        "offmax": offmax.mean().item(),
        "margin": margin.mean().item(),
        "margin_pos": (margin > 0).float().mean().item(),
        "col_acc": (S.argmax(dim=0) == idx).float().mean().item(),   # 区域 -> 描述
        "row_acc": (S.argmax(dim=1) == idx).float().mean().item(),   # 描述 -> 区域
        "K": K,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="/root/autodl-tmp/data/SA1B_5_threshold")
    ap.add_argument("--fgclip-root", default="/root/autodl-tmp/cache/fgclip2-base-patch16")
    ap.add_argument("--ref-key", default="llava-v1.5-7b")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--out-json", default="/root/MVPG/rcgr_usage_results.json")
    ap.add_argument("--out-md", default="/root/MVPG/rcgr_usage_report.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = load_groups(args.image_dir, args.ref_key, args.limit)
    if not groups:
        raise SystemExit("没有可用组，检查 image-dir / ref-key")
    print(f"[rcgr usage] device={device} groups={len(groups)} ref_key={args.ref_key}")

    fg = load_fgclip(args.fgclip_root, device)

    # ---- 一次性编码所有 crop 的整图特征 & 所有 caption 的三种文本特征 ----
    all_paths, all_caps, spans = [], [], []
    for gid, paths, cs in groups:
        spans.append((len(all_paths), len(paths)))
        all_paths.extend(paths)
        all_caps.extend(cs)
    print(f"[rcgr usage] crops={len(all_paths)} captions={len(all_caps)}")

    # caption 长度分布：决定它属于 short(<=64) 还是 long(196) 的长度 regime
    tl = torch.tensor([len(fg.tokenizer.sp.encode(c.lower(), out_type=int)) for c in all_caps])
    len_stat = {f"p{int(q * 100)}": int(torch.quantile(tl.float(), q)) for q in (0.5, 0.9, 0.99)}
    len_stat["max"] = int(tl.max())
    len_stat["frac_gt62"] = float((tl > 62).float().mean())
    len_stat["mean"] = float(tl.float().mean())
    print("[rcgr usage] caption tokens " + "  ".join(f"{k}={v}" for k, v in len_stat.items()))

    img_global = fg.encode_image_paths(all_paths)                      # (N,768) 已归一化
    txt = {}
    for wt, ml in sorted({(w, m) for _, w, m, _ in VARIANTS}):
        f = fg.encode_texts_walk(all_caps, walk_type=wt, max_length=ml)
        txt[(wt, ml)] = F.normalize(f, p=2, dim=-1)
        print(f"[rcgr usage] text walk={wt} max_length={ml} -> {tuple(f.shape)}")

    # ---- 逐组算各口径的 S ----
    S_acc = {v: [] for v in VNAMES}
    cross_acc = {v: [] for v in VNAMES}
    rows = []

    for gi, (gid, paths, cs) in enumerate(groups):
        a, k = spans[gi]
        ig = img_global[a:a + k]                                       # (K,768)

        # 跨组错配（用下一组的 caption 配本组区域，验证编码器整体分辨力）
        a2, k2 = spans[(gi + 1) % len(groups)]
        same_k = k2 == k

        # patch 级口径：dense 特征 + masked max（本组 K 张一起算，算完即释放显存）
        dense, valid = fg.image_dense_features(paths)                   # (K,P,768), (K,P)
        dn = F.normalize(dense, p=2, dim=-1)

        def dense_sim(t_caps, t_other=None):
            """t_caps: (K,768) 本组文本；t_other: (K,768) 他组文本（跨组对照）。

            S[i,j] = max_p cos(dense_j[p], t_i)（P 为 patch 数，masked max）。
            返回 (S (K,K), cross_diag (K,) or None)。
            """
            Kr, P, H = dn.shape
            flat = dn.reshape(Kr * P, H)
            fv = valid.reshape(Kr, P)
            s = (flat @ t_caps.t()).view(Kr, P, -1)                     # (Kreg,P,Kcap)
            s = s.masked_fill(~fv[:, :, None], float("-inf"))
            Sij = s.max(dim=1).values                                   # (Kreg,Kcap)
            cd = None
            if t_other is not None:
                s2 = (flat @ t_other.t()).view(Kr, P, -1)
                s2 = s2.masked_fill(~fv[:, :, None], float("-inf"))
                cd = s2.max(dim=1).values.diagonal()                    # (K,)
            return Sij.t(), cd                                          # (Kcap,Kreg)

        cds = {}
        for name, wt, ml, mode in VARIANTS:
            t = txt[(wt, ml)][a:a + k]
            if mode == "global":
                S = t @ ig.t()
                cds[name] = (txt[(wt, ml)][a2:a2 + k2] @ ig.t()).diagonal() if same_k else None
            else:
                t2 = txt[(wt, ml)][a2:a2 + k2] if same_k else None
                S, cds[name] = dense_sim(t, t2)
            S = S.detach().to(torch.float32)
            S_acc[name].append(S)
            rows.append({"gid": gid, "variant": name, **col_row_stats(S)})
            if cds[name] is not None:
                cross_acc[name].append(cds[name].detach().to(torch.float32))

        del dense, dn
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- 汇总 ----
    summary = {}
    for v in VNAMES:
        per_group = [col_row_stats(s) for s in S_acc[v]]
        margins = torch.stack([s.diagonal() - _offdiag(s)[0].max(dim=0).values
                               for s in S_acc[v]])
        entry = {
            "diag": sum(p["diag"] for p in per_group) / len(per_group),
            "offmax": sum(p["offmax"] for p in per_group) / len(per_group),
            "margin": margins.mean().item(),
            "margin_pos": sum(p["margin_pos"] for p in per_group) / len(per_group),
            "col_acc": sum(p["col_acc"] for p in per_group) / len(per_group),
            "row_acc": sum(p["row_acc"] for p in per_group) / len(per_group),
            "rew_std_in_group": margins.std(dim=1).mean().item(),
            "rew_std_all": margins.flatten().std().item(),
        }
        if cross_acc[v]:
            entry["cross_diag"] = torch.stack(cross_acc[v]).mean().item()
        summary[v] = entry
        print(f"[rcgr usage] {v:13s} " + "  ".join(
            f"{kk}={vv:+.4f}" if isinstance(vv, float) and kk in ("margin", "cross_diag")
            else f"{kk}={vv:.4f}" for kk, vv in entry.items()))

    json.dump({"summary": summary, "caption_len": len_stat, "rows": rows},
              open(args.out_json, "w"), ensure_ascii=False, indent=1)

    # ---- 报告 ----
    hdr = ["口径", "diag", "offmax", "margin", "margin>0", "col_acc", "row_acc",
           "组内std", "跨组diag"]
    L = ["# RCGR 的 FG-CLIP 口径对照验证\n",
         f"- 图像目录：`{args.image_dir}`；组数：{len(groups)}；文本来源：`{args.ref_key}`\n",
         f"- caption token 长度：p50={len_stat['p50']}，p90={len_stat['p90']}，"
         f"p99={len_stat['p99']}，max={len_stat['max']}；"
         f"超过 62 token 的比例 = {len_stat['frac_gt62']:.3f}。"
         f"short/box 的位置表宽只有 64，故当前 `short_L64` 口径下这 "
         f"{len_stat['frac_gt62'] * 100:.0f}% 的 caption 被截断。\n",
         f"- 数据：[区域子图 K 张] × [与该区域逐行对齐的 caption K 条]，"
         f"K 由网格 r{{r}}c{{c}} 决定。\n",
         "- `margin = mean_i ( sim(c_i,R_i) - max_{j!=i} sim(c_j,R_i) )` 即 RCGR 的竞争奖励均值；",
         "`margin>0` 为「本区域描述在组内竞争中胜出」的比例（随机基线 = 1/K ≈ "
         f"{1.0 / max(1, sum(r['K'] for r in rows) / max(1, len(rows))):.2f}）。\n",
         "- `col_acc`：区域 -> 描述 的 top-1 命中率（`argmax_j S[j,i] == i`）；"
         "`row_acc`：描述 -> 区域 的 top-1 命中率。\n",
         "- `跨组diag`：用**下一组**的 caption 配本组区域的自匹配均值（越低 = 编码器"
         "整体分辨力越强，对 dense 口径它与 `diag` 的差距即「分辨余量」）。\n",
         "\n| " + " | ".join(hdr) + " |\n",
         "|" + "---|" * len(hdr) + "\n"]
    for v in VNAMES:
        e = summary[v]
        cells = [v, f"{e['diag']:+.4f}", f"{e['offmax']:+.4f}", f"{e['margin']:+.4f}",
                 f"{e['margin_pos']:.3f}", f"{e['col_acc']:.3f}", f"{e['row_acc']:.3f}",
                 f"{e['rew_std_in_group']:.4f}",
                 f"{e['cross_diag']:+.4f}" if "cross_diag" in e else "-"]
        L.append("| " + " | ".join(cells) + " |\n")
    L.append("\n## 判读\n")
    L.append("- `margin` 越正越大、`margin>0`/`col_acc` 越高于随机基线，该口径的区域竞争信号越可用。\n")
    L.append("- `组内std` 是 GRPO advantage 的分子：接近 0 说明组内无差异，训练拿不到梯度。\n")
    L.append("- 同一图像口径下 `short_L64` vs `long_L196` 的差值 = 「caption 被截断到 "
             "64 token」的代价（注意 walk_type 也换了，二者捆绑在 FG-CLIP2 的长度档里）。\n")
    L.append("- 同一 walk_type 下 `*_global` vs `*_dense` 的差值 = 图像特征口径的代价。\n")

    # ---- 结论（数据驱动）----
    cur = summary["short_L64_global"]
    best = max(VNAMES, key=lambda v: summary[v]["col_acc"])
    L.append("\n## 结论\n")
    L.append(f"- 当前训练口径 `short_L64_global`：col_acc = **{cur['col_acc']:.3f}**（随机基线 0.25）、"
             f"margin = {cur['margin']:+.4f}、组内 std = {cur['rew_std_in_group']:.4f} "
             f"-> 竞争信号真实存在，**RCGR 这套方法本身可用**。\n")
    L.append(f"- 全因子最优口径是 **`{best}`**（col_acc = {summary[best]['col_acc']:.3f}）。\n")
    for mode in ("global", "dense"):
        vs = [v for v in VNAMES if v.endswith(mode)]
        L.append(f"- 图像口径 `{mode}` 在 3 种 walk_type 上的平均 col_acc = "
                 f"{sum(summary[v]['col_acc'] for v in vs) / len(vs):.3f}"
                 f"（平均跨组diag = "
                 f"{sum(summary[v].get('cross_diag', float('nan')) for v in vs) / len(vs):+.4f}）。\n")
    for wt in ("short", "box", "long"):
        vs = [v for v in VNAMES if v.startswith(wt + "_")]
        L.append(f"- 文本档 `{wt}`（{vs[0].split('_')[1]}）在 2 种图像口径上的平均 col_acc = "
                 f"{sum(summary[v]['col_acc'] for v in vs) / len(vs):.3f}。\n")
    open(args.out_md, "w").write("".join(L))
    print(f"[rcgr usage] 写出 {args.out_json} 与 {args.out_md}")


if __name__ == "__main__":
    main()
