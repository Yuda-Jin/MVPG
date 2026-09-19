"""FG-CLIP 换图测试：实体的视觉支持是否「因果依赖」于原图？

动机：
  既有的 patch grounding 信号 `s(e, 原图)` 只能提供**正证据**（这个 patch 像不像 e），
  且 FG-CLIP 是场景级匹配，分辨率不足以证明某个具体物体「不存在」（幻觉是否定命题）。
  换图测试把「存在性」改写成**因果增量**：

      Δ(e) = s(e, 原图) − mean_{其他图} s(e, 其他图)

  直觉：
    - 真实物体 e（图中确实存在）→ 匹配依赖原图内容，换到无关图后得分下降，Δ > 0；
    - 幻觉物体 e（图中不存在，源自语言先验）→ 在原图/无关图上都只是场景级泛化匹配，
      得分同样低且不随图变化，Δ ≈ 0。

  这绕开「相似度只能给正证据」的天花板：不需要绝对阈值，只看**同一短语在不同图上的差分**，
  图级偏置与短语级偏置都被减掉。

对照信号（同一批 12 组短语，可与既有报告直接对比）：
  1. `s(e, 原图)`：= fgclip_granular_report.md 的 box max_patch_sim（d=+0.72）
  2. `Δ_raw`：原图得分 − 其他 11 图得分均值
  3. `Δ_z`：先按「每张图上所有短语的得分分布」做 z-score 归一化再差分（去掉图级打分偏置）
  4. `Δ_max`：原图得分 − 其他图得分的**最大值**（更严格：只要任意一张无关图能匹配上就不算因果）

输出 markdown + json 到项目根目录。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/MVPG")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvpg_run.fgclip_local import load_fgclip, preprocess_images  # noqa: E402
from verify_fgclip_granular_hallucination import PHRASES, main_img_path  # noqa: E402
from verify_fgclip_split_compare import (  # noqa: E402
    FGCLIP_ROOT, dense_features, encode_text_walk, max_patch_sim, mean, std, cohens_d,
)

WALKS = ("box", "short")


def auc(pos, neg):
    """P(pos > neg)，pos=真实短语（期望得分高），neg=幻觉短语。"""
    n = 0.0
    tot = 0
    for a in pos:
        for b in neg:
            tot += 1
            if a > b:
                n += 1.0
            elif a == b:
                n += 0.5
    return n / tot if tot else 0.0


def acc_at(xs_hall, xs_real, thr):
    ok = sum(1 for x in xs_real if x >= thr) + sum(1 for x in xs_hall if x < thr)
    return ok / (len(xs_hall) + len(xs_real))


def best_acc(xs_hall, xs_real):
    """扫描所有候选阈值，返回 (最优准确率, 对应阈值)。"""
    cands = sorted(set(xs_hall + xs_real))
    mids = [(cands[i] + cands[i + 1]) / 2 for i in range(len(cands) - 1)] or cands
    best = (0.0, 0.0)
    for t in mids:
        a = acc_at(xs_hall, xs_real, t)
        if a > best[0]:
            best = (a, t)
    return best


def main():
    t0 = time.time()
    fg = load_fgclip(FGCLIP_ROOT, "cpu")
    print(f"fgclip loaded in {time.time() - t0:.1f}s")

    # ---- 1. 收集 12 组主图 ----
    gids = [g for g in PHRASES if main_img_path(g) is not None]
    print(f"groups with image: {len(gids)}/{len(PHRASES)}")

    # ---- 2. 全局短语池（去重，用于 z 归一化与批量编码）----
    all_phrases = []
    for g in gids:
        all_phrases.extend(PHRASES[g]["hall"] + PHRASES[g]["real"])
    uniq_phrases = sorted(set(all_phrases))
    p_to_idx = {p: i for i, p in enumerate(uniq_phrases)}

    # ---- 3. 预编码文本特征（每 walk 一次） ----
    text_feat = {}
    for walk in WALKS:
        feat = encode_text_walk(fg, uniq_phrases, walk)
        text_feat[walk] = F.normalize(feat, p=2, dim=-1)
        print(f"encoded {len(uniq_phrases)} phrases with walk={walk}")

    # ---- 4. 预计算每张图的 dense 特征，并构造相似度矩阵 S[phrase, img] ----
    dens = {}
    S = {w: np.zeros((len(uniq_phrases), len(gids)), dtype=np.float64) for w in WALKS}
    for j, g in enumerate(gids):
        img = main_img_path(g)
        dense, valid = dense_features(fg, img)
        dens[g] = (dense, valid)
        for w in WALKS:
            sims = max_patch_sim(dense, valid, text_feat[w]).cpu().numpy()
            S[w][:, j] = sims
        print(f"image {g}: n_patch={int(valid.sum())}")

    g_to_j = {g: j for j, g in enumerate(gids)}

    # ---- 5. 每张图的 z 归一化矩阵（按该图上所有短语的得分分布标准化）----
    Z = {}
    for w in WALKS:
        mu = S[w].mean(axis=0, keepdims=True)
        sd = S[w].std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Z[w] = (S[w] - mu) / sd

    # ---- 6. 逐短语计算四种信号 ----
    rows = []
    for g in gids:
        j = g_to_j[g]
        others = [k for k in range(len(gids)) if k != j]
        for cls, phrases in (("hall", PHRASES[g]["hall"]), ("real", PHRASES[g]["real"])):
            for p in phrases:
                i = p_to_idx[p]
                rec = {"gid": g, "class": cls, "phrase": p}
                for w in WALKS:
                    s_orig = S[w][i, j]
                    s_swaps = S[w][i, others]
                    z_orig = Z[w][i, j]
                    z_swaps = Z[w][i, others]
                    rec[f"s_orig_{w}"] = float(s_orig)
                    rec[f"s_swap_mean_{w}"] = float(s_swaps.mean())
                    rec[f"delta_raw_{w}"] = float(s_orig - s_swaps.mean())
                    rec[f"delta_max_{w}"] = float(s_orig - s_swaps.max())
                    rec[f"delta_z_{w}"] = float(z_orig - z_swaps.mean())
                rows.append(rec)

    # ---- 7. 汇总每条信号 ----
    SIGNALS = []
    for w in WALKS:
        SIGNALS += [(f"s_orig ({w})", f"s_orig_{w}", False),
                    (f"delta_raw ({w})", f"delta_raw_{w}", False),
                    (f"delta_z ({w})", f"delta_z_{w}", False),
                    (f"delta_max ({w})", f"delta_max_{w}", False)]

    summary = {}
    for name, key, _ in SIGNALS:
        hall = [r[key] for r in rows if r["class"] == "hall"]
        real = [r[key] for r in rows if r["class"] == "real"]
        ba, bt = best_acc(hall, real)
        # 组级 margin（每组 real 均值 − hall 均值）
        margins = []
        for g in gids:
            gr = [r for r in rows if r["gid"] == g]
            margins.append(mean([r[key] for r in gr if r["class"] == "real"]) -
                           mean([r[key] for r in gr if r["class"] == "hall"]))
        summary[name] = {
            "key": key,
            "hall_mean": mean(hall), "hall_std": std(hall),
            "real_mean": mean(real), "real_std": std(real),
            "cohens_d": cohens_d(hall, real),
            "auc": auc(real, hall),
            "acc_mid": acc_at(hall, real, (mean(hall) + mean(real)) / 2),
            "best_acc": ba, "best_tau": bt,
            "margin_mean": mean(margins),
            "margin_gt0": sum(1 for m in margins if m > 0),
            "n_groups": len(gids),
            "n_hall": len(hall), "n_real": len(real),
        }

    json.dump({"summary": summary, "rows": rows},
              open("/root/MVPG/fgclip_image_swap_results.json", "w"),
              ensure_ascii=False, indent=2)

    # ---- 8. 写 markdown ----
    L = []
    L.append("# FG-CLIP 换图测试：视觉支持的因果性检验\n")
    L.append(f"- 数据：`/root/autodl-tmp/data/SA1B_5_threshold` 的 {len(gids)} 张主图"
             f"（复用 `verify_fgclip_granular_hallucination.py` 的 PHRASES）\n")
    L.append(f"- 模型：FG-CLIP2-base（`{FGCLIP_ROOT}`），CPU 推理\n")
    L.append("- 信号：`s(e, 原图)`（既有 patch grounding 基线） vs "
             "`Δ(e) = s(e, 原图) − 其他图得分`（换图因果差分），"
             "`delta_z` 为图内 z 归一化后的差分，`delta_max` 用其他图得分最大值做被减项\n")
    L.append(f"- 短语数：幻觉 {summary[SIGNALS[0][0]]['n_hall']}，真实 "
             f"{summary[SIGNALS[0][0]]['n_real']}；每组各图均作为其他短语的「随机图」\n")

    L.append("\n## 1. 信号对比（可直接与 granular 报告的 d=+0.72 / acc=64.7% 对比）\n")
    L.append("| 信号 | 幻觉均值 | 真实均值 | Cohen's d | AUC | 中点阈值准确率 | 最优准确率 | 最优 tau | 组级 margin>0 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, key, _ in SIGNALS:
        m = summary[name]
        L.append(f"| `{name}` | {m['hall_mean']:+.4f} | {m['real_mean']:+.4f} | "
                 f"**{m['cohens_d']:+.2f}** | {m['auc']:.3f} | {m['acc_mid']*100:.1f}% | "
                 f"**{m['best_acc']*100:.1f}%** | {m['best_tau']:+.4f} | "
                 f"{m['margin_gt0']}/{m['n_groups']} |")

    # ---- 信号分解：原图信号里有多少「不依赖原图」----
    decomp = {}
    for w in WALKS:
        so = np.array([r[f"s_orig_{w}"] for r in rows])
        sw = np.array([r[f"s_swap_mean_{w}"] for r in rows])
        dl = np.array([r[f"delta_raw_{w}"] for r in rows])
        d_orig = mean([r[f"s_orig_{w}"] for r in rows if r["class"] == "real"]) - \
                 mean([r[f"s_orig_{w}"] for r in rows if r["class"] == "hall"])
        d_swap = mean([r[f"s_swap_mean_{w}"] for r in rows if r["class"] == "real"]) - \
                 mean([r[f"s_swap_mean_{w}"] for r in rows if r["class"] == "hall"])
        decomp[w] = {
            "corr_orig_swapmean": float(np.corrcoef(so, sw)[0, 1]),
            "corr_delta_orig": float(np.corrcoef(dl, so)[0, 1]),
            "gap_orig": d_orig,
            "gap_on_unrelated_images": d_swap,
            "gap_image_specific": d_orig - d_swap,
            "share_unrelated": d_swap / d_orig if d_orig else 0.0,
        }

    L.append("\n## 2. 为什么 Δ 反而更弱？——原图信号分解\n")
    L.append("若 `s(e, 原图)` 真的在度量「图像里有没有 e」，那把它替换成一张无关图，"
             "真实/幻觉的差距应当基本消失。实测并非如此：\n")
    L.append("| walk | corr(s_orig, s_swap_mean) | 真实−幻觉 gap(原图) | gap(无关图均值) | gap(原图−无关图) | 无关图占比 |")
    L.append("|---|---|---|---|---|---|")
    for w in WALKS:
        d = decomp[w]
        L.append(f"| {w} | {d['corr_orig_swapmean']:.3f} | {d['gap_orig']:+.4f} | "
                 f"{d['gap_on_unrelated_images']:+.4f} | {d['gap_image_specific']:+.4f} | "
                 f"**{d['share_unrelated']*100:.0f}%** |")
    L.append("")
    for w in WALKS:
        d = decomp[w]
        L.append(f"- **{w}**：相关性 {d['corr_orig_swapmean']:.3f}；真实−幻觉的原始差 "
                 f"{d['gap_orig']:+.4f} 中有 **{d['gap_on_unrelated_images']:+.4f}"
                 f"（{d['share_unrelated']*100:.0f}%）在完全不相关的图上同样存在**，"
                 f"真正依赖原图的分量只剩 {d['gap_image_specific']:+.4f}。\n")
    w0 = "box"
    d0 = decomp[w0]
    L.append(f"- 解读：`s(e, 原图)` 的主要成分是**短语的「泛化性」**（越通用的描述在任何图上都得分越高），"
             f"而非「这张图里有没有 e」。换图差分恰好把这个共模分量减掉，"
             f"剩下的图像特异性分量极小（{d0['gap_image_specific']:+.4f}，d 仅 "
             f"{summary['delta_raw (box)']['cohens_d']:+.2f}），所以 Δ 看起来「更差」——"
             f"**它更诚实，暴露了原始信号里的虚高成分**。\n")

    # 结论
    base_d = summary["s_orig (box)"]["cohens_d"]
    base_acc = summary["s_orig (box)"]["best_acc"]
    best_name = max((n for n, _, _ in SIGNALS), key=lambda n: summary[n]["cohens_d"])
    bm = summary[best_name]
    d_ok = bm["cohens_d"] >= 0.8
    L.append("\n## 3. 结论\n")
    L.append(f"- 既有基线 `s_orig (box)`：d={base_d:+.2f}，最优准确率 {base_acc*100:.1f}%"
             f"（与 granular 报告一致，作为对照）。\n")
    L.append(f"- 换图差分（box）`delta_raw`：d={summary['delta_raw (box)']['cohens_d']:+.2f}，"
             f"AUC={summary['delta_raw (box)']['auc']:.3f}，"
             f"最优准确率 {summary['delta_raw (box)']['best_acc']*100:.1f}%；"
             f"z 归一化版 `delta_z`：d={summary['delta_z (box)']['cohens_d']:+.2f}，"
             f"最优准确率 {summary['delta_z (box)']['best_acc']*100:.1f}%。\n")
    L.append(f"- 全部信号中效应量最大者：`{best_name}`（d={bm['cohens_d']:+.2f}，"
             f"最优准确率 {bm['best_acc']*100:.1f}%）。\n")
    L.append(f"- **判定：换图差分信号{'具备' if d_ok else '尚不具备'}可用强度"
             f"（判据 Cohen's d ≥ 0.8）**。\n")
    L.append(f"- 关键发现不在「Δ 弱」，而在**原图信号本身不干净**：`s(e, 原图)` 的真实−幻觉差距里，"
             f"box 头有 {decomp['box']['share_unrelated']*100:.0f}%、"
             f"short 头有 {decomp['short']['share_unrelated']*100:.0f}% "
             f"在无关图上同样存在，即**大部分差距来自短语泛化性而非图像证据**。"
             f"因此任何以 `s(e, 原图)` 的绝对值为基础的判据（含此前的子句级 d=1.20）都存在系统性虚高。\n")
    L.append(f"- 换言之，换图差分不是「更差的方案」，而是**对同一条信号做了去掉共模分量后的诚实度量**；"
             f"它揭示：FG-CLIP 在「某具体物体是否存在」这一粒度上的图像特异性证据极弱"
             f"（box {decomp['box']['gap_image_specific']:+.4f}，short {decomp['short']['gap_image_specific']:+.4f}）。\n")

    L.append("\n## 4. 分组明细（增量 d 与原始 d 对比）\n")
    L.append("| gid | hall 数 | real 数 | s_orig(box) margin | delta_raw(box) margin | delta_z(box) margin |")
    L.append("|---|---|---|---|---|---|")
    for g in gids:
        gr = [r for r in rows if r["gid"] == g]
        nh = sum(1 for r in gr if r["class"] == "hall")
        nr = sum(1 for r in gr if r["class"] == "real")

        def gm(key):
            return mean([r[key] for r in gr if r["class"] == "real"]) - \
                   mean([r[key] for r in gr if r["class"] == "hall"])
        L.append(f"| {g} | {nh} | {nr} | {gm('s_orig_box'):+.4f} | "
                 f"{gm('delta_raw_box'):+.4f} | {gm('delta_z_box'):+.4f} |")

    L.append("\n## 5. 逐短语明细\n")
    L.append("| gid | 类别 | 短语 | s_orig(box) | Δ_raw(box) | Δ_z(box) | Δ_max(box) | Δ_raw(short) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["gid"], r["class"])):
        L.append(f"| {r['gid']} | {r['class']} | `{r['phrase']}` | {r['s_orig_box']:.4f} | "
                 f"{r['delta_raw_box']:+.4f} | {r['delta_z_box']:+.4f} | {r['delta_max_box']:+.4f} | "
                 f"{r['delta_raw_short']:+.4f} |")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_image_swap_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_image_swap_report.md")
    for name, _, _ in SIGNALS:
        m = summary[name]
        print(f"{name:>18}: d={m['cohens_d']:+.2f} auc={m['auc']:.3f} "
              f"best_acc={m['best_acc']*100:.1f}%@{m['best_tau']:+.4f} margin>0={m['margin_gt0']}/{m['n_groups']}")


if __name__ == "__main__":
    main()
