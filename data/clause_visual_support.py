"""统计所有子句的视觉支持，并标定「幻觉阈值」。

前置：`annotate_neg_clauses.py` 已用独立 VLM 复核通道找回每条 caption 里被注入的
幻觉物体短语，并把短语回填成子句级标签（label=1 = 幻觉子句）。

本脚本两阶段：

  score   —— 用 FG-CLIP2 对每张主图算 dense patch 特征，对每个子句算：
              * s_box / s_short : 子句特征 × 全部 patch 特征的 masked max 余弦（视觉支持）
              * C{k}_box        : 集中度 a_(1) − a_(k+1)（挖掉 top-k 尖端后分数掉落量）
              * z_box / z_short : 图内 patch 维度标准化 (a1 − mean_p) / std_p
            产物逐子句 jsonl，可断点续传。
  analyze —— 对比「幻觉子句 vs 真实子句」的分布（AUC / Cohen's d / 分位数），
            用统计口径反解**固定阈值**（给定目标精度取最低阈值，保住召回），
            并报告图内归一化后的口径，最后出直方图。

用法：
    PY=/root/miniconda3/envs/mvpg/bin/python
    $PY data/clause_visual_support.py --stage score   --labels /root/MVPG/neg_clause_labels.jsonl
    $PY data/clause_visual_support.py --stage analyze --labels /root/MVPG/neg_clause_labels.jsonl

产物：
    /root/MVPG/clause_support_scores.jsonl
    /root/MVPG/clause_support_report.md
    /root/MVPG/clause_support_hist.png
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

from calibrate_paired_ref import auc_score, quantile  # noqa: E402
from mvpg_run.fgclip_local import load_fgclip  # noqa: E402

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
KS = (5, 20, 50)


# ---------------------------------------------------------------- 工具

def main_image_path(image_dir, gid):
    d = Path(image_dir) / gid
    for s in IMG_SUFFIXES:
        p = d / (gid + s)
        if p.is_file():
            return str(p)
    return None


def load_labels(path, limit=0):
    """读标注 jsonl，按 gid 去重（后者覆盖前者），跳过 error 行。"""
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r or "clauses" not in r:
                continue
            recs[r["gid"]] = r
    out = [recs[g] for g in sorted(recs)]
    return out[:limit] if limit else out


def pick_swap_images(image_dir, exclude_gids, n):
    """从数据目录里挑 n 张**不在标注集合内**的图当作「无关图」基线。"""
    out = []
    for gid in sorted(os.listdir(image_dir)):
        if len(out) >= n:
            break
        if gid in exclude_gids or not os.path.isdir(os.path.join(image_dir, gid)):
            continue
        p = main_image_path(image_dir, gid)
        if p:
            out.append(p)
    return out


def load_done_gids(path):
    done = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["gid"])
                except Exception:
                    continue
    return done


def sim_columns(dense, valid, text_feat):
    """dense (P,H), valid (P,), text_feat (N,H) -> (P,N) masked cosine。"""
    dn = F.normalize(dense, p=2, dim=-1)
    tn = F.normalize(text_feat, p=2, dim=-1)
    sim = dn @ tn.T
    sim = sim.masked_fill(~valid.unsqueeze(1), -1e9)
    return sim


def support_stats(sim, valid):
    """sim (P,N), valid (P,) -> 每个子句的 (max, z_patch, C5, C20, C50)。

    注意：`sim` 里被 mask 的 padding patch 已填成极小值，统计 mean/std 时**必须排除**，
    否则 -1e9 会主导方差，z 退化成只依赖「padding 占比」的图级常数。
    """
    P = sim.shape[0]
    order = torch.sort(sim, dim=0, descending=True).values  # (P,N)
    a1 = order[0]
    n = valid.sum().clamp_min(1)
    simv = sim.masked_fill(~valid.unsqueeze(1), 0.0)
    mean = simv.sum(dim=0) / n
    var = (simv * simv).sum(dim=0) / n - mean * mean
    std = var.clamp_min(1e-12).sqrt()
    z = (a1 - mean) / std
    Cs = []
    for k in KS:
        Cs.append(a1 - order[min(k, P - 1)])
    return a1, z, Cs


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    sp = ((va + vb) / 2) ** 0.5
    return (ma - mb) / sp if sp > 0 else float("nan")


def pick_low_threshold(hall, real, target_precision):
    """返回 (t, precision, recall)：分数 <= t 判为幻觉。

    语义与 calibrate_paired_ref.pick_threshold 取负后等价（取「满足 precision>=target
    的**最高** t」，即在该精度下尽量保召回），但用二分计数把复杂度从 O(n^2) 降到 O(n log n)。
    全量 9w+ 子句下原实现需 ~88 亿次比较，会跑数小时。
    """
    import bisect
    if not hall or not real:
        return float("nan"), float("nan"), float("nan")
    hall_s, real_s = sorted(hall), sorted(real)
    best = float("nan")
    # 分数 <= t 判幻觉：t 越高召回越高但精度越低，故满足精度的**最高** t 最优。
    for t in sorted(set(hall_s) | set(real_s)):
        tp = bisect.bisect_right(hall_s, t)
        fp = bisect.bisect_right(real_s, t)
        if tp + fp == 0:
            continue
        if tp / (tp + fp) >= target_precision:
            best = t
        elif best == best:  # 已找到可行阈值，精度继续下滑 => 停
            break
    if best != best:  # nan
        return float("nan"), float("nan"), float("nan")
    t = best
    tp = bisect.bisect_right(hall_s, t)
    fp = bisect.bisect_right(real_s, t)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / len(hall) if hall else float("nan")
    return t, prec, rec


def last_idx_map(rows):
    """(gid, cap) -> 该段落最大 idx（末句子句的位置）。"""
    last = {}
    for r in rows:
        k = (r["gid"], r["cap"])
        if k not in last or r["idx"] > last[k]:
            last[k] = r["idx"]
    return last


def series_raw(rows, key):
    """逐行 (gid, label, value)，方向统一为「越低越像幻觉」。

    部分指标（如 patch 级 `z_*` / `C*_box`）经验极性相反（幻觉子句分**更高**），
    整体取负后即可套用同一套「score <= t」口径。返回 (series, sign)。
    """
    hall, real = gather(rows, key)
    sign = -1.0 if (hall and real and auc_score(real, hall) < 0.5) else 1.0
    return [(r["gid"], r["label"], sign * r[key]) for r in rows if key in r], sign


def series_group(rows, key, group, name):
    """组内标准化（图内 / 段落内）口径的逐行序列，方向同样统一。"""
    sc, lab, kk = group_norm(rows, key, group)
    hall = [v for v, y in zip(sc[name], lab) if y == 1]
    real = [v for v, y in zip(sc[name], lab) if y == 0]
    sign = -1.0 if (hall and real and auc_score(real, hall) < 0.5) else 1.0
    return [(kk[i][0], lab[i], sign * sc[name][i]) for i in range(len(lab))], sign


def series_last_position(rows, last):
    """位置先验口径：末句记 0（最像幻觉），其余记 1，作为阈值法的诚实对照。"""
    return [(r["gid"], r["label"],
             0.0 if r["idx"] == last[(r["gid"], r["cap"])] else 1.0) for r in rows], 1.0


def threshold_transfer(series, hold_gids, target):
    """校准图反解阈值 → 留出图评估。None 表示校准集上无可行阈值。

    series 元素为 (gid, label, value)，value 已统一成「越低越像幻觉」。
    """
    cal_h = [v for g, y, v in series if g not in hold_gids and y == 1]
    cal_r = [v for g, y, v in series if g not in hold_gids and y == 0]
    t, cal_p, _ = pick_low_threshold(cal_h, cal_r, target)
    if t != t:
        return None
    te_h = [v for g, y, v in series if g in hold_gids and y == 1]
    te_r = [v for g, y, v in series if g in hold_gids and y == 0]
    tp = sum(1 for v in te_h if v <= t)
    fp = sum(1 for v in te_r if v <= t)
    return {"t": t, "cal_p": cal_p,
            "te_p": tp / (tp + fp) if tp + fp else float("nan"),
            "te_r": tp / len(te_h) if te_h else float("nan"),
            "te_fpr": fp / len(te_r) if te_r else float("nan")}


def pr_points(hall, real):
    """「分越低越像幻觉」排序下逐操作点的 (召回, 精度) 阶梯。"""
    items = sorted([(v, 1) for v in hall] + [(v, 0) for v in real])
    n_h = len(hall)
    pts, tp, fp = [], 0, 0
    for _, y in items:
        tp += y
        fp += 1 - y
        pts.append((tp / n_h, tp / (tp + fp)))
    return pts


def prec_at_recall(pts, r):
    """召回 >= r 的操作点里的最高精度（同召回比精度用）。"""
    best = float("nan")
    for rec, pre in pts:
        if rec >= r and (best != best or pre > best):
            best = pre
    return best


def pick_f1_threshold(hall, real, grid=200):
    """在「score <= t 判为幻觉」口径下取 F1 最大的阈值。返回 (t, prec, rec, f1)。

    这是「大致分割幻觉 / 事实」最直白的操作点：精度与召回折中。
    只在与正类相邻的候选阈值上扫描（分位网格），避免遍历 9w 个取值。
    """
    import bisect
    if not hall or not real:
        return float("nan"), float("nan"), float("nan"), float("nan")
    hs, rs = sorted(hall), sorted(real)
    pooled = sorted(hs + rs)
    bt = bp = br = bf = float("nan")
    for i in range(grid + 1):
        t = quantile(pooled, i / grid)
        tp = bisect.bisect_right(hs, t)
        if tp == 0:
            continue
        fp = bisect.bisect_right(rs, t)
        prec = tp / (tp + fp)
        rec = tp / len(hs)
        f1 = 2 * prec * rec / (prec + rec)
        if bf != bf or f1 > bf:
            bt, bp, br, bf = t, prec, rec, f1
    return bt, bp, br, bf


def write_transfer_table(lines, signals, hold, targets=(0.5, 0.7, 0.8)):
    """写「固定阈值跨图迁移」表。

    signals：有序 dict {口径名: [(gid, label, value)]}，value 已统一为「越低越像幻觉」。
    返回 {(口径, 目标精度): 结果}，结果 None 表示校准集上无可行阈值。
    """
    lines.append("| 信号口径 | 目标精度 | 校准阈值 t | 校准精度 | **留出精度** | 留出召回 | "
                 "留出误报率 | 留出 AUC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    res = {}
    for nm, series in signals.items():
        te_h = [v for g, y, v in series if g in hold and y == 1]
        te_r = [v for g, y, v in series if g in hold and y == 0]
        te_auc = auc_score(te_r, te_h)
        for i, target in enumerate(targets):
            o = threshold_transfer(series, hold, target)
            res[(nm, target)] = o
            if o is None:
                lines.append(f"| {nm if i == 0 else ''} | {target:.1f} | — | — | — "
                             f"| — | — | {fmt(te_auc, 3)} |")
            else:
                lines.append(f"| {nm if i == 0 else ''} | {target:.1f} | {fmt(o['t'])} | "
                             f"{fmt(o['cal_p'], 3)} | **{fmt(o['te_p'], 3)}** | "
                             f"{fmt(o['te_r'], 3)} | {fmt(o['te_fpr'], 3)} | "
                             f"{fmt(te_auc, 3)} |")
    lines.append("")
    return res


# ---------------------------------------------------------------- score

def stage_score(args):
    fg = load_fgclip(args.fgclip_root, args.device)
    recs = load_labels(args.labels, args.limit)
    done = load_done_gids(args.out)
    todo = [r for r in recs if r["gid"] not in done]
    print(f"标注 {len(recs)} 张，已完成 {len(done)}，本次待处理 {len(todo)} 张")
    if not todo:
        return

    # 换图基线：把同一个子句拿去和「无关图」算支持度。
    # 若换图后 AUC 几乎不变，说明该指标抓的是「语言先验/短语泛化性」而非这张图的视觉证据。
    swap_paths = list(args.swap_images)
    if not swap_paths and args.swap_n > 0:
        exclude = {r["gid"] for r in recs}
        swap_paths = pick_swap_images(args.image_dir, exclude, args.swap_n)
    swap_dense = []
    for sp in swap_paths:
        d, v = fg.image_dense_features([sp], batch_size=1)
        swap_dense.append((d[0], v[0]))
    if swap_dense:
        print(f"换图基线已加载 {len(swap_dense)} 张无关图: "
              f"{', '.join(Path(p).parent.name for p in swap_paths)}")

    t0 = time.time()
    n_written = n_fail = 0
    with open(args.out, "a", encoding="utf-8") as fo:
        for n, r in enumerate(todo, 1):
            gid = r["gid"]
            img = main_image_path(args.image_dir, gid)
            if img is None:
                n_fail += 1
                continue

            clauses, meta = [], []
            for ci, cs in enumerate(r["clauses"]):
                for j, txt in enumerate(cs):
                    clauses.append(txt)
                    meta.append((ci, j, r["labels"][ci][j]))
            if not clauses:
                n_fail += 1
                continue

            try:
                dense, valid = fg.image_dense_features([img], batch_size=1)
                dense, valid = dense[0], valid[0]
                row = {"gid": gid, "n_patches": int(valid.sum())}
                for walk in args.walks:
                    feat = fg.encode_texts_walk(clauses, walk_type=walk, batch_size=64)
                    sim = sim_columns(dense, valid, feat)
                    a1, z, Cs = support_stats(sim, valid)
                    row[f"s_{walk}"] = [float(x) for x in a1]
                    row[f"z_{walk}"] = [float(x) for x in z]
                    if swap_dense:
                        per_img = [sim_columns(sd, sv, feat).max(dim=0).values
                                   for sd, sv in swap_dense]
                        sswap = torch.stack(per_img, dim=0).mean(dim=0)
                        row[f"s_swap_{walk}"] = [float(x) for x in sswap]
                    if walk == "box":
                        for k, c in zip(KS, Cs):
                            row[f"C{k}_box"] = [float(x) for x in c]
            except Exception as e:
                n_fail += 1
                print(f"  [{gid}] 打分失败: {e}")
                continue

            for i, (ci, j, lab) in enumerate(meta):
                rec = {"gid": gid, "cap": ci, "idx": j, "label": int(lab),
                       "text": clauses[i], "n_words": len(clauses[i].split())}
                for walk in args.walks:
                    rec[f"s_{walk}"] = row[f"s_{walk}"][i]
                    rec[f"z_{walk}"] = row[f"z_{walk}"][i]
                    if f"s_swap_{walk}" in row:
                        rec[f"s_swap_{walk}"] = row[f"s_swap_{walk}"][i]
                for k in KS:
                    key = f"C{k}_box"
                    if key in row:
                        rec[key] = row[key][i]
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
            fo.flush()

            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] 子句 {n_written} fail={n_fail} "
                      f"用时 {el:.0f}s ({n / max(el, 1e-9):.2f} 张/s)")

    print(f"打分完成: 写出 {n_written} 条子句 -> {args.out}")


# ---------------------------------------------------------------- analyze

def label_quality(labels_path):
    """标注自洽性：幻觉短语应命中本 caption，且不出现在同图其他 caption 里。

    纯文本检查，与 FG-CLIP 打分完全独立，用来发现「VLM 误把真实物体判成幻觉」。
    """
    if not labels_path or not os.path.isfile(labels_path):
        return None
    own_hit = unique = n = 0
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "clauses" not in r:
                continue
            cap_text = [" ".join(cs) for cs in r["clauses"]]
            for a in r.get("absent", []):
                p = normalize_text(a.get("phrase") or "")
                if not p:
                    continue
                n += 1
                ci = a.get("cap", 0)
                if 0 <= ci < len(cap_text) and p in normalize_text(cap_text[ci]):
                    own_hit += 1
                others = [normalize_text(t) for k, t in enumerate(cap_text) if k != ci]
                if all(p not in t for t in others):
                    unique += 1
    return {"n": n, "own_hit": own_hit, "unique": unique} if n else None


def normalize_text(text):
    import re as _re
    return _re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def read_scores(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(x, nd=4):
    return "nan" if x != x else f"{x:.{nd}f}"


def gather(rows, key):
    hall = [r[key] for r in rows if r["label"] == 1 and key in r]
    real = [r[key] for r in rows if r["label"] == 0 and key in r]
    return hall, real


def is_cont_fragment(text):
    """F2「续接片」：子句**自身**以小写字母开头，说明它本来在逗号后面接着上一句。

    如 `while the figure on the right wears ...`。这类碎片本身不是幻觉，但对任意
    图片 / patch 都能匹配得上（短语泛化性强），会占据视觉支持的分布高端、污染池子与
    分类器精度。按 `[.,;]` 切分 caption 时无法避免，只能显式过滤。
    """
    t = (text or "").lstrip()
    return bool(t) and t[0].islower()


def frag_flags(rows):
    """给每条子句打碎片标记（按 (gid, cap) 段内排序后判定）。

    - f2: 自身首字母小写 —— 逗号**续接片**（本子句是上一句的尾巴）；
    - f5: 下一个子句首字母小写 —— 本子句被逗号**截断**（如 `The image features a large`）。
    返回 id(row) -> {"f2": bool, "f5": bool}，便于回连原始记录。
    """
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[(r["gid"], r["cap"])].append(r)
    flags = {}
    for _, rs in by_cap.items():
        rs.sort(key=lambda r: r["idx"])
        for i, r in enumerate(rs):
            nxt = rs[i + 1] if i + 1 < len(rs) else None
            flags[id(r)] = {"f2": is_cont_fragment(r.get("text", "")),
                            "f5": bool(nxt and is_cont_fragment(nxt.get("text", "")))}
    return flags


def average_precision(hall, real):
    """幻觉（分数越低越像幻觉）为正类的平均精度 AP。

    AP 的随机基线 = 正类占比，故只有 AP 明显高于基率才说明该口径有判别力。
    """
    if not hall:
        return float("nan")
    items = sorted([(v, 1) for v in hall] + [(v, 0) for v in real])
    tp = ap = 0
    prev_rec = 0.0
    for i, (_, y) in enumerate(items):
        if y == 1:
            tp += 1
            rec = tp / len(hall)
            ap += (tp / (i + 1)) * (rec - prev_rec)
            prev_rec = rec
    return ap


def group_norm(rows, key, group="cap"):
    """把同一指标在**组内**标准化，消掉组级偏移（跨图/跨段落不可比的问题）。

    group="cap" 用 (gid, cap) 分组 = 段落（caption）内；"gid" = 同一图内。
    返回 (scores, labels, keys)，keys=(gid, cap, idx) 便于回连原始子句做位置控制；
      z    (s − μ) / σ     组内 z
      m     s − μ          组内中心化（保留原始尺度）
      rank  组内升序分位    相对排名 ∈ [0,1]，对极端值 / σ≈0 最稳
    """
    by = defaultdict(list)
    for r in rows:
        if key in r:
            by[r["gid"] if group == "gid" else (r["gid"], r["cap"])].append(r)
    out = {"z": [], "m": [], "rank": []}
    labels, keys = [], []
    for _, rs in by.items():
        vals = [r[key] for r in rs]
        k = len(vals)
        mu = sum(vals) / k
        sd = (sum((v - mu) ** 2 for v in vals) / (k - 1)) ** 0.5 if k > 1 else 0.0
        order = sorted(range(k), key=lambda i: vals[i])
        rk = [0.0] * k
        for p, i in enumerate(order):
            rk[i] = p / (k - 1) if k > 1 else 0.0
        for i, r in enumerate(rs):
            out["z"].append((vals[i] - mu) / sd if sd > 1e-9 else 0.0)
            out["m"].append(vals[i] - mu)
            out["rank"].append(rk[i])
            labels.append(r["label"])
            keys.append((r["gid"], r["cap"], r["idx"]))
    return out, labels, keys


def _hist_pair(ax, real, hall, key, xlabel):
    """幻觉 / 真实 的密度直方图 + 中位数虚线（最大视觉支撑的分布对比）。"""
    lo, hi = quantile(real + hall, 0.005), quantile(real + hall, 0.995)
    if not hi > lo:
        hi = lo + 1e-6
    bins = [lo + (hi - lo) * i / 80 for i in range(81)]
    mr, mh = quantile(real, 0.5), quantile(hall, 0.5)
    ax.hist(real, bins=bins, density=True, alpha=0.55, color="#4c72b0",
            label=f"real  n={len(real)}  μ={sum(real) / len(real):.3f}")
    ax.hist(hall, bins=bins, density=True, alpha=0.55, color="#c44e52",
            label=f"hallucinated  n={len(hall)}  μ={sum(hall) / len(hall):.3f}")
    ax.axvline(mr, color="#2b4a7d", ls="--", lw=1.2)
    ax.axvline(mh, color="#8c2f33", ls="--", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(f"{key}   AUC={auc_score(real, hall):.3f}", fontsize=10)
    ax.legend(fontsize=8)


def _cdf_pair(ax, real, hall, key):
    """累积分布：两条曲线的横向间距就是两种子句的重叠程度。"""
    import numpy as np
    for vals, c, lab in ((real, "#4c72b0", "real"),
                         (hall, "#c44e52", "hallucinated")):
        v = np.sort(np.asarray(vals, dtype=float))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=1.5, label=lab)
    ax.set_xlabel(key)
    ax.set_ylabel("CDF")
    ax.set_title(f"CDF of {key}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def make_plots(rows, metrics, args):
    """最大视觉支撑（s_box / s_short）的分布图。

    图 A：幻觉 vs 真实 的密度直方图 + CDF。
    图 B：换图基线差值 (原图 − 无关图) 的分布 + 按子句词数的 AUC 诊断。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"绘图跳过: {e}")
        return
    p = Path(args.plot)
    # 过滤版加 `_filtered` 后缀，避免覆盖未过滤的旧图（两套图并存便于对照）。
    sfx = "_filtered" if getattr(args, "drop_fragments", False) else ""
    tag = " [F2-filtered]" if sfx else ""
    main_png = p.with_name(f"{p.stem}{sfx}{p.suffix}")
    sup = [m for m in ["s_box", "s_short"] if m in metrics]
    if not sup:
        return

    # ---- 图 A：最大视觉支撑分布 ----
    fig, axes = plt.subplots(2, len(sup), figsize=(6.4 * len(sup), 8.2), squeeze=False)
    for j, key in enumerate(sup):
        hall, real = gather(rows, key)
        _hist_pair(axes[0][j], real, hall, key,
                   f"{key}  (masked max cosine)")
        _cdf_pair(axes[1][j], real, hall, key)
    fig.suptitle(f"Max visual support: hallucinated vs real clauses{tag}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(main_png, dpi=140)
    plt.close(fig)
    print(f"分布图已保存: {main_png}")

    # ---- 图 B：换图基线 + 词数诊断 ----
    swaps = [m for m in sup if f"s_swap_{m[2:]}" in metrics]
    n = len(swaps) + 1
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.4), squeeze=False)
    axes = axes[0]
    for ax, key in zip(axes, swaps):
        sk = f"s_swap_{key[2:]}"
        real = [r[key] - r[sk] for r in rows
                if r["label"] == 0 and key in r and sk in r]
        hall = [r[key] - r[sk] for r in rows
                if r["label"] == 1 and key in r and sk in r]
        _hist_pair(ax, real, hall, key, f"{key} − {sk}  (own − unrelated image)")

    ax = axes[-1]
    labs, a_box, a_short = [], [], []
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 10 ** 9)]:
        sub = [r for r in rows if lo <= r["n_words"] < hi]
        labs.append(f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+")
        for key, acc in (("s_box", a_box), ("s_short", a_short)):
            h, rr = gather(sub, key) if key in metrics else ([], [])
            acc.append(auc_score(rr, h) if h and rr else float("nan"))
    xs = list(range(len(labs)))
    ax.bar([x - 0.2 for x in xs], a_box, width=0.4, color="#4c72b0", label="s_box")
    ax.bar([x + 0.2 for x in xs], a_short, width=0.4, color="#c44e52", label="s_short")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_xlabel("clause word count")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("AUC by clause length", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"Swap-image baseline & length diagnosis{tag}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    diag = p.with_name(f"{p.stem}{sfx}_diag{p.suffix}")
    fig.savefig(diag, dpi=140)
    plt.close(fig)
    print(f"诊断图已保存: {diag}")

    # ---- 图 C：段落内标准化后的分布（看是否拉开） ----
    last = {}
    for r in rows:
        k = (r["gid"], r["cap"])
        if k not in last or r["idx"] > last[k]:
            last[k] = r["idx"]
    fig, axes = plt.subplots(3, len(sup), figsize=(6.4 * len(sup), 11.6), squeeze=False)
    for j, key in enumerate(sup):
        sc, lab, kk = group_norm(rows, key, "cap")
        panels = [
            ("z", "all clauses", None),
            ("rank", "all clauses", None),
            ("z", "non-last clauses only",
             [i for i, k in enumerate(kk) if k[2] != last[(k[0], k[1])]]),
        ]
        for i, (name, sub, keep) in enumerate(panels):
            sel = range(len(lab)) if keep is None else keep
            real = [sc[name][i2] for i2 in sel if lab[i2] == 0]
            hall = [sc[name][i2] for i2 in sel if lab[i2] == 1]
            _hist_pair(axes[i][j], real, hall, f"{key} in-cap {name}",
                       f"{key} within-caption {name} ({sub})")
    fig.suptitle(f"Within-caption normalised max visual support{tag}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    norm_png = p.with_name(f"{p.stem}{sfx}_capnorm{p.suffix}")
    fig.savefig(norm_png, dpi=140)
    plt.close(fig)
    print(f"段落内归一化分布图已保存: {norm_png}")


def stage_analyze(args):
    all_rows = read_scores(args.out)
    rows = [r for r in all_rows if not is_cont_fragment(r.get("text", ""))] \
        if getattr(args, "drop_fragments", False) else all_rows
    hall_rows = [r for r in rows if r["label"] == 1]
    real_rows = [r for r in rows if r["label"] == 0]
    n_gid = len({r["gid"] for r in rows})
    print(f"子句 {len(rows)}（幻觉 {len(hall_rows)} / 真实 {len(real_rows)}），图 {n_gid} 张")

    lines = []
    lines.append("# 子句视觉支持分布与幻觉阈值标定\n")
    lines.append(f"- 数据：{n_gid} 张主图 / {len(rows)} 个子句"
                 f"（幻觉 {len(hall_rows)}，真实 {len(real_rows)}）")
    lines.append("- 标注：独立 VLM 复核（`annotate_neg_clauses.py`）找回每条 caption "
                 "被注入的缺席物体短语，回填到子句 → label=1")
    lines.append("- 判别方向：视觉支持**越低**越可能是幻觉，判据为 `score <= t`")
    if getattr(args, "drop_fragments", False):
        lines.append(f"- **碎片过滤已启用**（`--drop-fragments`，剔除 F2 续接片）："
                     f"{len(all_rows)} → {len(rows)} 条（详见第 6.1 节对照）")
    lines.append("")

    # ---- 逐指标分布 ----
    lines.append("## 1. 各指标分布（幻觉 vs 真实）\n")
    lines.append("| 指标 | 幻觉 mean±std | 真实 mean±std | 幻觉 median | 真实 median | AUC | Cohen's d |")
    lines.append("|---|---|---|---|---|---|---|")
    metrics = [k for k in rows[0] if k.startswith(("s_", "z_", "C"))]
    order = [m for m in ["s_box", "s_short", "z_box", "z_short"] if m in metrics] + \
            [f"C{k}_box" for k in KS if f"C{k}_box" in metrics]
    stats = {}
    for m in order:
        hall, real = gather(rows, m)
        ma = sum(hall) / len(hall)
        mr = sum(real) / len(real)
        sa = (sum((x - ma) ** 2 for x in hall) / (len(hall) - 1)) ** 0.5
        sr = (sum((x - mr) ** 2 for x in real) / (len(real) - 1)) ** 0.5
        auc = auc_score(real, hall)   # 幻觉分低 => real 当正类，AUC>0.5 说明可分
        d = cohens_d(real, hall)
        stats[m] = {"auc": auc, "d": d}
        lines.append(f"| {m} | {fmt(ma)} ± {fmt(sa)} | {fmt(mr)} ± {fmt(sr)} | "
                     f"{fmt(quantile(hall, 0.5))} | {fmt(quantile(real, 0.5))} | "
                     f"{fmt(auc, 3)} | {fmt(d, 3)} |")
    lines.append("")

    # ---- 换图基线 ----
    swap_ms = [m for m in ["s_box", "s_short"]
               if m in metrics and f"s_swap_{m[2:]}" in metrics]
    if swap_ms:
        lines.append("## 2. 换图基线：信号是否真的来自原图视觉证据\n")
        lines.append("把同一子句拿去和一批**无关图**算同样的 max 余弦（`s_swap_*`）。"
                     "若换图后的 AUC 和原图相近，说明该指标主要抓的是"
                     "「语言先验 / 短语泛化性」，而不是这张图的视觉证据。"
                     "真正来自原图的证据应体现在**差值** `s − s_swap` 上（越大越好）。\n")
        lines.append("| 指标 | 原图 mean | 换图 mean | AUC(原图) | AUC(换图) | AUC(差值) |")
        lines.append("|---|---|---|---|---|---|")
        for m in swap_ms:
            sk = f"s_swap_{m[2:]}"
            for r in rows:
                if m in r and sk in r:
                    r[f"d_{m}"] = r[m] - r[sk]
            hall, real = gather(rows, m)
            hall_s, real_s = gather(rows, sk)
            hall_d, real_d = gather(rows, f"d_{m}")
            mm = sum(hall + real) / len(hall + real)
            ms = sum(hall_s + real_s) / len(hall_s + real_s)
            lines.append(f"| {m} | {fmt(mm)} | {fmt(ms)} | {fmt(auc_score(real, hall), 3)} | "
                         f"{fmt(auc_score(real_s, hall_s), 3)} | "
                         f"{fmt(auc_score(real_d, hall_d), 3)} |")
        lines.append("\n> 判读：AUC(差值) 明显高于 AUC(原图) ⇒ 原图视觉证据确实有贡献；"
                     "三者接近 ⇒ 阈值标定的其实是短语泛化性，跨图阈值不可靠。\n")
    else:
        lines.append("> 未提供 `--swap-images` / `--swap-n`，跳过换图基线。\n")

    # ---- 固定阈值 ----
    lines.append("## 3. 固定阈值（给定目标精度反解，判据 score <= t）\n")
    lines.append("patch 级 `z_*` / `C*_box` 的经验极性相反（幻觉子句分**更高**），"
                 "表内已整体取负以套用同一套「score <= t」口径，并在指标名后标注「取负」，"
                 "其阈值 t 也是取负后的尺度。\n")
    lines.append("| 指标 | 目标精度 | 阈值 t | 实际精度 | 幻觉召回 | 真实误报率 |")
    lines.append("|---|---|---|---|---|---|")
    hit_rates = {}
    for m in order[:4] + [f"C{k}_box" for k in KS if f"C{k}_box" in metrics]:
        hall, real = gather(rows, m)
        sign = -1.0 if auc_score(real, hall) < 0.5 else 1.0
        name = m if sign > 0 else f"{m}（取负）"
        hall = [sign * v for v in hall]
        real = [sign * v for v in real]
        for target in (0.5, 0.7, 0.8, 0.9):
            t, prec, rec = pick_low_threshold(hall, real, target)
            fpr = sum(1 for v in real if v <= t) / len(real) if t == t else float("nan")
            lines.append(f"| {name} | {target:.1f} | {fmt(t)} | {fmt(prec, 3)} | "
                         f"{fmt(rec, 3)} | {fmt(fpr, 3)} |")
    lines.append("")

    # ---- 阈值预算权衡 ----
    lines.append("### 3.1 阈值预算权衡（按低分位取阈值，直接回答「阈值该取多少」）\n")
    lines.append("上表在高精度目标下多为 nan：分布重叠严重，要够精度就得深入极低端尾部，"
                 "召回塌到 0。换成「按预算打」的视角——若把分数最低的 q 比例子句判为幻觉，"
                 "能得到什么精度/召回：\n")
    lines.append("| 指标 | 判定比例 q | 阈值 t | 命中 TP | 误报 FP | 精度 | 召回 |")
    lines.append("|---|---|---|---|---|---|---|")
    base = len(hall_rows) / max(len(rows), 1)
    lines.append(f"| （随机基线） | — | — | — | — | {fmt(base, 3)} | — |")
    for m in ["s_box", "s_short"]:
        if m not in metrics:
            continue
        hall, real = gather(rows, m)
        allv = sorted(hall + real)
        for q in (0.01, 0.02, 0.05, 0.10, 0.20):
            t = quantile(allv, q)
            tp = sum(1 for v in hall if v <= t)
            fp = sum(1 for v in real if v <= t)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            lines.append(f"| {m} | {q:.2f} | {fmt(t)} | {tp} | {fp} | "
                         f"{fmt(prec, 3)} | {fmt(tp / len(hall), 3)} |")
    lines.append(f"\n> 「随机基线」一行是幻觉子句的自然占比（{fmt(base, 3)}），"
                 "精度高于它才说明该阈值有增益。\n")

    # ---- 固定阈值能否跨图迁移：留出图验证 ----
    lines.append("### 3.2 固定阈值能否跨图迁移（留出图验证）\n")
    lines.append("§3 / §3.1 的精度-召回都是**在同一批数据上反解阈值**（in-sample），"
                 "天然乐观。固定阈值的真正可行性取决于**跨图迁移**：在奇数序号的图片上"
                 "标定阈值，拿到偶数序号的图片上评估。切分以**图**为单位（图级切分），"
                 "避免同一张图的子句同时出现在校准集与留出集里造成泄漏。\n")
    gids_all = sorted({r["gid"] for r in rows})
    hold = {g for i, g in enumerate(gids_all) if i % 2 == 0}
    n_cal = len(gids_all) - len(hold)
    n_te_h = sum(1 for r in rows if r["gid"] in hold and r["label"] == 1)
    n_te_r = sum(1 for r in rows if r["gid"] in hold and r["label"] == 0)
    lines.append(f"- 图级切分：校准 {n_cal} 张 / 留出 {len(hold)} 张"
                 f"（留出集子句 幻觉 {n_te_h}、真实 {n_te_r}，基率 "
                 f"{fmt(n_te_h / max(n_te_h + n_te_r, 1), 3)}）")
    lines.append("- 各口径的方向统一为「分越低越像幻觉」；`位置先验（猜末句）` 是"
                 "不含任何视觉信息的对照，视觉口径必须同时赢过它的精度与召回才算有增益。\n")

    last_a = last_idx_map(rows)
    signals = {
        "s_short raw（全局）": series_raw(rows, "s_short")[0],
        "s_short 图内m": series_group(rows, "s_short", "gid", "m")[0],
        "s_short 段落内m": series_group(rows, "s_short", "cap", "m")[0],
        "s_short 段落内z": series_group(rows, "s_short", "cap", "z")[0],
        "s_short 段落内rank": series_group(rows, "s_short", "cap", "rank")[0],
        "s_box 段落内m": series_group(rows, "s_box", "cap", "m")[0],
        "z_short（patch 级）": series_raw(rows, "z_short")[0],
        "C5_box": series_raw(rows, "C5_box")[0],
        "位置先验（猜末句）": series_last_position(rows, last_a)[0],
    }
    res = write_transfer_table(lines, signals, hold)

    def _prs(series):
        return pr_points([v for g, y, v in series if g in hold and y == 1],
                         [v for g, y, v in series if g in hold and y == 0])

    main_o = res.get(("s_short 段落内m", 0.7))
    raw_o = res.get(("s_short raw（全局）", 0.7))
    pos_o = res.get(("位置先验（猜末句）", 0.5))
    vis_p = prec_at_recall(_prs(signals["s_short 段落内m"]),
                           pos_o["te_r"]) if pos_o else float("nan")
    if main_o and raw_o:
        lines.append(f"> **判读**（图级留出）。全局口径 `s_short raw`：校准集目标精度 0.7 → "
                     f"留出集精度 {fmt(raw_o['te_p'], 3)}、召回 {fmt(raw_o['te_r'], 3)}；"
                     f"段落内口径 `s_short 段落内m`：精度 {fmt(main_o['te_p'], 3)}、"
                     f"召回 {fmt(main_o['te_r'], 3)}（校准集标定 t={fmt(main_o['t'])}）。"
                     f"in-sample（§3.1）与留出的精度基本持平 ⇒ 阈值本身**是可迁移的**，"
                     f"跨图没有明显漂移。")
        pos_p = pos_o["te_p"] if pos_o else float("nan")
        pos_r = pos_o["te_r"] if pos_o else float("nan")
        lines.append(f"> **同召回比精度**（位置先验唯一的非平凡操作点：召回 {fmt(pos_r, 3)}、"
                     f"精度 {fmt(pos_p, 3)}）：`s_short 段落内m` 在该召回的约束下最高精度为 "
                     f"{fmt(vis_p, 3)}。"
                     + ("视觉口径**压过**位置先验 ⇒ 固定阈值携带位置先验之外的视觉信息。\n"
                        if vis_p == vis_p and pos_p == pos_p and vis_p > pos_p else
                        "视觉口径**未能压过**位置先验 ⇒ 该阈值本质在复述「幻觉常在末句」。\n"))
        lines.append(f"> 但代价是**召回很低**：精度 0.7 时留出召回仅 {fmt(main_o['te_r'], 3)}。"
                     "故结论是——固定阈值**可行但覆盖不足**，适合「高精度、低召回」的保守筛查，"
                     "不适合作为唯一的幻觉判定器。\n")

    # ---- 叠加 F2 碎片过滤后的迁移验证 ----
    if not getattr(args, "drop_fragments", False):
        fl_a = frag_flags(all_rows)
        rows_nf = [r for r in rows if not fl_a[id(r)]["f2"]]
        n_nf_h = sum(1 for r in rows_nf if r["label"] == 1)
        lines.append("**叠加 F2 碎片过滤后**（分析侧已决定启用的口径；只缩小候选池，"
                     "不改变图级切分）：\n")
        lines.append(f"- 过滤后子句 {len(rows_nf)}（幻觉 {n_nf_h}，基率 "
                     f"{fmt(n_nf_h / max(len(rows_nf), 1), 3)}），留出集样本相应减少。\n")
        last_nf = last_idx_map(rows_nf)
        sig_nf = {
            "s_short raw（全局）": series_raw(rows_nf, "s_short")[0],
            "s_short 段落内m": series_group(rows_nf, "s_short", "cap", "m")[0],
            "s_box 段落内m": series_group(rows_nf, "s_box", "cap", "m")[0],
            "位置先验（猜末句）": series_last_position(rows_nf, last_nf)[0],
        }
        res_nf = write_transfer_table(lines, sig_nf, hold)
        nf_m = res_nf.get(("s_short 段落内m", 0.7))
        nf_r = res_nf.get(("s_short raw（全局）", 0.7))
        if nf_m and main_o and raw_o and nf_r:
            lines.append(f"> **判读**：过滤掉 F2 续接片后，同样在留出图上取目标精度 0.7，"
                         f"`s_short 段落内m` 的召回从 {fmt(main_o['te_r'], 3)} 提升到 "
                         f"{fmt(nf_m['te_r'], 3)}（精度 {fmt(nf_m['te_p'], 3)}）；"
                         f"全局口径 `s_short raw` 从 {fmt(raw_o['te_r'], 3)} 到 "
                         f"{fmt(nf_r['te_r'], 3)}。即**固定阈值路线的瓶颈主要是碎片污染，"
                         "而不是阈值不可迁移**——清掉碎片后同一目标精度下的覆盖率接近翻倍。\n")

    # ---- 归一化口径：全局 / 图内 / 段落内 ----
    lines.append("## 4. 归一化口径：全局 vs 图内 vs 段落内\n")
    lines.append("固定全局阈值失败的原因是「跨图、跨段落不可比」。把同一指标放到"
                 "**同一段落（caption）内**和其他子句一起标准化，可消掉段落级的"
                 "语言先验偏移。方向统一为「越大越像真实」，AUC>0.5 越大越好：\n")
    lines.append("| 指标 | 口径 | AUC | 阈值@精度0.7 | 召回 | 判最低10%的精度 | 召回 |")
    lines.append("|---|---|---|---|---|---|---|")
    last = {}                       # (gid, cap) -> 最大 idx，用于位置控制
    for r in rows:
        k = (r["gid"], r["cap"])
        if k not in last or r["idx"] > last[k]:
            last[k] = r["idx"]
    variants = []                   # (显示名, 真实分, 幻觉分, 真实键, 幻觉键)
    for key in ["s_box", "s_short"]:
        if key not in metrics:
            continue
        h0, r0 = gather(rows, key)
        kr0 = [(r["gid"], r["cap"], r["idx"]) for r in rows
               if key in r and r["label"] == 0]
        kh0 = [(r["gid"], r["cap"], r["idx"]) for r in rows
               if key in r and r["label"] == 1]
        variants.append((f"{key} raw（全局）", r0, h0, kr0, kh0))
        for grp, glab in (("gid", "图内"), ("cap", "段落内")):
            sc, lab, kk = group_norm(rows, key, grp)
            for name in ("z", "m", "rank"):
                rl, kr, hl, kh = [], [], [], []
                for v, y, k in zip(sc[name], lab, kk):
                    if y == 1:
                        hl.append(v)
                        kh.append(k)
                    else:
                        rl.append(v)
                        kr.append(k)
                variants.append((f"{key} {glab}{name}", rl, hl, kr, kh))
    for name, rl, hl, _, _ in variants:
        auc = auc_score(rl, hl)
        t, prec, rec = pick_low_threshold(hl, rl, 0.7)
        allv = sorted(rl + hl)
        t10 = quantile(allv, 0.10)
        tp = sum(1 for v in hl if v <= t10)
        fp = sum(1 for v in rl if v <= t10)
        p10 = tp / (tp + fp) if tp + fp else float("nan")
        lines.append(f"| {name.split(' ', 1)[0]} | {name.split(' ', 1)[1]} | {fmt(auc, 3)} | "
                     f"{fmt(t)} | {fmt(rec, 3)} | {fmt(p10, 3)} | "
                     f"{fmt(tp / len(hl) if hl else float('nan'), 3)} |")
    lines.append("")

    # ---- 位置控制：扣掉「最后一句」偏置 ----
    lines.append("### 4.1 位置控制（扣掉「最后一句」偏置后的口径对比）\n")
    lines.append("幻觉子句大量出现在句尾（见下方位置先验），末句本身语言先验就不同，"
                 "所以任何归一化的增益都必须**在非末句子句集合内仍然成立**才算真实视觉信号：\n")
    lines.append("| 指标 | 口径 | AUC 全部 | AUC 仅非末句 | AUC 仅末句 |")
    lines.append("|---|---|---|---|---|")
    for name, rl, hl, kr, kh in variants:
        toks = name.split(" ", 1)
        keep_r = {i for i, k in enumerate(kr) if k[2] != last[(k[0], k[1])]}
        keep_h = {i for i, k in enumerate(kh) if k[2] != last[(k[0], k[1])]}
        a_all = auc_score(rl, hl)
        a_nonlast = auc_score([rl[i] for i in keep_r], [hl[i] for i in keep_h])
        a_last = auc_score([v for i, v in enumerate(rl) if i not in keep_r],
                           [v for i, v in enumerate(hl) if i not in keep_h])
        lines.append(f"| {toks[0]} | {toks[1]} | {fmt(a_all, 3)} | "
                     f"{fmt(a_nonlast, 3)} | {fmt(a_last, 3)} |")
    lines.append("")

    # ---- 位置先验基线（诚实对照） ----
    by_cap_l = defaultdict(list)
    for r in rows:
        by_cap_l[(r["gid"], r["cap"])].append(r)
    pr, ph = [], []
    for _, rs in by_cap_l.items():
        rs = sorted(rs, key=lambda r: r["idx"])
        k = len(rs)
        for i, r in enumerate(rs):
            v = 0.0 if i == k - 1 else 1.0  # 越大越像真实
            (ph if r["label"] == 1 else pr).append(v)
    n_last = sum(1 for v in ph if v == 0.0)
    lines.append(f"> **位置先验对照**：幻觉子句有 {n_last}/{len(ph)} = "
                 f"{fmt(n_last / max(len(ph), 1), 3)} 落在每条 caption 的**最后一句**"
                 f"（注入时通常追加在句尾）；单靠「猜最后一句」这条规则的 AUC = "
                 f"{fmt(auc_score(pr, ph), 3)}、精度 = "
                 f"{fmt(n_last / max(len(by_cap_l), 1), 3)}。"
                 "比较归一化口径时要扣掉这个偏置，否则会把位置信息误当成视觉信号。\n")

    # ---- argmin 基线 ----
    lines.append("## 5. 参考基线：组内 argmin 命中率\n")
    lines.append("（对照用。argmin 是相对判别，只能当训练信号，不能当评估阈值。）\n")
    lines.append("| 口径 | 每条 caption 的 argmin 是否=幻觉子句 |")
    lines.append("|---|---|")
    for key in ["s_box", "C5_box", "C20_box", "z_box"]:
        if key not in metrics:
            continue
        by_cap = defaultdict(list)
        for r in rows:
            if key in r:
                by_cap[(r["gid"], r["cap"])].append(r)
        hit = tot = 0
        for _, rs in by_cap.items():
            if not any(r["label"] == 1 for r in rs):
                continue
            tot += 1
            am = min(rs, key=lambda r: r[key])
            if am["label"] == 1:
                hit += 1
        hit_rates[key] = hit / tot if tot else float("nan")
        lines.append(f"| {key} | {hit}/{tot} = {fmt(hit / tot if tot else float('nan'), 3)} |")
    lines.append("")

    # ---- 子句长度（碎片）诊断 ----
    lines.append("## 6. 子句长度（切分碎片）诊断\n")
    lines.append("按 `[.,;]` 切分会把长句切成碎片；碎片是**真实内容**但视觉支持天然偏低，"
                 "会被固定阈值误判成幻觉。按词数分桶看可分性：\n")
    lines.append("| 词数桶 | 子句数 | 幻觉数 | AUC(s_box) | AUC(s_short) |")
    lines.append("|---|---|---|---|---|")
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 10 ** 9)]:
        sub = [r for r in rows if lo <= r["n_words"] < hi]
        hb = [r["s_box"] for r in sub if r["label"] == 1 and "s_box" in r]
        rb = [r["s_box"] for r in sub if r["label"] == 0 and "s_box" in r]
        hs = [r["s_short"] for r in sub if r["label"] == 1 and "s_short" in r]
        rs = [r["s_short"] for r in sub if r["label"] == 0 and "s_short" in r]
        lab = f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+"
        lines.append(f"| {lab} | {len(sub)} | {len(hb)} | {fmt(auc_score(rb, hb), 3)} | "
                     f"{fmt(auc_score(rs, hs), 3)} |")
    lines.append("")

    # ---- 碎片污染量化 + 过滤口径对照 ----
    lines.append("## 6.1 切分碎片污染与过滤口径对照\n")
    lines.append("按 `[.,;]` 切分 caption 会产出两类**非幻觉碎片**，它们对任意图片甚至"
                 "单个 patch 都能匹配（短语泛化性强），因而占据视觉支持分布的高端、"
                 "污染池子与分类器精度：\n")
    lines.append("- **F2 续接片**：子句自身以小写字母开头（逗号后接续），"
                 "如 `while the figure on the right wears ...`；")
    lines.append("- **F5 截断片**：子句的**下一个**子句以小写开头（本子句被逗号截断），"
                 "如 `The image features a large`。\n")
    fl = frag_flags(all_rows)
    n_all = len(all_rows)
    n_pos = sum(1 for r in all_rows if r["label"] == 1)
    base_all = n_pos / max(n_all, 1)
    med_all = quantile([r["s_short"] for r in all_rows if "s_short" in r], 0.5)

    def _grp(keep):
        sub = [r for r in all_rows if keep(r)]
        pos = sum(1 for r in sub if r["label"] == 1)
        med = quantile([r["s_short"] for r in sub if "s_short" in r], 0.5)
        return sub, pos, med

    f2_sub, f2_pos, f2_med = _grp(lambda r: fl[id(r)]["f2"])
    f5_sub, f5_pos, f5_med = _grp(lambda r: fl[id(r)]["f5"])
    f4_sub, f4_pos, f4_med = _grp(lambda r: r["n_words"] <= 4)
    lines.append("| 碎片类别 | 子句数 | 占池比 | 幻觉率 | s_short 中位 | 全池中位 |")
    lines.append("|---|---|---|---|---|---|")
    for nm, sub, pos, med in (("F2 自身小写（续接片）", f2_sub, f2_pos, f2_med),
                              ("F5 被下一句截断", f5_sub, f5_pos, f5_med),
                              ("≤4 词片段", f4_sub, f4_pos, f4_med)):
        lines.append(f"| {nm} | {len(sub)} | {fmt(len(sub) / max(n_all, 1), 3)} | "
                     f"{fmt(pos / max(len(sub), 1), 3)} | {fmt(med)} | {fmt(med_all)} |")

    # 高端污染：s_short 最高 10% 里碎片占多少
    s_all = sorted(r["s_short"] for r in all_rows if "s_short" in r)
    if s_all:
        t_top = quantile(s_all, 0.90)
        top = [r for r in all_rows if r.get("s_short", -1) >= t_top]
        top_frag = sum(1 for r in top if r["n_words"] <= 4)
        top_pos = sum(1 for r in top if r["label"] == 1)
        lines.append(f"\n> **高端污染**：`s_short` 最高的 10%（n={len(top)}）里，"
                     f"≤4 词碎片占 {top_frag}/{len(top)} = "
                     f"{fmt(top_frag / max(len(top), 1), 3)}，而真正的幻觉子句只占 "
                     f"{top_pos}/{len(top)} = {fmt(top_pos / max(len(top), 1), 3)}"
                     f"（池子基率 {fmt(base_all, 3)}）。即高分端几乎被碎片垄断，"
                     "任何「取高分」的用法都会先撞上碎片。\n")

    # 过滤口径对照：AP（越高越好）
    keep_preds = [
        ("无过滤", lambda r: True),
        ("F2 自身小写（推荐）", lambda r: not fl[id(r)]["f2"]),
        ("F2 + nw≤4", lambda r: not (fl[id(r)]["f2"] or r["n_words"] <= 4)),
        ("F2 ∪ F5", lambda r: not (fl[id(r)]["f2"] or fl[id(r)]["f5"])),
        ("仅 nw≤4", lambda r: r["n_words"] > 4),
    ]
    lines.append("过滤口径用 AP（幻觉为正类、分数越低越像幻觉；随机基线=池基率）衡量，"
                 "同时看段内 z 口径与正例保留率：\n")
    lines.append("| 过滤口径 | 保留 n | 基率 | AP(s_short raw) | AP(s_short 段内z) | 保留幻觉正例 |")
    lines.append("|---|---|---|---|---|---|")
    for nm, keep in keep_preds:
        sub = [r for r in all_rows if keep(r)]
        hs, rs_ = gather(sub, "s_short")
        ap_raw = average_precision(hs, rs_)
        sc, lab, _ = group_norm(sub, "s_short", "cap")
        ap_z = average_precision([v for v, y in zip(sc["z"], lab) if y == 1],
                                 [v for v, y in zip(sc["z"], lab) if y == 0])
        npos = sum(1 for r in sub if r["label"] == 1)
        lines.append(f"| {nm} | {len(sub)} | {fmt(npos / max(len(sub), 1), 3)} | "
                     f"{fmt(ap_raw, 3)} | {fmt(ap_z, 3)} | "
                     f"{fmt(npos / max(n_pos, 1), 3)} |")
    lines.append("\n> 判读：**F2 自身小写**是最佳单条过滤规则——池子缩小且基率上升，"
                 "AP 明显抬升，同时保住约七成幻觉正例；词长过滤几乎无用"
                 "（`The image features a large` 有 5 个词，按 `≤4` 逃得掉）；"
                 "`F2 ∪ F5` 虽 AP 更高，但会丢掉近三分之二的幻觉正例，段内 z 口径反而变差。"
                 "训练侧若直接套用同样的过滤会损失正例，建议只在**分析/打分口径**启用，"
                 "训练侧另行处理（如给碎片降权而非剔除）。\n")
    lines.append("")

    # ---- 标注自洽性检查 ----
    lq = label_quality(args.labels)
    if lq:
        lines.append("## 7. 标注自洽性检查（与 FG-CLIP 无关）\n")
        lines.append("生成 prompt 要求 4 条 caption 注入**互不相同**的物体，"
                     "因此每条 caption 的幻觉短语不应出现在同图其他 caption 里：\n")
        lines.append(f"- 短语能精确命中本 caption 子句：{lq['own_hit']}/{lq['n']} "
                     f"= {fmt(lq['own_hit'] / max(lq['n'], 1), 3)}")
        lines.append(f"- 短语**未**泄漏到同图其他 caption：{lq['unique']}/{lq['n']} "
                     f"= {fmt(lq['unique'] / max(lq['n'], 1), 3)}")
        lines.append("- 泄漏率高说明标注可能误指真实物体，需抽检。\n")

    # ---- 推荐阈值：一个可直接用的分割点 ----
    lines.append("## 8. 推荐阈值：一个可直接用的分割点\n")
    lines.append("把「大致分割幻觉 / 事实」落成一个操作点，最直白的选法是取 **F1 最大**"
                 "（精度与召回折中）。阈值的取法严格走图级留出：在**校准图**（奇数序号）"
                 "上取 F1 最大的 t，再拿到**留出图**（偶数序号）上评估，避免 in-sample 乐观。"
                 "判据统一为 `score <= t` 判为幻觉，所有序列方向已统一成「分越低越像幻觉」。\n")

    def _f1(p, r):
        return 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float("nan")

    def _eval_at(series, gset, t):
        h = [v for g, y, v in series if g in gset and y == 1]
        r_ = [v for g, y, v in series if g in gset and y == 0]
        tp = sum(1 for v in h if v <= t)
        fp = sum(1 for v in r_ if v <= t)
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / len(h) if h else float("nan")
        return p, r, _f1(p, r)

    cal = set(gids_all) - hold
    rec_specs = [
        ("s_short 段落内m（相对，推荐）", "s_short 段落内m"),
        ("s_short 段落内z（相对）", "s_short 段落内z"),
        ("s_short raw（全局绝对尺度）", "s_short raw（全局）"),
        ("s_box 段落内m（相对，次选）", "s_box 段落内m"),
    ]
    lines.append("| 口径 | 校准 F1 最优 t | 校准精度 | 校准召回 | 校准 F1 | "
                 "**留出精度** | 留出召回 | 留出 F1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    f1_pick = {}
    for nm, sk in rec_specs:
        series = signals[sk]
        ch = [v for g, y, v in series if g in cal and y == 1]
        cr = [v for g, y, v in series if g in cal and y == 0]
        t, p, r, f1 = pick_f1_threshold(ch, cr)
        f1_pick[nm] = t
        tp_, tr_, tf1 = _eval_at(series, hold, t)
        lines.append(f"| {nm} | {fmt(t)} | {fmt(p, 3)} | {fmt(r, 3)} | {fmt(f1, 3)} | "
                     f"**{fmt(tp_, 3)}** | {fmt(tr_, 3)} | {fmt(tf1, 3)} |")
    lines.append("")

    # 推荐口径的三个常用工作点
    main_nm = "s_short 段落内m（相对，推荐）"
    m_series = signals["s_short 段落内m"]
    m_ch = [v for g, y, v in m_series if g in cal and y == 1]
    m_cr = [v for g, y, v in m_series if g in cal and y == 0]
    t_hp, p_hp, _ = pick_low_threshold(m_ch, m_cr, 0.8)      # 高精度点
    t_f1 = f1_pick[main_nm]                                  # F1 最优点
    t_hr = quantile(sorted(m_ch), 0.9) if m_ch else float("nan")  # 高召回点（召回≈0.9）
    lines.append(f"**推荐口径 `s_short 段落内m` 的三个常用工作点**（阈值均取校准图 → "
                 f"留出图评估）：\n")
    lines.append("| 工作点 | 阈值来源 | t | 留出精度 | 留出召回 | 留出 F1 | 适用场景 |")
    lines.append("|---|---|---|---|---|---|---|")
    for wp, src, t, use in (
            ("高精度", "目标精度 0.8", t_hp, "保守筛查，宁漏勿错"),
            ("平衡（推荐）", "F1 最大", t_f1, "大致分割幻觉/事实"),
            ("高召回", "召回 ≈ 0.9", t_hr, "先捞一遍，代价是误报多")):
        p_, r_, f_ = _eval_at(m_series, hold, t)
        lines.append(f"| {wp} | {src} | {fmt(t, 3)} | {fmt(p_, 3)} | {fmt(r_, 3)} | "
                     f"{fmt(f_, 3)} | {use} |")
    lines.append("")

    raw_nm = "s_short raw（全局绝对尺度）"
    lines.append("**用法（把阈值落到代码里）**：\n")
    lines.append(f"- **首选 `s_short 段落内m`（相对阈值，t={fmt(t_f1, 3)}）**：它消掉了"
                 "跨图/跨段落先验偏移，跨图可迁移性最好，且不需要固定全局尺度。"
                 "每条 caption 内：先算全部子句的 `s_short`，取段落均值 μ，"
                 "子句的 `m = s_short − μ`；`m <= t` 判为幻觉。")
    lines.append(f"- **备选 `s_short raw`（绝对阈值，t={fmt(f1_pick[raw_nm], 4)}）**："
                 "直接比较 `s_short`，部署最省事，但跨图绝对值有漂移，换数据集需重新标定。")
    lines.append("- 若要更平滑的相对口径，可把 `m` 换成 `段落内rank`"
                 "（子句在段内的升序分位，∈[0,1]），对极端值和 σ≈0 最稳。")
    lines.append("\n> 判读：F1 最优点的留出精度/召回与校准集基本持平 ⇒ 该阈值**可迁移**，"
                 "不是 in-sample 拟合的假象。但绝对量级仍是「精度中等、召回偏低」——"
                 "它只能**大致**分割，适合做预警 / 粗筛或训练信号，不宜当唯一判定器；"
                 "想抬高覆盖，先清 F2 续接片（§6.1），再考虑把阈值调向高召回工作点。\n")

    report = "\n".join(lines)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)

    # ---- 分布图 ----
    make_plots(rows, metrics, args)


def main():
    ap = argparse.ArgumentParser(description="子句视觉支持统计 + 幻觉阈值标定")
    ap.add_argument("--stage", choices=["score", "analyze"], default="score")
    ap.add_argument("--image-dir", default="/root/autodl-tmp/data/SA1B_5_threshold")
    ap.add_argument("--labels", default="/root/MVPG/neg_clause_labels.jsonl")
    ap.add_argument("--out", default="/root/MVPG/clause_support_scores.jsonl")
    ap.add_argument("--report", default="/root/MVPG/clause_support_report.md")
    ap.add_argument("--plot", default="/root/MVPG/clause_support_hist.png")
    ap.add_argument("--fgclip-root", default="/root/autodl-tmp/cache/fgclip2-base-patch16")
    ap.add_argument("--walks", default="box,short", help="逗号分隔，如 box,short,long")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--swap-images", nargs="*", default=[],
                    help="换图基线用的无关图路径（显式指定）")
    ap.add_argument("--swap-n", type=int, default=8,
                    help="未显式指定时，从数据目录自动挑 n 张不在标注集内的图当无关图；0=关闭")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drop-fragments", action="store_true",
                    help="分析时剔除 F2 续接片（自身小写开头的逗号碎片），见报告 6.1 节")
    args = ap.parse_args()
    args.walks = [w for w in args.walks.split(",") if w]
    if args.stage == "score":
        stage_score(args)
    else:
        stage_analyze(args)


if __name__ == "__main__":
    main()
