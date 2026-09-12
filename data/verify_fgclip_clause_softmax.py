"""验证「子句切分 + box 扫描 + softmax」能否定位幻觉子句。

比上一轮更贴合真实：直接对 neg_captions 里的整句负样本 caption 按标点切成子句，
每个子句用 box 头 × dense patch 算相似度，再对同一 caption 的子句做 softmax(s/τ)，
看「幻觉子句」（含手工标注幻觉物体）的概率是否显著低于同句其它真实子句。

幻觉子句通过 hall 实体核心词（去掉冠词）子串匹配自动识别。
输出 markdown 到项目根目录。
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/MVPG")

import torch
import torch.nn.functional as F
from PIL import Image

from mvpg_run.fgclip_local import load_fgclip, preprocess_images  # noqa: E402

FGCLIP_ROOT = "/root/autodl-tmp/cache/fgclip2-base-patch16"
DATA = "/root/autodl-tmp/data/SA1B_5_threshold"
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TAUS = [0.5, 0.2, 0.1, 0.05]

# 每个 gid 的幻觉物体核心词（去掉冠词），用于在负样本 caption 里定位幻觉子句
HALL_CORES = {
    "sa_1000": ["brown dog", "woven basket", "brass oil lamp", "wooden ladder"],
    "sa_183": ["checkered flag"],
    "sa_2646": ["brown dog", "red bicycle", "child in a blue jacket", "wooden park bench"],
    "sa_3439": ["fire hydrant", "brown briefcase", "wooden bench", "street lamp"],
    "sa_4286": ["stone lion statue", "wooden rowboat", "metal flagpole", "campfire"],
    "sa_5122": ["black bird", "brass mailbox", "potted plant", "set of keys"],
    "sa_5954": ["yellow crane", "red bicycle", "green bench", "silver motorcycle"],
    "sa_6749": ["red double-decker bus", "group of cyclists", "large fountain", "yellow taxi"],
    "sa_7547": ["brown dog", "red bicycle", "cat", "wooden bench"],
    "sa_8367": ["wooden bench", "blue bicycle", "small fountain", "group of pigeons"],
    "sa_9166": ["red stop sign", "wooden bench", "brown dog", "yellow bicycle"],
    "sa_9998": ["wooden rowboat", "green rowboat", "white seagull", "red kayak"],
}


def main_img_path(gid):
    d = Path(DATA) / gid
    if not d.is_dir():
        return None
    for s in IMG_SUFFIXES:
        p = d / (gid + s)
        if p.is_file():
            return str(p)
    return None


def dense_features(fg, img_path):
    im = Image.open(img_path).convert("RGB")
    pv, mask, shapes = preprocess_images([im])
    pv = pv.to(fg.device); mask = mask.to(fg.device); shapes = shapes.to(fg.device)
    with torch.no_grad():
        dense = fg.model.get_image_dense_feature(
            pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=shapes)
    return dense[0], mask[0].bool()


def encode_text_walk(fg, texts, walk_type):
    texts = list(texts)
    tok = fg.tokenizer(texts)
    for k, v in tok.items():
        tok[k] = v.to(fg.device)
    with torch.no_grad():
        feat = fg.model.get_text_features(
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], walk_type=walk_type)
    return feat


def max_patch_sim(dense, valid, text_feat):
    dn = F.normalize(dense, p=2, dim=-1)
    tn = F.normalize(text_feat, p=2, dim=-1)
    sim = dn @ tn.T
    sim = sim.masked_fill(~valid.unsqueeze(1), -1e9)
    return sim.max(dim=0).values


def split_clauses(caption):
    parts = re.split(r"[.,;]", caption)
    out = []
    for p in parts:
        p = p.strip()
        if len(p.split()) >= 3:   # 至少 3 个词，过滤过短碎片
            out.append(p)
    return out


def softmax(x, tau):
    x = torch.tensor(x, dtype=torch.float32) / tau
    x = x - x.max()
    e = torch.exp(x)
    return (e / e.sum()).tolist()


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0


def cohens_d(a, b):
    sa, sb = std(a), std(b)
    sp = ((sa ** 2 + sb ** 2) / 2) ** 0.5
    return (mean(b) - mean(a)) / sp if sp > 0 else 0.0


def main():
    t0 = time.time()
    fg = load_fgclip(FGCLIP_ROOT, "cpu")
    neg = json.load(open(f"{DATA}/neg_captions.json"))
    print(f"loaded fgclip + neg in {time.time() - t0:.1f}s")

    clause_rows = []
    for gid, cores in HALL_CORES.items():
        img = main_img_path(gid)
        if img is None:
            continue
        dense, valid = dense_features(fg, img)
        for cap in neg.get(gid, []):
            sub = split_clauses(cap)
            if len(sub) < 2:
                continue
            s_box = max_patch_sim(dense, valid, encode_text_walk(fg, sub, "box")).tolist()
            s_short = max_patch_sim(dense, valid, encode_text_walk(fg, sub, "short")).tolist()
            rows = []
            for cl, sb, ss in zip(sub, s_box, s_short):
                low = cl.lower()
                is_hall = any(c in low for c in cores)
                rows.append({"is_hall": is_hall, "clause": cl, "s_box": sb, "s_short": ss})
            if not any(r["is_hall"] for r in rows):
                continue   # 该 caption 没匹配到幻觉子句，跳过
            for tau in TAUS:
                pb = softmax([r["s_box"] for r in rows], tau)
                ps = softmax([r["s_short"] for r in rows], tau)
                for r, pbi, psi in zip(rows, pb, ps):
                    r[f"p_box_tau{tau}"] = pbi
                    r[f"p_short_tau{tau}"] = psi
            for r in rows:
                r["gid"] = gid
            clause_rows.extend(rows)

    def summarize(pkey, skey):
        hall = [r[pkey] for r in clause_rows if r["is_hall"]]
        real = [r[pkey] for r in clause_rows if not r["is_hall"]]
        # 定位准确率：幻觉子句是否是该 caption 内概率最低的子句
        # 需要按 caption 分组；用近似：按 (gid, 连续段落) 分组较复杂，这里给全局 + 组级 margin
        gid_margins = []
        for gid in HALL_CORES:
            gr = [r for r in clause_rows if r["gid"] == gid]
            mh = mean([r[pkey] for r in gr if r["is_hall"]])
            mr = mean([r[pkey] for r in gr if not r["is_hall"]])
            gid_margins.append(mr - mh)
        # 原始相似度对照
        sh = [r[skey] for r in clause_rows if r["is_hall"]]
        sr = [r[skey] for r in clause_rows if not r["is_hall"]]
        return {
            "hall_mean": mean(hall), "real_mean": mean(real),
            "cohens_d": cohens_d(hall, real),
            "margin_mean": mean([mr - mh for mr, mh in
                                 [(mean([r[pkey] for r in clause_rows if r['gid'] == g and not r['is_hall']]),
                                   mean([r[pkey] for r in clause_rows if r['gid'] == g and r['is_hall']]))
                                  for g in HALL_CORES]]),
            "gid_margin_gt0": sum(1 for m in gid_margins if m > 0),
            "n_groups": len(HALL_CORES),
            "raw_s_hall": mean(sh), "raw_s_real": mean(sr),
        }

    n_hall = sum(1 for r in clause_rows if r["is_hall"])
    n_real = sum(1 for r in clause_rows if not r["is_hall"])
    summary = {"n_hall": n_hall, "n_real": n_real}
    summary["raw_box"] = summarize("s_box", "s_box")   # 复用，pkey=skey 即原始
    summary["raw_short"] = summarize("s_short", "s_short")
    for tau in TAUS:
        summary[f"softmax_box_tau{tau}"] = summarize(f"p_box_tau{tau}", "s_box")
    for tau in TAUS:
        summary[f"softmax_short_tau{tau}"] = summarize(f"p_short_tau{tau}", "s_short")

    json.dump({"summary": summary, "clauses": clause_rows},
              open("/root/MVPG/fgclip_clause_softmax_results.json", "w"),
              ensure_ascii=False, indent=2)

    L = []
    L.append("# 子句切分 + box 扫描 + softmax 幻觉定位验证报告\n")
    L.append(f"- 数据：`{DATA}` 负样本 caption（12 组，共 {len(clause_rows)} 子句，"
             f"幻觉子句 {n_hall} / 真实子句 {n_real}）\n")
    L.append("- 方法：每条负样本 caption 按标点切子句 → 每个子句 box/short 头 × dense patch 算相似度"
             "→ 同 caption 内 softmax(s/τ) → 比较幻觉子句 vs 真实子句的概率。\n")

    L.append("\n## 结果（Cohen's d 越大越好，目标 d≥1.0）\n")
    L.append("| 信号 | 幻觉子句 prob | 真实子句 prob | Cohen's d | 组级 margin>0 |")
    L.append("|---|---|---|---|---|")
    def row(name, m):
        L.append(f"| {name} | {m['hall_mean']:.4f} | {m['real_mean']:.4f} | "
                 f"**{m['cohens_d']:+.2f}** | {m['gid_margin_gt0']}/{m['n_groups']} |")
    row("raw s (box)", summary["raw_box"])
    row("raw s (short)", summary["raw_short"])
    for tau in TAUS:
        row(f"softmax box τ={tau}", summary[f"softmax_box_tau{tau}"])
    for tau in TAUS:
        row(f"softmax short τ={tau}", summary[f"softmax_short_tau{tau}"])

    best = max([(summary[k]["cohens_d"], k) for k in summary if k.startswith("softmax")])
    L.append("\n## 结论\n")
    L.append(f"- 最强组合 **{best[1]}**，Cohen's d={best[0]:+.2f}"
             f"{'，达稳定(d≥1.0)' if best[0] >= 1.0 else '，仍不达稳定(d<1.0)'}。\n")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_clause_softmax_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_clause_softmax_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
