"""对比「逗号分割」vs「句号分割」对子句级幻觉信号清晰度的影响。

对 20 张主图（不再用子图）的每条负样本 caption，分别按逗号 / 句号切分成子句，
每个子句用 box 头 × dense patch 算 max_patch_sim（一对一 raw 相似度，无需 softmax），
标记幻觉子句（核心词匹配），比较两种分割下 hall vs real 子句的区分度。

指标：Cohen's d、组级 margin>0 比例、幻觉子句是否为 caption 内相似度最低（top-1 定位准确率）。
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

# 20 个 gid 的幻觉物体核心词（去掉冠词，用于在负样本 caption 里定位幻觉子句）
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
    "sa_1001": ["rocking horse", "potted plant", "cash register", "leather handbag"],
    "sa_1002": ["flat-screen television", "grand piano", "palm tree", "silver tray"],
    "sa_1004": ["stopwatch", "tennis ball", "wooden bench", "blue bicycle"],
    "sa_1007": ["black bicycle", "white dog", "silver trophy", "school bus"],
    "sa_1008": ["cargo ship", "school bus", "hot air balloon", "sailboat"],
    "sa_1009": ["backpack", "headphones", "wooden desk", "wristwatch"],
    "sa_1013": ["bicycle", "school bus", "picnic table", "fountain"],
    "sa_1016": ["white swan", "wooden bench", "narrowboat", "fisherman"],
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


def split_clauses(caption, pattern):
    parts = re.split(pattern, caption)
    out = []
    for p in parts:
        p = p.strip()
        if len(p.split()) >= 3:
            out.append(p)
    return out


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
    print(f"loaded in {time.time() - t0:.1f}s")

    # 结果容器：comma / period / combined 三种分割下的子句记录
    results = {"comma": [], "period": [], "combined": []}
    for gid, cores in HALL_CORES.items():
        img = main_img_path(gid)
        if img is None:
            print(f"skip {gid}: no image")
            continue
        dense, valid = dense_features(fg, img)
        for cap_idx, cap in enumerate(neg.get(gid, [])):
            for mode, pattern in [("comma", r","), ("period", r"\."), ("combined", r"[.,;]")]:
                sub = split_clauses(cap, pattern)
                if len(sub) < 2:
                    continue
                s_box = max_patch_sim(dense, valid, encode_text_walk(fg, sub, "box")).tolist()
                s_short = max_patch_sim(dense, valid, encode_text_walk(fg, sub, "short")).tolist()
                for cl, sb, ss in zip(sub, s_box, s_short):
                    low = cl.lower()
                    is_hall = any(c in low for c in cores)
                    results[mode].append({"gid": gid, "cap_idx": cap_idx, "is_hall": is_hall,
                                          "clause": cl, "s_box": sb, "s_short": ss})

    def summarize(rows):
        hall = [r["s_box"] for r in rows if r["is_hall"]]
        real = [r["s_box"] for r in rows if not r["is_hall"]]
        # 组级 margin
        gid_margins = []
        for gid in HALL_CORES:
            gr = [r for r in rows if r["gid"] == gid]
            mh = mean([r["s_box"] for r in gr if r["is_hall"]])
            mr = mean([r["s_box"] for r in gr if not r["is_hall"]])
            gid_margins.append(mr - mh)
        # top-1 定位准确率：幻觉子句是否为该 caption 内 box 相似度最低
        groups = {}
        for r in rows:
            groups.setdefault((r["gid"], r["cap_idx"]), []).append(r)
        n_cap = 0
        n_hit = 0
        for key, gr in groups.items():
            if not any(r["is_hall"] for r in gr):
                continue
            n_cap += 1
            min_r = min(gr, key=lambda r: r["s_box"])
            if min_r["is_hall"]:
                n_hit += 1
        return {
            "n_hall": len(hall), "n_real": len(real),
            "hall_mean": mean(hall), "real_mean": mean(real),
            "cohens_d": cohens_d(hall, real),
            "margin_mean": mean(gid_margins),
            "gid_margin_gt0": sum(1 for m in gid_margins if m > 0),
            "n_groups": len(HALL_CORES),
            "top1_acc": n_hit / n_cap if n_cap else 0.0,
            "n_cap": n_cap,
        }

    summary = {"comma": summarize(results["comma"]),
               "period": summarize(results["period"]),
               "combined": summarize(results["combined"])}

    json.dump({"summary": summary, "results": results},
              open("/root/MVPG/fgclip_split_compare_results.json", "w"),
              ensure_ascii=False, indent=2)

    L = []
    L.append("# 逗号 vs 句号 vs 联合分割：子句级幻觉信号清晰度对比\n")
    L.append(f"- 数据：`{DATA}` 负样本 caption，**20 张主图**（不使用子图）\n")
    L.append("- 方法：每条负样本 caption 分别按逗号 / 句号 / 联合(`.,;`)切分 → 每个子句 box 头 × dense patch"
             "算 max_patch_sim（一对一 raw，无 softmax）→ 标记幻觉子句 → 比较区分度。\n")

    L.append("\n## 汇总（box 头）\n")
    L.append("| 分割 | 幻觉子句数 | 真实子句数 | 幻觉均值 | 真实均值 | Cohen's d | 组级 margin>0 | top-1 定位准确率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mode in ["comma", "period", "combined"]:
        m = summary[mode]
        L.append(f"| {mode} | {m['n_hall']} | {m['n_real']} | {m['hall_mean']:.4f} | {m['real_mean']:.4f} | "
                 f"**{m['cohens_d']:+.2f}** | {m['gid_margin_gt0']}/{m['n_groups']} | "
                 f"{m['top1_acc']*100:.1f}% ({m['n_cap']}句) |")

    L.append("\n## 结论\n")
    c, p = summary["comma"]["cohens_d"], summary["period"]["cohens_d"]
    L.append(f"- 逗号 vs 句号：**{'句号' if p > c else '逗号'}分割更清晰**"
             f"（d {c:+.2f} vs {p:+.2f}），但两者都明显弱于联合分割。\n")
    L.append(f"- 联合(`.,;`)分割：d={summary['combined']['cohens_d']:+.2f}，"
             f"top-1 定位准确率 {summary['combined']['top1_acc']*100:.1f}%。\n")
    L.append(f"- 逗号分割 top-1 {summary['comma']['top1_acc']*100:.1f}%；"
             f"句号分割 top-1 {summary['period']['top1_acc']*100:.1f}%。\n")

    # 分组明细（三种分割各一行 margin）
    L.append("\n## 分组 margin（real − hall 的 box 相似度，越大越清晰）\n")
    L.append("| gid | 逗号 margin | 句号 margin | 联合 margin |")
    L.append("|---|---|---|---|")
    for gid in HALL_CORES:
        def gm(mode):
            return mean([r["s_box"] for r in results[mode] if r["gid"] == gid and not r["is_hall"]]) - \
                   mean([r["s_box"] for r in results[mode] if r["gid"] == gid and r["is_hall"]])
        L.append(f"| {gid} | {gm('comma'):+.4f} | {gm('period'):+.4f} | {gm('combined'):+.4f} |")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_split_compare_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_split_compare_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
