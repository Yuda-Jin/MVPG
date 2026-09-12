"""验证 softmax 放大能否提升「幻觉 vs 真实」短语的区分度。

背景：原生 clip 相似度（box 头 × dense patch 的 max_patch_sim）对单物体存在性不敏感，
hall 与 real 短语的绝对余弦差异极小（~0.013，d≈0.7）。CLIP 做零样本分类时用 softmax
把「组内相对差异」指数放大。本脚本对每个 gid 的候选短语（base + real + hall）算相似度后
做 softmax(s/τ)，比较 hall 与 real 短语的 softmax 概率分布，看区分度是否被放大到 d≥1.0。

同时扫温度 τ，并输出原始相似度作对照。
"""

import json
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
TAUS = [1.0, 0.5, 0.2, 0.1, 0.05]

GROUPS = {
    "sa_1000": {"base": "a painted wall",
                "hall": ["a brown dog", "a woven basket", "a brass oil lamp", "a wooden ladder"],
                "real": ["two human figures", "a green jacket", "a striped skirt", "a headdress"]},
    "sa_183": {"base": "an asphalt track",
               "hall": ["a checkered flag"],
               "real": ["a white race car", "orange and blue livery", "green grass"]},
    "sa_2646": {"base": "a plaza",
                "hall": ["a brown dog", "a red bicycle", "a child in a blue jacket", "a wooden bench"],
                "real": ["women walking", "a white building", "green trees"]},
    "sa_3439": {"base": "a city street",
                "hall": ["a fire hydrant", "a brown briefcase", "a wooden bench", "a street lamp"],
                "real": ["a man in a blue blazer", "a yellow auto-rickshaw", "a motorcycle"]},
    "sa_4286": {"base": "a grassy field",
                "hall": ["a stone lion statue", "a wooden rowboat", "a metal flagpole", "a campfire"],
                "real": ["an ancient stone temple", "green mountains"]},
    "sa_5122": {"base": "a wall",
                "hall": ["a black bird", "a brass mailbox", "a potted plant", "a set of keys"],
                "real": ["a tiled street sign", "ceramic tiles", "blue and yellow patterns"]},
    "sa_5954": {"base": "an outdoor plaza",
                "hall": ["a yellow crane", "a red bicycle", "a green bench", "a silver motorcycle"],
                "real": ["a blue booth", "a crowd of people", "a man in a blue jacket"]},
    "sa_6749": {"base": "a city scene",
                "hall": ["a red double-decker bus", "a group of cyclists", "a large fountain", "a yellow taxi"],
                "real": ["a tall cylindrical tower", "a modern building", "a concrete walkway"]},
    "sa_7547": {"base": "a storefront",
                "hall": ["a brown dog", "a red bicycle", "a cat", "a wooden bench"],
                "real": ["parked motorcycles", "a metal shutter", "a group of people"]},
    "sa_8367": {"base": "a town square",
                "hall": ["a wooden bench", "a blue bicycle", "a small fountain", "a group of pigeons"],
                "real": ["a statue", "tiled roofs"]},
    "sa_9166": {"base": "a paved road",
                "hall": ["a red stop sign", "a wooden bench", "a brown dog", "a yellow bicycle"],
                "real": ["a police officer on a motorcycle", "a blue truck", "a silver car"]},
    "sa_9998": {"base": "a marina",
                "hall": ["a wooden rowboat", "a green rowboat", "a white seagull", "a red kayak"],
                "real": ["white sailboats", "wooden docks", "a person with a camera"]},
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
    print(f"fgclip loaded in {time.time() - t0:.1f}s")

    # 每个短语：gid, class(base/real/hall), 相似度 s（box 与 short）
    records = []
    for gid, d in GROUPS.items():
        img = main_img_path(gid)
        if img is None:
            continue
        dense, valid = dense_features(fg, img)
        phrases = [("base", d["base"])] + [("real", e) for e in d["real"]] + \
                  [("hall", e) for e in d["hall"]]
        texts = [p for _, p in phrases]
        s_box = max_patch_sim(dense, valid, encode_text_walk(fg, texts, "box")).tolist()
        s_short = max_patch_sim(dense, valid, encode_text_walk(fg, texts, "short")).tolist()
        for (cls, ph), sb, ss in zip(phrases, s_box, s_short):
            records.append({"gid": gid, "class": cls, "phrase": ph,
                            "s_box": sb, "s_short": ss})

    # 组内 softmax：对每个 gid，把 base+real+hall 的相似度做 softmax(·/τ)
    def group_softmax_probs(key, tau):
        probs = []
        for gid in GROUPS:
            gr = [r for r in records if r["gid"] == gid]
            s = [r[key] for r in gr]
            p = softmax(s, tau)
            for r, pi in zip(gr, p):
                probs.append({"gid": gid, "class": r["class"], "prob": pi, "s": r[key]})
        return probs

    def summarize(rows):
        hall = [r["prob"] for r in rows if r["class"] == "hall"]
        real = [r["prob"] for r in rows if r["class"] == "real"]
        margins = []
        for gid in GROUPS:
            g = [r for r in rows if r["gid"] == gid]
            mh = mean([r["prob"] for r in g if r["class"] == "hall"])
            mr = mean([r["prob"] for r in g if r["class"] == "real"])
            margins.append(mr - mh)
        return {
            "hall_mean": mean(hall), "real_mean": mean(real),
            "cohens_d": cohens_d(hall, real),
            "margin_mean": mean(margins),
            "margin_gt0": sum(1 for m in margins if m > 0),
            "n_groups": len(GROUPS),
        }

    summary = {"n_groups": len(GROUPS)}
    # 原始相似度对照（无 softmax）
    summary["raw_s_box"] = summarize([{"gid": r["gid"], "class": r["class"], "prob": r["s_box"]}
                                      for r in records if r["class"] != "base"])
    summary["raw_s_short"] = summarize([{"gid": r["gid"], "class": r["class"], "prob": r["s_short"]}
                                        for r in records if r["class"] != "base"])
    for tau in TAUS:
        for walk in ["box", "short"]:
            rows = group_softmax_probs(f"s_{walk}", tau)
            summary[f"softmax_{walk}_tau{tau}"] = summarize(rows)

    json.dump({"summary": summary}, open("/root/MVPG/fgclip_softmax_results.json", "w"),
              ensure_ascii=False, indent=2)

    L = []
    L.append("# softmax 放大验证报告\n")
    L.append(f"- 数据：`{DATA}`（同 12 组主图）；模型 FG-CLIP2-base（CPU）\n")
    L.append("- 方法：每组把 base+real+hall 短语对图像算 `max_patch_sim`（box/short 头），"
             "再组内 softmax(s/τ)，比较 hall 与 real 短语的概率分布。\n")

    L.append("\n## 结果（Cohen's d 越大越好，目标 d≥1.0）\n")
    L.append("| 信号 | hall prob | real prob | Cohen's d | 组级 margin>0 |")
    L.append("|---|---|---|---|---|")

    def row(name, m):
        L.append(f"| {name} | {m['hall_mean']:.4f} | {m['real_mean']:.4f} | "
                 f"**{m['cohens_d']:+.2f}** | {m['margin_gt0']}/{m['n_groups']} |")

    row("raw s (box，无softmax)", summary["raw_s_box"])
    row("raw s (short，无softmax)", summary["raw_s_short"])
    for tau in TAUS:
        row(f"softmax box τ={tau}", summary[f"softmax_box_tau{tau}"])
    for tau in TAUS:
        row(f"softmax short τ={tau}", summary[f"softmax_short_tau{tau}"])

    best = max([(summary[k]["cohens_d"], k) for k in summary if k.startswith("softmax")])
    L.append(f"\n## 结论\n")
    L.append(f"- 最强 softmax 组合 **{best[1]}**，Cohen's d={best[0]:+.2f}"
             f"{'，达稳定(d≥1.0)' if best[0] >= 1.0 else '，仍不达稳定(d<1.0)'}。\n")
    L.append(f"- 原始相似度 d_box={summary['raw_s_box']['cohens_d']:+.2f}、"
             f"d_short={summary['raw_s_short']['cohens_d']:+.2f}，作为 softmax 放大前的基线。\n")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_softmax_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_softmax_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
