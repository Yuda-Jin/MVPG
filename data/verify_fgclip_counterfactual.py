"""验证「反事实幻觉敏感度」信号是否稳定。

核心思想（利用 FG-CLIP 的对比/对齐能力，而非绝对相似度）：
  对 caption 中某个实体短语 e，构造一对句子：
    C_present = f"a photo of {base} with {e}"   # 含该实体
    C_absent  = f"a photo of {base}"            # 不含该实体（同一场景上下文）
  差分 Δ(e) = align(img, C_present) − align(img, C_absent)
    - 真实实体：加上它对齐分上升 -> Δ > 0
    - 幻觉实体：加上它对对齐几乎无贡献 -> Δ ≈ 0（甚至为负）
  这比「绝对相似度」更稳，因为抵消了所有短语分数的共模偏移，只度量「该实体
  是否对图文对齐有增量贡献」。

align 同时测三种：
  - patch-short：dense patch × 整句 short 文本特征，max_patch_sim
  - patch-box  ：dense patch × 整句 box  文本特征，max_patch_sim
  - global-short：整图特征 × 整句 short 文本特征，余弦

数据沿用上一轮 12 组，手工给出每组的 base（真实背景场景）与 hall/real 实体短语。
输出 markdown 到项目根目录。
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

# 每组：base（真实、笼统背景，尽量不含具体待测实体）+ hall（幻觉物体）+ real（真实物体）
GROUPS = {
    "sa_1000": {
        "base": "a painted wall",
        "hall": ["a brown dog", "a woven basket", "a brass oil lamp", "a wooden ladder"],
        "real": ["two human figures", "a green jacket", "a striped skirt", "a headdress"],
    },
    "sa_183": {
        "base": "an asphalt track",
        "hall": ["a checkered flag"],
        "real": ["a white race car", "orange and blue livery", "green grass"],
    },
    "sa_2646": {
        "base": "a plaza",
        "hall": ["a brown dog", "a red bicycle", "a child in a blue jacket", "a wooden bench"],
        "real": ["women walking", "a white building", "green trees"],
    },
    "sa_3439": {
        "base": "a city street",
        "hall": ["a fire hydrant", "a brown briefcase", "a wooden bench", "a street lamp"],
        "real": ["a man in a blue blazer", "a yellow auto-rickshaw", "a motorcycle"],
    },
    "sa_4286": {
        "base": "a grassy field",
        "hall": ["a stone lion statue", "a wooden rowboat", "a metal flagpole", "a campfire"],
        "real": ["an ancient stone temple", "green mountains"],
    },
    "sa_5122": {
        "base": "a wall",
        "hall": ["a black bird", "a brass mailbox", "a potted plant", "a set of keys"],
        "real": ["a tiled street sign", "ceramic tiles", "blue and yellow patterns"],
    },
    "sa_5954": {
        "base": "an outdoor plaza",
        "hall": ["a yellow crane", "a red bicycle", "a green bench", "a silver motorcycle"],
        "real": ["a blue booth", "a crowd of people", "a man in a blue jacket"],
    },
    "sa_6749": {
        "base": "a city scene",
        "hall": ["a red double-decker bus", "a group of cyclists", "a large fountain", "a yellow taxi"],
        "real": ["a tall cylindrical tower", "a modern building", "a concrete walkway"],
    },
    "sa_7547": {
        "base": "a storefront",
        "hall": ["a brown dog", "a red bicycle", "a cat", "a wooden bench"],
        "real": ["parked motorcycles", "a metal shutter", "a group of people"],
    },
    "sa_8367": {
        "base": "a town square",
        "hall": ["a wooden bench", "a blue bicycle", "a small fountain", "a group of pigeons"],
        "real": ["a statue", "tiled roofs"],
    },
    "sa_9166": {
        "base": "a paved road",
        "hall": ["a red stop sign", "a wooden bench", "a brown dog", "a yellow bicycle"],
        "real": ["a police officer on a motorcycle", "a blue truck", "a silver car"],
    },
    "sa_9998": {
        "base": "a marina",
        "hall": ["a wooden rowboat", "a green rowboat", "a white seagull", "a red kayak"],
        "real": ["white sailboats", "wooden docks", "a person with a camera"],
    },
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
    return dense[0], mask[0].bool(), int(mask.sum().item())


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


def global_sim(fg, img_path, texts):
    img_feat = fg.encode_image_paths([img_path], batch_size=1)   # (1,D) normalized
    txt_feat = fg.encode_texts(texts)                            # (B,D) normalized, short
    return (img_feat @ txt_feat.T)[0]


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

    rows = []
    for gid, d in GROUPS.items():
        img = main_img_path(gid)
        if img is None:
            print(f"skip {gid}: no image")
            continue
        base = d["base"]
        dense, valid, n_patch = dense_features(fg, img)
        gfeat = fg.encode_image_paths([img], batch_size=1)

        phrases = [("hall", e) for e in d["hall"]] + [("real", e) for e in d["real"]]
        present = [f"a photo of {base} with {e}" for _, e in phrases]
        absent = [f"a photo of {base}"] * len(phrases)

        # patch-short / patch-box
        sp = max_patch_sim(dense, valid, encode_text_walk(fg, present, "short"))
        sa = max_patch_sim(dense, valid, encode_text_walk(fg, absent, "short"))
        bp = max_patch_sim(dense, valid, encode_text_walk(fg, present, "box"))
        ba = max_patch_sim(dense, valid, encode_text_walk(fg, absent, "box"))

        # global-short
        gp = global_sim(fg, img, present)
        ga = global_sim(fg, img, absent)

        for (cls, e), i in zip(phrases, range(len(phrases))):
            rows.append({
                "gid": gid, "class": cls, "entity": e,
                "d_patch_short": (sp[i] - sa[i]).item(),
                "d_patch_box": (bp[i] - ba[i]).item(),
                "d_global_short": (gp[i] - ga[i]).item(),
                "n_patch": n_patch,
            })
        hb = [r["d_patch_short"] for r in rows if r["gid"] == gid and r["class"] == "hall"]
        rb = [r["d_patch_short"] for r in rows if r["gid"] == gid and r["class"] == "real"]
        print(f"{gid}: n_patch={n_patch} d_patch_short hall={mean(hb):+.4f} real={mean(rb):+.4f}")

    # 汇总三类 align 度量
    def summarize(key):
        hall = [r[key] for r in rows if r["class"] == "hall"]
        real = [r[key] for r in rows if r["class"] == "real"]
        margins = []
        for gid in GROUPS:
            g = [r for r in rows if r["gid"] == gid]
            h = mean([r[key] for r in g if r["class"] == "hall"])
            rr = mean([r[key] for r in g if r["class"] == "real"])
            margins.append(rr - h)
        thr = (mean(hall) + mean(real)) / 2
        acc = sum(1 for r in rows if (r[key] >= thr) == (r["class"] == "real")) / len(rows)
        return {
            "hall_mean": mean(hall), "real_mean": mean(real),
            "hall_std": std(hall), "real_std": std(real),
            "cohens_d": cohens_d(hall, real),
            "margin_mean": mean(margins), "margin_gt0": sum(1 for m in margins if m > 0),
            "acc": acc, "n_hall": len(hall), "n_real": len(real),
        }

    summary = {
        "n_groups": len(GROUPS),
        "patch_short": summarize("d_patch_short"),
        "patch_box": summarize("d_patch_box"),
        "global_short": summarize("d_global_short"),
    }

    json.dump({"summary": summary, "rows": rows},
              open("/root/MVPG/fgclip_counterfactual_results.json", "w"),
              ensure_ascii=False, indent=2)

    def metric_block(name, m):
        L = []
        L.append(f"### {name}\n")
        L.append(f"- 真实实体 Δ 均值 **{m['real_mean']:+.4f}**（σ={m['real_std']:.4f}），"
                 f"幻觉实体 Δ 均值 **{m['hall_mean']:+.4f}**（σ={m['hall_std']:.4f}）。\n")
        L.append(f"- Cohen's d **{m['cohens_d']:+.2f}**；margin(real−hall) 均值 **{m['margin_mean']:+.4f}**，"
                 f"margin>0 的组 **{m['margin_gt0']}/{summary['n_groups']}**；"
                 f"短语级分类正确率 **{m['acc']*100:.1f}%**。\n")
        return L

    L = []
    L.append("# 反事实幻觉敏感度（Δ）信号验证报告\n")
    L.append(f"- 数据：`{DATA}`（上一轮同 12 组主图）\n")
    L.append(f"- 模型：FG-CLIP2-base（`{FGCLIP_ROOT}`），CPU 推理\n")
    L.append("- 方法：对每个实体短语构造 `a photo of {base} with {e}` 与 `a photo of {base}`，"
             "算对齐分差 `Δ = align(present) − align(absent)`。真实实体 Δ>0（贡献对齐），幻觉实体 Δ≈0（无贡献）。\n")
    L.append("- align 度量：patch-short / patch-box（dense patch 的 max_patch_sim）、global-short（整图余弦）。\n")

    L.append("\n## 结论\n")
    stable = summary["patch_short"]["cohens_d"] >= 1.0 and summary["patch_short"]["acc"] >= 0.8
    L.append(f"- **判定：patch-short 反事实差分信号"
             f"{'能' if stable else '尚不能'}提供稳定幻觉信号**"
             f"（Cohen's d={summary['patch_short']['cohens_d']:+.2f}，"
             f"准确率={summary['patch_short']['acc']*100:.1f}%）。\n")
    L.append("\n## 三种 align 度量对比\n")
    for name in ["patch_short", "patch_box", "global_short"]:
        L += metric_block(name, summary[name])

    L.append("\n## 分组结果（patch-short 的 Δ：hall vs real）\n")
    L.append("| gid | 幻觉 Δ（min/mean/max） | 真实 Δ（min/mean/max） | margin(real−hall) |")
    L.append("|---|---|---|---|")
    for gid in GROUPS:
        g = [r for r in rows if r["gid"] == gid]
        hb = [r["d_patch_short"] for r in g if r["class"] == "hall"]
        rb = [r["d_patch_short"] for r in g if r["class"] == "real"]
        L.append(f"| {gid} | {min(hb):+.4f} / {mean(hb):+.4f} / {max(hb):+.4f} "
                 f"| {min(rb):+.4f} / {mean(rb):+.4f} / {max(rb):+.4f} "
                 f"| {mean(rb)-mean(hb):+.4f} |")

    L.append("\n## 逐实体明细（Δ = align(present) − align(absent)）\n")
    L.append("| gid | 类别 | 实体 | Δ_patch_short | Δ_patch_box | Δ_global_short |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['gid']} | {r['class']} | `{r['entity']}` | "
                 f"{r['d_patch_short']:+.4f} | {r['d_patch_box']:+.4f} | {r['d_global_short']:+.4f} |")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_counterfactual_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_counterfactual_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
