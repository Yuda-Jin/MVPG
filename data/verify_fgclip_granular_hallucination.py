"""验证「短语级 patch grounding」能否提供稳定幻觉信号。

与上一轮（整图级 FG-CLIP 余弦）对比，本脚本验证假设：
  即使整图级 embedding 无法区分「正确场景 + 一个幻觉物体」与「正常描述」，
  短语级 patch grounding 仍可：把 caption 切成实体短语，用
    - `get_text_features(phrase, walk_type="box")` 得到短语文本特征
    - `get_image_dense_feature()` 得到 patch 级视觉特征
  对每个短语计算 `max_patch_sim = max over valid patches of cos(box_feat, dense_patch)`。
  若某短语（如 dog/basket）在图像任何 patch 上相似度都低，则判为幻觉。

实验设计（关键点）：
  对上一轮 12 组，手工从数据中抽取：
    - 「幻觉物体短语」：容易负样本中额外注入、图中不存在的物体（dog/basket/lamp...）
    - 「真实物体短语」：各组负样本共享的、图中确实存在的场景元素（car/building/statue...）
  比较二者的 max_patch_sim 分布，并同时测 short 与 box 两种 walk_type 作对照。

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

# 12 组手工抽取的短语：hall = 幻觉物体（图中不存在），real = 真实场景元素（图中存在）
# 短语统一为简洁名词短语，适合 box 头（局部区域描述）。
PHRASES = {
    "sa_1000": {
        "hall": ["a brown dog", "a woven basket", "a brass oil lamp", "a wooden ladder"],
        "real": ["two human figures", "a wall painting", "a green jacket", "a striped skirt", "a headdress"],
    },
    "sa_183": {
        "hall": ["a checkered flag"],
        "real": ["a white race car", "an asphalt track", "green grass", "orange and blue livery"],
    },
    "sa_2646": {
        "hall": ["a brown dog", "a red bicycle", "a child in a blue jacket", "a wooden bench"],
        "real": ["women walking", "a white building", "green trees", "a paved plaza"],
    },
    "sa_3439": {
        "hall": ["a fire hydrant", "a brown briefcase", "a wooden bench", "a street lamp"],
        "real": ["a man in a blue blazer", "a yellow auto-rickshaw", "a motorcycle"],
    },
    "sa_4286": {
        "hall": ["a stone lion statue", "a wooden rowboat", "a metal flagpole", "a campfire"],
        "real": ["an ancient stone temple", "green mountains", "a grassy field"],
    },
    "sa_5122": {
        "hall": ["a black bird", "a brass mailbox", "a potted plant", "a set of keys"],
        "real": ["a tiled street sign", "ceramic tiles", "blue and yellow patterns"],
    },
    "sa_5954": {
        "hall": ["a yellow crane", "a red bicycle", "a green bench", "a silver motorcycle"],
        "real": ["a blue booth", "a crowd of people", "a man in a blue jacket"],
    },
    "sa_6749": {
        "hall": ["a red double-decker bus", "a group of cyclists", "a large fountain", "a yellow taxi"],
        "real": ["a tall cylindrical tower", "a modern building", "a concrete walkway"],
    },
    "sa_7547": {
        "hall": ["a brown dog", "a red bicycle", "a cat", "a wooden bench"],
        "real": ["parked motorcycles", "a metal shutter", "a group of people"],
    },
    "sa_8367": {
        "hall": ["a wooden bench", "a blue bicycle", "a small fountain", "a group of pigeons"],
        "real": ["a statue", "a cobblestone square", "tiled roofs"],
    },
    "sa_9166": {
        "hall": ["a red stop sign", "a wooden bench", "a brown dog", "a yellow bicycle"],
        "real": ["a police officer on a motorcycle", "a blue truck", "a silver car"],
    },
    "sa_9998": {
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
    """返回单图的 (dense_feat (N,D), valid_mask (N,), num_valid)。"""
    im = Image.open(img_path).convert("RGB")
    pv, mask, shapes = preprocess_images([im])
    pv = pv.to(fg.device)
    mask = mask.to(fg.device)
    shapes = shapes.to(fg.device)
    with torch.no_grad():
        dense = fg.model.get_image_dense_feature(
            pixel_values=pv, pixel_attention_mask=mask, spatial_shapes=shapes
        )
    dense = dense[0]                # (N, D)
    mask = mask[0]                  # (N,)
    return dense, mask.bool(), int(mask.sum().item())


def encode_text_walk(fg, texts, walk_type):
    texts = list(texts)
    tok = fg.tokenizer(texts)
    for k, v in tok.items():
        tok[k] = v.to(fg.device)
    with torch.no_grad():
        feat = fg.model.get_text_features(
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], walk_type=walk_type
        )
    return feat  # (B, D)


def max_patch_sim(dense, valid, text_feat):
    """dense (N,D), valid (N,), text_feat (B,D) -> (B,) max cosine over valid patches."""
    dn = F.normalize(dense, p=2, dim=-1)
    tn = F.normalize(text_feat, p=2, dim=-1)
    sim = dn @ tn.T                    # (N, B)
    sim = sim.masked_fill(~valid.unsqueeze(1), -1e9)
    return sim.max(dim=0).values       # (B,)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    t0 = time.time()
    fg = load_fgclip(FGCLIP_ROOT, "cpu")
    print(f"fgclip loaded in {time.time() - t0:.1f}s")

    rows = []          # 每 (gid, phrase, class, box_sim, short_sim, n_patch)
    for gid, d in PHRASES.items():
        img = main_img_path(gid)
        if img is None:
            print(f"skip {gid}: no image")
            continue
        dense, valid, n_patch = dense_features(fg, img)
        phrases = [("hall", p) for p in d["hall"]] + [("real", p) for p in d["real"]]
        texts = [p for _, p in phrases]
        box_feat = encode_text_walk(fg, texts, "box")
        short_feat = encode_text_walk(fg, texts, "short")
        box_sim = max_patch_sim(dense, valid, box_feat).tolist()
        short_sim = max_patch_sim(dense, valid, short_feat).tolist()
        for (cls, phrase), bs, ss in zip(phrases, box_sim, short_sim):
            rows.append({"gid": gid, "class": cls, "phrase": phrase,
                         "box_sim": bs, "short_sim": ss, "n_patch": n_patch})
        print(f"{gid}: n_patch={n_patch} box_hall={mean([b for (c,_),b in zip(phrases, box_sim) if c=='hall']):.4f} "
              f"box_real={mean([b for (c,_),b in zip(phrases, box_sim) if c=='real']):.4f}")

    # 汇总
    def agg(cls, key):
        xs = [r[key] for r in rows if r["class"] == cls]
        return xs

    hall_box = agg("hall", "box_sim")
    real_box = agg("real", "box_sim")
    hall_short = agg("hall", "short_sim")
    real_short = agg("real", "short_sim")

    # 每组 margin（mean(real)-mean(hall)）
    margins_box, margins_short = [], []
    for gid in PHRASES:
        g = [r for r in rows if r["gid"] == gid]
        hb = mean([r["box_sim"] for r in g if r["class"] == "hall"])
        rb = mean([r["box_sim"] for r in g if r["class"] == "real"])
        hs = mean([r["short_sim"] for r in g if r["class"] == "hall"])
        rs = mean([r["short_sim"] for r in g if r["class"] == "real"])
        margins_box.append(rb - hb)
        margins_short.append(rs - hs)

    def std(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0

    def cohens_d(a, b):
        sa, sb = std(a), std(b)
        sp = ((sa ** 2 + sb ** 2) / 2) ** 0.5
        return (mean(b) - mean(a)) / sp if sp > 0 else 0.0

    # 阈值取两类均值的中间点，统计短语级分类正确率
    thr_box = (mean(hall_box) + mean(real_box)) / 2
    thr_short = (mean(hall_short) + mean(real_short)) / 2
    box_acc = sum(1 for r in rows if (r["box_sim"] >= thr_box) == (r["class"] == "real")) / len(rows)
    short_acc = sum(1 for r in rows if (r["short_sim"] >= thr_short) == (r["class"] == "real")) / len(rows)

    summary = {
        "n_groups": len(PHRASES),
        "n_hall": len(hall_box), "n_real": len(real_box),
        "box_hall_mean": mean(hall_box), "box_real_mean": mean(real_box),
        "box_hall_std": std(hall_box), "box_real_std": std(real_box),
        "box_hall_min": min(hall_box), "box_hall_max": max(hall_box),
        "box_real_min": min(real_box), "box_real_max": max(real_box),
        "box_margin_mean": mean(margins_box), "box_margin_gt0": sum(1 for m in margins_box if m > 0),
        "box_cohens_d": cohens_d(hall_box, real_box),
        "box_acc": box_acc,
        "short_hall_mean": mean(hall_short), "short_real_mean": mean(real_short),
        "short_hall_std": std(hall_short), "short_real_std": std(real_short),
        "short_margin_mean": mean(margins_short), "short_margin_gt0": sum(1 for m in margins_short if m > 0),
        "short_cohens_d": cohens_d(hall_short, real_short),
        "short_acc": short_acc,
    }

    json.dump({"summary": summary, "rows": rows},
              open("/root/MVPG/fgclip_granular_results.json", "w"),
              ensure_ascii=False, indent=2)

    # 写 markdown
    L = []
    L.append("# 短语级 Patch Grounding 幻觉信号验证报告\n")
    L.append(f"- 数据：`{DATA}`（上一轮同 12 组主图）\n")
    L.append(f"- 模型：FG-CLIP2-base（`{FGCLIP_ROOT}`），CPU 推理\n")
    L.append("- 方法：`get_text_features(phrase, walk_type=box/short)` × `get_image_dense_feature()`，"
             "对每个短语取「所有有效 patch 上余弦相似度的最大值」`max_patch_sim`\n")
    L.append("- 对比：**幻觉物体短语**（图中不存在） vs **真实物体短语**（图中存在），"
             "分数越低表示越缺乏图像局部支撑\n")

    # 结论
    L.append("\n## 结论\n")
    box_sep = summary["box_margin_mean"]
    box_ok = summary["box_margin_gt0"]
    box_d = summary["box_cohens_d"]
    short_sep = summary["short_margin_mean"]
    short_ok = summary["short_margin_gt0"]
    short_d = summary["short_cohens_d"]
    L.append(f"- **box 头**：真实短语 max_patch_sim 均值 **{summary['box_real_mean']:.4f}**（σ={summary['box_real_std']:.4f}），"
             f"幻觉短语 **{summary['box_hall_mean']:.4f}**（σ={summary['box_hall_std']:.4f}）；每组 margin(real−hall) 均值 "
             f"**{box_sep:+.4f}**，margin>0 的组 **{box_ok}/{summary['n_groups']}**；效应量 Cohen's d **{box_d:+.2f}**；"
             f"短语级分类正确率 **{summary['box_acc']*100:.1f}%**。\n")
    L.append(f"- **short 头（对照）**：真实短语 **{summary['short_real_mean']:.4f}**（σ={summary['short_real_std']:.4f}），"
             f"幻觉短语 **{summary['short_hall_mean']:.4f}**（σ={summary['short_hall_std']:.4f}）；margin 均值 **{short_sep:+.4f}**，"
             f"margin>0 的组 **{short_ok}/{summary['n_groups']}**；Cohen's d **{short_d:+.2f}**；"
             f"短语级分类正确率 **{summary['short_acc']*100:.1f}%**。\n")
    # 稳定判据：效应量 d>=0.8（大效应）且 margin 几乎全正，才视为稳定信号
    stable_box = box_d >= 0.8 and box_ok == summary["n_groups"]
    stable_short = short_d >= 0.8 and short_ok == summary["n_groups"]
    verdict_box = "**稳定**" if stable_box else "**不稳定/弱**"
    verdict_short = "**稳定**" if stable_short else "**不稳定/弱**"
    L.append(f"- box 头信号判定：{verdict_box}（Cohen's d={box_d:+.2f}，margin 强度 {box_sep:+.4f}）。\n")
    L.append(f"- short 头信号判定：{verdict_short}（Cohen's d={short_d:+.2f}，margin 强度 {short_sep:+.4f}）。\n")
    L.append(f"- **总体判定：短语级 patch grounding（box 头 max_patch_sim）"
             f"{'能' if stable_box else '不能'}提供稳定幻觉信号**。\n")

    # 分组表
    L.append("\n## 分组结果（每组：幻觉 vs 真实 的 box max_patch_sim）\n")
    L.append("| gid | 幻觉短语 box（min/mean/max） | 真实短语 box（min/mean/max） | margin(box) | margin(short) |")
    L.append("|---|---|---|---|---|")
    for gid in PHRASES:
        g = [r for r in rows if r["gid"] == gid]
        hb = [r["box_sim"] for r in g if r["class"] == "hall"]
        rb = [r["box_sim"] for r in g if r["class"] == "real"]
        hs = [r["short_sim"] for r in g if r["class"] == "hall"]
        rs = [r["short_sim"] for r in g if r["class"] == "real"]
        L.append(
            f"| {gid} | {min(hb):.4f} / {mean(hb):.4f} / {max(hb):.4f} "
            f"| {min(rb):.4f} / {mean(rb):.4f} / {max(rb):.4f} "
            f"| {mean(rb)-mean(hb):+.4f} | {mean(rs)-mean(hs):+.4f} |"
        )

    # 逐短语明细（box）
    L.append("\n## 逐短语明细（box max_patch_sim）\n")
    L.append("| gid | 类别 | 短语 | box_sim | short_sim |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['gid']} | {r['class']} | `{r['phrase']}` | {r['box_sim']:.4f} | {r['short_sim']:.4f} |")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_granular_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_granular_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
