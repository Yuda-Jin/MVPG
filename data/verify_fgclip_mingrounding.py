"""验证「短语级最小支撑 min-grounding」句级信号是否稳定。

核心（组件 A）：把生成的 caption 切成短语，逐短语算 patch 支撑度 g_k，
用 min_k g_k 作为整句的 grounding 分数。正常句每个短语都有支撑 -> min 高；
幻觉句里的幻觉短语支撑低 -> min 被拖低。min 自动定位最不可信的短语。

与上一轮「短语级判别」的区别：这里做的是**句级聚合**——比较
  normal_score = min(g over [base] + real)      # 正常句
  hall_score   = min(g over [base] + real + [hall_j])  # 幻觉句（每次掺一个幻觉短语）
若幻觉短语支撑度稳定低于 base+real 的最小支撑度，则 hall_score < normal_score，
即句级 min 信号能区分幻觉句。

聚合方式同时测 min / mean 两种，文本头同时测 box / short，横向对比。
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
        dense, valid, n_patch = dense_features(fg, img)
        base = d["base"]
        normal_phrases = [base] + d["real"]
        hall_phrases = d["hall"]
        all_phrases = normal_phrases + hall_phrases

        for walk in ["box", "short"]:
            g = max_patch_sim(dense, valid, encode_text_walk(fg, all_phrases, walk)).tolist()
            g_normal = g[:len(normal_phrases)]
            g_hall = g[len(normal_phrases):]
            n_min = min(g_normal)
            n_mean = mean(g_normal)
            for hj, gh in zip(hall_phrases, g_hall):
                hall_min = min(n_min, gh)                      # min 聚合后的幻觉句分数
                hall_mean = mean(g_normal + [gh])              # mean 聚合后的幻觉句分数
                rows.append({
                    "gid": gid, "walk": walk, "entity": hj,
                    "normal_min": n_min, "hall_min": hall_min,
                    "margin_min": n_min - hall_min,            # >=0；>0 表示幻觉句 min 更低
                    "normal_mean": n_mean, "hall_mean": hall_mean,
                    "margin_mean": n_mean - hall_mean,
                    "g_hall": gh, "g_normal_min": n_min,
                })

    def summarize(key_normal, key_hall, key_margin):
        walk_rows = {"box": [], "short": []}
        for w in walk_rows:
            rs = [r for r in rows if r["walk"] == w]
            normals = [r[key_normal] for r in rs]
            halls = [r[key_hall] for r in rs]
            margins = [r[key_margin] for r in rs]
            # 组级 margin（每 gid 平均）
            gid_margins = []
            for gid in GROUPS:
                gr = [r[key_margin] for r in rs if r["gid"] == gid]
                gid_margins.append(mean(gr))
            walk_rows[w] = {
                "normal_mean": mean(normals), "hall_mean": mean(halls),
                "cohens_d": cohens_d(halls, normals),
                "margin_mean": mean(margins),
                "margin_gt0_ratio": sum(1 for m in margins if m > 0) / len(margins),
                "gid_margin_gt0": sum(1 for m in gid_margins if m > 0),
                "n_groups": len(GROUPS), "n_hall": len(margins),
            }
        return walk_rows

    summary = {
        "n_groups": len(GROUPS),
        "min": summarize("normal_min", "hall_min", "margin_min"),
        "mean": summarize("normal_mean", "hall_mean", "margin_mean"),
    }

    json.dump({"summary": summary, "rows": rows},
              open("/root/MVPG/fgclip_mingrounding_results.json", "w"),
              ensure_ascii=False, indent=2)

    L = []
    L.append("# 短语级最小支撑（min-grounding）句级信号验证报告\n")
    L.append(f"- 数据：`{DATA}`（同 12 组主图）\n")
    L.append(f"- 模型：FG-CLIP2-base（`{FGCLIP_ROOT}`），CPU 推理\n")
    L.append("- 方法：正常句 = `[base] + real`，幻觉句 = `[base] + real + [1个hall]`；"
             "逐短语算 patch 支撑度 `g_k = max_patch cos(text(box/short), dense)`，"
             "句级分数取 `min_k g_k`（或 `mean_k g_k` 作对照）。幻觉句应显著更低。\n")

    L.append("\n## 结论\n")
    for agg in ["min", "mean"]:
        for walk in ["box", "short"]:
            m = summary[agg][walk]
            L.append(f"- **{agg} + {walk}**：正常句 {m['normal_mean']:.4f} → 幻觉句 {m['hall_mean']:.4f}；"
                     f"Cohen's d **{m['cohens_d']:+.2f}**；margin 均值 {m['margin_mean']:+.4f}；"
                     f"单短语 margin>0 比例 **{m['margin_gt0_ratio']*100:.1f}%**；"
                     f"组级 margin>0 **{m['gid_margin_gt0']}/{m['n_groups']}**。\n")
    best = max([(summary[a][w]["cohens_d"], a, w) for a in ["min", "mean"] for w in ["box", "short"]])
    L.append(f"- **判定：最强组合为 {best[1]}+{best[2]}，Cohen's d={best[0]:+.2f}"
             f"{'，达到稳定(d≥1.0)' if best[0] >= 1.0 else '，仍不达稳定(d<1.0)'}**。\n")

    L.append("\n## 分组结果（min+box 的句级 min 分数）\n")
    L.append("| gid | 正常句 min | 幻觉句 min（min/mean） | 组级 margin |")
    L.append("|---|---|---|---|")
    for gid in GROUPS:
        rs = [r for r in rows if r["gid"] == gid and r["walk"] == "box"]
        nm = rs[0]["normal_min"]
        hs = [r["hall_min"] for r in rs]
        mg = mean([r["margin_min"] for r in rs])
        L.append(f"| {gid} | {nm:.4f} | {min(hs):.4f} / {mean(hs):.4f} | {mg:+.4f} |")

    L.append("\n## 逐短语明细（patch 支撑度 g_k，box 头）\n")
    L.append("| gid | 类型 | 短语 | g_k(box) | g_k(short) |")
    L.append("|---|---|---|---|---|")
    seen = set()
    for r in rows:
        key = (r["gid"], r["entity"])
        if key in seen:
            continue
        seen.add(key)
        # 找 box 和 short 的 g_hall
        b = next(x for x in rows if x["gid"] == r["gid"] and x["entity"] == r["entity"] and x["walk"] == "box")
        s = next(x for x in rows if x["gid"] == r["gid"] and x["entity"] == r["entity"] and x["walk"] == "short")
        L.append(f"| {r['gid']} | hall | `{r['entity']}` | {b['g_hall']:.4f} | {s['g_hall']:.4f} |")

    out = "\n".join(L) + "\n"
    Path("/root/MVPG/fgclip_mingrounding_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_mingrounding_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
