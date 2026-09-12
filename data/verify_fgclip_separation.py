"""验证 FG-CLIP 相似度能否区分「容易负样本（幻觉）」与「原模型生成的难负样本」。

在 CPU 上运行（本沙箱无 CUDA）。对若干组图片，对同一张主图计算：
  - 容易负样本 neg_captions[gid]（整图级「图中不存在物体」描述）的相似度
  - 难负样本 model_captions[gid][7b/13b][gid]（原模型对主图的正常描述）的相似度
核心判断：容易负样本的相似度是否系统性地低于难负样本。
输出 markdown 报告到项目根目录。
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/MVPG")

from mvpg_run.fgclip_local import load_fgclip  # noqa: E402

FGCLIP_ROOT = "/root/autodl-tmp/cache/fgclip2-base-patch16"
DATA = "/root/autodl-tmp/data/SA1B_5_threshold"
N_GROUPS = 12
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
M7 = "llava-v1.5-7b"
M13 = "llava-v1.5-13b"


def main_img_path(gid):
    d = Path(DATA) / gid
    if not d.is_dir():
        return None
    for s in IMG_SUFFIXES:
        p = d / (gid + s)
        if p.is_file():
            return str(p)
    return None


def resolve_sub_images(gid):
    d = Path(DATA) / gid
    out = {}
    if not d.is_dir():
        return out
    for p in d.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMG_SUFFIXES:
            continue
        if p.stem == gid:
            continue
        if p.stem.startswith(gid + "_r"):
            out[p.stem] = str(p)
    return out


def main():
    neg = json.load(open(f"{DATA}/neg_captions.json"))
    mc = json.load(open(f"{DATA}/model_captions.json"))

    valid = []
    for gid in sorted(neg):
        g_refs = mc.get(gid)
        if not g_refs:
            continue
        if M7 not in g_refs or M13 not in g_refs:
            continue
        if not g_refs[M7].get(gid) or not g_refs[M13].get(gid):
            continue
        if not neg.get(gid):
            continue
        if main_img_path(gid) is None:
            continue
        valid.append(gid)

    idxs = [int(round(i * (len(valid) - 1) / (N_GROUPS - 1))) for i in range(N_GROUPS)]
    groups = [valid[i] for i in idxs]

    print(f"valid groups: {len(valid)}, selected: {len(groups)}")

    t0 = time.time()
    fg = load_fgclip(FGCLIP_ROOT, "cpu")
    print(f"fgclip loaded in {time.time() - t0:.1f}s")

    rows = []
    sub_rows = []
    for gid in groups:
        img = main_img_path(gid)
        g_refs = mc[gid]
        h7 = g_refs[M7][gid]
        h13 = g_refs[M13][gid]
        easies = neg[gid]

        img_feat = fg.encode_image_paths([img], batch_size=1)
        texts = list(easies) + [h7, h13]
        txt_feat = fg.encode_texts(texts)
        sims = (img_feat @ txt_feat.T)[0].tolist()
        easy_sims = sims[: len(easies)]
        s7 = sims[len(easies)]
        s13 = sims[len(easies) + 1]
        margin = min(s7, s13) - max(easy_sims)

        rows.append({
            "gid": gid, "easy_sims": easy_sims, "s7": s7, "s13": s13,
            "margin": margin, "h7": h7, "h13": h13, "easies": easies,
        })

        # 子图级难负样本（区域 caption 对其区域图的相似度）
        sub_imgs = resolve_sub_images(gid)
        for stem, path in sorted(sub_imgs.items()):
            c7 = g_refs[M7].get(stem)
            c13 = g_refs[M13].get(stem)
            if not c7 or not c13:
                continue
            f = fg.encode_image_paths([path], batch_size=1)
            t = fg.encode_texts([c7, c13])
            s = (f @ t.T)[0].tolist()
            sub_rows.append({"gid": gid, "stem": stem, "s7": s[0], "s13": s[1]})

    # 汇总
    all_easy = [s for r in rows for s in r["easy_sims"]]
    all_s7 = [r["s7"] for r in rows]
    all_s13 = [r["s13"] for r in rows]
    margins = [r["margin"] for r in rows]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "n_groups": len(rows),
        "mean_easy": mean(all_easy),
        "min_easy": min(all_easy),
        "max_easy": max(all_easy),
        "mean_s7": mean(all_s7),
        "mean_s13": mean(all_s13),
        "mean_margin": mean(margins),
        "min_margin": min(margins),
        "n_margin_gt0": sum(1 for m in margins if m > 0),
        "n_margin_le0": sum(1 for m in margins if m <= 0),
        "sub_n": len(sub_rows),
        "sub_mean_s7": mean([r["s7"] for r in sub_rows]),
        "sub_mean_s13": mean([r["s13"] for r in sub_rows]),
    }

    json.dump({"summary": summary, "rows": rows, "sub_rows": sub_rows},
              open("/root/MVPG/fgclip_separation_results.json", "w"),
              ensure_ascii=False, indent=2)

    # 写 markdown
    lines = []
    lines.append("# FG-CLIP 区分「易负样本 vs 难负样本」验证报告\n")
    lines.append(f"- 数据：`{DATA}`（主图 + 子图 + `neg_captions.json` + `model_captions.json`）\n")
    lines.append(f"- 模型：FG-CLIP2-base（`{FGCLIP_ROOT}`），CPU 推理\n")
    lines.append(f"- 抽样：从 {len(valid)} 个有效组中均匀抽 {len(rows)} 组\n")
    lines.append("- 指标：FG-CLIP 图文余弦相似度（L2 归一化后点积），越高越接近图像内容\n")
    lines.append("\n## 结论\n")
    easy_mean = summary["mean_easy"]
    s7m = summary["mean_s7"]
    s13m = summary["mean_s13"]
    verdict = (
        "能区分" if summary["n_margin_gt0"] == len(rows)
        else f"部分区分（{summary['n_margin_gt0']}/{len(rows)} 组 margin>0）"
    )
    lines.append(
        f"- 容易负样本（幻觉）平均相似度 **{easy_mean:.4f}**，"
        f"难负样本 7B **{s7m:.4f}**、13B **{s13m:.4f}**。\n"
    )
    lines.append(f"- 每组「最难的易负样本」相对「最弱的难负样本」的分离 margin："
                 f"均值 {summary['mean_margin']:.4f}，最小 {summary['min_margin']:.4f}；"
                 f"margin>0 的组 **{summary['n_margin_gt0']}/{len(rows)}**。\n")
    lines.append(f"- **判定：{verdict}**。\n")

    lines.append("\n## 主图结果（每组：易负样本相似度列表 vs 7B/13B 难负样本）\n")
    lines.append("| gid | 易负样本 sim（min/mean/max） | 7B 难负 sim | 13B 难负 sim | margin |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        e = r["easy_sims"]
        lines.append(
            f"| {r['gid']} | {min(e):.4f} / {mean(e):.4f} / {max(e):.4f} "
            f"| {r['s7']:.4f} | {r['s13']:.4f} | {r['margin']:+.4f} |"
        )

    lines.append("\n## 汇总统计\n")
    lines.append("| 类别 | 平均相似度 | 最小 | 最大 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 易负样本（幻觉，整图级） | {summary['mean_easy']:.4f} "
                 f"| {summary['min_easy']:.4f} | {summary['max_easy']:.4f} |")
    lines.append(f"| 难负样本 7B（原模型主图 caption） | {summary['mean_s7']:.4f} | - | - |")
    lines.append(f"| 难负样本 13B（原模型主图 caption） | {summary['mean_s13']:.4f} | - | - |")

    lines.append("\n## 子图级难负样本（区域 caption vs 其区域图）\n")
    lines.append(f"共 {summary['sub_n']} 张子图；7B 区域 caption 平均相似度 "
                 f"**{summary['sub_mean_s7']:.4f}**，13B **{summary['sub_mean_s13']:.4f}**。\n")

    lines.append("\n## 抽样示例（前 3 组）\n")
    for r in rows[:3]:
        lines.append(f"\n### {r['gid']}\n")
        lines.append("- 7B 难负样本 caption：`" + r["h7"][:200] + "`\n")
        lines.append("- 13B 难负样本 caption：`" + r["h13"][:200] + "`\n")
        lines.append("- 易负样本（幻觉）示例：`" + r["easies"][0][:200] + "`\n")

    out = "\n".join(lines) + "\n"
    Path("/root/MVPG/fgclip_separation_report.md").write_text(out, encoding="utf-8")
    print("wrote /root/MVPG/fgclip_separation_report.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
