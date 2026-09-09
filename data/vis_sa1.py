"""SA-1B 批量筛选：mask -> 紧贴 bbox -> bbox 面积/宽高比过滤 -> bbox IoU 去重 -> 构图检查 -> 导出叠加框图。

筛选规则：
  1. 每个 mask 解码后导出紧贴 bbox
  2. bbox 面积占比 ∈ (area_min, area_max]（大于 10% 且不超过 60%）
  3. bbox 宽高比 ∈ [1:3, 3:1]
  4. 两两 bbox IoU <= 0.3 去重（按面积降序贪心 NMS）
  5. 去重后 >= 4 个才导出，且只保留面积最大的 4 个 bbox 叠加图
"""
import argparse
import glob
import json
import os
import numpy as np
from PIL import Image, ImageDraw


def rle_decode(size, counts_str):
    """解码 COCO RLE 字符串为 (H, W) 的 0/1 mask。

    实现对齐 pycocotools C 层 maskApi.c 的 rleFrString：
    LEB128 风格变长编码，每字符 6 bit（ASCII 48-111），
    bit5 是续传位、bit4 是符号扩展位；数值间再做与前前个
    run 的差分还原。cnts 交替表示 0/1 像素段（行优先扫描）。
    """
    h, w = size
    cnts = []
    p = 0
    while p < len(counts_str):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(counts_str[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)  # 符号扩展
        if len(cnts) > 2:
            x += cnts[-2]  # 差分还原
        cnts.append(x & 0xFFFFFFFF)
    total = h * w
    mask = np.zeros(total, dtype=np.uint8)
    pos = 0
    for i, ln in enumerate(cnts):
        if pos >= total:
            break
        if i % 2 == 1:
            mask[pos:min(pos + ln, total)] = 1
        pos += ln
    return mask.reshape((h, w), order='F')


def tight_bbox(mask):
    """由 mask (H, W) bool 计算紧贴 bbox，返回 (x, y, w, h)；空 mask 返回 None。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def bbox_iou(a, b):
    """两 bbox (x,y,w,h) 的 IoU。"""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def process_one(json_path, img_path, out_dir, cfg):
    """单张图：解码 masks -> 规则 1/2/3 筛选 -> IoU 去重 -> 构图检查 -> 导出。
    返回 (去重后个数, 是否保留并导出)。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    anns = data["annotations"]
    if not anns:
        return 0, False

    # 规则 1：解码 mask 并计算紧贴 bbox
    items = []  # (mask, tight_bbox, mask_area)
    for a in anns:
        m = rle_decode(a["segmentation"]["size"], a["segmentation"]["counts"]) > 0
        bb = tight_bbox(m)
        if bb is None:
            continue
        items.append((m, bb, int(m.sum())))

    img_area = float(data["image"]["width"]) * data["image"]["height"]

    # 规则 2：bbox 面积占比 ∈ (area_min, area_max]
    keep = [
        it for it in items
        if cfg.area_min < it[1][2] * it[1][3] / img_area <= cfg.area_max
    ]

    # 规则 3：宽高比 ∈ [1/aspect, aspect]
    keep = [
        it for it in keep
        if 1 / cfg.aspect <= it[1][2] / it[1][3] <= cfg.aspect
    ]

    # 规则 4：两两 bbox IoU <= iou 去重（按面积降序贪心 NMS）
    keep.sort(key=lambda it: -it[2])
    nms = []
    for it in keep:
        if all(bbox_iou(it[1], k[1]) <= cfg.iou for k in nms):
            nms.append(it)

    # 规则 6（先判数量）：去重后 < min_keep 个直接淘汰
    if len(nms) < cfg.min_keep:
        return len(nms), False

    # 规则 5：已弃用（top-4 并集覆盖检查被禁用）
    # top4 = nms[:4]
    # h_img, w_img = data["image"]["height"], data["image"]["width"]
    # union = np.zeros((h_img, w_img), dtype=bool)
    # for m, bb, area in top4:
    #     x, y, w, h = bb
    #     union[y:y + h, x:x + w] = True
    # coverage = union.sum() / float(w_img * h_img)
    # if coverage <= cfg.union_min or coverage >= cfg.union_ratio:
    #     return len(nms), False


    # 规则 6：只保留面积最大的 4 个
    n_kept = len(nms)
    nms = nms[:4]

    img = Image.open(img_path).convert("RGB")
    # 仅绘制 bbox 和编号（不渲染 mask）
    colors = [(255, 90, 90), (90, 255, 90), (90, 150, 255), (255, 210, 90)]
    draw = ImageDraw.Draw(img)
    for i, (m, bb, area) in enumerate(nms):
        x, y, w, h = bb
        draw.rectangle([x, y, x + w, y + h], outline=colors[i % len(colors)], width=3)
        draw.text((x, max(y - 14, 0)), str(i + 1), fill=colors[i % len(colors)])
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img.save(os.path.join(out_dir, f"{stem}.png"))
    return n_kept, True


def main():
    p = argparse.ArgumentParser(description="SA-1B 批量筛选并导出 bbox 叠加图")
    p.add_argument("--input-dir", "-i",
                   default=r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\sa1b",
                   help="输入目录（含 *.jpg 与同名 *.json 标注）")
    p.add_argument("--output-dir", "-o",
                   default=r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\sa1b_boxes",
                   help="输出目录（符合条件的叠加框图片）")
    p.add_argument("--area-min", type=float, default=0.10, help="bbox 面积占比下限（默认 10%%）")
    p.add_argument("--area-max", type=float, default=0.60, help="bbox 面积占比上限（默认 60%%）")
    p.add_argument("--aspect", type=float, default=3.0, help="宽高比上限，区间 [1/aspect, aspect]")
    p.add_argument("--iou", type=float, default=0.3, help="两两 bbox IoU 去重阈值")
    p.add_argument("--union-min", type=float, default=0.6, help="(unused) top-4 并集覆盖率下限，已禁用")
    p.add_argument("--union-ratio", type=float, default=0.85,
                   help="(unused) top-4 并集覆盖率上限（保留参数便于后续启用）")
    p.add_argument("--min-keep", type=int, default=4, help="去重后最少保留数，不足则整图丢弃")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    imgs = sorted(
        p for p in glob.glob(os.path.join(args.input_dir, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    total = len(imgs)
    if total == 0:
        print(f"输入目录没有图片: {args.input_dir}")
        return

    print(f"共 {total} 张图，输出到 {args.output_dir}")
    print(f"规则: 框面积 ∈ ({args.area_min:.0%}, {args.area_max:.0%}]  宽高比∈[1:{args.aspect},{args.aspect}:1]  "
          f"框IoU≤{args.iou}  top4框并集 ∈ ({args.union_min:.0%}, {args.union_ratio:.0%})  保留≥{args.min_keep}")

    kept = no_json = failed = 0
    for idx, img_path in enumerate(imgs):
        name = os.path.basename(img_path)
        json_path = os.path.splitext(img_path)[0] + ".json"
        if not os.path.exists(json_path):
            no_json += 1
            print(f"[{idx+1}/{total}] {name}: 缺 json，跳过")
            continue
        try:
            n_box, is_kept = process_one(json_path, img_path, args.output_dir, args)
        except Exception as e:
            failed += 1
            print(f"[{idx+1}/{total}] {name}: 处理失败 {e}")
            continue
        if is_kept:
            kept += 1
        print(f"[{idx+1}/{total}] {name}: 去重后 {n_box} 个 -> {'保留' if is_kept else '丢弃'}")

    # 最终汇报：符合要求的图片数量
    with_ann = total - no_json - failed
    print("=" * 70)
    print(f"最终结果: 输入 {total} 张，有标注 {with_ann} 张，缺 json {no_json} 张，失败 {failed} 张")
    print(f"符合要求并导出: {kept} 张（占 {kept / max(with_ann, 1):.1%}）")
    n_files = len(glob.glob(os.path.join(args.output_dir, "*.png")))
    print(f"输出目录文件数: {n_files}（{args.output_dir}）")
    print("=" * 70)


if __name__ == "__main__":
    main()
