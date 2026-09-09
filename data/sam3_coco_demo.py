import os
import glob
import time
import torch
import numpy as np
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ---------------- 配置 ----------------
import argparse
_argp = argparse.ArgumentParser(description="SAM3 + 词表逐类提示批量分割（输出 bbox 叠加图）")
_argp.add_argument("--raw-dir", "-i",
                   default=r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\raw_images",
                   help="输入图片目录")
_argp.add_argument("--out-dir", "-o",
                   default=r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\bbox_images",
                   help="输出目录（叠加图保存于此）")
_argp.add_argument("--checkpoint", "-c",
                   default=r"C:\Users\yuda_\.cache\modelscope\models\facebook--sam3\snapshots\master\sam3.pt",
                   help="SAM3 checkpoint（ModelScope 本地权重）")
_argp.add_argument("--tag-list",
                   default=r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\tag_list.txt",
                   help="提示词表路径；缺失时回退到 COCO 80")
_argp.add_argument("--confidence", type=float, default=0.3,
                   help="置信度阈值，调低→召回更多")
_argp.add_argument("--limit", "-l", type=int, default=0,
                   help="限制处理张数（0=全部）")
_args = _argp.parse_args()
CHECKPOINT = _args.checkpoint
RAW_DIR = _args.raw_dir
OUT_DIR = _args.out_dir
CONFIDENCE_THRESHOLD = _args.confidence
IMAGE_LIMIT = _args.limit
TAG_LIST_PATH = _args.tag_list

MIN_AREA_RATIO = 0.01        # 面积占比 <1% → 剔除
MAX_AREA_RATIO = 0.9         # 面积占比 >90% → 纯背景，剔除
NMS_IOU = 0.5                # 去重阈值：不同类别检测到同一实例时只留得分高的


def load_tags(path):
    """读取 tag_list.txt，每行一个标签，去除空行。"""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def dedup(seq):
    """保序去重。"""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# COCO 80 个实例类别（官方顺序）——仅作 tag_list 缺失时的回退
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / max(union, 1)


def process_one(processor, path, out_path, prompts):
    """单张图：逐词表提示分割 → 过滤去重 → 叠加图（含 bbox+标签）保存。返回实例数。"""
    image = Image.open(path).convert("RGB")
    w_img, h_img = image.size
    state = processor.set_image(image)

    detections = []  # (label, mask, box, score)
    for cls in prompts:
        out = processor.set_text_prompt(prompt=cls, state=state)
        n = out["scores"].numel()
        if n == 0:
            continue
        masks = out["masks"].cpu().numpy()[:, 0]  # (N,H,W) bool
        boxes = out["boxes"].cpu().numpy()        # (N,4) xyxy
        scores = out["scores"].cpu().numpy()      # (N,)
        for i in range(n):
            detections.append((cls, masks[i], boxes[i], float(scores[i])))

    # 过滤：面积占比 + 跨类别去重（NMS）
    keep = [
        d for d in detections
        if MIN_AREA_RATIO <= d[1].sum() / (w_img * h_img) <= MAX_AREA_RATIO
    ]
    keep.sort(key=lambda d: d[3], reverse=True)  # 按得分降序做 NMS
    nms = []
    for d in keep:
        if all(_iou(d[1], k[1]) < NMS_IOU for k in nms):
            nms.append(d)
    keep = sorted(nms, key=lambda d: d[1].sum(), reverse=True)  # 按面积降序输出

    # 渲染：原图 + 彩色掩码 + bbox + 类别/得分标签
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.array(image)
    for n, (label, mask, box, score) in enumerate(keep):
        color = (np.array(plt.cm.tab20(n % 20)[:3]) * 255).astype(np.uint8)
        alpha = 0.5
        img[mask] = (alpha * color + (1 - alpha) * img[mask]).astype(np.uint8)
        eroded = np.zeros_like(mask)
        eroded[1:-1, 1:-1] = (
            mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
            & mask[1:-1, :-2] & mask[1:-1, 2:]
        )
        img[mask & ~eroded] = color

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for n, (label, mask, box, score) in enumerate(keep):
        color = tuple((np.array(plt.cm.tab20(n % 20)[:3]) * 255).astype(int))
        x0, y0, x1, y1 = [int(v) for v in box]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, max(y0 - 14, 0)), f"{label} {score:.2f}", fill=color)

    img.save(out_path)
    del state
    torch.cuda.empty_cache()
    return len(keep)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(
        p for p in glob.glob(os.path.join(RAW_DIR, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )
    if IMAGE_LIMIT > 0:
        paths = paths[:IMAGE_LIMIT]
    total = len(paths)
    if total == 0:
        print(f"raw_images 中没有图片: {RAW_DIR}")
        return
    print(f"共 {total} 张图待处理，输出到 {OUT_DIR}")

    # 词表：优先 tag_list.txt（去重），缺失时回退 COCO 80
    # prompts = dedup(load_tags(TAG_LIST_PATH)) or COCO_CLASSES
    prompts = COCO_CLASSES
    print(f"提示词表: {len(prompts)} 个（来源: {TAG_LIST_PATH if os.path.exists(TAG_LIST_PATH) else 'COCO80 回退'}）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    print("加载 SAM3（最强 ckpt），只需一次...")
    model = build_sam3_image_model(
        bpe_path=None,
        device=device,
        eval_mode=True,
        checkpoint_path=CHECKPOINT,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=CONFIDENCE_THRESHOLD)

    t0 = time.time()
    ok = fail = 0
    for idx, path in enumerate(paths):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(OUT_DIR, f"{stem}.png")
        try:
            n_inst = process_one(processor, path, out_path, prompts)
            ok += 1
        except Exception as e:
            fail += 1
            torch.cuda.empty_cache()
            print(f"[{idx+1}/{total}] {name}: 处理失败: {e}")
            continue
        elapsed = (time.time() - t0) / (idx + 1)
        eta = elapsed * (total - idx - 1) / 60
        print(f"[{idx+1}/{total}] {name}: {n_inst} 实例 -> {stem}.png  "
              f"单张 {elapsed:.1f}s 剩余约 {eta:.1f} 分钟")
    print(f"完成: 成功 {ok}，失败 {fail}，总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
