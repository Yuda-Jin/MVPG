import torch
import numpy as np
from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ---------------- 配置 ----------------
CHECKPOINT = r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\models\sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"          # SAM1 最强（vit_h > vit_l > vit_b）；显存紧张可换 "vit_l"/"vit_b"
IMAGE_PATH = r"D:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\raw_images\COCO_train2014_000000576702.jpg"
OUT_PATH = r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\sam1_auto_entities.png"

# 显著实例过滤
MIN_AREA_RATIO = 0.02  # 面积占比 <2% → 剔除；主体常被切成小块，阈值太高会把主体整块丢掉
MAX_AREA_RATIO = 0.9    # 面积 > 90% → 纯背景，剔除

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

# 加载 SAM1 vit_h
print("加载 SAM1 vit_h ...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT).to(device)
sam.eval()

# 自动掩码生成器：网格采样提示点，零提示自动找出全图所有实体
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,          # 网格锚点密度（32x32=1024）；调小→更粗颗粒
    points_per_batch=64,
    pred_iou_thresh=0.88,        # 质量阈值，过滤低质量/背景碎片
    stability_score_thresh=0.95,
    stability_score_offset=1.0,
    box_nms_thresh=0.7,
    crop_n_layers=0,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=500,    # opencv 已装，启用后处理去除小碎片/填洞
    output_mode="binary_mask",
)

image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
img_area = image.shape[0] * image.shape[1]
print("正在生成掩码（零提示自动分割）...")
# 只包 no_grad，不要包 autocast（bf16 会导致 numpy 报错）
with torch.inference_mode():
    masks = mask_generator.generate(image)

# 显著实例过滤：面积占比 + 剔除横贯全图的背景带
def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()

h_img, w_img = image.shape[:2]
keep = []
for m in masks:
    ratio = m["area"] / img_area
    if not (MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO):
        continue
    x0, y0, x1, y1 = _bbox(m["segmentation"])
    if (x0 == 0 and x1 == w_img - 1) or (y0 == 0 and y1 == h_img - 1):
        continue  # 横贯画面左→右或上→下的背景带（地面/墙面）
    keep.append(m)
keep.sort(key=lambda m: m["area"], reverse=True)
print(f"共生成 {len(masks)} 个实例，过滤后保留 {len(keep)} 个")

# 合并过细分割：小块高度嵌入大块 → 并入（恢复被切碎的主体）
MERGE_OVERLAP = 0.6

def _bbox_area(bb):
    return (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)

def _bbox_inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x1 - x0 + 1) * (y1 - y0 + 1) if (x1 >= x0 and y1 >= y0) else 0.0

merged, used = [], [False] * len(keep)
for i, m in enumerate(keep):
    if used[i]:
        continue
    seg = m["segmentation"].copy()
    bbi, ai = _bbox(seg), _bbox_area(_bbox(seg))
    for j in range(i + 1, len(keep)):
        if used[j]:
            continue
        aj = _bbox_area(_bbox(keep[j]["segmentation"]))
        if _bbox_inter(bbi, _bbox(keep[j]["segmentation"])) / min(ai, aj) >= MERGE_OVERLAP:
            seg |= keep[j]["segmentation"]
            bbi, ai = _bbox(seg), _bbox_area(_bbox(seg))
            used[j] = True
    merged.append({"segmentation": seg, "area": int(seg.sum()), "predicted_iou": m["predicted_iou"]})

keep = [m for m in merged if MIN_AREA_RATIO <= m["area"] / img_area <= MAX_AREA_RATIO]
keep.sort(key=lambda m: m["area"], reverse=True)
print(f"合并后保留 {len(keep)} 个实例")
for m in keep:
    print(f"  面积占比 {m['area']/img_area*100:5.1f}%  置信度 {m['predicted_iou']:.3f}")

# 为每个掩码单独出图：只保留掩码区域，其余涂黑
import os
vis_dir = os.path.join(os.path.dirname(OUT_PATH), "mask_vis")
os.makedirs(vis_dir, exist_ok=True)
for n, m in enumerate(keep):
    seg = m["segmentation"]
    masked = np.zeros_like(image)
    masked[seg] = image[seg]
    Image.fromarray(masked).save(os.path.join(vis_dir, f"mask_{n:02d}.png"))
print(f"单实例掩码图已保存到: {vis_dir}")

# 渲染叠加图
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

img = image.copy()
for n, m in enumerate(keep):
    seg = m["segmentation"]
    color = (np.array(plt.cm.tab20(n % 20)[:3]) * 255).astype(np.uint8)
    alpha = 0.5
    img[seg] = (alpha * color + (1 - alpha) * img[seg]).astype(np.uint8)
    eroded = np.zeros_like(seg)
    eroded[1:-1, 1:-1] = (
        seg[1:-1, 1:-1] & seg[:-2, 1:-1] & seg[2:, 1:-1]
        & seg[1:-1, :-2] & seg[1:-1, 2:]
    )
    img[seg & ~eroded] = color

Image.fromarray(img).save(OUT_PATH)
print(f"结果已保存: {OUT_PATH}")

plt.figure(figsize=(12, 12))
plt.imshow(img)
plt.title(f"SAM1 Auto Panoptic ({len(keep)} entities)")
plt.axis("off")
plt.show()
