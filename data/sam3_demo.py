import torch
import numpy as np
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 本地权重（Meta 官方 CDN 下载，无需 HuggingFace）
CHECKPOINT = r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\models\sam2.1_hiera_large.pt"
MODEL_CFG = "sam2.1_hiera_l.yaml"
IMAGE_PATH = r"D:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\raw_images\COCO_train2014_000000519685.jpg"
OUT_PATH = r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\sam2_auto_entities.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

# 加载模型。apply_postprocessing=False：本机未装 opencv，关闭依赖 cv2 的后处理
sam2 = build_sam2(MODEL_CFG, CHECKPOINT, device=device, apply_postprocessing=False)

# 自动掩码生成器：在图像网格上采样提示点，找出全部实体
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=16,        # 锚点数量：16x16=256 个（32x32=1024 会产生大量背景碎片）
    points_per_batch=64,
    pred_iou_thresh=0.85,      # 提高阈值，滤掉低质量/背景区域
    stability_score_thresh=0.9,
    stability_score_offset=1.0,
    box_nms_thresh=0.7,
    crop_n_layers=0,
    crop_nms_thresh=0.7,
    min_mask_region_area=0,    # >0 需要 opencv，保持 0
    output_mode="binary_mask",
)

image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
print("正在生成实体掩码...")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    masks = mask_generator.generate(image)

# 显著实例过滤：剔除面积过小的背景碎片（按相对图像面积占比）
img_area = image.shape[0] * image.shape[1]
MIN_AREA_RATIO = 0.01   # 掩码面积 < 1% 图像面积 → 视为背景碎片；调大→更少更显著
MAX_AREA_RATIO = 0.9     # 面积 > 90% → 视为纯背景（天空/墙面等），剔除
keep = [
    m for m in masks
    if MIN_AREA_RATIO <= m["area"] / img_area <= MAX_AREA_RATIO
]
keep.sort(key=lambda m: m["area"], reverse=True)  # 按面积降序，最显著的排前面
print(f"共检测到 {len(masks)} 个实例，显著过滤后保留 {len(keep)} 个")
for m in keep:
    print(f"  面积占比 {m['area']/img_area*100:5.1f}%  置信度 {m['predicted_iou']:.3f}")

# 渲染叠加图：每个实体随机色半透明覆盖，边界高亮
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

img = image.copy()
for i, m in enumerate(keep):
    seg = m["segmentation"]
    color = (np.array(plt.cm.tab20(i % 20)[:3]) * 255).astype(np.uint8)
    alpha = 0.5
    img[seg] = (alpha * color + (1 - alpha) * img[seg]).astype(np.uint8)
    # 边界高亮（纯 numpy 4-邻域腐蚀）
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
plt.title(f"SAM2 Salient Instances ({len(keep)} entities)")
plt.axis("off")
plt.show()
