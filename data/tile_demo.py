import os
import glob
import time
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForCausalLM

# ---------------- 配置 ----------------
# FG-CLIP2 模型（本地权重目录）：vit-large-patch16（FG-CLIP2-Large）
MODEL_ROOT = r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\models\fgclip2-large-patch16"

# CLI 参数：python tile_demo.py --raw-dir X --out-dir Y --limit 100
import argparse
_argp = argparse.ArgumentParser(description="网格切图 + FG-CLIP2 内部相似度统计")
_argp.add_argument("--raw-dir", "-i", default=r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\raw_images",
                   help="输入图片目录")
_argp.add_argument("--out-dir", "-o", default=r"d:\OneDriveBackup\OneDrive\PHD\work3\proj\MVPG\data\5_images",
                   help="输出目录（子图/直方图/txt 均保存于此）")
_argp.add_argument("--n-split", type=int, default=2,
                   help="每边分割份数 n：n=2 即 2x2=4 等分，n=3 即 3x3=9 等分")
_argp.add_argument("--limit", "-l", type=int, default=0,
                   help="限制处理张数（0=全部）")
_argp.add_argument("--threshold", "-t", type=float, default=0.0,
                   help="相似度过滤阈值：max_sim > threshold 的图片不保存子图（0=不过滤）")
_args = _argp.parse_args()
RAW_DIR = _args.raw_dir
OUT_DIR = _args.out_dir
N_SPLIT = _args.n_split
IMAGE_LIMIT = _args.limit
THRESHOLD = _args.threshold

# 直方图保存路径
HIST_SAVE = os.path.join(OUT_DIR, "max_similarity_hist.png")


def load_encoder(model_root, device):
    """加载 FG-CLIP2。返回 (model, image_processor)。"""
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True)
    model.to(device).eval()
    proc = AutoImageProcessor.from_pretrained(model_root)
    return model, proc


def encode_image(model, proc, img):
    """FG-CLIP2 编码单张图，返回 L2 归一化特征 (D,)。"""
    w, h = img.size
    max_patches = (w // 16) * (h // 16)
    if max_patches > 784:
        n = 1024
    elif max_patches > 576:
        n = 784
    elif max_patches > 256:
        n = 576
    elif max_patches > 128:
        n = 256
    else:
        n = 128
    inp = proc(images=img, max_num_patches=n, return_tensors="pt").to(model.device)
    with torch.no_grad():
        feat = model.get_image_features(**inp)
    return torch.nn.functional.normalize(feat, p=2, dim=-1)[0]


def process_one(path, out_dir, model, proc, n_split, threshold):
    """单张图：n x n 网格切分 + FG-CLIP2 编码全部子图。
    返回 (子图数, 保存目录, 内部相似度的最大值 或 None, 是否保存)。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.join(out_dir, stem)

    img = Image.open(path).convert("RGB")
    w, h = img.size
    cw, ch = w // n_split, h // n_split
    n_tiles = n_split * n_split

    # n x n 网格子图（先不保存，等相似度算完再决定），命名 r{行}c{列}，如 r0c0 / r2c1
    tiles = {}
    if w >= n_split and h >= n_split:
        for r in range(n_split):
            for c in range(n_split):
                box = (c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
                tiles[f"r{r}c{c}"] = img.crop(box)

    # FG-CLIP2 编码全部子图，计算两两相似度（C(n_tiles,2) 对），取最大（值越大内容越接近）
    max_sim = None
    if len(tiles) == n_tiles:
        names = list(tiles.keys())
        feats = [encode_image(model, proc, tiles[nm]) for nm in names]
        S = torch.stack(feats) @ torch.stack(feats).T  # (n_tiles, n_tiles)
        tri = torch.triu_indices(n_tiles, n_tiles, offset=1)
        max_sim = float(S[tri[0], tri[1]].max())
        del feats, S

    # 过滤：max_sim > threshold（且 threshold != 0）的图片不保存
    saved = True
    if threshold != 0 and max_sim is not None and max_sim > threshold:
        saved = False
    else:
        os.makedirs(folder, exist_ok=True)
        # 主图 1/n 下采样，与子图保持同一规格
        img.resize((cw, ch), Image.LANCZOS).save(os.path.join(folder, f"{stem}.png"))
        for nm, crop in tiles.items():
            crop.save(os.path.join(folder, f"{stem}_{nm}.png"))
    return len(tiles), folder, max_sim, saved


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    print(f"加载 FG-CLIP2: {MODEL_ROOT} ...")
    model, proc = load_encoder(MODEL_ROOT, device)
    print("FG-CLIP2 加载完成")

    t0 = time.time()
    max_sims = []  # (img_name, max_sim)
    ok = fail = skipped = 0
    for idx, path in enumerate(paths):
        name = os.path.basename(path)
        try:
            n_quads, folder, max_sim, saved = process_one(path, OUT_DIR, model, proc, N_SPLIT, THRESHOLD)
            ok += 1
            if not saved:
                skipped += 1
        except Exception as e:
            fail += 1
            torch.cuda.empty_cache()
            print(f"[{idx+1}/{total}] {name}: 处理失败: {e}")
            continue
        if max_sim is not None:
            max_sims.append((name, max_sim))
        elapsed = (time.time() - t0) / (idx + 1)
        eta = elapsed * (total - idx - 1) / 60
        sim_str = f"  max_sim={max_sim:.3f}" if max_sim is not None else ""
        saved_str = "  [未保存: 超过阈值]" if not saved else ""
        print(f"[{idx+1}/{total}] {name}: 原图+{n_quads} 子图{sim_str}{saved_str}  "
              f"单张 {elapsed:.1f}s 剩余约 {eta:.1f} 分钟")

    # 汇报直方图
    if max_sims:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = np.array([v for _, v in max_sims])
        n_tiles = N_SPLIT * N_SPLIT
        npairs = n_tiles * (n_tiles - 1) // 2
        plt.figure(figsize=(8, 5))
        plt.hist(vals, bins=30, edgecolor="black")
        plt.xlabel(f"max internal similarity (FG-CLIP2, {npairs} pairwise)")
        plt.ylabel("image count")
        plt.title(f"Distribution of max internal similarity (n={len(vals)})")
        plt.savefig(HIST_SAVE, dpi=120, bbox_inches="tight")
        plt.close()

        print("=" * 60)
        print(f"直方图已保存: {HIST_SAVE}")
        print(f"图片数: {len(vals)}  最小: {vals.min():.3f}  最大: {vals.max():.3f}")
        print(f"均值: {vals.mean():.3f}  中位数: {np.median(vals):.3f}  标准差: {vals.std():.3f}")
        print("分位数: 10%={:.3f}  25%={:.3f}  75%={:.3f}  90%={:.3f}".format(
            *np.percentile(vals, [10, 25, 75, 90])))
        print("=" * 60)

        # 保存名单到同一文件：最相似按 5/10/15/20/25/30% 多档 + 最不相似 top 5%
        top_pcts = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        bottom_pct = 0.05
        top_sorted = sorted(max_sims, key=lambda x: x[1], reverse=True)      # 最相似，降序
        bottom_sorted = sorted(max_sims, key=lambda x: x[1])                  # 最不相似，升序
        top_bottom_save = os.path.join(OUT_DIR, "top_pct_similarity.txt")
        with open(top_bottom_save, "w", encoding="utf-8") as f:
            for p in top_pcts:
                n = max(1, int(round(len(vals) * p)))
                f.write(f"=== most similar (top {p:.0%}, n={n}) ===\n")
                for img_name, sim in top_sorted[:n]:
                    f.write(f"{img_name}\t{sim:.4f}\n")
                f.write("\n")
            n_bottom = max(1, int(round(len(vals) * bottom_pct)))
            f.write(f"=== least similar (top {bottom_pct:.0%}, n={n_bottom}) ===\n")
            for img_name, sim in bottom_sorted[:n_bottom]:
                f.write(f"{img_name}\t{sim:.4f}\n")
        print(f"名单已保存: {top_bottom_save}")
        print("最相似 top 5%（示例）:")
        for img_name, sim in top_sorted[:5]:
            print(f"  {img_name}  max_sim={sim:.4f}")
        print("最不相似 top 5%（示例）:")
        for img_name, sim in bottom_sorted[:5]:
            print(f"  {img_name}  max_sim={sim:.4f}")
    print(f"完成: 成功 {ok}，失败 {fail}，未保存(超过阈值) {skipped}，总耗时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
