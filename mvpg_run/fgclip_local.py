"""FG-CLIP2 本地加载器（transformers 4.34 兼容，纯官方权重，无额外依赖）。

背景：官方 qihoo360/fg-clip2-base 的 modeling/processor 需要 transformers 4.53+，
而本仓库 LLaVA-1.5 锁 transformers 4.34。二者无法同进程共存，故：
  - `mvpg_run/fgclip2/` 是对官方 `modeling_fgclip2.py` 的最小补丁副本（仅替换
    新版 transformers API 为 4.34 等价 shim，权重结构/前向逻辑完全一致）。
  - 本模块复刻官方 Siglip2ImageProcessorFast 的预处理：
        resize(二分求最大<=max_num_patches 的 16 倍数尺寸, BILINEAR)
        -> /255 -> (x-0.5)/0.5 -> patchify(patch16) -> pad 到 max_num_patches
    文本用 sentencepiece 读 tokenizer.model（Gemma 风格，vocab 256000），
    按 README：lower + max_length=64 截断/填充 + walk_type="short"。
"""

import json
import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根，便于直接运行本模块

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mvpg_run.fgclip2 import modeling_fgclip2 as _M

# walk_type/short 文本最大 token 数（README：short captions -> max_length=64）
TEXT_MAX_LEN = 64
PATCH_SIZE = 16


# -----------------------------------------------------------------------------
# Siglip2 处理器复刻
# -----------------------------------------------------------------------------

def _max_num_patches_for(w, h):
    """README determine_max_value：按 (w//16)*(h//16) 分桶。"""
    n = (w // 16) * (h // 16)
    if n > 784:
        return 1024
    if n > 576:
        return 784
    if n > 256:
        return 576
    if n > 128:
        return 256
    return 128


def _image_size_for_max_num_patches(h, w, max_num_patches, patch_size=16, eps=1e-5):
    """官方 get_image_size_for_max_num_patches（二分找最大可用 scale）。"""

    def scaled(size, scale):
        s = max(patch_size, math.ceil(size * scale / patch_size) * patch_size)
        return int(s)

    lo, hi = eps / 10.0, 100.0
    while hi - lo >= eps:
        scale = (lo + hi) / 2.0
        if ((scaled(h, scale) // patch_size) * (scaled(w, scale) // patch_size)) <= max_num_patches:
            lo = scale
        else:
            hi = scale
    return scaled(h, lo), scaled(w, lo)


def preprocess_images(images, max_num_patches=None):
    """list[PIL.Image] -> (pixel_values, pixel_attention_mask, spatial_shapes)。

    pixel_values: (B, max_patches, C*P*P) float32，[-1,1]（按官方复刻）。
    每张图用自己的 max_num_patches 桶，不足的做单 batch 时在 batch 维同样补齐，
    但调用方通常保证同组同尺寸 -> 同桶。
    """
    pixel_values, masks, shapes = [], [], []
    if max_num_patches is None:
        buckets = [_max_num_patches_for(im.width, im.height) for im in images]
    else:
        buckets = [max_num_patches] * len(images)
    for im, mnp in zip(images, buckets):
        w, h = im.size
        th, tw = _image_size_for_max_num_patches(h, w, mnp, PATCH_SIZE)
        im = im.convert("RGB").resize((tw, th), Image.BILINEAR)
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - 0.5) / 0.5  # (h,w,3)
        ph, pw = th // PATCH_SIZE, tw // PATCH_SIZE
        patches = arr.reshape(ph, PATCH_SIZE, pw, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4)
        patches = patches.reshape(ph * pw, -1)
        cur = ph * pw
        if cur < mnp:
            pad = np.zeros((mnp - cur, patches.shape[1]), dtype=np.float32)
            patches = np.concatenate([patches, pad], axis=0)
        mask = np.ones(mnp, dtype=np.int32)
        mask[cur:] = 0
        pixel_values.append(patches)
        masks.append(mask)
        shapes.append((ph, pw))
    pixel_values = torch.from_numpy(np.stack(pixel_values))          # (B,mnp,768)
    masks = torch.from_numpy(np.stack(masks))                        # (B,mnp)
    shapes = torch.tensor(shapes, dtype=torch.long)                  # (B,2)
    return pixel_values, masks, shapes


# -----------------------------------------------------------------------------
# 文本 tokenizer（Gemma 风格 sentencepiece；vocab=256000，pad=0,eos=1,bos=2,unk=3）
# -----------------------------------------------------------------------------

class _SpTokenizer:
    def __init__(self, model_file):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.pad_id = 0
        self.eos_id = 1

    def __call__(self, texts, max_length=TEXT_MAX_LEN):
        ids_list, attn_list = [], []
        for t in texts:
            ids = self.sp.encode(str(t).lower(), out_type=int)  # README 要求 lower
            # 与官方 tokenizer(...) 默认 add_special_tokens=True 一致：末尾加 eos
            ids = ids[: max_length - 1] + [self.eos_id]
            n = len(ids)
            ids = ids + [self.pad_id] * (max_length - n)
            attn = [1] * n + [0] * (max_length - n)
            ids_list.append(ids)
            attn_list.append(attn)
        return {
            "input_ids": torch.tensor(ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attn_list, dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# 模型加载
# -----------------------------------------------------------------------------

class FGChip:
    """轻量封装：encode_image_paths / encode_texts -> L2 归一化特征。"""

    def __init__(self, root, device):
        root = Path(root)
        cfg = json.loads((root / "config.json").read_text())
        config = _M.Fgclip2Config(
            text_config=cfg["text_config"], vision_config=cfg["vision_config"]
        )
        # 强制 eager attention（本机无需 flash/sdpa 后端）
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"

        model = _M.Fgclip2Model(config)

        from safetensors.torch import load_file
        sd = load_file(str(root / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"FG-CLIP2 权重缺失键: {missing[:10]} ...")
        if unexpected:
            raise RuntimeError(f"FG-CLIP2 意外键: {unexpected[:10]} ...")
        model.to(device).eval()
        self.model = model
        self.device = device
        self.tokenizer = _SpTokenizer(root / "tokenizer.model")

    # ---- 图像编码 ----
    def encode_image_paths(self, paths, batch_size=4):
        feats = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            ims = [Image.open(p).convert("RGB") for p in batch]
            feats.append(self._image_batch(ims))
        return F.normalize(torch.cat(feats, dim=0), p=2, dim=-1)

    def encode_images(self, pil_images):
        return self._image_batch([im.convert("RGB") for im in pil_images])

    def _image_batch(self, ims):
        pixel_values, masks, shapes = preprocess_images(ims)
        pixel_values = pixel_values.to(self.device)
        masks = masks.to(self.device)
        shapes = shapes.to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(
                pixel_values=pixel_values, pixel_attention_mask=masks, spatial_shapes=shapes
            )
        return F.normalize(feat, p=2, dim=-1)

    # ---- 文本编码 ----
    def encode_texts(self, texts, batch_size=8):
        texts = list(texts)
        feats = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i:i + batch_size])
            for k, v in tok.items():
                tok[k] = v.to(self.device)
            with torch.no_grad():
                feat = self.model.get_text_features(
                    input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], walk_type="short"
                )
            feats.append(feat)
        return F.normalize(torch.cat(feats, dim=0), p=2, dim=-1)


_load_lock = threading.Lock()
_cache = {"model": None}


def load_fgclip(root, device):
    """线程安全单例加载。"""
    with _load_lock:
        if _cache["model"] is None:
            _cache["model"] = FGChip(root, device)
        return _cache["model"]


# 兼容旧 _load_fgclip 返回结构（fg 需支持如下键/字段）
def _load_fgclip(root, device):
    return load_fgclip(root, device)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "models/fgclip2-base-patch16")
    fg = load_fgclip(root, "cpu")
    print("FG-CLIP2 loaded:", fg.model.config.model_type)
    d = Path("/root/autodl-tmp/data/SA1B_5_threshold/sa_1000")
    paths = [str(d / f"sa_1000_r0c0.png"), str(d / f"sa_1000_r1c1.png"),
             str(d / f"sa_1000_r0c1.png"), str(d / f"sa_1000_r1c0.png")]
    img = fg.encode_image_paths(paths)
    print("image feat", img.shape)
    txt = fg.encode_texts([
        "A wooden table with small figures.",
        "A dark shadowed corner with a chair.",
        "Textured pattern on the wall.",
        "Bright object near the window.",
        "A golden teapot sits on a wooden table beside the figures.",  # 负例
    ])
    sim = img @ txt.T
    print("sim (rows=images, cols=texts):\n", sim.numpy().round(3))
