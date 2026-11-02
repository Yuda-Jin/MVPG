"""Source-image indexing for the retained grouped caption diagnostics.

These helpers preserve the existing group-index, negative-pool and image-order
semantics without importing the retired MVPG training program or a model stack.
"""

import json
import re
from pathlib import Path


# Preserve the prompt used by the existing diagnostic model_captions.json files.
DEFAULT_QUESTION = "Describe the image content in detail."
_IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _clean_negs(negs):
    if isinstance(negs, str):
        negs = [negs]
    return [str(n).strip() for n in (negs or []) if n and str(n).strip()]


def _read_root_neg_pool(image_dir):
    """读取 image-dir 根级总 json（当前数据为 neg_captions.json）。找不到返回 None。"""
    image_dir = Path(image_dir)
    for name in ("neg_captions.json", "groups.json"):
        fp = image_dir / name
        if not fp.is_file():
            continue
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(raw, dict):
            pool = {str(k): _clean_negs(v) for k, v in raw.items()}
            # 只保留确实存在目录的组，避免索引与磁盘不一致
            return {gid: negs for gid, negs in pool.items() if (image_dir / gid).is_dir()}
    return None


_group_resolve_cache = {}


def resolve_group_images(image_dir, gid):
    """解析单个组目录 -> (main_img, regions)（与旧 scan_group_images 同语义，
    但仅针对被采样到的组，不做全量目录扫描，也不读 meta.json）。"""
    key = (str(image_dir), gid)
    if key in _group_resolve_cache:
        return _group_resolve_cache[key]
    gd = Path(image_dir) / gid
    main_img, tiles, others = None, {}, []
    if gd.is_dir():
        for p in sorted(gd.iterdir()):
            if not p.is_file() or p.suffix.lower() not in _IMG_SUFFIXES:
                continue
            if p.stem == gid:
                main_img = str(p)
                continue
            m = re.fullmatch(rf"{re.escape(gid)}_r(\d+)c(\d+)", p.stem, re.IGNORECASE)
            if m:
                tiles[(int(m.group(1)), int(m.group(2)))] = str(p)
            else:
                others.append(str(p))
    regions = [tiles[k] for k in sorted(tiles)] if tiles else others
    out = (main_img, regions)
    _group_resolve_cache[key] = out
    return out


def load_groups(image_dir):
    """返回组 dict 列表：{gid, negs}（main_img/regions 用到时再 resolve）。

    优先：image-dir 根级总 json 的 key（当前数据 neg_captions.json 含全部组，
    与子目录一一对应）——启动只做一次 json.load，负例直接用根 json。
    回退：无总 json 时按子目录名构造（负例为空，兼容无索引数据）。
    """
    pool = _read_root_neg_pool(image_dir)
    if pool:
        return [{"gid": gid, "negs": pool[gid]} for gid in sorted(pool)]
    return [{"gid": sub.name, "negs": []}
            for sub in sorted(Path(image_dir).iterdir()) if sub.is_dir()]
