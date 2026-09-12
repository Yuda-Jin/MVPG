#!/usr/bin/env python3
"""
用原模型（LLaVA-1.5-7B / LLaVA-1.5-13B）为 SA1B 数据集的每个数据组生成 caption，
作为「原模型生成的样本」补充进数据（不调用任何 API，纯本地模型推理）。

目录约定（与 run_mvpg.py / tile_demo.py 一致）：
    image-dir/
      neg_captions.json        # 根级新版组级负例池 {gid: [长句负例...]}（只读，不改）
      sa_XXXX/
        sa_XXXX.png            # 主图（文件名 == 文件夹名）
        sa_XXXX_r0c0.png ...   # 网格子图
        meta.json              # 逐组 meta（更新：替换 neg_captions + 写 model_captions）

推理规则：
  - 使用与训练一致的 prompt：DEFAULT_QUESTION = "Describe the image content in detail."
  - 对每组的主图 + 全部子图（网格 crop）逐一推理，含主图。
  - 生成参数与训练 rollout 一致（do_sample、temperature=1.0、top_p=1.0、top_k=0）。

输出：
  1) 独立根文件 model_captions.json：{gid: {model_name: {image_stem: caption}}}
  2) 更新每个子目录 meta.json：
       - neg_captions 用根级新版长句负例替换（旧短句负例不再保留）；
       - 新增 model_captions 字段（原模型推理 caption）。

用法：
    python data/generate_model_captions.py \
        --image-dir /root/autodl-tmp/data/SA1B_5_threshold \
        --model-dir /root/autodl-tmp/cache \
        --models llava-v1.5-7b,llava-v1.5-13b

    冒烟测试（只跑 2 组、只跑 7B）：
        python data/generate_model_captions.py \
            --image-dir /root/autodl-tmp/data/SA1B_5_threshold \
            --models llava-v1.5-7b --limit 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from mvpg_run.run_mvpg import (
    DEFAULT_QUESTION,
    build_query,
    load_groups,
    load_model_and_tokenizer,
    process_image,
    resolve_group_images,
)


def generate_batch(model, tokenizer, image_processor, dtype, device, query0, qlen,
                   paths, max_new_tokens, temperature):
    """对 paths（单批）推理，返回与 paths 等长的 caption 列表。

    与训练 rollout 相同的生成方式：同一 query 模板重复 B 次 + 批量化像素前向，
    do_sample=True、top_p=1.0、top_k=0、temperature（默认 1.0）。
    """
    pixels = torch.stack([process_image(image_processor, p) for p in paths]).to(
        device, dtype=dtype)
    b = pixels.size(0)
    query = query0.unsqueeze(0).repeat(b, 1)
    attn = torch.ones_like(query, dtype=torch.long)

    gen_kwargs = dict(
        inputs=query, images=pixels, attention_mask=attn,
        max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        top_p=1.0, top_k=0,
    )
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        seqs = model.generate(**gen_kwargs)

    responses = seqs[:, qlen:]
    captions = []
    for r in responses:
        r = r[r != tokenizer.pad_token_id]
        captions.append(tokenizer.decode(r, skip_special_tokens=True).strip())
    return captions


def main():
    p = argparse.ArgumentParser(description="原模型生成 caption（7B/13B，不调 API）")
    p.add_argument("--image-dir", "-i", required=True, help="数据目录（含根级 neg_captions.json 与子目录）")
    p.add_argument("--model-dir", default="/root/autodl-tmp/cache", help="模型所在目录")
    p.add_argument("--models", default="llava-v1.5-7b,llava-v1.5-13b",
                   help="逗号分隔的模型子目录名（位于 model-dir 下）")
    p.add_argument("--question", default=DEFAULT_QUESTION, help="生成 prompt（默认与训练一致）")
    p.add_argument("--max-new-tokens", type=int, default=128, help="生成最大 token 数")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="采样温度；<=0 时退化为贪心解码（训练为 1.0）")
    p.add_argument("--batch-size", type=int, default=40, help="一次 generate 的图片数")
    p.add_argument("--save-every", type=int, default=1000,
                   help="每生成多少张图落盘一次 model_captions.json（图片级断点续传）")
    p.add_argument("--limit", "-l", type=int, default=0, help="限制处理的数据组数（0=全部）")
    p.add_argument("--out-json", default="model_captions.json", help="独立输出 json（写到 image-dir 下）")
    p.add_argument("--meta-name", default="meta.json", help="子目录 meta 文件名")
    p.add_argument("--bits", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true", help="与训练一致使用 bf16")
    p.add_argument("--force", action="store_true",
                   help="忽略已有 model_captions.json 结果，重新生成全部（默认跳过已完成的图）")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = Path(args.image_dir)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    # ---- 1) 组索引 + 新版负例（来自根级 neg_captions.json）----
    groups = load_groups(args.image_dir)
    if args.limit > 0:
        groups = groups[:args.limit]
    if not groups:
        raise SystemExit("没有找到数据组（请检查根级 neg_captions.json 或子目录）")
    print(f"数据组数: {len(groups)}")

    # ---- 2) 解析每组的主图 + 全部子图 ----
    group_images = {}  # gid -> [(stem, path), ...]（主图在前，子图按 row/col 排序）
    for g in groups:
        gid = g["gid"]
        main_img, regions = resolve_group_images(args.image_dir, gid)
        items = []
        if main_img:
            items.append((Path(main_img).stem, main_img))
        for r in regions:
            items.append((Path(r).stem, r))
        group_images[gid] = items

    total_images = sum(len(v) for v in group_images.values())
    print(f"每组图像数: 主图 + {len(next(iter(group_images.values()))) - 1} 子图，"
          f"合计 {total_images} 张图")

    # ---- 3) 已有结果（断点续传）----
    out_path = image_dir / args.out_json
    all_caps = {}
    if out_path.is_file() and not args.force:
        try:
            all_caps = json.loads(out_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            all_caps = {}

    # ---- 4) 逐模型推理 ----
    for mname in model_names:
        model_path = Path(args.model_dir) / mname
        if not model_path.is_dir():
            print(f"[skip] 模型目录不存在，跳过: {model_path}")
            continue

        print("=" * 78)
        print(f"加载模型: {mname}  ({model_path})")
        model, tokenizer, image_processor, dtype = load_model_and_tokenizer(
            str(model_path), args.bits, args.fp16, args.bf16, device)
        model.config.use_cache = True
        model.eval()

        # 收集该模型待生成的 (gid, stem, path)
        pending = []
        for g in groups:
            gid = g["gid"]
            per_model = all_caps.setdefault(gid, {}).setdefault(mname, {})
            for stem, path in group_images[gid]:
                if args.force or not per_model.get(stem):
                    pending.append((gid, stem, path))
        print(f"{mname}: 待生成 {len(pending)} / {total_images} 张图")

        # 批量生成（图片级断点续传：每 save_every 张落盘一次）
        query0 = build_query(tokenizer, args.question).to(device)
        qlen = query0.size(0)
        pbar = tqdm(total=len(pending), desc=mname, unit="img", ncols=100)
        since_save = 0
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            caps = generate_batch(model, tokenizer, image_processor, dtype, device,
                                  query0, qlen, [x[2] for x in chunk],
                                  args.max_new_tokens, args.temperature)
            for (gid, stem, _), cap in zip(chunk, caps):
                all_caps[gid][mname][stem] = cap
            pbar.update(len(chunk))
            since_save += len(chunk)
            if since_save >= args.save_every:
                out_path.write_text(json.dumps(all_caps, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                since_save = 0
        pbar.close()
        # 模型收尾：确保最后一小段也落盘
        out_path.write_text(json.dumps(all_caps, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存模型 {mname} 结果 -> {out_path}")

        del model
        torch.cuda.empty_cache()

    # ---- 5) 更新每个 meta.json：替换负例 + 写模型 caption ----
    print("=" * 78)
    updated = 0
    for g in groups:
        gid = g["gid"]
        meta_path = image_dir / gid / args.meta_name
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                meta = {}
        meta["neg_captions"] = g["negs"]  # 根级新版长句负例（替换旧短句）
        meta["model_captions"] = all_caps.get(gid, {})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        updated += 1
    print(f"meta.json 更新完成: {updated} 组")
    print(f"独立输出: {out_path}")


if __name__ == "__main__":
    main()
