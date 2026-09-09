#!/usr/bin/env python3
"""
批量调用 VLM，为每个图片子文件夹生成「图中不存在的物体」描述（硬负例），
写入该子文件夹下的 meta.json（字段 neg_caption，供 RL 训练使用）。

目录约定（与 tile_demo.py 输出一致）：
    image-dir/
      001/
        001.png          # 主图（文件名与文件夹名一致）
        001_r0c0.png     # 子图（网格切图，忽略）
        ...
      002/
        002.png
        ...

用法：
    python data/generate_neg_captions.py --image-dir <子文件夹目录> \
        --vlm-base-url https://api.openai.com/v1 \
        --vlm-model gpt-4o-mini --vlm-api-key sk-xxx
"""

import argparse
import base64
import io
import json
import os
import time

import requests
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_main_image(folder):
    """返回文件夹中的主图路径：优先取「文件名 == 文件夹名」的图片。

    若找不到则回退为第一个图片文件。
    """
    imgs = [
        f for f in sorted(os.listdir(folder))
        if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f))
    ]
    if not imgs:
        return None

    folder_stem = os.path.basename(folder.rstrip(os.sep))
    for f in imgs:
        if os.path.splitext(f)[0] == folder_stem:
            return os.path.join(folder, f)
    return os.path.join(folder, imgs[0])


def _parse_neg_list(content):
    """把 VLM 返回解析成负样本字符串列表，尽量取 4 个。"""
    c = content.strip()
    # 去掉 markdown 代码块
    if c.startswith("```"):
        c = c.strip("`").strip()
        if c.lower().startswith("json"):
            c = c[4:].strip()
    # 优先按 JSON 数组解析
    try:
        arr = json.loads(c)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
        # 兼容返回 {"setting": ..., "hallucinations": [...]} 这类对象
        if isinstance(arr, dict):
            for key in ("hallucinations", "neg_captions", "neg_caption",
                        "captions", "items", "descriptions", "result"):
                v = arr.get(key)
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
                if isinstance(v, str):
                    return [s.strip() for s in v.splitlines() if s.strip()]
    except Exception:
        pass
    # 回退：按行切分，去掉编号/项目符号
    lines = []
    for ln in c.splitlines():
        s = ln.strip().lstrip("0123456789.-*#) ").strip().strip('"')
        if s:
            lines.append(s)
    if len(lines) >= 2:
        return lines
    # 再回退：按逗号切分
    parts = [s.strip().strip('"') for s in c.split(",") if s.strip()]
    return parts or ([c] if c else [])


def generate_neg_captions(path, base_url, model_name, api_key, n_neg=4):
    """调用 OpenAI 兼容 VLM，生成图中不存在的 n_neg 个物体描述（硬负例）。"""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)  # 压缩体积，避免超限
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "You are annotating a dataset for vision-language hallucination detection.\n"
        "Step 1 (internal, do NOT write it out): infer the scene — the setting, environment, and the objects that are present.\n"
        f"Step 2: write {n_neg} DISTINCT hallucination descriptions that satisfy ALL of the following:\n"
        "  1. They sound natural and plausible in THIS scene (correct environment, likely co-occurring objects);\n"
        "  2. Each mentions a concrete object/entity that is actually NOT present in the image;\n"
        "  3. They are concise but complete sentences (subject + verb), not bare nouns.\n"
        "Do NOT mention any object that actually appears in the image.\n"
        f"Output ONLY a raw JSON array of exactly {n_neg} strings. "
        "No keys, no scene summary, no explanation, no markdown, nothing else.\n\n"
        "Example 1 (a kitchen with a stove, pots and a sink, but no kettle, fruit or refrigerator):\n"
        '  ["A silver kettle is boiling on the stove.", '
        '"A bowl of fresh fruit sits on the counter.", '
        '"A white refrigerator stands in the corner.", '
        '"A wooden cutting board lies next to the sink."]\n'
        "Example 2 (a city street with cars and pedestrians, but no bicycle, hydrant or traffic light):\n"
        '  ["A person is riding a bicycle along the curb.", '
        '"A red fire hydrant stands by the sidewalk.", '
        '"A traffic light hangs above the intersection.", '
        '"A parked motorcycle leans against the wall."]'
    )
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    if "deepseek" in model_name.lower():
        # DeepSeek V4 默认开启思考模式：思维链全部输出到 reasoning_content，
        # 且会先耗尽 max_tokens，导致最终 JSON 还没写 content 就已截断（content 为空）。
        # 这里显式关闭思考模式，让模型直接输出最终答案。
        payload["thinking"] = {"type": "disabled"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    # content 为空通常是思考模式把 max_tokens 耗尽所致。
    # reasoning_content 是思维链草稿而非答案，只有它本身是合法 JSON 数组时才兜底使用。
    if not content:
        rc = (msg.get("reasoning_content") or "").strip()
        if rc:
            try:
                arr = json.loads(rc)
                if isinstance(arr, list):
                    content = rc
            except Exception:
                pass
    negs = _parse_neg_list(content)
    # 补齐或截断到 n_neg 个
    while len(negs) < n_neg:
        negs.append(negs[0] if negs else "an object not present in this image")
    return negs[:n_neg]


def main():
    p = argparse.ArgumentParser(description="批量 VLM 生成缺席物体描述（写 meta.json）")
    p.add_argument("--image-dir", "-i", required=True, help="包含图片子文件夹的目录")
    p.add_argument("--vlm-base-url", default="https://api.openai.com/v1",
                   help="OpenAI 兼容 API 地址（deepseek/gpt 等）")
    p.add_argument("--vlm-model", default="gpt-4o-mini", help="VLM 模型名")
    p.add_argument("--vlm-api-key", default="",
                   help="API Key（默认读环境变量 OPENAI_API_KEY / DEEPSEEK_API_KEY）")
    p.add_argument("--n-neg", type=int, default=4, help="每张图生成的负样本数量")
    p.add_argument("--meta-name", default="meta.json", help="输出的 json 文件名")
    p.add_argument("--agg-json", default="neg_captions.json",
                   help="汇总所有负样本的 json 文件名（保存到 image-dir 下）")
    p.add_argument("--limit", "-l", type=int, default=0, help="限制处理文件夹数（0=全部）")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                   help="已有 meta.json 时跳过推理（断点续传，默认关闭即重新生成覆盖）；开启用 --resume")
    args = p.parse_args()

    api_key = args.vlm_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    if not api_key:
        raise SystemExit("未提供 API Key（--vlm-api-key 或环境变量 OPENAI_API_KEY / DEEPSEEK_API_KEY）")

    folders = sorted(
        d for d in (os.path.join(args.image_dir, n) for n in os.listdir(args.image_dir))
        if os.path.isdir(d)
    )
    if args.limit > 0:
        folders = folders[:args.limit]
    print(f"共 {len(folders)} 个子文件夹，VLM: {args.vlm_base_url} ({args.vlm_model})")

    ok = fail = skipped = 0
    t0 = time.time()
    for idx, folder in enumerate(folders):
        name = os.path.basename(folder)
        meta_path = os.path.join(folder, args.meta_name)

        # 断点续传（默认开启）：已有 meta.json 则跳过推理；--no-resume 时覆盖重新生成
        if args.resume and os.path.isfile(meta_path):
            skipped += 1
            print(f"[{idx + 1}/{len(folders)}] {name}: meta.json 已存在，跳过")
            continue

        main_img = find_main_image(folder)
        if main_img is None:
            fail += 1
            print(f"[{idx + 1}/{len(folders)}] {name}: 未找到主图，跳过")
            continue

        try:
            negs = generate_neg_captions(main_img, args.vlm_base_url, args.vlm_model, api_key, args.n_neg)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"neg_captions": negs}, f, ensure_ascii=False, indent=2)
            ok += 1
            print(f"[{idx + 1}/{len(folders)}] {name}: {negs}")
        except Exception as e:
            fail += 1
            print(f"[{idx + 1}/{len(folders)}] {name}: 失败: {e}")

    # 汇总：读取所有 meta.json（兼容 neg_captions / neg_caption 字段）
    all_neg = {}  # image_id -> [neg_caption, ...]
    for folder in folders:
        name = os.path.basename(folder)
        meta_path = os.path.join(folder, args.meta_name)
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            negs = data.get("neg_captions") or data.get("neg_caption") or []
            if isinstance(negs, str):
                negs = [negs]
            all_neg[name] = negs
        except Exception as e:
            print(f"[汇总] {name}: 读取 meta.json 失败: {e}")

    agg_path = os.path.join(args.image_dir, args.agg_json)
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(all_neg, f, ensure_ascii=False, indent=2)

    print(f"完成: 成功 {ok}，失败 {fail}，跳过 {skipped}，耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"负样本汇总已保存: {agg_path}（共 {len(all_neg)} 张）")


if __name__ == "__main__":
    main()
