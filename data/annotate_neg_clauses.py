"""为 neg_captions.json 补上「哪条子句带幻觉」的标注（子句级 GT）。

背景：generate_neg_captions.py 让 VLM 在每条 caption 里**恰好注入 1 个图中不存在的物体**，
但生成时只保存了 caption 文本，没有保存被注入的物体名。于是 neg_captions 只有
「caption 级」的阴性标签，缺「子句级」的幻觉位置标注。

本脚本用一个**独立于 FG-CLIP 的 VLM 复核通道**把该物体找回来：
    输入：主图 + 该图的 4 条 caption
    输出：每条 caption 里那个「被描述成存在、但图中其实没有」的名词短语（exact phrase）
再把短语回填到子句上（子句切分与训练/验证脚本一致：`[.,;]` 切分 + 过滤 <3 词碎片），
得到 `label=1` 的幻觉子句，其余子句 `label=0`。

关键性质：标注来自 VLM 看图复核，**与 FG-CLIP 的视觉支持打分完全独立**，
不存在「用打分反推标签、再用标签标定打分」的循环论证。

用法：
    /root/miniconda3/envs/mvpg/bin/python data/annotate_neg_clauses.py \
        --image-dir /root/autodl-tmp/data/SA1B_5_threshold \
        --out /root/MVPG/neg_clause_labels.jsonl \
        --base-url https://ai.stackway.org/v1 --model deepseek-v4-flash-vision-exp \
        --api-key sk-xxx --limit 40 --workers 8

断点续传：`--out` 已存在的 gid 会跳过，可反复追加直到 4848 张跑完。

产物：`neg_clause_labels.jsonl`，每行一张图：
    {"gid":..., "clauses":[["c0","c1",...], ...], "labels":[[0,1,0],...],
     "absent":[{"cap":0,"phrase":"...","clause_idx":1,"match":"substr"}, ...]}
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, "/root/MVPG")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_paired_ref import content_words, split_clauses  # noqa: E402

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PROMPT = (
    "You are auditing a dataset of deliberately corrupted image captions.\n"
    "Each caption below describes THIS image, but exactly ONE concrete object/entity in "
    "each caption was deliberately inserted and is NOT actually present in the image.\n"
    "For EACH caption, identify that one inserted object and quote the EXACT noun phrase "
    "as it appears in the caption text (copy the wording, do not paraphrase).\n"
    "Judge strictly by what is actually visible in the image; ignore style and setting.\n"
    'Output ONLY raw JSON: [{"caption_index":0,"absent_phrase":"..."}, ...] with one entry '
    "per caption, in order. No keys, no explanation, no markdown.\n\n"
)


def main_image_path(image_dir, gid):
    d = Path(image_dir) / gid
    for s in IMG_SUFFIXES:
        p = d / (gid + s)
        if p.is_file():
            return p
    return None


def load_done(out_path):
    done = set()
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                # 失败行不算完成，重跑同一命令时会自动补做
                if "error" in r or "clauses" not in r:
                    continue
                done.add(r["gid"])
    return done


def encode_image_b64(path, max_side=1024):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json_array(content):
    if not content:
        return None
    s = content.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        arr = json.loads(s[i:j + 1])
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def call_vlm(base_url, model, api_key, b64, captions, timeout=120, retries=3):
    prompt = PROMPT + "".join(
        f"caption[{i}]: {c}\n\n" for i, c in enumerate(captions)
    )
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 1024,
    }
    if "deepseek" in model.lower():
        payload["thinking"] = {"type": "disabled"}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = base_url.rstrip("/") + "/chat/completions"
    last = None
    for k in range(retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                msg = r.json()["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if not content and msg.get("reasoning_content"):
                    content = (msg.get("reasoning_content") or "").strip()
                return content, None
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = str(e)
        time.sleep(1.5 * (k + 1))
    return None, last


def normalize(text):
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def locate_clause(phrase, clauses):
    """把 VLM 返回的短语回填到子句下标。

    优先精确子串匹配；失败则退化为内容词 Jaccard 最大者（阈值 0.5）。
    返回 (clause_idx, match_method)。
    """
    pn = normalize(phrase)
    if not pn:
        return -1, "empty"
    for i, c in enumerate(clauses):
        if pn in normalize(c):
            return i, "substr"
    pw = content_words(phrase)
    if pw:
        best, best_j = -1, 0.0
        for i, c in enumerate(clauses):
            cw = content_words(c)
            if not cw:
                continue
            j = len(pw & cw) / len(pw | cw)
            if j > best_j:
                best, best_j = i, j
        if best >= 0 and best_j >= 0.5:
            return best, "jaccard"
    return -1, "fail"


def process_one(image_dir, gid, captions, base_url, model, api_key):
    img = main_image_path(image_dir, gid)
    if img is None:
        return {"gid": gid, "error": "no main image"}
    b64 = encode_image_b64(img)
    content, err = call_vlm(base_url, model, api_key, b64, captions)
    if err is not None:
        return {"gid": gid, "error": err}
    arr = parse_json_array(content)
    if arr is None:
        return {"gid": gid, "error": f"unparsable: {content[:150] if content else 'empty'}"}

    # VLM 可能不按顺序返回，按 caption_index 对齐；缺失的按位置兜底
    phrases = {}
    for pos, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        ci = item.get("caption_index", pos)
        try:
            ci = int(ci)
        except Exception:
            ci = pos
        phrases[ci] = str(item.get("absent_phrase") or "")

    clause_lists = [split_clauses(c) for c in captions]
    labels = [[0] * len(cs) for cs in clause_lists]
    absent = []
    for ci, cs in enumerate(clause_lists):
        phrase = phrases.get(ci, "")
        idx, method = locate_clause(phrase, cs)
        if 0 <= idx < len(cs):
            labels[ci][idx] = 1
        absent.append({"cap": ci, "phrase": phrase, "clause_idx": idx, "match": method})

    return {
        "gid": gid,
        "n_clauses": sum(len(cs) for cs in clause_lists),
        "clauses": clause_lists,
        "labels": labels,
        "absent": absent,
    }


def main():
    ap = argparse.ArgumentParser(description="VLM 复核找回 neg_captions 的幻觉子句标注")
    ap.add_argument("--image-dir", default="/root/autodl-tmp/data/SA1B_5_threshold")
    ap.add_argument("--captions-json", default="neg_captions.json",
                    help="image-dir 下的汇总 json（{gid: [caption,...]}）")
    ap.add_argument("--out", default="/root/MVPG/neg_clause_labels.jsonl")
    ap.add_argument("--base-url", default=os.environ.get("VLM_BASE_URL", "https://ai.stackway.org/v1"))
    ap.add_argument("--model", default=os.environ.get("VLM_MODEL", "deepseek-v4-flash-vision-exp"))
    ap.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("未提供 API Key（--api-key 或环境变量 VLM_API_KEY）")

    with open(os.path.join(args.image_dir, args.captions_json), encoding="utf-8") as f:
        caps = json.load(f)
    caps = {g: c for g, c in caps.items() if isinstance(c, list) and c}

    done = load_done(args.out)
    todo = [g for g in caps if g not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"总计 {len(caps)} 张，已完成 {len(done)} 张，本次待处理 {len(todo)} 张")
    if not todo:
        return

    lock = threading.Lock()
    ok = fail = 0
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as fo, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, args.image_dir, g, caps[g],
                          args.base_url, args.model, args.api_key): g for g in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
            if "error" in rec:
                fail += 1
                if fail <= 5:
                    print(f"  [{rec['gid']}] 失败: {rec['error']}")
            else:
                ok += 1
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"[{n}/{len(todo)}] ok={ok} fail={fail} 用时 {el:.0f}s "
                      f"({n / max(el, 1e-9):.1f} 张/s)")
    print(f"完成: ok={ok} fail={fail} -> {args.out}")
    print("提示：失败项未写入结果，重跑同一命令会自动补做。")


if __name__ == "__main__":
    main()
