#!/usr/bin/env python3
"""
MVPG RL 训练（自研，不依赖 verl）。

方法对应 paper/story.md：
  §3.3  策略梯度：REINFORCE + leave-one-out 基线（组 = 同图多区域视角），
        梯度只回传投影层 mm_projector，视觉编码器与 LLM 冻结。
  §3.2  区域对比接地奖励 RCGR：
        r_i = alpha * [sim(a_i, R_i) - max_{j!=i} sim(c_j, R_i)]
            + beta  * [sim(a_i, R_i) - sim(b, R_i)]
            - eta   * len(a_i)
        sim = FG-CLIP 文本/区域特征余弦相似度。

数据格式（image-dir 下每个子目录 = 一张原图的一组区域视角）：
    image-dir/
      001/
        region_0.png   # mask-only 区域图 M_0（只保留区域 R_0，其余涂黑）
        region_1.png
        ...
        meta.json      # {"neg_caption": "a photo of <缺席难负例>"}

用法：
    python mvpg_run/run_mvpg.py --image-dir <区域图目录> --model-name <LLaVA路径> \
        --output-dir <checkpoint目录> --steps 1000 --lr 1e-4
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import transformers

from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from utils.constants import DEFAULT_IMAGE_TOKEN
from utils.common_utils import compute_logprobs

# ---------------- RCGR 奖励系数（story.md §3.2） ----------------
ALPHA = 1.0   # 跨区域竞争项
BETA = 0.5    # 负样本基线项
ETA = 0.0     # 长度惩罚（先取 0，后续消融）
FGCLIP_ROOT = str(ROOT / "models" / "fgclip2-base-patch16")

# 生成 prompt（对每个 mask-only 区域图提问）
DEFAULT_QUESTION = "Describe the highlighted object or region in this image."


# =============================================================================
# FG-CLIP2：图像/文本编码（用于奖励）
# =============================================================================

_FGCLIP = {"model": None, "proc": None, "tokenizer": None}


def _load_fgclip(model_root, device):
    if _FGCLIP["model"] is not None:
        return _FGCLIP
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True)
    model.to(device).eval()
    proc = AutoImageProcessor.from_pretrained(model_root)
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    _FGCLIP.update(model=model, proc=proc, tokenizer=tokenizer)
    return _FGCLIP


def _max_num_patches(w, h):
    n = (w // 16) * (h // 16)
    return 1024 if n > 784 else 784 if n > 576 else 576 if n > 256 else 256 if n > 128 else 128


def _encode_image(fg, img_path, device):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    inp = fg["proc"](images=img, max_num_patches=_max_num_patches(w, h),
                     return_tensors="pt").to(device)
    with torch.no_grad():
        feat = fg["model"].get_image_features(**inp)
    return F.normalize(feat, p=2, dim=-1)[0]


def _encode_text(fg, text, device):
    tok = fg["tokenizer"](text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feat = fg["model"].get_text_features(**tok)
    return F.normalize(feat, p=2, dim=-1)[0]


def _cos(a, b):
    return float((a * b).sum().item())


def compute_group_rewards(fg, region_paths, captions, neg_caption, device):
    """组级 RCGR 奖励。region_paths/captions 一一对应（同图 K 个区域）。
    返回 list[float]（长度 K）。

    - sim(a_i, R_i)：caption_i 文本 vs 区域 i 视觉
    - max_{j!=i} sim(c_j, R_i)：同图其余区域描述的竞争项
    - sim(b, R_i)：缺席难负例基线
    """
    K = len(region_paths)
    img_feats = [_encode_image(fg, p, device) for p in region_paths]
    txt_feats = [_encode_text(fg, c, device) for c in captions]

    rewards = []
    for i in range(K):
        sim_self = _cos(txt_feats[i], img_feats[i])

        # 跨区域竞争项：其余区域描述 vs 本区域视觉，取最大
        sim_others = max(_cos(txt_feats[j], img_feats[i]) for j in range(K) if j != i)

        # 负样本基线
        if neg_caption:
            neg_feat = _encode_text(fg, neg_caption, device)
            sim_neg = _cos(neg_feat, img_feats[i])
        else:
            sim_neg = 0.0

        r = ALPHA * (sim_self - sim_others) + BETA * (sim_self - sim_neg) - ETA * len(captions[i])
        rewards.append(r)
    return rewards


# =============================================================================
# 模型加载（复用 opa_train.py 验证过的接口），只训 mm_projector
# =============================================================================


def load_model_and_tokenizer(model_name, bits, fp16, bf16, device):
    compute_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)

    bnb_kwargs = {}
    if bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_kwargs = dict(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["mm_projector", "lm_head"],
            ),
        )

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name,
        use_flash_attention_2=False,
        torch_dtype=compute_dtype,
        device_map={"": device} if bits in [4, 8] else None,
        trust_remote_code=True,
        **bnb_kwargs,
    )
    if bits not in [4, 8]:
        model = model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # 视觉塔 + 图像处理器
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=compute_dtype, device=device)
    image_processor = vision_tower.image_processor

    # 初始化视觉 tokenizer（把 <image> 映射为 IMAGE_TOKEN_INDEX）
    from argparse import Namespace
    model_args = Namespace(mm_use_im_start_end=False, mm_use_im_patch_token=True)
    model.config.mm_use_im_start_end = False
    model.config.mm_use_im_patch_token = True
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 只训 mm_projector
    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    return model, tokenizer, image_processor, compute_dtype


# =============================================================================
# 数据：扫描 image-dir，每个子目录 = 一组区域
# =============================================================================


def scan_groups(image_dir):
    """返回 [(group_id, [region_path...], meta)]，meta 含 neg_caption。"""
    groups = []
    for sub in sorted(Path(image_dir).iterdir()):
        if not sub.is_dir():
            continue
        region_paths = sorted(
            p for p in glob.glob(os.path.join(str(sub), "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )
        if len(region_paths) < 2:  # leave-one-out 需要至少 2 个区域
            continue
        meta = {}
        meta_path = os.path.join(str(sub), "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        groups.append((sub.name, region_paths, meta))
    return groups


# =============================================================================
# prompt 构造（LLaVA v1 对话模板）
# =============================================================================


def build_query(tokenizer, question):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return tokenizer_image_token(prompt, tokenizer, return_tensors="pt")


def process_image(image_processor, img_path, image_aspect_ratio="pad"):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    if image_aspect_ratio == "pad":
        def expand2square(pil_img, bg):
            w, h = pil_img.size
            if w == h:
                return pil_img
            if w > h:
                r = Image.new(pil_img.mode, (w, w), bg)
                r.paste(pil_img, (0, (w - h) // 2))
                return r
            r = Image.new(pil_img.mode, (h, h), bg)
            r.paste(pil_img, ((h - w) // 2, 0))
            return r
        img = expand2square(img, tuple(int(x * 255) for x in image_processor.image_mean))
    return image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]


# =============================================================================
# 训练循环
# =============================================================================


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("MVPG RL 训练（自研 REINFORCE + leave-one-out，只训投影层）")
    print(f"  model  : {args.model_name}")
    print(f"  data   : {args.image_dir}")
    print(f"  device : {device}")
    print("=" * 78)

    groups = scan_groups(args.image_dir)
    print(f"共 {len(groups)} 个区域组（每组 >=2 个区域）")
    if len(groups) == 0:
        raise SystemExit("没有找到有效数据组，请检查 image-dir 结构。")

    model, tokenizer, image_processor, dtype = load_model_and_tokenizer(
        args.model_name, args.bits, args.fp16, args.bf16, device
    )
    model.config.use_cache = True  # 生成阶段需要

    # FG-CLIP（用于奖励）
    fg = _load_fgclip(args.fgclip_root, device)

    # 优化器只包含可训参数（= mm_projector）
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"可训参数（mm_projector）: {n_params}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    os.makedirs(args.output_dir, exist_ok=True)

    question = args.question
    max_new = args.max_new_tokens

    step = 0
    while step < args.steps:
        # 随机取一组（可循环多 epoch）
        gid, region_paths, meta = groups[step % len(groups)]
        K = len(region_paths)
        neg_caption = meta.get("neg_caption", "")

        # ---- 1) 构造 batch：K 张区域图 -> queries + pixel_values ----
        query = build_query(tokenizer, question)  # (L,)
        query = query.unsqueeze(0).repeat(K, 1).to(device)          # (K, L)
        query_attn = torch.ones_like(query, dtype=torch.long)
        pixels = torch.stack(
            [process_image(image_processor, p) for p in region_paths]
        ).to(device, dtype=dtype)

        # ---- 2) rollout：生成 K 个区域描述 ----
        model.eval()
        with torch.no_grad():
            seqs = model.generate(
                inputs=query, images=pixels, attention_mask=query_attn,
                do_sample=True, max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id,
                top_p=1.0, top_k=0, temperature=args.temperature,
            )
        qlen = query.size(1)
        responses = seqs[:, qlen:]  # (K, max_new)
        captions = []
        for r in responses:
            r = r[r != tokenizer.pad_token_id]
            captions.append(tokenizer.decode(r, skip_special_tokens=True).strip())

        # ---- 3) 组级 RCGR 奖励 + leave-one-out 基线 ----
        rewards = compute_group_rewards(fg, region_paths, captions, neg_caption, device)
        r = torch.tensor(rewards, device=device, dtype=torch.float32)
        baseline = (r.sum() - r) / (K - 1)          # leave-one-out
        advantage = r - baseline

        # ---- 4) 前向算 logprob（response 部分）----
        model.train()
        model.config.use_cache = False
        input_ids = torch.cat([query, responses], dim=1)  # (K, L+max_new)
        attn = torch.cat(
            [query_attn, responses.ne(tokenizer.pad_token_id).long()], dim=1
        )
        outputs = model(input_ids=input_ids, attention_mask=attn, images=pixels)
        logits = outputs.logits  # (K, L+max_new, V)
        resp_logits = logits[:, qlen - 1 : -1]  # 预测 responses[:, 0:max_new]
        lp = compute_logprobs(resp_logits, responses, ignore_index=tokenizer.pad_token_id)
        seq_lp = lp.sum(dim=1)  # (K,)

        # REINFORCE：loss = -mean(advantage * logprob)
        loss = -(advantage * seq_lp).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        model.config.use_cache = True

        if step % args.log_interval == 0:
            print(
                f"[{step}/{args.steps}] gid={gid} K={K} loss={loss.item():.4f} "
                f"adv_mean={advantage.mean().item():.4f} "
                f"r_mean={r.mean().item():.4f} r_min={r.min().item():.4f} r_max={r.max().item():.4f}"
            )

        if (step + 1) % args.save_interval == 0 or step == args.steps - 1:
            save_path = os.path.join(args.output_dir, f"mm_projector_step{step + 1}.bin")
            torch.save(model.get_model().mm_projector.state_dict(), save_path)
            print(f"已保存投影层: {save_path}")

        step += 1

    print("训练完成。投影层保存在", args.output_dir)


def main():
    p = argparse.ArgumentParser(description="MVPG RL（REINFORCE + leave-one-out，只训投影层）")
    p.add_argument("--image-dir", required=True, help="区域图目录（每子目录一组区域）")
    p.add_argument("--model-name", required=True, help="LLaVA 模型路径")
    p.add_argument("--output-dir", default=str(ROOT / "output" / "mvpg"), help="checkpoint 输出目录")
    p.add_argument("--fgclip-root", default=FGCLIP_ROOT, help="FG-CLIP2 权重目录")
    p.add_argument("--question", default=DEFAULT_QUESTION, help="生成 prompt")
    p.add_argument("--steps", type=int, default=1000, help="训练步数")
    p.add_argument("--lr", type=float, default=1e-4, help="投影层学习率")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--bits", type=int, default=16, choices=[4, 8, 16], help="量化位数（8GB 显存建议 4）")
    p.add_argument("--fp16", action="store_true", help="使用 fp16")
    p.add_argument("--bf16", action="store_true", help="使用 bf16")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=100)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
