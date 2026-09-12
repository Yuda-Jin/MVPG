#!/usr/bin/env python3
"""
MVPG RL 训练（自研，不依赖 verl）。

方法对应 paper/story.md：
  §3.3  策略梯度：REINFORCE + 组内归一化基线（组 = 同图多区域视角：
        advantage = (r_i - mean(r)) / (std(r) + eps)，消除组间奖励量纲差异），
        梯度只回传投影层 mm_projector，视觉编码器与 LLM 冻结。
        另加对原模型（训练前 mm_projector）的 KL 正则（仅主图行），防止策略
        偏离语言分布坠入「堆词/重复句式」低质量吸引子。
  §3.2  区域对比接地奖励 RCGR：
        r_i = alpha * [sim(a_i, R_i) - max_{j!=i} sim(c_j, R_i)]      # 区域竞争
            + beta  * [sim(a_i, R_i) - sim(b, R_i)]                    # 缺席负样本(易)
            + gamma * [sim(a_i, R_i) - sim(ref_i, R_i)]               # 原模型 caption(难)
        sim = FG-CLIP 文本/区域特征余弦相似度。
        ref 为「与当前训练策略同基座」的原模型（7B 训 7B、13B 训 13B）对各区域
        （含主图）预生成的 caption，区域级对齐；作为难负样本，要求策略 caption 比原
        模型 caption 更贴合图像（self-play / relative reward）。

数据格式（image-dir 下每个子目录 = 一张主图的一组网格子图视角）：
    image-dir/
      neg_captions.json    # 组级总 json：{gid: ["缺席难负例1", ...]}（同时也是组索引，
                           # 训练用它一次列出全部组，不再遍历子目录）
      001/
        001.png            # 主图（整图，已下采样到与子图同尺寸）
        001_r0c0.png       # 子图视角 R_0（主图按 n 等分网格切分得到的 crop）
        001_r0c1.png       # 子图视角 R_1
        ...
        meta.json          # 旧版逐组负例（保留文件，训练不再读取；负例一律取自根 json）
    neg_captions 是对整张主图生成的「图中不存在物体」描述，作为组级缺席难负例池。
    无 neg_captions.json 时回退为按子目录名罗列组（负例为空）。

用法：
    python mvpg_run/run_mvpg.py --image-dir <数据目录> --model-name <LLaVA路径> \
        --output-dir <checkpoint目录> --steps 1000 --lr 1e-4
"""

import argparse
import csv
import json
import os
import re
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
from utils.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

# ---------------- RCGR 奖励默认系数（可通过 CLI 覆盖做消融） ----------------
FGCLIP_ROOT = str(ROOT / "models" / "fgclip2-base-patch16")

# 生成 prompt（对每个子图视角提问；子图是网格 crop，非 mask 高亮图）
DEFAULT_QUESTION = "Describe the image content in detail."

# 原模型 caption 难负样本在 model_captions.json 里的 key（与 data/generate_model_captions.py
# 的 --models 默认一致）。每个区域（含主图）各有一条 7B/13B caption，区域级对齐。
REF7_KEY = "llava-v1.5-7b"
REF13_KEY = "llava-v1.5-13b"


def _ref_key_from_model_name(model_name):
    """由当前训练策略的模型名/路径推断应使用哪个原模型 caption 作为难负样本。

    策略是 7B → 用原 7B caption；策略是 13B → 用原 13B caption（同基座自对弈，
    避免跨模型风格差异污染 reward）。无法判断（路径不含 13b）时回退到 7B。
    """
    if "13b" in os.path.basename(str(model_name)).lower():
        return REF13_KEY
    return REF7_KEY


# =============================================================================
# FG-CLIP2：图像/文本编码（用于奖励）
# 本地补丁版加载器（mvpg_run/fgclip_local.py）：官方权重 + Siglip2 风格预处理
# 复刻，规避 transformers 4.34 与官方 4.53+ 建模代码的 API 冲突。
# =============================================================================

_fgclip = None  # 惰性加载单例（FGChip 对象）


def _load_fgclip(root, device):
    global _fgclip
    if _fgclip is None:
        from mvpg_run.fgclip_local import load_fgclip
        _fgclip = load_fgclip(root, device)
    return _fgclip


def _cos(a, b):
    return float((a * b).sum().item())


def _select_neg_sim(neg_feats, img_feat, neg_mode):
    """按 neg_mode 从负例特征池选一条与 img_feat 的余弦：
    first=池首条（训练默认，稳定可复现）；worst=最难区分的一条。"""
    if neg_mode == "worst":
        return max(_cos(nf, img_feat) for nf in neg_feats)
    return _cos(neg_feats[0], img_feat)


def compute_group_rewards(fg, region_paths, captions, neg_captions, device,
                          alpha=1.0, beta=0.5, neg_mode="first",
                          reg_weight=0.0, ref_captions=None, ref_weights=None):
    """组级 RCGR 奖励。region_paths/captions 一一对应（同图 K 个子图视角）。
    返回 (list[float], float)：奖励列表（长度 K）与区域结构正则均值
    （reg_weight<=0 时为 nan，仅关闭时）。

    - sim(a_i, R_i)：caption_i 文本 vs 区域 i 视觉
    - max_{j!=i} sim(c_j, R_i)：同图其余区域描述的竞争项
    - sim(b, R_i)：缺席难负例基线（b 来自组级 neg_captions 池；
      neg_mode="first" 固定取池首条；neg_mode="worst" 逐区域取最难一条）
    - ref_captions：长度 K 的列表，每元素 {model_name: caption}（原模型对第 i 个区域
      预生成的 caption，区域级对齐、含主图行）。ref_weights：{model_name: w}，w<=0 的
      模型跳过。对应难负样本项 w*(sim(a_i,R_i) - sim(ref_i,R_i))，要求策略 caption
      比原模型 caption 更贴合图像（self-play / relative reward）。
    - reg_weight>0：区域结构对齐正则（reward shaping）。caption 由离散采样得到，
      FG-CLIP 文本编码在 no_grad 下不可微，无法作为直接 loss 反传到投影层，
      故并入奖励经 advantage 产生梯度：
        S_t[i,j] = sim(caption_i, R_j)   文本→视觉相似度矩阵（随策略变化）
        S_v[i,j] = sim(R_i, R_j)         视觉→视觉结构先验（组内固定）
      两矩阵均去对角线后 Frobenius 归一化（单位范数，量纲对齐），区域 i 的
      奖励扣减 reg_weight * Σ_{j≠i} (S_t_n[i,j] - S_v_n[i,j])²。
      约束语义：描述间的相对相似结构应贴合区域视觉间的相对相似结构。
    """
    K = len(region_paths)
    img_feats = fg.encode_image_paths(region_paths)                # (K, D)

    # 原模型 ref caption：逐行展开成 (文本, (行号, 模型名))，仅保留权重>0且有文本的
    ref_txt, ref_idx = [], []
    ref_weights = ref_weights or {}
    if ref_captions:
        for i, d in enumerate(ref_captions):
            if not d:
                continue
            for mname, w in ref_weights.items():
                c = d.get(mname)
                if w > 0 and c:
                    ref_txt.append(c)
                    ref_idx.append((i, mname))

    neg_captions = list(neg_captions or [])
    all_txt = list(captions) + neg_captions + ref_txt
    feats = fg.encode_texts(all_txt)                               # (K+N+R, D)
    txt_feats = feats[:K]
    neg_feats = feats[K:K + len(neg_captions)] if neg_captions else []
    ref_feats = feats[K + len(neg_captions):]

    # sim_ref[i][model_name] = cos(ref_caption, region_i)
    sim_ref = {}
    for f, (i, mname) in zip(ref_feats, ref_idx):
        sim_ref.setdefault(i, {})[mname] = _cos(f, img_feats[i])

    reg_rows = None
    if reg_weight > 0 and K > 1:
        eye = torch.eye(K, dtype=torch.bool, device=img_feats.device)
        S_t = (txt_feats @ img_feats.t()).masked_fill(eye, 0.0)    # 文本→视觉，去对角
        S_v = (img_feats @ img_feats.t()).masked_fill(eye, 0.0)    # 视觉→视觉，去对角
        S_t = S_t / (S_t.norm() + 1e-6)                            # Frobenius 归一化
        S_v = S_v / (S_v.norm() + 1e-6)
        reg_rows = ((S_t - S_v) ** 2).sum(dim=1)                   # (K,) 逐区域失配

    rewards = []
    for i in range(K):
        sim_self = _cos(txt_feats[i], img_feats[i])

        # 跨区域竞争项：其余区域描述 vs 本区域视觉，取最大
        sim_others = max(_cos(txt_feats[j], img_feats[i]) for j in range(K) if j != i)

        # 缺席难负例基线（图级负例池）
        sim_neg = _select_neg_sim(neg_feats, img_feats[i], neg_mode) if len(neg_feats) > 0 else 0.0

        r = alpha * (sim_self - sim_others) + beta * (sim_self - sim_neg)

        # 原模型 caption 难负样本：逐区域 sim_ref_i，要求策略 caption 超过原模型
        for mname, w in ref_weights.items():
            s = (sim_ref.get(i) or {}).get(mname)
            if w > 0 and s is not None:
                r = r + w * (sim_self - s)

        if reg_rows is not None:
            r = r - reg_weight * reg_rows[i].item()
        rewards.append(r)

    reg_mean = reg_rows.mean().item() if reg_rows is not None else float("nan")
    return rewards, reg_mean


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

    # LLaVA-1.5 checkpoint 不含 CLIP 视觉塔权重（0 keys），config 默认指向
    # openai/clip-vit-large-patch14-336 远程仓库。加载前改写为本地目录避免联网。
    # 本地 CLIP 默认放在模型目录同级 clip-vit-large-patch14-336，可用 LLAVA_CLIP_DIR 覆盖。
    clip_dir = os.environ.get("LLAVA_CLIP_DIR")
    if clip_dir is None:
        clip_dir = str(Path(model_name).resolve().parent / "clip-vit-large-patch14-336")
    if os.path.isdir(clip_dir):
        llava_cfg = LlavaLlamaForCausalLM.config_class.from_pretrained(model_name)
        llava_cfg.mm_vision_tower = clip_dir
    else:
        llava_cfg = None

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name,
        config=llava_cfg,
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
    # 与 checkpoint config 保持一致：llava-v1.5-7b 的 mm_use_im_start_end / mm_use_im_patch_token
    # 均为 False（<image> 由 prompt 层处理，词表 32000 不含额外 special token）。
    from argparse import Namespace
    model_args = Namespace(
        mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
        mm_use_im_patch_token=bool(getattr(model.config, "mm_use_im_patch_token", False)),
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    )
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 只训 mm_projector
    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    return model, tokenizer, image_processor, compute_dtype


# =============================================================================
# 数据：组索引优先来自 image-dir 根级总 json（{gid: [缺席难负例]}），
# 启动只做一次 json.load，不再遍历全部子目录、不再读子目录 meta.json（文件保留）。
# 子图路径仅在组被采样到时解析（单目录，按 `<gid>_r<row>c<col>` 识别并排序）。
# =============================================================================

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


def _read_model_captions(image_dir, ref_json="model_captions.json"):
    """读取 image-dir 根级 model_captions.json（原模型逐区域 caption）。
    结构 {gid: {model_name: {image_stem: caption}}}。找不到返回 None。"""
    fp = Path(image_dir) / ref_json
    if not fp.is_file():
        return None
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return {str(gid): models for gid, models in raw.items() if isinstance(models, dict)}


def _ref_captions_for_paths(model_captions, gid, paths, model_keys):
    """按 paths 顺序，取每个区域（含主图）的原模型 caption dict {model_name: caption}。

    paths 与 compute_group_rewards 的 region_paths 严格同序；主图行的 stem == gid，
    子图行的 stem == gid_r{r}c{c}，直接按 Path(p).stem 从 model_captions[gid] 查询。
    """
    g_refs = model_captions.get(gid, {}) or {}
    out = []
    for p in paths:
        stem = Path(p).stem
        d = {}
        for mname in model_keys:
            c = (g_refs.get(mname) or {}).get(stem)
            if c:
                d[mname] = c
        out.append(d)
    return out


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


def compute_logprobs(logits, labels, ignore_index):
    """逐 token logprob，padding(ignore_index) 位置置 0。同 utils.common_utils。"""
    return -F.cross_entropy(
        logits.permute(0, 2, 1), labels, reduction="none", ignore_index=ignore_index
    )


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_key = _ref_key_from_model_name(args.model_name)
    print("=" * 78)
    print("MVPG RL 训练（自研 REINFORCE + 组均值基线，只训投影层）")
    print(f"  model  : {args.model_name}")
    print(f"  data   : {args.image_dir}")
    print(f"  device : {device}")
    print(f"  RCGR   : alpha={args.alpha} beta={args.beta} neg_mode={args.neg_mode} "
          f"use_main_image={args.use_main_image}")
    print(f"  batch  : groups_per_step={args.groups_per_step} reg_weight={args.reg_weight} "
          f"kl_weight={args.kl_weight}")
    print(f"  ref    : ref_key={ref_key}(auto，同基座) ref7_weight={args.ref7_weight} "
          f"ref13_weight={args.ref13_weight}")
    print("=" * 78)

    groups = load_groups(args.image_dir)
    print(f"共 {len(groups)} 个数据组")
    if len(groups) == 0:
        raise SystemExit("没有找到有效数据组，请检查 image-dir 结构或根级 json。")
    for g in groups[:3]:
        print(f"  gid={g['gid']} negs={len(g['negs'])}")

    # 原模型 caption 难负样本（区域级）：只取与当前训练策略同基座的原模型 caption
    # （7B 训 7B、13B 训 13B），避免跨模型风格差异污染 reward。权重<=0 时关闭。
    model_captions = _read_model_captions(args.image_dir, args.ref_json) or {}
    ref_weight = args.ref7_weight if ref_key == REF7_KEY else args.ref13_weight
    ref_weights = {ref_key: ref_weight} if ref_weight > 0 else {}
    if ref_weights:
        n_ref_groups = sum(
            1 for g in model_captions
            if any((model_captions[g].get(k) or {}) for k in ref_weights))
        print(f"原模型难负样本: ref_key={ref_key} 权重={ref_weights}，"
              f"含 caption 的组数={n_ref_groups}/{len(groups)}")
    else:
        print(f"原模型难负样本: 关闭（ref_key={ref_key} 对应 weight<=0）")

    model, tokenizer, image_processor, dtype = load_model_and_tokenizer(
        args.model_name, args.bits, args.fp16, args.bf16, device
    )
    model.config.use_cache = True  # 生成阶段需要

    # 参考策略（原模型）：训练前的 mm_projector 权重，用于主图行的 KL 正则。
    # 深拷贝到内存，避免后续训练权重变化时被覆盖（KL 项据此计算分布偏离）。
    ref_projector_sd = {k: v.detach().clone()
                        for k, v in model.get_model().mm_projector.state_dict().items()}

    # FG-CLIP（用于奖励）
    fg = _load_fgclip(args.fgclip_root, device)

    # 优化器只包含可训参数（= mm_projector）
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"可训参数（mm_projector）: {n_params}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    os.makedirs(args.output_dir, exist_ok=True)

    # 每步指标写 CSV（不含 adv_mean），供 mvpg_run/plot_train_log.py 可视化
    log_path = os.path.join(args.output_dir, "train_log.csv")
    log_fp = open(log_path, "w", newline="", encoding="utf-8")
    log_csv = csv.writer(log_fp)
    log_csv.writerow(["step", "gid", "loss", "r_mean", "r_min", "r_max", "r_std", "r_main", "kl"])
    log_fp.flush()
    print(f"训练指标将写入: {log_path}")

    # 每步采样输出写文本 log（不打印到终端），供坍缩/退化诊断：
    # 追溯「正常描述→重复句式→乱码→单 token 循环」全过程。
    captions_log_path = os.path.join(args.output_dir, "train_captions.log")
    captions_fp = open(captions_log_path, "w", encoding="utf-8")
    print(f"模型采样输出将写入: {captions_log_path}")

    question = args.question
    max_new = args.max_new_tokens

    step = 0
    gcursor = 0  # 组游标：顺序循环多 epoch；每个优化器步消耗 G 个组
    collapse_steps = 0  # 连续全组 logp 饱和的步数（输出坍缩检测，>=5 硬停止）
    G = max(1, args.groups_per_step)
    print(f"每次更新同时处理 {G} 个数据组（数据并行；组间独立，reward/advantage 仍组内计算）")
    while step < args.steps:
        # ---- 0) 顺序取 G 个有效组（坏组跳过；游标循环）----
        batch_raw, metas, tries = [], [], 0
        while len(batch_raw) < G and tries < len(groups):
            tries += 1
            g = groups[gcursor % len(groups)]
            gcursor += 1
            main_img, regions = resolve_group_images(args.image_dir, g["gid"])
            if len(regions) < 2:
                print(f"[skip] gid={g['gid']} regions={len(regions)} < 2，跳过")
                continue
            batch_raw.append((g, main_img, regions))
        if not batch_raw:
            print("[stop] 没有可用数据组，提前结束")
            break

        # ---- 1) 构造合并 rollout batch：每组 K 张训练视角（主图行始终参与训练）----
        # use_main_image 关闭时，主图作为额外槽位一起生成，并同样参与奖励/advantage/梯度
        # （整图级监督信号）；r_main 直接取主图行在组内竞争中的奖励值（同口径汇报）。
        # 所有行共享同一 query 模板，行间独立，可直接拼大批次。
        query0 = build_query(tokenizer, question).to(device)
        qlen = query0.size(0)  # (L,) 单条 prompt 长度
        gen_pixels_list, row0 = [], 0
        for g, main_img, regions in batch_raw:
            region_paths = list(regions)
            if args.use_main_image and main_img:   # 可选：整图并入组内作为额外视角
                region_paths.insert(0, main_img)
            metric_main_path = main_img if (not args.use_main_image and main_img) else None
            gen_paths = region_paths + ([metric_main_path] if metric_main_path else [])
            reward_paths = region_paths + ([metric_main_path] if metric_main_path else [])
            ref_captions = _ref_captions_for_paths(
                model_captions, g["gid"], reward_paths, ref_weights.keys())
            metas.append({
                "gid": g["gid"], "main_img": main_img,
                "region_paths": region_paths, "neg_pool": g["negs"],
                "metric_main_path": metric_main_path,
                "reward_paths": reward_paths, "ref_captions": ref_captions,
                # K = 参与训练的行数（含主图行），与 gen_paths/reward 计算的行序严格对齐
                "row0": row0, "n_rows": len(gen_paths), "K": len(gen_paths),
            })
            row0 += len(gen_paths)
            gen_pixels_list.extend(process_image(image_processor, p) for p in gen_paths)
        B = row0
        gen_pixels = torch.stack(gen_pixels_list).to(device, dtype=dtype)
        query = query0.unsqueeze(0).repeat(B, 1)
        query_attn = torch.ones_like(query, dtype=torch.long)

        # ---- 2) rollout：一次生成全部组的各视角描述 ----
        model.eval()
        with torch.no_grad():
            seqs = model.generate(
                inputs=query, images=gen_pixels, attention_mask=query_attn,
                do_sample=True, max_new_tokens=max_new,
                pad_token_id=tokenizer.pad_token_id,
                top_p=1.0, top_k=0, temperature=args.temperature,
            )
        responses_all = seqs[:, qlen:]  # (B, max_new)
        captions_all = []
        for r_ in responses_all:
            r_ = r_[r_ != tokenizer.pad_token_id]
            captions_all.append(tokenizer.decode(r_, skip_special_tokens=True).strip())

        # ---- 3) 逐组 RCGR 奖励 + 组内归一化 advantage（组间严格独立）----
        for m in metas:
            caps = captions_all[m["row0"]: m["row0"] + m["n_rows"]]
            resp = responses_all[m["row0"]: m["row0"] + m["n_rows"]]
            # 参与奖励/advantage/梯度的是全部行（含主图行）；路径顺序与 gen_paths 一致：
            # use_main_image=True 时主图在 region_paths[0]，否则主图行在末尾。
            reward_paths = m["reward_paths"]
            if m["metric_main_path"] is not None:
                cap_main = caps[-1]
            else:
                cap_main = caps[0] if (args.use_main_image and m["main_img"]) else None
            rewards, reg_mean = compute_group_rewards(
                fg, reward_paths, caps, m["neg_pool"], device,
                alpha=args.alpha, beta=args.beta,
                neg_mode=args.neg_mode, reg_weight=args.reg_weight,
                ref_captions=m["ref_captions"], ref_weights=ref_weights)
            r = torch.tensor(rewards, device=device, dtype=torch.float32)
            # 组内归一化（GRPO 风格）：mean=0、单位方差，消除组间量纲差异，
            # 使每步梯度尺度一致；std→0（组内无差异）时 advantage→0，自然跳过更新。
            # adv clamp：组内 std 很小时 (r-mean)/std 会爆炸，单步大梯度会把投影层
            # 推向输出坍缩吸引子（乱码→单 token 循环→logp 饱和为 0 死锁）。
            m.update(resp=resp, r=r, reg_mean=reg_mean,
                     adv=((r - r.mean()) / (r.std() + 1e-6)).clamp(-5.0, 5.0))
            m["caps"] = caps
            # 主图进度指标：把主图当作组内一个视角，直接取其竞争奖励值（与 r_min/r_max
            # 同口径，含跨区域竞争项，落在同一尺度区间内，可直接比较）。
            if cap_main is not None and m["main_img"]:
                main_idx = -1 if m["metric_main_path"] is not None else 0
                m["r_main"] = r[main_idx].item()
            else:
                m["r_main"] = float("nan")

        # ---- 4) 逐组前向算 logprob，梯度累积后一次更新（显存 O(单组)，与全 batch 反传等价）----
        # REINFORCE：loss = -mean_rows(advantage·logprob) = Σ_g -(adv_g·lp_g).sum()/N_train。
        model.train()
        model.config.use_cache = False
        n_train = sum(m["K"] for m in metas)
        # 带图像前向时，query 里的每个 <image> 占位符会被展开成 m_img 个视觉 token，
        # 预测 responses[:, 0] 的 logits 下标 = (qlen-1) + s*(m_img-1)。
        s_img = int((query0 == IMAGE_TOKEN_INDEX).sum().item())
        m_img = model.get_model().get_vision_tower().num_patches if s_img > 0 else 0
        resp_start = qlen - 1 + s_img * (m_img - 1)

        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        n_degenerate = 0
        for m in metas:
            k = m["K"]

            # KL 参考 logprob（仅主图行）：先把投影层临时换成训练前权重，no_grad 重算
            # lp_ref 后立即恢复当前权重。必须放在本组「当前权重前向」之前——当前前向会
            # 建立依赖投影层权重的反向图，之后再做 load_state_dict 的 inplace copy 会
            # 让权重版本计数不一致，触发 “modified by an inplace operation” 错误。
            main_idx = None
            lp_ref_main = None
            if args.kl_weight > 0 and m["main_img"]:
                main_idx = -1 if m["metric_main_path"] is not None else 0
                main_gidx = m["row0"] + (m["n_rows"] - 1 if main_idx == -1 else 0)
                mm_proj = model.get_model().mm_projector
                curr_sd = {kk: vv.detach().clone() for kk, vv in mm_proj.state_dict().items()}
                mm_proj.load_state_dict(ref_projector_sd)
                main_ids = torch.cat([query0.unsqueeze(0), m["resp"][main_idx].unsqueeze(0)], dim=1)
                main_attn = torch.cat(
                    [torch.ones(1, qlen, dtype=torch.long, device=query0.device),
                     m["resp"][main_idx].ne(tokenizer.pad_token_id).long().unsqueeze(0)], dim=1)
                with torch.no_grad():
                    ref_logits = model(input_ids=main_ids, attention_mask=main_attn,
                                       images=gen_pixels[main_gidx].unsqueeze(0)).logits
                lp_ref_main = compute_logprobs(ref_logits[:, resp_start:-1],
                                               m["resp"][main_idx].unsqueeze(0),
                                               ignore_index=tokenizer.pad_token_id)
                mm_proj.load_state_dict(curr_sd)

            ids = torch.cat([query0.unsqueeze(0).repeat(k, 1), m["resp"]], dim=1)
            attn = torch.cat(
                [torch.ones(k, qlen, dtype=torch.long, device=query0.device),
                 m["resp"].ne(tokenizer.pad_token_id).long()], dim=1
            )
            logits = model(input_ids=ids, attention_mask=attn,
                           images=gen_pixels[m["row0"]: m["row0"] + k]).logits
            resp_logits = logits[:, resp_start:-1]  # 预测 responses[:, 0:max_new]
            assert resp_logits.size(1) == m["resp"].size(1), (
                f"resp_logits 长度 {resp_logits.size(1)} != responses 长度 {m['resp'].size(1)}"
            )
            lp = compute_logprobs(resp_logits, m["resp"], ignore_index=tokenizer.pad_token_id)
            # 坍缩检测：logp 全零 = 模型对采样响应的置信度在 fp16 下饱和
            # （输出退化为确定性 token 循环），此时 loss/梯度恒 0，训练无法自恢复。
            # 跳过该组并在 step 末尾统计；连续多步全退化则硬停止，提示回滚 ckpt。
            if lp.abs().max().item() < 1e-5:
                n_degenerate += 1
                continue

            # KL 正则（仅主图行）：对原模型（训练前 mm_projector）的分布偏离惩罚。
            # KL[π_θ||π_ref] ≈ lp_curr - lp_ref（样本来自 π_θ）。lp_ref 已在 no_grad 下
            # 用参考权重重算并恢复当前权重；(lp - lp_ref) 作为常数乘子（detach）再乘以
            # lp，反向得到标准 KL 惩罚梯度 β*(lp-lp_ref)*∇lp。
            kl_main = 0.0
            kl_val = float("nan")
            if lp_ref_main is not None:
                kl_main = ((lp[main_idx] - lp_ref_main[0]).detach() * lp[main_idx]).sum()
                # 实际 KL 散度样本估计：Σ_t (log π_θ - log π_ref)，仅主图行整句求和，
                # 期望 >= 0；单样本可正可负。kl_main 是其梯度对应的 loss 项，kl_val 用于汇报。
                kl_val = (lp[main_idx] - lp_ref_main[0]).detach().sum().item()
            m["kl"] = kl_val

            loss_g = -(m["adv"] * lp.sum(dim=1)).sum() / n_train
            if lp_ref_main is not None:
                loss_g = loss_g + args.kl_weight * kl_main / n_train
            loss_g.backward()
            loss_val += loss_g.item()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        model.config.use_cache = True

        # 坍缩硬停止：连续多步所有组 logp 饱和（输出坍缩、梯度恒 0），继续跑无意义
        collapse_steps = collapse_steps + 1 if n_degenerate == len(metas) else 0
        if collapse_steps >= 5:
            raise SystemExit(
                f"[collapse] 连续 {collapse_steps} 步所有响应 logp 饱和（输出已坍缩为"
                f"确定性 token 循环），训练无法自恢复。请回滚到最后一个有效 ckpt"
                f"（权重范数仍在变化的最近一步），并降低 --lr 后重启。")

        if step % args.log_interval == 0:
            rs = torch.cat([m["r"] for m in metas])
            # r_std：各组内 std 的均值 = advantage 归一化分母的平均水平。
            # 持续收缩 = 输出趋同（探索度衰减/熵坍缩前兆）；极小非零 = 归一化放大噪声。
            stds = [m["r"].std().item() for m in metas]
            r_mains = [m["r_main"] for m in metas if m["r_main"] == m["r_main"]]
            regs = [m["reg_mean"] for m in metas if m["reg_mean"] == m["reg_mean"]]
            reg_str = f" reg={sum(regs) / len(regs):.4f}" if regs else ""
            kls = [m["kl"] for m in metas if m["kl"] == m["kl"]]
            kl_str = f" kl={sum(kls) / len(kls):.4f}" if kls else ""
            r_main_str = f"{sum(r_mains) / len(r_mains):.4f}" if r_mains else "nan"
            deg_str = f" degenerate={n_degenerate}/{len(metas)}" if n_degenerate else ""
            print(
                f"[{step}/{args.steps}] G={len(metas)} B={n_train} loss={loss_val:.4f} "
                f"r_mean={rs.mean().item():.4f} r_min={rs.min().item():.4f} r_max={rs.max().item():.4f} "
                f"r_std={sum(stds) / len(stds):.4f} "
                f"r_main={r_main_str}{kl_str}{reg_str}{deg_str}"
            )
            # 采样输出写 log（不打印）：每步一组，含各视角 caption，便于诊断坍缩/退化
            captions_fp.write(f"===== step {step} =====\n")
            for m in metas:
                for p, c in zip(m["reward_paths"], m["caps"]):
                    stem = os.path.splitext(os.path.basename(p))[0]
                    tag = "main" if stem == m["gid"] else stem[len(m["gid"]) + 1:]
                    captions_fp.write(f"[{m['gid']}] {tag}: {c}\n")
            captions_fp.write("\n")
            captions_fp.flush()

        # 每组指标落盘（同 step 多行；adv_mean 恒为 0，无信息量，不记录）
        for m in metas:
            log_csv.writerow([
                step, m["gid"],
                f"{loss_val:.6f}",
                f"{m['r'].mean().item():.6f}",
                f"{m['r'].min().item():.6f}",
                f"{m['r'].max().item():.6f}",
                f"{m['r'].std().item():.6f}",
                f"{m['r_main']:.6f}",
                f"{m['kl']:.6f}",
            ])
        log_fp.flush()

        if (step + 1) % args.save_interval == 0 or step == args.steps - 1:
            save_path = os.path.join(args.output_dir, f"mm_projector_step{step + 1}.bin")
            torch.save(model.get_model().mm_projector.state_dict(), save_path)
            print(f"已保存投影层: {save_path}")

        step += 1

    log_fp.close()
    captions_fp.close()
    print("训练完成。投影层保存在", args.output_dir)
    print("指标曲线: python mvpg_run/plot_train_log.py --log", log_path)


def main():
    p = argparse.ArgumentParser(description="MVPG RL（REINFORCE + 组均值基线，只训投影层）")
    p.add_argument("--image-dir", required=True,
                   help="数据目录（含根级 neg_captions.json 组索引；每子目录 = 主图 + 网格子图）")
    p.add_argument("--model-name", required=True, help="LLaVA 模型路径")
    p.add_argument("--output-dir", default=str(ROOT / "output" / "mvpg"), help="checkpoint 输出目录")
    p.add_argument("--fgclip-root", default=FGCLIP_ROOT, help="FG-CLIP2 权重目录")
    p.add_argument("--question", default=DEFAULT_QUESTION, help="生成 prompt")
    p.add_argument("--steps", type=int, default=4848,
                   help="训练步数（优化器更新次数；每步消耗 groups-per-step 个数据组）")
    p.add_argument("--groups-per-step", type=int, default=4,
                   help="每次更新同时处理的数据组数（数据并行维度；组间独立，"
                        "reward/advantage 仍组内计算；G 个组拼接为一个大 rollout batch，"
                        "前向反向按组累积梯度，显存占用与 G=1 相同）")
    p.add_argument("--lr", type=float, default=3e-4, help="投影层学习率")
    p.add_argument("--alpha", type=float, default=1.0, help="RCGR 跨区域竞争项系数")
    p.add_argument("--beta", type=float, default=0.5,
                   help="RCGR 幻觉负样本基线项系数（整图级 neg_captions；设为 0 取消 β 项，消融）")
    p.add_argument("--ref7-weight", type=float, default=0.5,
                   help="原模型(7B) caption 难负样本竞争项系数 gamma（区域级；"
                        "仅当训练策略为 7B 时生效，<=0 关闭）")
    p.add_argument("--ref13-weight", type=float, default=0.5,
                   help="原模型(13B) caption 难负样本竞争项系数 gamma（区域级；"
                        "仅当训练策略为 13B 时生效，<=0 关闭）")
    p.add_argument("--ref-json", default="model_captions.json",
                   help="根级原模型 caption json（由 data/generate_model_captions.py 生成）")
    p.add_argument("--kl-weight", type=float, default=0.1,
                   help="对原模型（训练前 mm_projector）的 KL 正则权重，仅主图行；<=0 关闭。"
                        "过大(>=1)会压死策略、奖励长期不涨，建议 0.05~0.2")
    p.add_argument("--neg-mode", choices=["first", "worst"], default="first",
                   help="缺席难负例池用法：first=固定取池首条；worst=逐区域取最难一条")
    p.add_argument("--reg-weight", type=float, default=0.0,
                   help="区域结构对齐正则权重（reward shaping；<=0 关闭）")
    p.add_argument("--use-main-image", action="store_true",
                   help="把整图（下采样主图）并入组内作为额外视角参与 reward/优化")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=128)
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
