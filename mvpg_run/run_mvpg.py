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

  --reward-mode clause  子句级视觉支持阈值奖励（替代 RCGR，只训主图）：
        caption 切成「规范打分单元」后，每单元到 dense patch 查最大视觉支持 s（FG-CLIP
        short walk，raw 余弦），s>=--clause-threshold 记 +1、否则 -1；优势经字符偏移
        对齐映射到 token 区间，loss = -Σ_t adv_t · logπ(o_t)（逐 token REINFORCE）。
        adv 非零均值 -> 全幻觉 caption 被整体压低，可直接压制幻觉。
        规范切分（见 clause_spans）：不直接采用模型输出的标点切分，而是「标点断句 +
        长段强制按词再切 + 短段并回相邻段」——否则 ±1 阈值下「合并子句 / 不写标点」
        是严格占优策略（段越少，max-patch-sim 越容易越过阈值，且能把低支撑内容藏进
        同一段），策略会退化成 run-on 长句并把奖励架空成 self-imitation。

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
from bisect import bisect_left, bisect_right
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

FGCLIP_ROOT = str(ROOT / "models" / "fgclip2-base-patch16")

# =============================================================================
# 训练超参数（单一事实来源）
#   调参只改本块：下方 argparse 的默认值全部引用这些常量，run/train_mvpg.sh 不再
#   重复给值（只留路径 / 步数与 --reward-mode 等开关），避免两处默认值分叉。
#   子句级奖励取值依据：阈值为「长度校正后 s_adj」的口径（见 HP_CLAUSE_LEN_MU），
#   不再是 raw s_short 的绝对阈值；max_words=16 落在词数可分化性最优的
#   10~20 词桶中部。
# =============================================================================

# ---- 优化 / 生成 / 日志 ----
HP_LR = 1e-4              # 只训 mm_projector；1e-4 稳定，显著更大易坍缩
HP_QUESTION = "Describe the image concisely."   # 训练 rollout 的提问
HP_TEMPERATURE = 1.0
HP_MAX_NEW_TOKENS = 128   # 生成上限（安全阀）：简洁提问下模型通常远早于此自然结束
HP_MAX_GRAD_NORM = 1.0
HP_STEPS = 5000           # 优化器更新次数（每步 HP_GROUPS_PER_STEP 个组，共 4848 组）
HP_GROUPS_PER_STEP = 4
HP_KL_WEIGHT = 0.1        # 对原模型（训练前 mm_projector）的 KL 正则，仅主图行；建议 0.05~0.2
HP_BITS = 16
HP_LOG_INTERVAL = 5
HP_SAVE_INTERVAL = 50

# ---- 奖励模式与 RCGR 系数 ----
HP_REWARD_MODE = "rcgr"   # rcgr=组级区域竞争（默认）；clause=子句级视觉支持阈值奖励；
                          # sentence=句级全图特征相似度 - 参考模型基线（只训主图）
HP_ALPHA = 1.0            # RCGR 跨区域竞争项系数
HP_BETA = 0.5             # RCGR 缺席负样本基线项系数（整图级；0 关闭）
HP_REF7_WEIGHT = 0.5      # 原 7B caption 难负样本竞争项（7B 策略生效；<=0 关闭）
HP_REF13_WEIGHT = 0.5     # 原 13B caption 难负样本竞争项（13B 策略生效；<=0 关闭）
HP_NEG_MODE = "first"     # 缺席难负例池用法：first=池首条；worst=每区域取最难一条
HP_REG_WEIGHT = 0.0       # 区域结构对齐正则（reward shaping；<=0 关闭）

# ---- 子句级奖励（--reward-mode clause）----
HP_CLAUSE_WALK = "short"       # FG-CLIP 文本 walk 类型（子句级标定推荐 short）
HP_CLAUSE_THRESHOLD = -0.0251  # 长度校正后 s_adj >= 该值记 +1，否则 -1
HP_CLAUSE_MAX_WORDS = 16       # 规范切分的打分单元词数上限
# 长度校正表：打分单元词数上限 -> 该长度的 max-patch-sim 期望 mu(n)。
# 打分用 s_adj = s_short - mu(n_words)：raw s_short 随词数单调上升
# （corr(n_words, s_short)=+0.369），配合固定绝对阈值会让「把每个单元顶满
# max_words」成为严格占优策略，策略因此退化成不写标点的 run-on 长句。
# mu 取「桶内全体子句」均值：按构造使各长度期望收益平在 0，策略无法靠 padding
# 长度薅奖励。刻意不用「桶内仅真实子句」均值——那样判别力更高（留出 F1 0.739 /
# AUC 0.925 vs 0.664 / 0.888），但会重新引入随长度的正偏置，而长度正是策略手里的
# 旋钮（且池中「长子句更易为幻觉」的相关性疑为标注注入过程的产物）。
# 桶界与取值来自 clause_support_scores.jsonl 的 3~18 词池（校准/留出按 gid 奇偶
# 切分，mu 只在校准集拟合）；越界（<3 词或 >18 词）就近 clamp。
HP_CLAUSE_LEN_MU = ((5, 0.1479), (9, 0.1600), (15, 0.1688), (18, 0.1990))

# ---- 句级奖励（--reward-mode sentence）----
HP_SENTENCE_WALK = "short"      # FG-CLIP 文本 walk 类型（句级推荐 short）
HP_SENTENCE_MIN_WORDS = 3       # 句子最小词数；更短的碎片并入相邻句，避免碎片各自占一个打分单元

# 原模型 ref caption 生成用的 prompt（data/generate_model_captions.py 导入本常量）。
# 刻意保留长句版：已落盘的 model_captions.json 是用它生成的，改动它会让新旧 ref 口径漂移。
# 训练 rollout 的提问用上面的 HP_QUESTION（简洁版，抑制长句与被 max_new_tokens 截断）。
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
# 子句级 grounding reward（阈值型）
# 对单图生成的 caption 先做「规范化切分」（见 clause_spans），每个打分单元到图中
# dense patch 查询最大视觉支持 s（FG-CLIP short walk，raw 余弦），按阈值判优势：
# s>=t -> +1，否则 -> -1。优势再按「字符偏移对齐」映射回 token 区间，作为逐 token
# advantage。
#
# 与 RCGR 的关键差异：adv 非零均值（±1），整条 caption 的整体奖励被保留，因此
# 全幻觉 caption 会被整体压低，可直接压制幻觉；而句内中心化（r=s-mean(s)）只做
# caption 内信用再分配，无法整体压制幻觉。
#
# 为什么必须「规范化」而不能直接采用模型输出的标点切分：±1 阈值 + max-patch 池化
# 下，段数越少，max-patch-sim 越容易越过阈值，且低支撑内容可以藏进同一段被高支撑
# 内容带着一起记 +1。于是「合并子句 / 不写标点」成为严格占优策略，策略退化成
# run-on 长句，整条 caption 只产出一个分数并全记 +1（r_std=0，奖励被架空为
# self-imitation）。规范切分把打分粒度从「模型手里」收回，使该逃逸路径失效。
# =============================================================================

_CLAUSE_TOKEN = re.compile(r"[^.,;]+")   # 标点初切：`.,;` 为分隔符
_WORD = re.compile(r"\S+")               # 词边界（用于长段再切 / 短段判定）


def _n_words(caption, a, b):
    return len(_WORD.findall(caption, a, b))


def _len_mu(n_words):
    """打分单元词数 -> 该长度的 max-patch-sim 期望 mu(n)。

    桶界见 HP_CLAUSE_LEN_MU；小于首桶或大于末桶的就近 clamp。
    """
    for hi, mu in HP_CLAUSE_LEN_MU:
        if n_words <= hi:
            return mu
    return HP_CLAUSE_LEN_MU[-1][1]


def clause_spans(caption, max_words=16, min_words=3):
    """规范化切分 caption，返回 [(原始片段, 起点, 终点)]（字符偏移均指向原 caption）。

    三步（不改写文本、只重划边界，故 span 可直接用于 map_clause_adv_to_tokens 的
    字符对齐）：

    1) 标点初切：按 `.,;` 切成原始片段；
    2) 长段强制再切：> `max_words` 词的段按词边界切成若干 ≤ max_words 的连续窗口
       ——拆掉「不写标点把整条 caption 折叠成一个打分单元」这条占优路径；
    3) 短段并回邻居：< `min_words` 词的段并入前一段（不再丢弃），使打分单元普遍落在
       `[min_words, max_words + min_words - 1]` 词——否则写一堆 `a. b. c.` 短碎片可
       拿 adv=0 的零梯度逃逸口（旧实现是直接丢弃），而短单元本身也不可分化
       （0~5 词桶 AUC 仅 0.473）。

    片段文本保留原始形态（含首尾空白），FG-CLIP 打分时另行 strip。
    """
    caption = caption or ""
    spans = [(m.start(), m.end()) for m in _CLAUSE_TOKEN.finditer(caption)
             if m.group(0).split()]
    if not spans:
        return []

    # 2) 长段按词边界强制再切（保留原字符跨度）
    split = []
    for a, b in spans:
        words = list(_WORD.finditer(caption, a, b))
        if len(words) <= max_words:
            split.append((a, b))
        else:
            for i in range(0, len(words), max_words):
                grp = words[i:i + max_words]
                split.append((grp[0].start(), grp[-1].end()))

    # 3) 短段并回邻居：避免 < min_words 的碎片单独打分（此前是直接丢弃，等于给了
    #    「写一堆 a. b. c. 短碎片」的零梯度逃逸口）。封顶 cap 防止连续碎片链式合并
    #    反而长成新的 run-on；同时保证长段再切的尾部残段被吸收（否则 17 词会切出
    #    「16 + 1」并让 1 词残段单独打分）。
    cap = max_words + min_words - 1
    units = []
    for a, b in split:
        n = _n_words(caption, a, b)
        if units:
            la, lb = units[-1]
            ln = _n_words(caption, la, lb)
            if ln < min_words or (n < min_words and ln + n <= cap):
                units[-1] = (la, b)
                continue
        units.append((a, b))

    return [(caption[a:b], a, b) for a, b in units]


def compute_clause_adv(fg, img_path, caption, walk_type="short", threshold=-0.0251,
                       max_words=16):
    """单图 caption 的子句级阈值优势（长度校正口径）。

    打分单元 = clause_spans 的规范化切分，单元分数取
        s_adj = s_short - mu(n_words)
    其中 s_short 是该单元对图像 patch 的最大相似度，mu(n) 是该词数下的期望值
    （见 HP_CLAUSE_LEN_MU）。扣掉 mu 后各长度的期望收益平在 0，「把每个单元顶满
    max_words」不再有红利——raw s_short 随词数单调上升，配合固定绝对阈值会把
    padding 变成严格占优策略，策略因此退化成不写标点的 run-on 长句。

    返回 (spans, advs, s_vals)：spans 为 [(片段, 起, 终)]（规范化切分，见 clause_spans）；
    advs 为 ±1；s_vals 为打分单元的 s_adj（用于诊断阈值是否合适）。
    """
    spans = clause_spans(caption, max_words=max_words)
    if not spans or not img_path:
        return [], [], []
    texts = [frag.strip() for frag, _, _ in spans]
    dense, valid = fg.image_dense_features([img_path])
    dense, valid = dense[0], valid[0]
    tn = F.normalize(fg.encode_texts_walk(texts, walk_type=walk_type), p=2, dim=-1)
    dn = F.normalize(dense, p=2, dim=-1)
    sim = dn @ tn.T                                  # (P, N)
    sim = sim.masked_fill(~valid.unsqueeze(1), -1e9)
    raw = sim.max(dim=0).values.tolist()             # (N,) 每单元 raw 最大视觉支持
    s_vals = [v - _len_mu(_n_words(caption, a, b))
              for v, (_, a, b) in zip(raw, spans)]   # (N,) 长度校正后
    advs = [1.0 if v >= threshold else -1.0 for v in s_vals]
    return spans, advs, s_vals


# =============================================================================
# 句级 grounding reward（全图特征 + 参考模型基线）
# 对单图生成的 caption 按句号切句，每句算「全图相似度」s_glob：
#   s_glob = cos(句向量, 整图全局特征)   （FG-CLIP pooler 输出，方案B）
# 再减去参考模型（同基座）对同图 caption 的句级全图相似度均值 mu_ref：
#   adv_i = s_glob(sent_i) - mu_ref
# adv_i 作为该句成员 token 的逐 token advantage（连续、非零均值），整条 caption
# 的奖励 = 各句 adv 之和。与 clause 模式的差别：打分粒度是「句号切分」而非阈值 ±1，
# 奖励是连续值且以参考模型为基线（self-play relative reward），直接压制整体幻觉。
#
# 为什么用全图特征而非 max-patch：全量评测显示方案B（全图 pooler 特征）在 AUC、
# Cohen's d、F1 上全面领先 max-patch（方案A），且对 walk 类型更鲁棒（见
# support_vs_global_report.md）。mu_ref 提供「同图同基座」的绝对基线，使
# s_glob 的量纲在同一图内可比，逐句落后于参考模型的句子被压低。
# =============================================================================

_SENT_END = re.compile(r"\.(?=\s|$)")   # 句界：句号后接空白或串尾


def sentence_spans(caption, min_words=3):
    """按句号切分 caption，返回 [(原始片段, 起点, 终点)]（字符偏移均指向原 caption）。

    仅以句号（`.` 后接空白或串尾）为句界，尾段（无句号）作为独立句。相邻过短
    片段（< min_words 词）并入前一句（首句过短则并入后一句），避免 "a. b. c."
    碎片各自占一个打分单元、稀释句级信号。

    与 clause_spans 一样只重划边界、不改写文本，故 span 可直接用于
    map_clause_adv_to_tokens 的字符对齐。
    """
    caption = caption or ""
    bounds, start = [], 0
    for m in _SENT_END.finditer(caption):
        end = m.end()                                   # 含句号
        if caption[start:end].strip():
            bounds.append((start, end))
        start = end
    if caption[start:].strip():                         # 尾段（无句号）
        bounds.append((start, len(caption)))
    if not bounds:
        return []

    units = []
    for a, b in bounds:
        if units and _n_words(caption, a, b) < min_words:
            units[-1] = (units[-1][0], b)               # 短碎片并入前一句
        else:
            units.append((a, b))
    if len(units) >= 2 and _n_words(caption, *units[0]) < min_words:
        units[1] = (units[0][0], units[1][1])           # 首句过短则并入后一句
        units.pop(0)
    return [(caption[a:b], a, b) for a, b in units]


def _ref_sentence_mean(fg, img_feat, ref_caption, walk_type="short", min_words=3):
    """参考模型 caption 的句级全图相似度均值 mu_ref（无参考/无有效句 -> 0.0）。"""
    if not ref_caption:
        return 0.0
    sents = [frag.strip() for frag, _, _ in sentence_spans(ref_caption, min_words=min_words)]
    sents = [s for s in sents if s]
    if not sents:
        return 0.0
    tn = F.normalize(fg.encode_texts_walk(sents, walk_type=walk_type), p=2, dim=-1)
    return float((tn @ img_feat).mean().item())


def compute_sentence_adv(fg, img_path, caption, ref_captions=None, ref_key=None,
                         walk_type="short", min_words=3):
    """单图 caption 的句级连续优势：adv_i = s_glob(sent_i) - mu_ref。

    - s_glob：句子向量 × 整图全局特征（FG-CLIP pooler，方案B）的余弦。
    - mu_ref：参考模型（同基座）对同图 caption 的句级全图相似度均值；无参考时取 0。
    返回 (spans, advs, s_vals, mu_ref)：spans 见 sentence_spans；advs 为连续句级奖励；
    s_vals 为各句 s_glob（诊断用）；mu_ref 为所用基线。
    """
    spans = sentence_spans(caption, min_words=min_words)
    if not spans or not img_path:
        return [], [], [], float("nan")
    sents = [frag.strip() for frag, _, _ in spans]
    img_feat = fg.encode_image_paths([img_path])[0]                  # (D,) 已归一化
    tn = F.normalize(fg.encode_texts_walk(sents, walk_type=walk_type), p=2, dim=-1)
    s_vals = (tn @ img_feat).tolist()                               # (N,) 每句全图相似度
    mu_ref = _ref_sentence_mean(
        fg, img_feat, (ref_captions or {}).get(ref_key), walk_type, min_words)
    advs = [v - mu_ref for v in s_vals]
    return spans, advs, s_vals, mu_ref


def map_clause_adv_to_tokens(tokenizer, resp_ids, caption, spans, advs):
    """按字符偏移把子句优势映射到 token 区间，返回 (adv (len(resp_ids),), stats)。

    长度取 resp_ids 实际宽度（而非 --max-new-tokens）：generate 在整批提前命中 EOS
    时返回序列会短于 max_new，adv 必须与 logprob 的宽度严格一致。

    做法：用「解码前缀长度」建立 token->字符右端点表 ends（ends[j] = 前 j+1 个
    token 解码后的字符长度），子句 [a,b) 覆盖的 token 区间即
        lo = #{j: ends[j] <= a}, hi = #{j: ends[j-1] < b}
    与 caption 由同一 decode 产生，严格同源，规避「重新 tokenize + 子序列匹配」的
    前导空格丢失、重复子句、往返不幂等等问题。
    """
    full = [int(t) for t in resp_ids if int(t) != tokenizer.pad_token_id]
    idxs = [k for k, t in enumerate(resp_ids) if int(t) != tokenizer.pad_token_id]
    adv = torch.zeros(len(resp_ids), dtype=torch.float32)
    stats = {"n_clauses": 0, "n_matched": 0, "match_rate": 0.0, "coverage": 0.0,
             "pos_frac": 0.0, "avg_words": 0.0}
    if not full:
        return adv, stats

    ends = [len(tokenizer.decode(full[:j + 1], skip_special_tokens=True))
            for j in range(len(full))]
    n_tok = len(full)
    matched = 0
    for (_, a, b), r_ in zip(spans, advs):
        lo = bisect_right(ends, a)
        hi = bisect_left(ends, b) + 1
        lo, hi = max(0, min(lo, n_tok)), max(0, min(hi, n_tok))
        if hi <= lo:
            continue
        adv[idxs[lo]: idxs[hi - 1] + 1] = r_
        matched += 1

    covered = int((adv != 0).sum().item())
    n_clauses = len(spans)
    stats.update(
        n_clauses=n_clauses,
        n_matched=matched,
        match_rate=matched / n_clauses if n_clauses else 0.0,
        coverage=covered / len(idxs) if idxs else 0.0,
        pos_frac=(sum(1 for v in advs if v > 0) / n_clauses) if n_clauses else 0.0,
        avg_words=(sum(len(frag.split()) for frag, _, _ in spans) / n_clauses)
        if n_clauses else 0.0,
    )
    return adv, stats


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
    print(f"  reward : mode={args.reward_mode}")
    if args.reward_mode == "clause":
        print(f"  clause : walk={args.clause_walk} threshold(校正后 s_adj)={args.clause_threshold} "
              f"max_words={args.clause_max_words}（mu(n) 见 HP_CLAUSE_LEN_MU；逐 token advantage=±1）")
    elif args.reward_mode == "sentence":
        print(f"  sentence: walk={args.sentence_walk} min_words={args.sentence_min_words} "
              f"（句号切句 -> 全图特征相似度 - 参考模型句级均值 mu_ref；逐 token advantage）")
    else:
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
            if args.reward_mode in ("clause", "sentence"):
                # 子句级 / 句级奖励只用主图，不依赖网格子图
                if not main_img:
                    print(f"[skip] gid={g['gid']} 无主图，跳过")
                    continue
            elif len(regions) < 2:
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
            if args.reward_mode in ("clause", "sentence"):
                # 子句级 / 句级奖励：只训主图单行（K=1），不使用区域竞争/负例。
                # 句级模式额外取主图行的参考模型 caption（同基座）算句级均值 mu_ref。
                main_ref = None
                if args.reward_mode == "sentence":
                    rc = _ref_captions_for_paths(
                        model_captions, g["gid"], [main_img], ref_weights.keys())
                    main_ref = rc[0] if rc else {}
                metas.append({
                    "gid": g["gid"], "main_img": main_img,
                    "region_paths": [main_img], "neg_pool": [],
                    "metric_main_path": None,
                    "reward_paths": [main_img], "ref_captions": main_ref,
                    "row0": row0, "n_rows": 1, "K": 1,
                })
                row0 += 1
                gen_pixels_list.append(process_image(image_processor, main_img))
                continue
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

        # ---- 3) 逐组计算奖励 + advantage（组间严格独立）----
        for m in metas:
            caps = captions_all[m["row0"]: m["row0"] + m["n_rows"]]
            resp = responses_all[m["row0"]: m["row0"] + m["n_rows"]]

            if args.reward_mode == "clause":
                # 子句级阈值奖励：caption 用 decode(full) 的原始形态（不 strip），
                # 与 map_clause_adv_to_tokens 的字符偏移表严格同源。
                full = [int(t) for t in resp[0] if int(t) != tokenizer.pad_token_id]
                caption = tokenizer.decode(full, skip_special_tokens=True)
                spans, advs, s_vals = compute_clause_adv(
                    fg, m["main_img"], caption,
                    walk_type=args.clause_walk, threshold=args.clause_threshold,
                    max_words=args.clause_max_words)
                adv_token, cstats = map_clause_adv_to_tokens(
                    tokenizer, resp[0], caption, spans, advs)
                adv_token = adv_token.to(device)
                m.update(resp=resp, caps=caps, r=adv_token, reg_mean=float("nan"),
                         adv_token=adv_token, r_main=float("nan"),
                         clause_stats=cstats, clause_svals=s_vals)
                continue

            if args.reward_mode == "sentence":
                # 句级连续奖励：reward_i = s_glob(sent_i) - mu_ref（mu_ref 为参考模型
                # 句级均值）。caption 用 decode(full) 原始形态（不 strip），与
                # map_clause_adv_to_tokens 的字符偏移表严格同源。
                full = [int(t) for t in resp[0] if int(t) != tokenizer.pad_token_id]
                caption = tokenizer.decode(full, skip_special_tokens=True)
                spans, advs, s_vals, mu_ref = compute_sentence_adv(
                    fg, m["main_img"], caption,
                    ref_captions=m["ref_captions"], ref_key=ref_key,
                    walk_type=args.sentence_walk, min_words=args.sentence_min_words)
                adv_token, sstats = map_clause_adv_to_tokens(
                    tokenizer, resp[0], caption, spans, advs)
                adv_token = adv_token.to(device)
                m.update(resp=resp, caps=caps, r=adv_token, reg_mean=float("nan"),
                         adv_token=adv_token, r_main=float("nan"),
                         clause_stats=sstats, sentence_svals=s_vals, sentence_mu=mu_ref)
                continue

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

            if args.reward_mode in ("clause", "sentence"):
                # 逐 token advantage：loss = -Σ_t adv_t · logπ(o_t)（pad 位 logp 恒 0）
                loss_g = -(m["adv_token"].unsqueeze(0) * lp).sum() / n_train
            else:
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
            clause_str = ""
            if args.reward_mode == "clause":
                # 映射验收指标：match=子句成功映射比例（应≈1）；coverage=被覆盖的非 pad
                # token 占比；ncl=平均打分单元数；avgw=单元平均词数（长段已按
                # --clause-max-words 强制再切，run-on 时 ncl 应随 avgw 一起受控）；
                # pos_frac=判 +1 的单元占比（长期偏离 0.4~0.6 说明阈值失配）
                cst = [m["clause_stats"] for m in metas]
                n = len(cst)
                clause_str = (
                    f" match={sum(s['match_rate'] for s in cst) / n:.3f}"
                    f" cov={sum(s['coverage'] for s in cst) / n:.3f}"
                    f" ncl={sum(s['n_clauses'] for s in cst) / n:.2f}"
                    f" avgw={sum(s['avg_words'] for s in cst) / n:.1f}"
                    f" pos_frac={sum(s['pos_frac'] for s in cst) / n:.3f}")
            elif args.reward_mode == "sentence":
                # 句级诊断：nsen=平均句数；avgw=句平均词数（run-on 时 nsen 塌成 1）；
                # mu_ref=各图参考模型句级均值（基线水平，随图/模型而定）
                cst = [m["clause_stats"] for m in metas]
                n = len(cst)
                mus = [m["sentence_mu"] for m in metas if m["sentence_mu"] == m["sentence_mu"]]
                mu_str = f"{sum(mus) / len(mus):.3f}" if mus else "nan"
                clause_str = (
                    f" match={sum(s['match_rate'] for s in cst) / n:.3f}"
                    f" cov={sum(s['coverage'] for s in cst) / n:.3f}"
                    f" nsen={sum(s['n_clauses'] for s in cst) / n:.2f}"
                    f" avgw={sum(s['avg_words'] for s in cst) / n:.1f}"
                    f" mu_ref={mu_str}")
            print(
                f"[{step}/{args.steps}] G={len(metas)} B={n_train} loss={loss_val:.4f} "
                f"r_mean={rs.mean().item():.4f} r_min={rs.min().item():.4f} r_max={rs.max().item():.4f} "
                f"r_std={sum(stds) / len(stds):.4f} "
                f"r_main={r_main_str}{kl_str}{reg_str}{clause_str}{deg_str}"
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
    p.add_argument("--question", default=HP_QUESTION, help="训练 rollout 的生成 prompt")
    p.add_argument("--steps", type=int, default=HP_STEPS,
                   help="训练步数（优化器更新次数；每步消耗 groups-per-step 个数据组）")
    p.add_argument("--groups-per-step", type=int, default=HP_GROUPS_PER_STEP,
                   help="每次更新同时处理的数据组数（数据并行维度；组间独立，"
                        "reward/advantage 仍组内计算；G 个组拼接为一个大 rollout batch，"
                        "前向反向按组累积梯度，显存占用与 G=1 相同）")
    p.add_argument("--lr", type=float, default=HP_LR, help="投影层学习率")
    p.add_argument("--reward-mode", choices=["rcgr", "clause", "sentence"], default=HP_REWARD_MODE,
                   help="奖励模式：rcgr=组级区域竞争接地奖励（默认）；"
                        "clause=子句级视觉支持阈值奖励（只训主图，逐 token advantage=±1）；"
                        "sentence=句级全图特征相似度 - 参考模型句级均值（只训主图，逐 token 连续 advantage）")
    p.add_argument("--sentence-walk", choices=["short", "box", "long"], default=HP_SENTENCE_WALK,
                   help="句级奖励的 FG-CLIP 文本 walk 类型（句级推荐 short）")
    p.add_argument("--sentence-min-words", type=int, default=HP_SENTENCE_MIN_WORDS,
                   help="句级奖励的最小句词数：句号切句后 < 该值的碎片并入相邻句，"
                        "避免 'a. b. c.' 碎片各自占一个打分单元")
    p.add_argument("--clause-walk", choices=["short", "box", "long"], default=HP_CLAUSE_WALK,
                   help="子句级奖励的 FG-CLIP 文本 walk 类型（子句级标定推荐 short）")
    p.add_argument("--clause-threshold", type=float, default=HP_CLAUSE_THRESHOLD,
                   help="子句级奖励阈值（长度校正口径）：s_adj = s_short - mu(n_words) >= 阈值"
                        " 记 +1，否则 -1。mu(n) 见 HP_CLAUSE_LEN_MU（桶内全体均值，"
                        "消除「顶满 max_words」的 padding 红利）")
    p.add_argument("--clause-max-words", type=int, default=HP_CLAUSE_MAX_WORDS,
                   help="子句级奖励规范切分的打分单元词数上限：标点初切后 > 该值的段按"
                        "词边界强制再切（拆掉「不写标点/合并子句」的占优逃逸路径）。"
                        "默认 16，落在子句可分化性最优的 10~20 词桶中部")
    p.add_argument("--alpha", type=float, default=HP_ALPHA, help="RCGR 跨区域竞争项系数")
    p.add_argument("--beta", type=float, default=HP_BETA,
                   help="RCGR 幻觉负样本基线项系数（整图级 neg_captions；设为 0 取消 β 项，消融）")
    p.add_argument("--ref7-weight", type=float, default=HP_REF7_WEIGHT,
                   help="原模型(7B) caption 难负样本竞争项系数 gamma（区域级；"
                        "仅当训练策略为 7B 时生效，<=0 关闭）")
    p.add_argument("--ref13-weight", type=float, default=HP_REF13_WEIGHT,
                   help="原模型(13B) caption 难负样本竞争项系数 gamma（区域级；"
                        "仅当训练策略为 13B 时生效，<=0 关闭）")
    p.add_argument("--ref-json", default="model_captions.json",
                   help="根级原模型 caption json（由 data/generate_model_captions.py 生成）")
    p.add_argument("--kl-weight", type=float, default=HP_KL_WEIGHT,
                   help="对原模型（训练前 mm_projector）的 KL 正则权重，仅主图行；<=0 关闭。"
                        "过大(>=1)会压死策略、奖励长期不涨，建议 0.05~0.2")
    p.add_argument("--neg-mode", choices=["first", "worst"], default=HP_NEG_MODE,
                   help="缺席难负例池用法：first=固定取池首条；worst=逐区域取最难一条")
    p.add_argument("--reg-weight", type=float, default=HP_REG_WEIGHT,
                   help="区域结构对齐正则权重（reward shaping；<=0 关闭）")
    p.add_argument("--use-main-image", action="store_true",
                   help="把整图（下采样主图）并入组内作为额外视角参与 reward/优化")
    p.add_argument("--temperature", type=float, default=HP_TEMPERATURE)
    p.add_argument("--max-new-tokens", type=int, default=HP_MAX_NEW_TOKENS)
    p.add_argument("--max-grad-norm", type=float, default=HP_MAX_GRAD_NORM)
    p.add_argument("--bits", type=int, default=HP_BITS, choices=[4, 8, 16],
                   help="量化位数（8GB 显存建议 4）")
    p.add_argument("--fp16", action="store_true", help="使用 fp16")
    p.add_argument("--bf16", action="store_true", help="使用 bf16")
    p.add_argument("--log-interval", type=int, default=HP_LOG_INTERVAL)
    p.add_argument("--save-interval", type=int, default=HP_SAVE_INTERVAL)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
