#!/bin/bash
# MVPG RL 训练启动脚本（真实 GPU 运行，需在沙箱外/autodl 实例上执行）
#
# 超参数的唯一事实来源是 mvpg_run/run_mvpg.py 顶部的 HP_* 常量块；本脚本只负责
# 指定路径、步数与少量开关，不再重复给 --lr / --temperature / 各奖励系数等默认值
# （避免两处默认值分叉）。需要临时覆盖时直接加对应命令行参数即可。
set -e

# ===== 路径 =====
PY=/root/miniconda3/envs/mvpg/bin/python
DATA_DIR=/root/autodl-tmp/data/SA1B_5_threshold
MODEL_DIR=/root/autodl-tmp/cache/llava-v1.5-7b
FGCLIP_DIR=/root/autodl-tmp/cache/fgclip2-base-patch16
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-tmp/output/mvpg_ckpt}

# ===== 规模 / 开关 =====
# steps 为优化器更新次数；每步消耗 GROUPS_PER_STEP 个数据组（共 4848 组）
# GROUPS_PER_STEP=4 时 1212 步 ≈ 1 epoch
STEPS=${STEPS:-5000}
GROUPS_PER_STEP=${GROUPS_PER_STEP:-4}
# 建议先冒烟：STEPS=2 MAXNEW=8 bash run/train_mvpg.sh
MAXNEW=${MAXNEW:-128}
# 奖励模式：rcgr=组级区域竞争（默认）；clause=子句级视觉支持阈值奖励（只训主图）；
# sentence=句级全图特征相似度 - 参考模型句级均值（只训主图）
REWARD_MODE=${REWARD_MODE:-rcgr}
# KL 正则权重：留空则用 run_mvpg.py 的 HP_KL_WEIGHT 默认值；KL_WEIGHT=0 可关闭 KL 项
KL_WEIGHT=${KL_WEIGHT:-}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"
echo "=== MVPG 训练 ==="
echo "model : $MODEL_DIR"
echo "data  : $DATA_DIR"
echo "fgclip: $FGCLIP_DIR"
echo "steps : $STEPS  groups/step: $GROUPS_PER_STEP  max_new_tokens: $MAXNEW"
echo "reward: mode=$REWARD_MODE  (其余超参数见 run_mvpg.py 的 HP_* 常量块)"
echo "================="

$PY mvpg_run/run_mvpg.py \
    --image-dir "$DATA_DIR" \
    --model-name "$MODEL_DIR" \
    --fgclip-root "$FGCLIP_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --groups-per-step "$GROUPS_PER_STEP" \
    --max-new-tokens "$MAXNEW" \
    --bf16 \
    --reward-mode "$REWARD_MODE" \
    ${KL_WEIGHT:+--kl-weight "$KL_WEIGHT"}
