#!/bin/bash
# MVPG RL 训练启动脚本（真实 GPU 运行，需在沙箱外/autodl 实例上执行）
set -e

# ===== 路径 =====
PY=/root/miniconda3/envs/mvpg/bin/python
DATA_DIR=/root/autodl-tmp/data/SA1B_5_threshold
MODEL_DIR=/root/autodl-tmp/cache/llava-v1.5-7b
FGCLIP_DIR=/root/autodl-tmp/cache/fgclip2-base-patch16
OUTPUT_DIR=/root/autodl-tmp/output/mvpg_ckpt
# steps 为优化器更新次数；每步消耗 GROUPS_PER_STEP 个数据组（共 4848 组）
# GROUPS_PER_STEP=4 时 1212 步 ≈ 1 epoch
STEPS=${STEPS:-5000}
GROUPS_PER_STEP=${GROUPS_PER_STEP:-4}

# 建议先冒烟：STEPS=2 MAXNEW=8 bash run/train_mvpg.sh
# 默认对齐 ref caption 的生成预算（128），避免策略输出被截断导致难负样本项带负偏置
MAXNEW=${MAXNEW:-128}
# KL 正则权重（仅主图行）：过大(>=1)会压死策略、奖励长期不涨，建议 0.05~0.2
KL_WEIGHT=${KL_WEIGHT:-0.1}
# RCGR 幻觉负样本基线项系数（β，整图级 neg_captions）：设为 0 取消 β 项（消融）
BETA=${BETA:-0.5}
# 原模型 caption 难负样本竞争项（区域级）：训练脚本按 MODEL_DIR 自动选择同基座 ref
# （7B 训 7B、13B 训 13B），只使用对应的 REF*_WEIGHT；设为 0 可关闭（消融）
REF7_WEIGHT=${REF7_WEIGHT:-0.5}
REF13_WEIGHT=${REF13_WEIGHT:-0.5}

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
    --lr 1e-4 \
    --temperature 1.0 \
    --kl-weight "$KL_WEIGHT" \
    --beta "$BETA" \
    --ref7-weight "$REF7_WEIGHT" \
    --ref13-weight "$REF13_WEIGHT" \
    --log-interval 5 \
    --save-interval 50
