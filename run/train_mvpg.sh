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
MAXNEW=${MAXNEW:-64}

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
    --log-interval 5 \
    --save-interval 50
