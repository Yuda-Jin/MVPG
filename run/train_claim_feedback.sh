#!/usr/bin/env bash
set -euo pipefail
# Run from repository root; each run must use a new output directory.
: "${MODEL_NAME:?Set MODEL_NAME to a local patched LLaVA checkpoint}"
: "${FEEDBACK_JSONL:?Set FEEDBACK_JSONL to the verified fixed feedback pool}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}"
python -m mvpg_run.claim_feedback train \
  --model-name "$MODEL_NAME" \
  --input "$FEEDBACK_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --image-root "${IMAGE_ROOT:-.}" \
  --mode "${MODE:-modulated}" \
  --device "${DEVICE:-cuda}" \
  --precision "${PRECISION:-bf16}" \
  --lr "${LR:-0.001}" \
  --max-update-norm "${MAX_UPDATE_NORM:-0.001}" \
  --seed "${SEED:-42}" "$@"
