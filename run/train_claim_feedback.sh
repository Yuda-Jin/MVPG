#!/usr/bin/env bash
set -euo pipefail
# Run from repository root; each run must use a new output directory.
: "${MODEL_NAME:?Set MODEL_NAME to a local patched LLaVA checkpoint}"
: "${FEEDBACK_JSONL:?Set FEEDBACK_JSONL to the verified fixed feedback pool}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}"
EXTRA=()
if [[ -n "${RESUME_FROM:-}" ]]; then EXTRA+=(--resume-from "$RESUME_FROM"); fi
if [[ "${ALLOW_LEGACY_RESUME:-0}" == 1 ]]; then EXTRA+=(--allow-legacy-resume); fi
if [[ "${TENSORBOARD:-0}" == 1 ]]; then EXTRA+=(--tensorboard); fi
# Only explicit environment overrides should replace restored settings.
if [[ -n "${MODE:-}" ]]; then EXTRA+=(--mode "$MODE"); fi
if [[ -n "${PRECISION:-}" ]]; then EXTRA+=(--precision "$PRECISION"); fi
if [[ -n "${LR:-}" ]]; then EXTRA+=(--lr "$LR"); fi
if [[ -n "${MAX_UPDATE_NORM:-}" ]]; then EXTRA+=(--max-update-norm "$MAX_UPDATE_NORM"); fi
if [[ -n "${SEED:-}" ]]; then EXTRA+=(--seed "$SEED"); fi
"${PYTHON_BIN:-python}" -m mvpg_run.claim_feedback train \
  --model-name "$MODEL_NAME" \
  --input "$FEEDBACK_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --image-root "${IMAGE_ROOT:-.}" \
  --device "${DEVICE:-cuda}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-500}" \
  --log-every "${LOG_EVERY:-10}" \
  "${EXTRA[@]}" "$@"
