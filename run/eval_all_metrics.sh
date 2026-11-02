#!/usr/bin/env bash
set -euo pipefail
# Run from repository root after preparing benchmark datasets and dependencies.
# MODEL_LORA_BASE is the current claim-feedback run's final/ adapter directory.
: "${MODEL_BASE:?Set MODEL_BASE to the local LLaVA-7B or LLaVA-13B checkpoint}"
: "${MODEL_LORA_BASE:?Set MODEL_LORA_BASE to the current run final adapter directory}"
: "${IMAGE_FOLDER_LB:?Set IMAGE_FOLDER_LB to COCO train2017 for LLaVA-Bench}"
: "${IMAGE_FOLDER_POPE:?Set IMAGE_FOLDER_POPE to COCO val2014 for POPE}"
: "${IMAGE_DIR_AMBER:?Set IMAGE_DIR_AMBER to the AMBER image directory}"
: "${MMHAL_DATA_ROOT:?Set MMHAL_DATA_ROOT to the MMHal response_template.json/images directory}"

MODEL_SUFFIX="${MODEL_SUFFIX:-claim_feedback_eval}"
EVAL_ROOT="${EVAL_ROOT:-./output/evaluation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
POPE_CAT="${POPE_CAT:-adversarial}"
GPT_MODEL="${GPT_MODEL:-gpt-4}"
RUN_GPT="${RUN_GPT:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENAI_ENDPOINT="${OPENAI_ENDPOINT:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export MMHAL_DATA_ROOT
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

case "$RUN_GPT" in
    0) ;;
    1)
        : "${OPENAI_API_KEY:?Set OPENAI_API_KEY for GPT scoring, or RUN_GPT=0 for generation/local metrics only}"
        : "${ANNOTATION_FILE:?Set ANNOTATION_FILE to the COCO annotations directory for Object HalBench}"
        ;;
    *) echo "RUN_GPT must be 0 or 1" >&2; exit 2 ;;
esac
case "$POPE_CAT" in
    random|popular|adversarial) ;;
    *) echo "POPE_CAT must be random, popular or adversarial" >&2; exit 2 ;;
esac
if [[ ! -f "$MODEL_LORA_BASE/adapter_config.json" ]] ||
   [[ ! -f "$MODEL_LORA_BASE/adapter_model.bin" && ! -f "$MODEL_LORA_BASE/adapter_model.safetensors" ]]; then
    echo "MODEL_LORA_BASE must contain adapter_config.json and adapter weights" >&2
    exit 2
fi

# Stage 1: MMHal-Bench generation and optional GPT scoring.
OUTPUT_DIR="$EVAL_ROOT/mmhal_bench"
"$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa_mmhal.py \
    --model-path "$MODEL_BASE" \
    --use-qlora True \
    --qlora-path "$MODEL_LORA_BASE" \
    --temperature 0.0 \
    --answers-file "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
    --image_aspect_ratio pad \
    --test-prompt ''
if [[ "$RUN_GPT" == 1 ]]; then
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/eval_gpt_mmhal.py \
        --response "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
        --evaluation "$OUTPUT_DIR/review/review-file-$MODEL_SUFFIX.jsonl" \
        --gpt-model "$GPT_MODEL"
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/summarize_gpt_mmhal.py \
        --evaluation "$OUTPUT_DIR/review/review-file-$MODEL_SUFFIX.jsonl"
fi

# Stage 2: LLaVA-Bench generation and optional GPT scoring.
OUTPUT_DIR="$EVAL_ROOT/llava_bench"
"$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa.py \
    --model-path "$MODEL_BASE" \
    --use-qlora True \
    --qlora-path "$MODEL_LORA_BASE" \
    --question-file ./eval_llava_rlhf_coco/llava/qa90_questions.jsonl \
    --image-folder "$IMAGE_FOLDER_LB" \
    --answers-file "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
    --image_aspect_ratio pad \
    --test-prompt ''
if [[ "$RUN_GPT" == 1 ]]; then
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/eval_gpt_review_visual.py \
        --question ./eval_llava_rlhf_coco/llava/qa90_questions.jsonl \
        --context ./eval_llava_rlhf_coco/table/caps_boxes_coco2014_val_80.jsonl \
        --answer-list \
        ./eval_llava_rlhf_coco/llava/qa90_gpt4_answer.jsonl \
        "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
        --rule ./eval_llava_rlhf_coco/table/rule.json \
        --output "$OUTPUT_DIR/review/review-file-$MODEL_SUFFIX.jsonl"
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/summarize_gpt_review.py \
        -d "$OUTPUT_DIR/review/" -f "review-file-$MODEL_SUFFIX.jsonl"
fi

# Stage 3: POPE local metrics (choose a category with POPE_CAT).
OUTPUT_DIR="$EVAL_ROOT/pop_bench"
"$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa.py \
    --short_eval True \
    --model-path "$MODEL_BASE" \
    --use-qlora True \
    --qlora-path "$MODEL_LORA_BASE" \
    --question-file "./eval_llava_rlhf_coco/pope/coco_pope_$POPE_CAT.jsonl" \
    --image-folder "$IMAGE_FOLDER_POPE" \
    --answers-file "$OUTPUT_DIR/answer/answer-file-${MODEL_SUFFIX}_$POPE_CAT.jsonl" \
    --image_aspect_ratio pad \
    --test-prompt '\nAnswer the question using a single word or phrase.'
"$PYTHON_BIN" ./eval_llava_rlhf_coco/summarize_eval_pope.py \
    --answers-file "$OUTPUT_DIR/answer/answer-file-${MODEL_SUFFIX}_$POPE_CAT.jsonl" \
    --label-file "./eval_llava_rlhf_coco/pope/coco_pope_$POPE_CAT.jsonl"

# Stage 4: AMBER local metrics. Prepare en_core_web_lg before running.
OUTPUT_DIR="$EVAL_ROOT/AMBER_bench"
"$PYTHON_BIN" ./eval_llava_rlhf_coco/AMBER_generate.py \
    --model-path "$MODEL_BASE" \
    --use-qlora True \
    --qlora-path "$MODEL_LORA_BASE" \
    --temperature 0.0 \
    --answers-file "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
    --image-file "$IMAGE_DIR_AMBER" \
    --image_aspect_ratio pad \
    --test-prompt ''
"$PYTHON_BIN" ./eval_llava_rlhf_coco/AMBER_eval.py \
    --inference_data "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
    --evaluation_type 'g'

# Stage 5: Object HalBench generation and optional GPT-assisted scoring.
# Prepare jsonlines and en_core_web_trf in the evaluation environment beforehand.
OUTPUT_DIR="$EVAL_ROOT/objhal_bench"
"$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa_objectHal.py \
    --model-path "$MODEL_BASE" \
    --use-qlora True \
    --qlora-path "$MODEL_LORA_BASE" \
    --question-file ./eval_llava_rlhf_coco/object_hal/obj_halbench_300_with_image.jsonl \
    --answers-file "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
    --image_aspect_ratio pad \
    --test-prompt ''
if [[ "$RUN_GPT" == 1 ]]; then
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/eval_gpt_obj_halbench.py \
        --coco_path "$ANNOTATION_FILE" \
        --answers_file "$OUTPUT_DIR/answer/answer-file-$MODEL_SUFFIX.jsonl" \
        --evaluation_file "$OUTPUT_DIR/review/review-file-${MODEL_SUFFIX}_eval.jsonl" \
        --gpt-model "$GPT_MODEL" \
        --use_gpt
else
    echo "GPT scoring skipped (RUN_GPT=0); generated responses are not scored MMHal/LLaVA/Object HalBench results."
fi
