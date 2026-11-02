#!/usr/bin/env bash
# Run from the repository root. Complete stages are reused after checking output
# IDs/counts and the adapter/data manifest. Failures do not stop other benchmarks.
# Partial generation files are preserved before restarting that stage.
set -uo pipefail
# Local service credentials are ignored by Git. Do not enable shell tracing.
if [[ -f "${ENV_FILE:-.env}" ]]; then
    set -a
    source "${ENV_FILE:-.env}"
    set +a
fi
: "${MODEL_BASE:?Set MODEL_BASE to the local LLaVA checkpoint}"
: "${MODEL_LORA_BASE:?Set MODEL_LORA_BASE to the trained final adapter directory}"
: "${IMAGE_FOLDER_LB:?Set IMAGE_FOLDER_LB to the qa90 COCO image directory}"
: "${IMAGE_FOLDER_POPE:?Set IMAGE_FOLDER_POPE to COCO val2014}"
: "${IMAGE_DIR_AMBER:?Set IMAGE_DIR_AMBER to the AMBER image directory}"
: "${MMHAL_DATA_ROOT:?Set MMHAL_DATA_ROOT to the MMHal template/images directory}"

MODEL_SUFFIX="${MODEL_SUFFIX:-claim_feedback_eval}"
EVAL_ROOT="${EVAL_ROOT:-./output/evaluation}"
PYTHON_BIN="${PYTHON_BIN:-python}"
POPE_CAT="${POPE_CAT:-all}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RUN_GPT="${RUN_GPT:-auto}"
export GPT_MODEL="${GPT_MODEL:-gpt-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MMHAL_DATA_ROOT
# An empty base_url differs from an absent base_url in the OpenAI SDK.
if [[ -n "${OPENAI_ENDPOINT:-${OPENAI_BASE_URL:-}}" ]]; then
    export OPENAI_ENDPOINT="${OPENAI_ENDPOINT:-$OPENAI_BASE_URL}"
else
    unset OPENAI_ENDPOINT
fi
case "$MODEL_SUFFIX" in *[!a-zA-Z0-9_.-]*|'') echo "Invalid MODEL_SUFFIX" >&2; exit 2 ;; esac
case "$RUN_GPT" in auto|0|1) ;; *) echo "RUN_GPT must be auto, 0 or 1" >&2; exit 2 ;; esac
case "$POPE_CAT" in
    all) POPE_CATEGORIES=(random popular adversarial) ;;
    random|popular|adversarial) POPE_CATEGORIES=("$POPE_CAT") ;;
    *) echo "POPE_CAT must be all, random, popular or adversarial" >&2; exit 2 ;;
esac

SUMMARY_TOOL=./run/summarize_eval_results.py
LOG_DIR="$EVAL_ROOT/logs/$MODEL_SUFFIX"
STATUS_FILE="$LOG_DIR/stages.tsv"
mkdir -p "$LOG_DIR" || exit 2
SOURCES=(./eval_llava_rlhf_coco/llava/qa90_questions.jsonl
    ./eval_llava_rlhf_coco/AMBER_data/query/query_all.json
    ./eval_llava_rlhf_coco/object_hal/obj_halbench_300_with_image.jsonl
    "$MMHAL_DATA_ROOT/response_template.json")
for category in "${POPE_CATEGORIES[@]}"; do
    SOURCES+=("./eval_llava_rlhf_coco/pope/coco_pope_$category.jsonl")
done
"$PYTHON_BIN" "$SUMMARY_TOOL" --prepare-run "$LOG_DIR/manifest.json" \
    --model "$MODEL_BASE" --adapter "$MODEL_LORA_BASE" \
    --judge-model "$GPT_MODEL" --sources "${SOURCES[@]}" || exit 2
if [[ ! -f "$STATUS_FILE" ]]; then printf 'stage\tstatus\texit_code\tmessage\tlog\n' > "$STATUS_FILE"; fi

record_status() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$LOG_DIR/$1.log" >> "$STATUS_FILE"
    printf '[%s] %s: %s\n' "$1" "$2" "$4"
}
was_completed() {
    "$PYTHON_BIN" "$SUMMARY_TOOL" --stage-completed "$STATUS_FILE" "$1"
}
valid() { "$PYTHON_BIN" "$SUMMARY_TOOL" --validate "$1" "$2" "$3" >/dev/null 2>&1; }
preserve_partial() {
    if [[ -f "$1" ]]; then
        mv -- "$1" "$1.incomplete.$(date +%Y%m%d_%H%M%S).$$" || return 1
    fi
}
preserve_reviews() {
    local review suffix
    for review in "$@"; do
        for suffix in '' '.meta.json' '.gpt_objects.jsonl'; do
            preserve_partial "$review$suffix" || return 1
        done
    done
}
generate() {
    local stage_name="$1" kind="$2" output="$3" source="$4" code
    shift 4
    if valid "$kind" "$output" "$source"; then
        record_status "$stage_name" completed 0 "reused complete output"
        return 0
    fi
    # Scores from an earlier answer file cannot survive its regeneration.
    if [[ "$stage_name" == object_generate ]]; then
        record_status object_score_local blocked 2 "answers are being regenerated"
        record_status object_score_gpt blocked 2 "answers are being regenerated"
        preserve_reviews "$EVAL_ROOT/objhal_bench/review-local/review-file-$MODEL_SUFFIX.jsonl" \
            "$EVAL_ROOT/objhal_bench/review-gpt/review-file-$MODEL_SUFFIX.jsonl" || return 1
    else
        record_status "${stage_name%_generate}_score" blocked 2 "answers are being regenerated"
        if [[ "$stage_name" == llava_generate ]]; then
            record_status llava_review blocked 2 "answers are being regenerated"
            preserve_reviews "$EVAL_ROOT/llava_bench/review/review-file-$MODEL_SUFFIX.jsonl" || return 1
        elif [[ "$stage_name" == mmhal_generate ]]; then
            preserve_reviews "$EVAL_ROOT/mmhal_bench/review/review-file-$MODEL_SUFFIX.jsonl" || return 1
        fi
    fi
    preserve_partial "$output" || { record_status "$stage_name" failed 2 "could not preserve incomplete output"; return 1; }
    printf '\n--- %s ---\n' "$(date -Is)" >> "$LOG_DIR/$stage_name.log"
    "$@" >> "$LOG_DIR/$stage_name.log" 2>&1
    code=$?
    if [[ "$code" == 0 ]] && valid "$kind" "$output" "$source"; then
        record_status "$stage_name" completed 0 "complete output validated"
        return 0
    fi
    [[ "$code" != 0 ]] || code=3
    record_status "$stage_name" failed "$code" "generation failed or output IDs/counts are incomplete"
    return 1
}
score() {
    local stage_name="$1" code resume_valid=1
    shift
    if [[ "$stage_name" == llava_review ]] && ! valid qa_review "$REVIEW" "$QUESTIONS"; then
        resume_valid=0
    fi
    if [[ "$resume_valid" == 1 ]] && was_completed "$stage_name" && "$PYTHON_BIN" "$SUMMARY_TOOL" \
        --validate-score "$stage_name" "$LOG_DIR/$stage_name.log"; then
        record_status "$stage_name" completed 0 "reused completed score"
        return 0
    fi
    preserve_partial "$LOG_DIR/$stage_name.log" || return 1
    printf '\n--- %s ---\n' "$(date -Is)" >> "$LOG_DIR/$stage_name.log"
    "$@" >> "$LOG_DIR/$stage_name.log" 2>&1
    code=$?
    if [[ "$code" == 0 ]] && "$PYTHON_BIN" "$SUMMARY_TOOL" \
        --validate-score "$stage_name" "$LOG_DIR/$stage_name.log"; then
        record_status "$stage_name" completed 0 "scorer completed"
        return 0
    fi
    [[ "$code" != 0 ]] || code=3
    record_status "$stage_name" failed "$code" "scorer failed; see stage log"
    return 1
}
judge_available() {
    local stage_name="$1"
    if [[ "$RUN_GPT" == 0 ]]; then
        record_status "$stage_name" skipped 0 "RUN_GPT=0; no judge score"
        return 1
    fi
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        record_status "$stage_name" blocked 2 "OPENAI_API_KEY is absent; responses will still be generated"
        return 1
    fi
    return 0
}

COMMON=(--model-path "$MODEL_BASE" --use-qlora True --qlora-path "$MODEL_LORA_BASE"
    --temperature 0.0 --image_aspect_ratio pad)

OUT="$EVAL_ROOT/mmhal_bench/answer/answer-file-$MODEL_SUFFIX.jsonl"
if generate mmhal_generate mmhal "$OUT" "$MMHAL_DATA_ROOT/response_template.json" \
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa_mmhal.py "${COMMON[@]}" --answers-file "$OUT" --test-prompt ''; then
    if judge_available mmhal_score; then
        REVIEW="$EVAL_ROOT/mmhal_bench/review/review-file-$MODEL_SUFFIX.jsonl"
        score mmhal_score "$PYTHON_BIN" ./eval_llava_rlhf_coco/eval_gpt_mmhal.py \
            --response "$OUT" --evaluation "$REVIEW" --gpt-model "$GPT_MODEL"
    fi
else record_status mmhal_score blocked 2 "complete generated answers required"; fi

QUESTIONS=./eval_llava_rlhf_coco/llava/qa90_questions.jsonl
OUT="$EVAL_ROOT/llava_bench/answer/answer-file-$MODEL_SUFFIX.jsonl"
if generate llava_generate qa "$OUT" "$QUESTIONS" \
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa.py "${COMMON[@]}" --question-file "$QUESTIONS" \
    --image-folder "$IMAGE_FOLDER_LB" --answers-file "$OUT" --test-prompt '' --batch-size "$EVAL_BATCH_SIZE"; then
    if judge_available llava_score; then
        REVIEW="$EVAL_ROOT/llava_bench/review/review-file-$MODEL_SUFFIX.jsonl"
        if score llava_review "$PYTHON_BIN" ./eval_llava_rlhf_coco/eval_gpt_review_visual.py \
            --question "$QUESTIONS" --context ./eval_llava_rlhf_coco/table/caps_boxes_coco2014_val_80.jsonl \
            --answer-list ./eval_llava_rlhf_coco/llava/qa90_gpt4_answer.jsonl "$OUT" \
            --rule ./eval_llava_rlhf_coco/table/rule.json --output "$REVIEW"; then
            if valid qa_review "$REVIEW" "$QUESTIONS"; then
                score llava_score "$PYTHON_BIN" ./eval_llava_rlhf_coco/summarize_gpt_review.py \
                    -d "$(dirname "$REVIEW")" -f "$(basename "$REVIEW")"
            else record_status llava_score failed 3 "incomplete or invalid qa90 judge reviews"; fi
        else record_status llava_score blocked 2 "qa90 judge reviews failed"; fi
    fi
else record_status llava_score blocked 2 "complete generated answers required"; fi

for category in "${POPE_CATEGORIES[@]}"; do
    QUESTIONS="./eval_llava_rlhf_coco/pope/coco_pope_$category.jsonl"
    OUT="$EVAL_ROOT/pop_bench/answer/answer-file-${MODEL_SUFFIX}_$category.jsonl"
    if generate "pope_${category}_generate" pope "$OUT" "$QUESTIONS" \
        "$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa.py "${COMMON[@]}" --short_eval True \
        --question-file "$QUESTIONS" --image-folder "$IMAGE_FOLDER_POPE" --answers-file "$OUT" \
        --test-prompt $'\nAnswer the question using a single word or phrase.' --batch-size "$EVAL_BATCH_SIZE"; then
        score "pope_${category}_score" "$PYTHON_BIN" ./eval_llava_rlhf_coco/summarize_eval_pope.py \
            --answers-file "$OUT" --label-file "$QUESTIONS"
    else record_status "pope_${category}_score" blocked 2 "complete generated answers required"; fi
done

QUESTIONS=./eval_llava_rlhf_coco/AMBER_data/query/query_all.json
OUT="$EVAL_ROOT/AMBER_bench/answer/answer-file-$MODEL_SUFFIX-all.jsonl"
if generate amber_generate amber "$OUT" "$QUESTIONS" \
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/AMBER_generate.py "${COMMON[@]}" --question-file "$QUESTIONS" \
    --answers-file "$OUT" --image-file "$IMAGE_DIR_AMBER" --test-prompt '' --batch-size "$EVAL_BATCH_SIZE"; then
    score amber_score "$PYTHON_BIN" ./eval_llava_rlhf_coco/AMBER_eval.py \
        --inference_data "$OUT" --evaluation_type a
else record_status amber_score blocked 2 "complete generated answers required"; fi

QUESTIONS=./eval_llava_rlhf_coco/object_hal/obj_halbench_300_with_image.jsonl
OUT="$EVAL_ROOT/objhal_bench/answer/answer-file-$MODEL_SUFFIX.jsonl"
if generate object_generate object "$OUT" "$QUESTIONS" \
    "$PYTHON_BIN" ./eval_llava_rlhf_coco/model_vqa_objectHal.py "${COMMON[@]}" \
    --question-file "$QUESTIONS" --answers-file "$OUT" --test-prompt ''; then
    if [[ -n "${ANNOTATION_FILE:-}" ]]; then
        score object_score_local "$PYTHON_BIN" ./eval_llava_rlhf_coco/eval_gpt_obj_halbench.py \
            --coco_path "$ANNOTATION_FILE" --answers_file "$OUT" \
            --evaluation_file "$EVAL_ROOT/objhal_bench/review-local/review-file-$MODEL_SUFFIX.jsonl"
        if judge_available object_score_gpt; then
            score object_score_gpt "$PYTHON_BIN" ./eval_llava_rlhf_coco/eval_gpt_obj_halbench.py \
                --coco_path "$ANNOTATION_FILE" --answers_file "$OUT" \
                --evaluation_file "$EVAL_ROOT/objhal_bench/review-gpt/review-file-$MODEL_SUFFIX.jsonl" \
                --gpt-model "$GPT_MODEL" --use_gpt
        fi
    else
        record_status object_score_local blocked 2 "ANNOTATION_FILE must contain COCO train/val2014 captions and instances"
        record_status object_score_gpt blocked 2 "COCO annotations required"
    fi
else
    record_status object_score_local blocked 2 "complete generated answers required"
    record_status object_score_gpt blocked 2 "complete generated answers required"
fi

"$PYTHON_BIN" "$SUMMARY_TOOL" --logs-dir "$LOG_DIR" --status "$STATUS_FILE" \
    --out-root "$EVAL_ROOT" --model-suffix "$MODEL_SUFFIX"
