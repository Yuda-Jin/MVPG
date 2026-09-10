#!/bin/bash
# ! MVPG 全指标评测（与 run/eval_all_metrics.sh 同一五阶段流水线，
# ! 模型侧从 QLoRA-LoRA 改为 base + 训练得到的 mm_projector 权重）。
#
# 用法：
#   CKPT=/root/autodl-tmp/output/mvpg_ckpt/mm_projector_step1000.bin bash run/eval_all_metrics_mvpg.sh
#
# 可选环境变量：
#   GPU_ID            使用的 GPU 编号（默认 0）
#   MODEL_BASE        LLaVA 底模路径（默认 /root/autodl-tmp/cache/llava-v1.5-7b）
#   CKPT              mm_projector 权重 .bin（必填，或用 MVPG_CKPT_DIR+STEP 组合）
#   MODEL_SUFFIX      输出文件标签（默认取 CKPT 文件名）
#   DATA_ROOT         评测数据根目录（默认 /root/autodl-tmp/data/eval）
#   OUT_ROOT          评测输出根目录（默认放在 CKPT 同目录下：$CKPT所在目录/eval_${MODEL_SUFFIX}）
#   OPENAI_ENDPOINT / OPENAI_API_KEY   Stage1/2/5 的 GPT 判分所需；未设置则跳过对应判分
#
# 数据依赖（不在仓库内，需提前准备）：
#   $DATA_ROOT/coco/train2017        LLaVA-Bench(qa90) 图片（缺则跳过 Stage2）
#   $DATA_ROOT/coco/val2014          POPE 图片（缺则跳过 Stage3）
#   $DATA_ROOT/AMBER/image           AMBER 图片（缺则跳过 Stage4）
#   MMHal / ObjHal 问题自带（HF datasets 下载 / base64 内嵌），无需本地图片
#   仅 Stage3(POPE)、Stage4(AMBER) 生成+评分不需要 OpenAI API

# 切到仓库根（脚本内的 ./eval_llava_rlhf_coco/... 相对路径依赖）
cd "$(dirname "$0")/.." || exit 1

# 防呆：所有依赖（torch/shortuuid/openai/spacy 等）装在 conda mvpg 环境，
# 若当前不在 mvpg 则自动切换，避免 (base) 下裸跑导致大面积 ModuleNotFoundError。
if [ "${CONDA_DEFAULT_ENV:-}" != "mvpg" ] && [ -d "/root/miniconda3/envs/mvpg" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate mvpg
    echo "[auto] 已切换 conda 环境: $CONDA_DEFAULT_ENV ($(which python))"
fi
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}
# DeepSeek API（OpenAI 兼容格式）：key 需填入下方默认值或运行时 export OPENAI_API_KEY
GPT_MODEL=${GPT_MODEL:-deepseek-v4-flash-vision-exp}
export OPENAI_ENDPOINT=${OPENAI_ENDPOINT:-"https://api.deepseek.com/v1"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"sk-b075bda7454b411d87027f5f19fde236"}
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------------- 模型与 checkpoint ----------------
MODEL_BASE=${MODEL_BASE:-/root/autodl-tmp/cache/llava-v1.5-7b}
CKPT=${CKPT:-${MVPG_CKPT_DIR:+$MVPG_CKPT_DIR/mm_projector_step${STEP:-1000}.bin}}
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "!! 缺少 mm_projector 权重：请设置 CKPT=/path/to/mm_projector_stepN.bin"
    exit 1
fi
MODEL_SUFFIX=${MODEL_SUFFIX:-$(basename "$CKPT" .bin)}
echo "MODEL_BASE=$MODEL_BASE"
echo "CKPT=$CKPT  SUFFIX=$MODEL_SUFFIX"

# ---------------- 评测数据路径 ----------------
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/eval}
# 输出根目录默认与 ckpt 同路径并带 ckpt 名，多 ckpt 的评测结果天然分目录隔离
OUT_ROOT=${OUT_ROOT:-"$(dirname "$CKPT")/eval_${MODEL_SUFFIX}"}
IMAGE_FOLDER_LB=${IMAGE_FOLDER_LB:-$DATA_ROOT/coco/train2017}
IMAGE_FOLDER_POPE=${IMAGE_FOLDER_POPE:-$DATA_ROOT/coco/val2014}
IMAGE_DIR_AMBER=${IMAGE_DIR_AMBER:-$DATA_ROOT/AMBER/image}
ANNOTATION_FILE=${ANNOTATION_FILE:-$DATA_ROOT/coco/annotations}

# 全程日志落盘：末尾据此解析汇总各阶段指标（多 ckpt 对比写入 summary.csv）
mkdir -p "$OUT_ROOT"
exec > >(tee "$OUT_ROOT/eval_log_${MODEL_SUFFIX}.log") 2>&1

# Stage 分隔符：终端/日志监督时快速定位各阶段（带时间戳便于估算耗时）
stage_header() {
    echo ""
    echo "================================================================================"
    echo "  $1  |  start at $(date '+%F %T')"
    echo "================================================================================"
}

MVPG_ARGS=(--model-path "$MODEL_BASE" --mm-projector "$CKPT" --image_aspect_ratio pad)
GPT_READY=false
[ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "" ] && GPT_READY=true

# ! Stage1 Eval mm-hal bench (requires OpenAI API for scoring; generation downloads
#   the MMHal-Bench dataset from HF on first run)
stage_header "Stage 1/5: MMHal-Bench"
OUTPUT_DIR=$OUT_ROOT/mmhal
if $GPT_READY; then
    python ./eval_llava_rlhf_coco/model_vqa_mmhal.py \
        "${MVPG_ARGS[@]}" \
        --temperature 0.0 \
        --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
        --test-prompt ''

    python ./eval_llava_rlhf_coco/eval_gpt_mmhal.py \
        --response ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
        --evaluation ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}.jsonl \
        --gpt-model $GPT_MODEL

    python ./eval_llava_rlhf_coco/summarize_gpt_mmhal.py \
        --evaluation ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}.jsonl
else
    echo "[skip] Stage1 MMHal：未设置 OPENAI_API_KEY（生成可跑，GPT 判分无意义）"
fi

# ! Stage2 Eval LLaVA bench qa90 (requires OpenAI API for scoring)
stage_header "Stage 2/5: LLaVA-Bench (qa90)"
OUTPUT_DIR=$OUT_ROOT/llava_bench
if [ -d "$IMAGE_FOLDER_LB" ]; then
    python ./eval_llava_rlhf_coco/model_vqa.py \
        "${MVPG_ARGS[@]}" \
        --question-file ./eval_llava_rlhf_coco/llava/qa90_questions.jsonl \
        --image-folder ${IMAGE_FOLDER_LB} \
        --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
        --test-prompt ''

    if $GPT_READY; then
        python ./eval_llava_rlhf_coco/eval_gpt_review_visual.py \
            --question ./eval_llava_rlhf_coco/llava/qa90_questions.jsonl \
            --context ./eval_llava_rlhf_coco/table/caps_boxes_coco2014_val_80.jsonl \
            --answer-list \
            ./eval_llava_rlhf_coco/llava/qa90_gpt4_answer.jsonl \
            ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
            --rule ./eval_llava_rlhf_coco/table/rule.json \
            --output ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}.jsonl

        python ./eval_llava_rlhf_coco/summarize_gpt_review.py -d ${OUTPUT_DIR}/review/ -f review-file-${MODEL_SUFFIX}.jsonl
    else
        echo "[skip] Stage2 GPT 判分：未设置 OPENAI_API_KEY（答案文件已生成）"
    fi
else
    echo "[skip] Stage2 LLaVA-Bench：缺图片目录 $IMAGE_FOLDER_LB"
fi

# ! Stage3 Eval POPE bench (No need for OpenAI API)
stage_header "Stage 3/5: POPE (adversarial)"
OUTPUT_DIR=$OUT_ROOT/pope
if [ -d "$IMAGE_FOLDER_POPE" ]; then
    POPE_CAT="adversarial"
    echo ${MODEL_SUFFIX} ${POPE_CAT}
    python ./eval_llava_rlhf_coco/model_vqa.py \
        "${MVPG_ARGS[@]}" \
        --short_eval True \
        --question-file ./eval_llava_rlhf_coco/pope/coco_pope_${POPE_CAT}.jsonl \
        --image-folder ${IMAGE_FOLDER_POPE} \
        --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}_${POPE_CAT}.jsonl \
        --test-prompt '\nAnswer the question using a single word or phrase.'

    python ./eval_llava_rlhf_coco/summarize_eval_pope.py \
        --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}_${POPE_CAT}.jsonl \
        --label-file ./eval_llava_rlhf_coco/pope/coco_pope_${POPE_CAT}.jsonl
else
    echo "[skip] Stage3 POPE：缺图片目录 $IMAGE_FOLDER_POPE"
fi

# ! Stage4 Eval AMBER bench (No need for OpenAI API)
stage_header "Stage 4/5: AMBER (generative)"
OUTPUT_DIR=$OUT_ROOT/amber
if [ -d "$IMAGE_DIR_AMBER" ]; then
    # en_core_web_lg 已安装则跳过；未安装时走 ghfast 镜像下载，避免 GitHub 直连超时
    if ! python -c "import spacy; spacy.load('en_core_web_lg')" >/dev/null 2>&1; then
        pip install "https://ghfast.top/https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
    fi

    python ./eval_llava_rlhf_coco/AMBER_generate.py \
        "${MVPG_ARGS[@]}" \
        --temperature 0.0 \
        --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
        --image-file $IMAGE_DIR_AMBER \
        --test-prompt ''

    python ./eval_llava_rlhf_coco/AMBER_eval.py \
        --inference_data ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
        --evaluation_type 'g'
else
    echo "[skip] Stage4 AMBER：缺图片目录 $IMAGE_DIR_AMBER"
fi

# ! Stage5 Eval Obj-Hal bench (GPT scoring optional; question file embeds images,
#   generation itself needs no local dataset)
stage_header "Stage 5/5: Object HalBench"
OUTPUT_DIR=$OUT_ROOT/object_hal
python ./eval_llava_rlhf_coco/model_vqa_objectHal.py \
    "${MVPG_ARGS[@]}" \
    --question-file ./eval_llava_rlhf_coco/object_hal/obj_halbench_300_with_image.jsonl \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --test-prompt ''

pip install jsonlines
# GitHub 直连被墙，spacy 模型 wheel 走 ghproxy 镜像（trf 会连带装 spacy-transformers
# 降级 transformers，脚本末尾统一恢复 4.34.1）
pip install "https://ghfast.top/https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl"

if $GPT_READY; then
    python ./eval_llava_rlhf_coco/eval_gpt_obj_halbench.py \
        --coco_path ${ANNOTATION_FILE} \
        --answers_file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
        --evaluation_file ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}_eval.jsonl \
        --gpt-model $GPT_MODEL \
        --use_gpt
else
    echo "[skip] Stage5 GPT 判分：未设置 OPENAI_API_KEY（答案文件已生成）"
fi

# there is some version conflict in the package, en_core_web_trf requires tokenizers-0.13.3 transformers-4.30.2.
# We need transformers-4.34.1 and tokenizers-0.14.1 for the model. Thereby we install back the two packages after the evaluation.
# huggingface_hub 一并锁定：trf wheel 会把它拖到 0.17.x，而 datasets 2.19 需要 >=0.21.2 的 insecure_hashlib
pip install transformers==4.34.1 tokenizers==0.14.1 huggingface_hub==0.23.5

stage_header "Summary: parsing eval log"
# 汇总各阶段指标 → $OUT_ROOT/summary_${MODEL_SUFFIX}.json + summary.csv（append，多 ckpt 对比）
python ./run/summarize_eval_results.py \
    --log "$OUT_ROOT/eval_log_${MODEL_SUFFIX}.log" \
    --out-root "$OUT_ROOT" \
    --model-suffix "$MODEL_SUFFIX"
