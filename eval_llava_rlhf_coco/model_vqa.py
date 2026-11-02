import argparse
import json
import math
import os

import shortuuid
import torch
from tqdm import tqdm

from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from mvpg_run.eval_batch import (BoundedImageCache, generate_equal_length_batch,
    ordered_batch_map, validate_batch_size)
from mvpg_run.llava_inference import process_image


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def eval_model(args):
    validate_batch_size(args.batch_size)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    compute_dtype = torch.float16
    if args.use_qlora:
        from mvpg_run.llava_inference import load_lora_for_inference
        model, tokenizer, image_processor, compute_dtype = load_lora_for_inference(
            model_path, args.qlora_path, device="cuda", precision="bf16")
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name)
    model.eval()

    with open(os.path.expanduser(args.question_file), "r") as source:
        questions = [json.loads(q) for q in source]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    def prepared():
        for line in questions:
            qs = line["text"]
            image_token = DEFAULT_IMAGE_TOKEN
            if model.config.mm_use_im_start_end:
                image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
            qs = image_token + "\n" + qs
            if args.test_prompt:
                qs += args.test_prompt
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            yield {
                "line": line,
                "input_ids": tokenizer_image_token(conv.get_prompt(), tokenizer,
                    IMAGE_TOKEN_INDEX, return_tensors="pt"),
                "image_path": os.path.join(args.image_folder, line["image"]),
                "stop_str": conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2,
                "max_new_tokens": 64 if args.short_eval else 1024,
            }

    def load_image(path):
        try:
            return process_image(image_processor, path, args.image_aspect_ratio)
        except OSError as exc:
            fallback = path.replace("train2017", "val2017")
            if fallback == path:
                raise
            print(exc)
            image = process_image(image_processor, fallback, args.image_aspect_ratio)
            print("retry val2017 file succeed")
            return image

    cache = BoundedImageCache(load_image, max_size=32)
    def predict(batch):
        return generate_equal_length_batch(model, tokenizer, batch, cache,
            device="cuda", compute_dtype=compute_dtype, temperature=args.temperature,
            top_p=args.top_p, num_beams=args.num_beams)

    results = ordered_batch_map(prepared(), predict, batch_size=args.batch_size,
        batch_key=lambda item: (item["input_ids"].numel(), item["max_new_tokens"]))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)
    with open(answers_file, "x") as answers:
        for item, text in tqdm(results, total=len(questions)):
            line = item["line"]
            answers.write(json.dumps({
                "question_id": line["question_id"], "prompt": line["text"],
                "text": text, "answer_id": shortuuid.uuid(),
                "model_id": model_name, "metadata": {},
            }) + "\n")
            answers.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Equal-token-length batches; default retains singleton evaluation")
    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")
    parser.add_argument("--short_eval", type=bool, default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--test-prompt", type=str,
                        default="\nAnswer the question using a single word or phrase.")
    args = parser.parse_args()
    if os.path.exists(args.answers_file):
        parser.exit(1, f"{args.answers_file} already exists. Please use a new output path.\n")
    eval_model(args)
