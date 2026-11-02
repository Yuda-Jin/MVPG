import argparse
import json
import math
import os

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
    if args.discriminative_max_new_tokens < 1:
        raise ValueError("discriminative-max-new-tokens must be positive")
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
        questions = get_chunk(json.load(source), args.num_chunks, args.chunk_idx)
    # Preserve the existing inclusive ID-based eval_steps convention.
    if args.eval_steps >= 0:
        for index, line in enumerate(questions):
            if line["id"] >= args.eval_steps:
                questions = questions[:index + 1]
                break

    def prepared():
        for line in questions:
            # Official AMBER query_all IDs1..1004 are generative; >=1005 are
            # discriminative. The released rows have no separate type field.
            discriminative = line["id"] >= 1005
            image_token = DEFAULT_IMAGE_TOKEN
            if model.config.mm_use_im_start_end:
                image_token = DEFAULT_IM_START_TOKEN + image_token + DEFAULT_IM_END_TOKEN
            qs = image_token + "\n" + line["query"]
            if args.test_prompt:
                qs += args.test_prompt
            if discriminative and args.discriminative_prompt:
                qs += "\n" + args.discriminative_prompt
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            yield {
                "line": line,
                "input_ids": tokenizer_image_token(conv.get_prompt(), tokenizer,
                    IMAGE_TOKEN_INDEX, return_tensors="pt"),
                "image_path": os.path.join(args.image_folder, args.image_file, line["image"]),
                "stop_str": conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2,
                "max_new_tokens": (args.discriminative_max_new_tokens if discriminative
                                   else 64 if args.short_eval else 1024),
            }

    cache = BoundedImageCache(
        lambda path: process_image(image_processor, path, args.image_aspect_ratio), max_size=32)
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
            # Keep the generated response unchanged; strict/normalized scoring
            # is an explicit choice in AMBER_eval rather than a hidden rewrite.
            answers.write(json.dumps({"id": item["line"]["id"], "response": text}) + "\n")
            answers.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str,
                        default="./eval_llava_rlhf_coco/AMBER_data/query/query_generative.json")
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Equal-token-length batches; default retains singleton evaluation")
    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--qlora-path", type=str, default="")
    parser.add_argument("--short_eval", type=bool, default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--test-prompt", type=str, default="")
    parser.add_argument("--discriminative-prompt", type=str,
                        default="Answer the question using a single word, Yes or No.",
                        help="Appended only to the official discriminative query IDs (>=1005)")
    parser.add_argument("--discriminative-max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if os.path.exists(args.answers_file):
        parser.exit(1, f"{args.answers_file} already exists. Please use a new output path.\n")
    eval_model(args)
