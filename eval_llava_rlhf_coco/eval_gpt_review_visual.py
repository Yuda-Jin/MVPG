import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from tqdm import tqdm


def parse_score(review):
    if not isinstance(review, str) or not review.strip():
        raise ValueError("LLaVA-Bench judge returned no text")
    values = review.splitlines()[0].replace(',', ' ').split()
    if len(values) != 2:
        raise ValueError("LLaVA-Bench first review line must contain two numeric scores")
    scores = [float(value) for value in values]
    if any(not math.isfinite(value) or not 0 <= value <= 10 for value in scores):
        raise ValueError("LLaVA-Bench scores must be finite and in [0, 10]")
    return scores


def judge_request_options(model):
    """Opt in to DeepSeek's non-thinking mode without changing GPT requests."""
    thinking = os.getenv("JUDGE_THINKING", "").strip().lower()
    if not str(model).lower().startswith("deepseek-") or not thinking:
        return {}
    if thinking != "disabled":
        raise ValueError("JUDGE_THINKING must be unset or disabled")
    # DeepSeek documents this OpenAI SDK extension explicitly.
    return {"extra_body": {"thinking": {"type": "disabled"}}}


def get_eval(client, content: str, max_tokens: int, model=None, max_retries=3):
    model = model or os.getenv('GPT_MODEL', 'gpt-4')
    request_options = judge_request_options(model)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {'role': 'user', 'content': content}],
                temperature=0.2,
                max_tokens=max_tokens,
                **request_options,
            )
            review = response.choices[0].message.content
            parse_score(review)
            return review
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            fatal = status is not None and 400 <= status < 500 and status not in (408, 409, 429)
            if fatal or attempt + 1 == max_retries:
                raise RuntimeError(
                    f"LLaVA-Bench judging failed ({type(exc).__name__}, status={status}); no score was assigned"
                ) from exc
            time.sleep(min(2 ** attempt, 10))
    raise ValueError("max_retries must be positive")


def read_jsonl(path):
    with Path(path).expanduser().open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_answers(questions, answers, name):
    question_ids = [question['question_id'] for question in questions]
    answer_ids = [answer['question_id'] for answer in answers]
    if (len(set(question_ids)) != len(question_ids) or len(set(answer_ids)) != len(answer_ids)
            or len(answers) != len(questions) or set(answer_ids) != set(question_ids)):
        raise ValueError(f"{name}: every question ID must occur exactly once; counts and IDs must match")
    by_id = {answer['question_id']: answer for answer in answers}
    return [by_id[question_id] for question_id in question_ids]


def main():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question', required=True)
    parser.add_argument('-c', '--context', required=True)
    parser.add_argument('-a', '--answer-list', nargs=2, required=True)
    parser.add_argument('-r', '--rule', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--gpt-model', default=os.getenv('GPT_MODEL', 'gpt-4'))
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if args.max_retries < 1 or args.timeout <= 0 or args.max_tokens < 1:
        raise ValueError("max-retries, timeout and max-tokens must be positive")

    questions = read_jsonl(args.question)
    if not questions:
        raise ValueError("Question set is empty")
    answers1 = align_answers(questions, read_jsonl(args.answer_list[0]), 'reference answers')
    answers2 = align_answers(questions, read_jsonl(args.answer_list[1]), 'model answers')
    rule_dict = json.loads(Path(args.rule).expanduser().read_text(encoding='utf-8'))
    contexts = read_jsonl(args.context)
    image_to_context = {context['image']: context for context in contexts}
    if len(image_to_context) != len(contexts):
        raise ValueError("Context image IDs are duplicated")
    for question in questions:
        if question['image'] not in image_to_context or question['category'] not in rule_dict:
            raise ValueError("Every question requires matching image context and scoring rule")

    paths = [args.question, args.context, *args.answer_list, args.rule]
    fingerprint = hashlib.sha256()
    for path in paths:
        data = Path(path).expanduser().read_bytes()
        fingerprint.update(len(data).to_bytes(8, 'big'))
        fingerprint.update(data)
    metadata = {'inputs_sha256': fingerprint.hexdigest(), 'judge_model': args.gpt_model,
                'max_tokens': args.max_tokens, 'schema': 'llava_bench_reviews_v1',
                'judge_request_options': judge_request_options(args.gpt_model)}
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_name(output.name + '.meta.json')
    if output.exists():
        if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError("Existing LLaVA-Bench reviews have different or missing provenance; choose a new output")
        reviews = read_jsonl(output)
        if len(reviews) > len(questions):
            raise ValueError("Resume file has more reviews than questions")
        for index, review in enumerate(reviews):
            if (review['question_id'] != questions[index]['question_id']
                    or review['category'] != questions[index]['category']
                    or review['tuple'] != parse_score(review['content'])):
                raise ValueError("Resume reviews have mismatched IDs, categories or invalid scores")
    else:
        reviews = []
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    client = None
    if len(reviews) < len(questions):
        from openai import OpenAI
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError("OPENAI_API_KEY is required for LLaVA-Bench GPT scoring")
        client = OpenAI(base_url=os.getenv('OPENAI_ENDPOINT') or os.getenv('OPENAI_BASE_URL') or None,
                        api_key=key, timeout=args.timeout, max_retries=0)
    with output.open('a', encoding='utf-8') as review_file:
        for index in tqdm(range(len(reviews), len(questions))):
            question, ans1, ans2 = questions[index], answers1[index], answers2[index]
            context = image_to_context[question['image']]
            captions = '\n'.join(context['captions'])
            boxes = '\n'.join(f'{item["category"]}: {item["bbox"]}' for item in context['instances'])
            rule = rule_dict[question['category']]
            role, prompt = rule['role'], rule['prompt']
            content = (f'[Context]\n{captions}\n\n{boxes}\n\n'
                       f'[Question]\n{question["text"]}\n\n'
                       f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                       f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                       f'[System]\n{prompt}\n\n')
            review = get_eval(client, content, args.max_tokens, args.gpt_model, args.max_retries)
            result = {'id': index + 1, 'question_id': question['question_id'],
                      'answer1_id': ans1.get('answer_id', ans1['question_id']),
                      'answer2_id': ans2.get('answer_id', ans2['question_id']),
                      'category': question['category'], 'content': review, 'tuple': parse_score(review)}
            review_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            review_file.flush()
            reviews.append(result)
    if len(reviews) != len(questions):
        raise RuntimeError("Incomplete LLaVA-Bench reviews; no aggregate score is valid")


if __name__ == '__main__':
    main()
