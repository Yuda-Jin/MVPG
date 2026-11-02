import argparse
import json
import time
import os
import hashlib
import re
from pathlib import Path

template = '''Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
'''


def parse_rating(response):
    if not isinstance(response, str):
        raise ValueError("MMHal judge returned no text")
    ratings = {int(value) for value in re.findall(r"\brating\s*:\s*([0-6])\b", response, re.I)}
    if len(ratings) != 1:
        raise ValueError("MMHal judge must return exactly one unambiguous Rating: 0..6")
    return ratings.pop()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def judge_request_options(model):
    """Opt in to DeepSeek's non-thinking mode without changing GPT requests."""
    thinking = os.getenv("JUDGE_THINKING", "").strip().lower()
    if not str(model).lower().startswith("deepseek-") or not thinking:
        return {}
    if thinking != "disabled":
        raise ValueError("JUDGE_THINKING must be unset or disabled")
    # DeepSeek documents this OpenAI SDK extension explicitly.
    return {"extra_body": {"thinking": {"type": "disabled"}}}


def ask_judge(client, content, model, max_retries=3):
    request_options = judge_request_options(model)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": content}], temperature=0.0,
                **request_options,
            )
            text = response.choices[0].message.content
            parse_rating(text)
            return text
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            fatal = status is not None and 400 <= status < 500 and status not in (408, 409, 429)
            if fatal or attempt + 1 == max_retries:
                raise RuntimeError(
                    f"MMHal judging failed ({type(exc).__name__}, status={status}); no score was assigned"
                ) from exc
            time.sleep(min(2 ** attempt, 10))
    raise ValueError("max_retries must be positive")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=str, default='responses/idefics_80b.json')
    parser.add_argument('--evaluation', type=str, required=True)
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--gpt-model', type=str, default=os.getenv('GPT_MODEL', 'gpt-4'))
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if args.max_retries < 1 or args.timeout <= 0:
        raise ValueError("max-retries and timeout must be positive")

    source = Path(args.response)
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 96:
        raise ValueError("MMHal requires all 96 responses in official order")
    for record in records:
        for field in ('image_content', 'question', 'gt_answer', 'model_answer'):
            if field not in record:
                raise ValueError(f"MMHal response is missing {field}")

    evaluation = Path(args.evaluation)
    metadata_path = evaluation.with_name(evaluation.name + ".meta.json")
    metadata = {"response_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "judge_model": args.gpt_model, "schema": "mmhal_ordered_reviews_v1",
                "judge_request_options": judge_request_options(args.gpt_model)}
    if evaluation.exists():
        if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Existing MMHal reviews have different or missing provenance; choose a new output")
        responses = json.loads(evaluation.read_text(encoding="utf-8"))
        if not isinstance(responses, list) or len(responses) > len(records):
            raise ValueError("Invalid MMHal resume file")
        for response in responses:
            parse_rating(response)
    else:
        responses = []
        save_json(metadata_path, metadata)

    if len(responses) < len(records):
        from openai import OpenAI
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for MMHal GPT scoring")
        client = OpenAI(base_url=os.getenv("OPENAI_ENDPOINT") or os.getenv("OPENAI_BASE_URL") or None,
                        api_key=api_key, timeout=args.timeout, max_retries=0)
        for index in range(len(responses), len(records)):
            record = records[index]
            content = template.format(', '.join(record['image_content']), record['question'],
                                      record['gt_answer'], record['model_answer'])
            review = ask_judge(client, content, args.gpt_model, args.max_retries)
            responses.append(review)
            # Publish every valid answer atomically, including partial progress.
            save_json(evaluation, responses)
            print(f"MMHal reviewed {len(responses)}/{len(records)}", flush=True)

    scores = [parse_rating(response) for response in responses]
    if len(scores) != 96:
        raise RuntimeError("Incomplete MMHal judging; no aggregate score is valid")
    hallucination = [int(score < 3) for score in scores]
    scores_each = [scores[index::8] for index in range(8)]
    print('Average score: {:.2f}'.format(sum(scores) / len(scores)))
    print('Hallucination rate: {:.2f}'.format(sum(hallucination) / len(hallucination)))
    print('Average score for each question type:', ','.join(
        str(round(sum(values) / len(values), 2)) for values in scores_each), flush=True)


if __name__ == '__main__':
    main()
