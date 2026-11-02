"""Batching contracts: no padding, exact IDs/order, bounded image retention."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from mvpg_run.eval_batch import (BoundedImageCache, generate_equal_length_batch,
                                ordered_batch_map)


def test_length_buckets_restore_order_and_keep_windows_bounded():
    consumed, calls = [], []
    def items():
        for index in range(19):
            consumed.append(index)
            yield {"id": index, "length": index % 3, "limit": 64 if index % 2 else 1024}
    def predict(batch):
        assert len(batch) <= 2
        assert len({(item["length"], item["limit"]) for item in batch}) == 1
        calls.append([item["id"] for item in batch])
        return [str(item["id"]) for item in batch]
    results = ordered_batch_map(items(), predict, batch_size=2, window_batches=4,
        batch_key=lambda item: (item["length"], item["limit"]))
    first, text = next(results)
    assert first["id"] == 0 and text == "0"
    assert consumed == list(range(8))  # only a bounded window was read ahead
    remaining = list(results)
    assert [first["id"]] + [item["id"] for item, _ in remaining] == list(range(19))
    assert [text] + [text for _, text in remaining] == [str(i) for i in range(19)]
    assert len(calls) < 19


def test_singleton_mode_preserves_execution_order_and_streams_immediately():
    calls = []
    def predict(batch):
        calls.append(batch[0])
        return batch
    results = ordered_batch_map([2, 1, 2, 1], predict, batch_size=1,
                                batch_key=lambda item: item)
    assert next(results) == (2, 2)
    assert calls == [2]
    assert list(results) == [(1, 1), (2, 2), (1, 1)]
    assert calls == [2, 1, 2, 1]


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_sizes_are_rejected(batch_size):
    with pytest.raises(ValueError, match="positive integer"):
        list(ordered_batch_map([], lambda batch: batch, batch_size=batch_size,
                               batch_key=lambda item: item))


def test_mismatched_prediction_count_fails_instead_of_misaligning_ids():
    with pytest.raises(ValueError, match="prediction count"):
        list(ordered_batch_map([1, 1], lambda batch: ["only one"],
                               batch_size=2, batch_key=lambda item: item))


def test_image_cache_reuses_and_evicts_the_least_recent_entry():
    loaded = []
    def load(path):
        loaded.append(path)
        return object()
    cache = BoundedImageCache(load, max_size=2)
    first = cache.get("a")
    cache.get("b")
    assert cache.get("a") is first
    cache.get("c")  # evicts b, since a was just used
    assert cache.get("a") is first
    cache.get("b")
    assert loaded == ["a", "b", "c", "b"]
    assert len(cache._values) == 2


def generation_items():
    return [{"input_ids": torch.tensor([1, -200, 3]), "image_path": path,
             "stop_str": "</s>", "max_new_tokens": 64}
            for path in ("one.jpg", "two.jpg")]


def test_generation_preserves_greedy_settings_prefix_and_raw_answer_text():
    items = generation_items()
    class Model:
        config = SimpleNamespace()
        def generate(self, **kwargs):
            assert torch.equal(kwargs["input_ids"], torch.tensor([[1, -200, 3]] * 2))
            assert torch.equal(kwargs["attention_mask"], torch.ones(2, 3, dtype=torch.long))
            assert kwargs["images"].shape == (2, 3, 2, 2)
            assert kwargs["images"].dtype == torch.bfloat16
            assert kwargs["do_sample"] is False and kwargs["temperature"] == 1.0
            assert kwargs["max_new_tokens"] == 64 and kwargs["use_cache"] is True
            return torch.cat((kwargs["input_ids"], torch.tensor([[4, 2], [5, 2]])), dim=1)
    class Tokenizer:
        def batch_decode(self, ids, skip_special_tokens):
            assert ids.tolist() == [[4, 2], [5, 2]] and skip_special_tokens
            return [" Yes. </s> ", "No</s>"]
    cache = BoundedImageCache(lambda path: torch.zeros(3, 2, 2))
    outputs = generate_equal_length_batch(Model(), Tokenizer(), items, cache,
        device="cpu", compute_dtype=torch.bfloat16, temperature=0.0)
    assert outputs == ["Yes.", "No"]  # punctuation is not silently normalized


@pytest.mark.parametrize("invalid", ["length", "limit", "prefix"])
def test_unsupported_batch_or_generation_output_fails_explicitly(invalid):
    items = generation_items()
    if invalid == "length":
        items[1]["input_ids"] = torch.tensor([1, -200])
    elif invalid == "limit":
        items[1]["max_new_tokens"] = 1024
    class Model:
        config = SimpleNamespace()
        def generate(self, **kwargs):
            assert invalid == "prefix"  # other errors must fail before generation
            return torch.zeros(2, 4, dtype=torch.long)
    cache = BoundedImageCache(lambda path: torch.zeros(3, 2, 2))
    with pytest.raises(ValueError, match="equal-length|same generation limit|prompt prefix"):
        generate_equal_length_batch(Model(), object(), items, cache,
            device="cpu", compute_dtype=torch.float32, temperature=0.0)


@pytest.fixture
def fake_llava_scripts(monkeypatch):
    prompts = []
    model = SimpleNamespace(config=SimpleNamespace(mm_use_im_start_end=False), eval=lambda: None)
    tokenizer, processor = object(), object()
    class Conversation:
        roles, sep, sep2, sep_style = ("user", "assistant"), "###", "</s>", 2
        def copy(self):
            return Conversation()
        def append_message(self, role, text):
            if role == "user":
                self.prompt = text
        def get_prompt(self):
            return self.prompt
    def tokenize(prompt, *args, **kwargs):
        prompts.append(prompt)
        return torch.ones(len(prompt), dtype=torch.long)
    modules = {name: ModuleType(name) for name in ("llava", "llava.constants",
        "llava.conversation", "llava.model", "llava.model.builder", "llava.utils", "llava.mm_utils")}
    for key, value in {"IMAGE_TOKEN_INDEX": -200, "DEFAULT_IMAGE_TOKEN": "<image>",
        "DEFAULT_IM_START_TOKEN": "<start>", "DEFAULT_IM_END_TOKEN": "<end>"}.items():
        setattr(modules["llava.constants"], key, value)
    modules["llava.conversation"].conv_templates = {"llava_v1": Conversation()}
    modules["llava.conversation"].SeparatorStyle = SimpleNamespace(TWO=2)
    modules["llava.model.builder"].load_pretrained_model = lambda *a: (tokenizer, model, processor, 2048)
    modules["llava.utils"].disable_torch_init = lambda: None
    modules["llava.mm_utils"].tokenizer_image_token = tokenize
    modules["llava.mm_utils"].get_model_name_from_path = lambda path: "fake-model"
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    # shortuuid is an evaluation dependency, not needed by the mechanics tests.
    monkeypatch.setitem(sys.modules, "shortuuid", SimpleNamespace(uuid=lambda: "test-answer"))
    def load_script(name):
        path = Path(__file__).resolve().parents[1] / "eval_llava_rlhf_coco" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"test_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return load_script, prompts


def script_args(tmp_path, **kwargs):
    values = dict(batch_size=2, model_path="fake", use_qlora=False, model_base=None,
        question_file=str(tmp_path / "questions.json"), num_chunks=1, chunk_idx=0,
        test_prompt="", conv_mode="llava_v1", short_eval=False, image_folder="",
        temperature=0.0, top_p=None, num_beams=1, image_aspect_ratio="pad",
        answers_file=str(tmp_path / "answers.jsonl"))
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_amber_discriminative_only_instruction_and_raw_original_order(
        tmp_path, fake_llava_scripts, monkeypatch):
    load, prompts = fake_llava_scripts
    module = load("AMBER_generate")
    args = script_args(tmp_path, image_file="images", eval_steps=-1,
        discriminative_prompt="Answer the question using a single word, Yes or No.",
        discriminative_max_new_tokens=64)
    rows = [{"id": 1, "image": "a.jpg", "query": "Describe this image."},
            {"id": 1005, "image": "a.jpg", "query": "Is there a cat?"}]
    Path(args.question_file).write_text(json.dumps(rows))
    limits = {}
    def predict(model, tokenizer, batch, cache, **kwargs):
        limits.update({item["line"]["id"]: item["max_new_tokens"] for item in batch})
        return ["Yes." if item["line"]["id"] >= 1005 else "Description." for item in batch]
    monkeypatch.setattr(module, "generate_equal_length_batch", predict)
    module.eval_model(args)
    assert prompts == ["<image>\nDescribe this image.",
        "<image>\nIs there a cat?\nAnswer the question using a single word, Yes or No."]
    assert limits == {1: 1024, 1005: 64}
    assert [json.loads(line) for line in Path(args.answers_file).read_text().splitlines()] == [
        {"id": 1, "response": "Description."}, {"id": 1005, "response": "Yes."}]


def test_vqa_bucket_reordering_retains_question_ids_prompts_and_row_order(
        tmp_path, fake_llava_scripts, monkeypatch):
    load, prompts = fake_llava_scripts
    module = load("model_vqa")
    args = script_args(tmp_path, short_eval=True, test_prompt="\nAnswer briefly.")
    rows = [{"question_id": i, "image": "a.jpg", "text": text}
            for i, text in [(10, "A?"), (1, "Longer question?"), (9, "B?")]]
    Path(args.question_file).write_text("".join(json.dumps(row) + "\n" for row in rows))
    executed = []
    def predict(model, tokenizer, batch, cache, **kwargs):
        executed.extend(item["line"]["question_id"] for item in batch)
        assert all(item["max_new_tokens"] == 64 for item in batch)
        return [str(item["line"]["question_id"]) for item in batch]
    monkeypatch.setattr(module, "generate_equal_length_batch", predict)
    module.eval_model(args)
    assert executed == [10, 9, 1]
    answers = [json.loads(line) for line in Path(args.answers_file).read_text().splitlines()]
    assert [row["question_id"] for row in answers] == [10, 1, 9]
    assert [row["text"] for row in answers] == ["10", "1", "9"]
    assert [row["prompt"] for row in answers] == [row["text"] for row in rows]
    assert prompts == ["<image>\n" + row["text"] + "\nAnswer briefly." for row in rows]
