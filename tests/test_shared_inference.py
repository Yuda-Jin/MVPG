"""Regression contracts for helpers extracted from the retired training recipe."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from data.grouped_caption_sources import DEFAULT_QUESTION, load_groups, resolve_group_images
from mvpg_run.llava_inference import build_query, load_model_and_tokenizer, process_image


@pytest.fixture
def fake_llava(monkeypatch):
    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.is_loaded = False
            self.image_processor = object()
        def load_model(self):
            self.is_loaded = True
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_projector = torch.nn.Linear(2, 2)
            self.vision = Vision()
            self.config = SimpleNamespace()
        def get_vision_tower(self):
            return self.vision
        def initialize_vision_tokenizer(self, args, tokenizer):
            self.initialized = (args, tokenizer)
    class Conversation:
        roles = ("user", "assistant")
        def copy(self):
            self.messages = []
            return self
        def append_message(self, role, content):
            self.messages.append((role, content))
        def get_prompt(self):
            return self.messages[0][1] + "\nASSISTANT:"
    model, tokenizer = Model(), SimpleNamespace(unk_token="<unk>")
    calls = {}
    class Factory:
        config_class = SimpleNamespace(from_pretrained=lambda path: SimpleNamespace())
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.update(path=path, kwargs=kwargs)
            return model
    modules = {name: ModuleType(name) for name in
               ("llava", "llava.conversation", "llava.model", "llava.mm_utils", "transformers")}
    modules["llava.conversation"].conv_templates = {"vicuna_v1": Conversation()}
    modules["llava.model"].LlavaLlamaForCausalLM = Factory
    modules["llava.mm_utils"].tokenizer_image_token = lambda prompt, tok, **kw: prompt
    modules["transformers"].AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return model, tokenizer, calls


def test_loader_preserves_local_clip_and_freezes_projector(fake_llava, tmp_path, monkeypatch):
    model, tokenizer, calls = fake_llava
    clip = tmp_path / "clip"
    clip.mkdir()
    monkeypatch.setenv("LLAVA_CLIP_DIR", str(clip))
    result, tok, processor, dtype = load_model_and_tokenizer("local-base", 16, False, False, "cpu")
    assert result is model and tok is tokenizer
    assert processor is model.vision.image_processor and dtype == torch.float32
    assert calls["kwargs"]["config"].mm_vision_tower == str(clip)
    assert calls["kwargs"]["use_flash_attention_2"] is False
    assert tokenizer.pad_token == "<unk>"
    assert model.vision.is_loaded
    assert not any(p.requires_grad for p in model.parameters())
    assert build_query(tokenizer, "What is present?") == "<image>\nWhat is present?\nASSISTANT:"


def test_loader_rejects_ambiguous_precision(fake_llava):
    with pytest.raises(ValueError, match="at most one"):
        load_model_and_tokenizer("base", 16, True, True, "cpu")
    with pytest.raises(ValueError, match="bits"):
        load_model_and_tokenizer("base", 3, False, False, "cpu")


def test_padding_preserves_image_content_and_mean_background(tmp_path):
    image = tmp_path / "wide.png"
    Image.new("RGB", (4, 2), (255, 0, 0)).save(image)
    class Processor:
        image_mean = (0.5, 0.5, 0.5)
        def preprocess(self, image, return_tensors):
            assert image.size == (4, 4)
            assert image.getpixel((0, 0)) == (127, 127, 127)
            assert image.getpixel((0, 1)) == (255, 0, 0)
            return {"pixel_values": torch.zeros(1, 3, 4, 4)}
    assert process_image(Processor(), image).shape == (3, 4, 4)


def test_grouped_diagnostic_inputs_keep_existing_order_and_negative_pool(tmp_path):
    group = tmp_path / "sample"
    group.mkdir()
    for name in ("sample.png", "sample_r1c0.jpg", "sample_r0c1.jpg", "unrelated.png"):
        (group / name).touch()
    (tmp_path / "neg_captions.json").write_text(json.dumps({
        "sample": [" absent dog ", ""], "missing_dir": ["cat"]}), encoding="utf-8")
    assert load_groups(tmp_path) == [{"gid": "sample", "negs": ["absent dog"]}]
    main, tiles = resolve_group_images(tmp_path, "sample")
    assert main == str(group / "sample.png")
    assert tiles == [str(group / "sample_r0c1.jpg"), str(group / "sample_r1c0.jpg")]
    assert DEFAULT_QUESTION == "Describe the image content in detail."
