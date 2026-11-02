"""Offline contracts for the current scheme's shared LLaVA inference helpers."""
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from mvpg_run.llava_inference import (build_query, load_lora_for_inference,
                                     load_model_and_tokenizer, process_image)


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
        config_class = SimpleNamespace(from_pretrained=lambda path: calls.setdefault("source_config", SimpleNamespace()))
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
    assert calls["kwargs"]["config"].image_checkpoint == {
        "_target_": "llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower",
        "download_method_for_image_model": "local", "image_model": str(clip),
    }
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


def test_local_clip_updates_stale_checkpoint_without_editing_model_files(fake_llava, tmp_path, monkeypatch):
    _, _, calls = fake_llava
    model_dir, clip = tmp_path / "base", tmp_path / "clip"
    model_dir.mkdir()
    clip.mkdir()
    original = {"_target_": "llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower",
                "download_method_for_image_model": "old-method",
                "image_model": "/another-machine/clip", "extra": {"keep": True}}
    unchanged = copy.deepcopy(original)
    config_file = model_dir / "config.json"
    text = json.dumps({"image_checkpoint": original})
    config_file.write_text(text, encoding="utf-8")
    calls["source_config"] = SimpleNamespace(image_checkpoint=original)
    monkeypatch.setenv("LLAVA_CLIP_DIR", str(clip))
    load_model_and_tokenizer(str(model_dir), 16, False, False, "cpu")
    cfg = calls["kwargs"]["config"]
    assert cfg.mm_vision_tower == str(clip.resolve())
    assert cfg.image_checkpoint["image_model"] == str(clip.resolve())
    assert cfg.image_checkpoint["download_method_for_image_model"] == "local"
    assert cfg.image_checkpoint["extra"] == {"keep": True}
    assert original == unchanged
    assert config_file.read_text(encoding="utf-8") == text


def test_default_sibling_clip_directory_supports_official_config(fake_llava, tmp_path, monkeypatch):
    _, _, calls = fake_llava
    model_dir, clip = tmp_path / "llava-v1.5-7b", tmp_path / "clip-vit-large-patch14-336"
    model_dir.mkdir()
    clip.mkdir()
    monkeypatch.delenv("LLAVA_CLIP_DIR", raising=False)
    load_model_and_tokenizer(str(model_dir), 16, False, False, "cpu")
    cfg = calls["kwargs"]["config"]
    assert cfg.mm_vision_tower == str(clip.resolve())
    assert cfg.image_checkpoint["image_model"] == str(clip.resolve())


@pytest.mark.parametrize("checkpoint", ["invalid", {"_target_": "llava.model.multimodal_encoder.dino.DINOVisionTower"}])
def test_local_clip_rejects_incompatible_vision_configuration(fake_llava, tmp_path, monkeypatch, checkpoint):
    _, _, calls = fake_llava
    calls["source_config"] = SimpleNamespace(image_checkpoint=checkpoint)
    monkeypatch.setenv("LLAVA_CLIP_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="mapping|non-CLIP"):
        load_model_and_tokenizer("base", 16, False, False, "cpu")


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


@pytest.mark.parametrize("weights", ["adapter_model.bin", "adapter_model.safetensors"])
def test_current_adapter_evaluation_accepts_both_formats(fake_llava, tmp_path, monkeypatch, weights):
    model, tokenizer, _ = fake_llava
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / weights).touch()
    def attach(base, path, is_trainable):
        assert base is model and path == str(tmp_path.resolve())
        assert is_trainable is False
        return base
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=attach)))
    loaded, tok, _, dtype = load_lora_for_inference("base", tmp_path, device="cpu", precision="fp32")
    assert loaded is model and tok is tokenizer and dtype == torch.float32
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_missing_adapter_does_not_fall_back_to_base(tmp_path):
    with pytest.raises(FileNotFoundError, match="Incomplete LoRA checkpoint"):
        load_lora_for_inference("nonexistent-base", tmp_path)
