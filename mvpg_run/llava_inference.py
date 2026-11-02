"""Shared patched-LLaVA loading for claim feedback and hallucination evaluation.

Kept independent of the claim-feedback optimizer so loading, prompting and image
preprocessing have one implementation for the current scheme.
Framework imports are lazy; calling this module's functions needs the patched
LLaVA environment described in the repository README.
"""
from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
import copy
import os
from pathlib import Path


def load_model_and_tokenizer(model_name, bits, fp16, bf16, device):
    """Load the same local CLIP/projector architecture used by old checkpoints.

    All parameters are frozen. The caller explicitly adds its own trainable
    adapters; loading a model for evaluation never enables projector training.
    """
    import torch
    import transformers
    from llava import conversation as conversation_lib
    from llava.model import LlavaLlamaForCausalLM

    if bits not in (4, 8, 16):
        raise ValueError("bits must be 4, 8 or 16")
    if fp16 and bf16:
        raise ValueError("Choose at most one of fp16 and bf16")
    compute_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
    bnb_kwargs = {}
    if bits in (4, 8):
        from transformers import BitsAndBytesConfig
        bnb_kwargs = dict(
            load_in_4bit=bits == 4, load_in_8bit=bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4, load_in_8bit=bits == 8,
                llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["mm_projector", "lm_head"],
            ),
        )
    clip_dir = os.environ.get("LLAVA_CLIP_DIR")
    if clip_dir is None:
        clip_dir = str(Path(model_name).resolve().parent / "clip-vit-large-patch14-336")
    clip_path = Path(clip_dir).expanduser()
    if clip_path.is_dir():
        clip_dir = str(clip_path.resolve())
        config = LlavaLlamaForCausalLM.config_class.from_pretrained(model_name)
        config.mm_vision_tower = clip_dir
        # The bundled patch rebuilds mm_vision_tower from image_checkpoint in
        # LlavaMetaModel.__init__. Update both representations in memory so a
        # stale training-machine path cannot override the local CLIP directory.
        checkpoint = getattr(config, "image_checkpoint", None)
        if checkpoint is None:
            checkpoint = {}
        elif not isinstance(checkpoint, Mapping):
            raise ValueError("config.image_checkpoint must be a mapping")
        checkpoint = copy.deepcopy(dict(checkpoint))
        clip_target = "llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower"
        if checkpoint.get("_target_", clip_target) != clip_target:
            raise ValueError("Local CLIP override is incompatible with the configured non-CLIP vision target")
        checkpoint.update(_target_=clip_target,
                          download_method_for_image_model="local", image_model=clip_dir)
        config.image_checkpoint = checkpoint
    else:
        config = None
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, config=config, use_flash_attention_2=False,
        torch_dtype=compute_dtype,
        device_map={"": device} if bits in (4, 8) else None,
        trust_remote_code=True, **bnb_kwargs,
    )
    if bits not in (4, 8):
        model = model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=compute_dtype, device=device)
    model_args = Namespace(
        mm_use_im_start_end=bool(getattr(model.config, "mm_use_im_start_end", False)),
        mm_use_im_patch_token=bool(getattr(model.config, "mm_use_im_patch_token", False)),
        tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None,
    )
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.requires_grad_(False)
    return model, tokenizer, vision_tower.image_processor, compute_dtype


def build_query(tokenizer, question):
    from llava import conversation as conversation_lib
    from llava.mm_utils import tokenizer_image_token
    from utils.constants import DEFAULT_IMAGE_TOKEN
    conv = conversation_lib.conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, return_tensors="pt")


def load_lora_for_inference(model_name, adapter_path, *, device="cuda", precision="bf16"):
    """Load the current scheme's adapter, failing if its checkpoint is missing.

    Both PEFT serialization formats are accepted. Evaluation must never silently
    substitute the unchanged base model for a misspelled adapter directory.
    """
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    adapter = Path(adapter_path).expanduser().resolve()
    if not (adapter / "adapter_config.json").is_file() or not any(
            (adapter / name).is_file() for name in ("adapter_model.bin", "adapter_model.safetensors")):
        raise FileNotFoundError(f"Incomplete LoRA checkpoint: {adapter}")
    from peft import PeftModel
    model, tokenizer, processor, dtype = load_model_and_tokenizer(
        model_name, 16, precision == "fp16", precision == "bf16", device)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, processor, dtype


def process_image(image_processor, img_path, image_aspect_ratio="pad"):
    from PIL import Image
    with Image.open(img_path) as source:
        image = source.convert("RGB")
    if image_aspect_ratio == "pad" and image.width != image.height:
        width, height = image.size
        canvas = Image.new("RGB", (max(width, height), max(width, height)),
                           tuple(int(x * 255) for x in image_processor.image_mean))
        canvas.paste(image, ((canvas.width-width)//2, (canvas.height-height)//2))
        image = canvas
    return image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
