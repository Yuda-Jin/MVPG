"""Conservative tokenizer bridge and optional frozen visual / LLaVA backends."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def prepare_multimodal(base, input_ids, attention_mask, labels, images):
    """Normalize the two LLaVA-1.5 interfaces used by the repo and cloud host.

    The repository runtime patch returns (embeddings, labels, mask); existing
    LLaVA-1.5 installations return (ids, mask, cache, embeddings, labels).
    Inspect the signature instead of swallowing TypeError from model internals.
    """
    import inspect
    prepare = base.prepare_inputs_labels_for_multimodal
    parameters = inspect.signature(prepare).parameters
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  labels=labels, images=images)
    legacy = "past_key_values" in parameters
    if "position_ids" in parameters:
        raise ValueError("Unsupported multimodal position_ids API; expected the repository patch or LLaVA-1.5 five-item API")
    if legacy:
        kwargs["past_key_values"] = None
    prepared = prepare(**kwargs)
    if not isinstance(prepared, (tuple, list)):
        raise ValueError("Multimodal preparation must return a tuple or list")
    if legacy and len(prepared) == 5:
        ids, mask, cache, embeds, expanded = prepared
        if ids is not None or cache is not None:
            raise ValueError("Expected expanded image embeddings without an autoregressive cache")
        return embeds, expanded, mask
    if not legacy and len(prepared) == 3:
        return prepared
    raise ValueError("Unsupported LLaVA multimodal preparation return format")


def decode_exact(tokenizer, ids):
    return tokenizer.decode(list(ids), skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)


def response_tokens(tokenizer, record):
    """Recover exact sampled IDs and trustworthy, non-rewriting character offsets.

    Slow SentencePiece tokenizers have no offset mapping. Incremental decoding is
    accepted only when every prefix is a prefix of the complete original text.
    Unicode byte fragments that rewrite a prefix cause an explicit rejection.
    No retokenization of a sampled response or approximate word search is used.
    """
    text = record["response"]
    ids = record.get("response_ids")
    if ids is None:
        ids = tokenizer.encode(text, add_special_tokens=False)
    ids = list(ids)
    if decode_exact(tokenizer, ids) != text:
        raise ValueError("Response IDs do not decode exactly to response; use the generating tokenizer and preserve text without stripping")
    offsets, previous = [], 0
    for end in range(1, len(ids) + 1):
        prefix = decode_exact(tokenizer, ids[:end])
        if not text.startswith(prefix) or len(prefix) < previous:
            raise ValueError("Tokenizer rewrites decoded prefixes; source alignment is unresolved for this response")
        offsets.append((previous, len(prefix)))
        previous = len(prefix)
    return ids, offsets


def llm_lora_targets(model):
    """Full module paths prevent suffix matching from tuning the vision tower."""
    import torch
    forbidden = ("vision_tower", "mm_projector", "vision_resampler", "lm_head")
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear)
               and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj")
               and not any(part in name.split(".") for part in forbidden)]
    if not targets:
        raise ValueError("No language-model q_proj/v_proj linear modules were found")
    return sorted(targets)


class FGClipVerifier:
    """A frozen cosine proxy; no absence or relation-entailment guarantee."""
    def __init__(self, root, device="cpu", score_mode="global", image_root=None,
                 walk_type="short"):
        if score_mode not in ("global", "patch"):
            raise ValueError("score_mode must be global or patch")
        from mvpg_run.fgclip_local import FGChip
        self.encoder = FGChip(root, device)
        self.encoder.model.requires_grad_(False)
        self.mode, self.walk_type = score_mode, walk_type
        self.image_root = Path(image_root or ".").resolve()

    def __call__(self, image, texts):
        import torch
        import torch.nn.functional as F
        if not texts:
            return []
        path = Path(image)
        path = path if path.is_absolute() else self.image_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        with torch.no_grad():
            if self.mode == "global":
                visual = self.encoder.encode_image_paths([str(path)])
                text = F.normalize(self.encoder.encode_texts_walk(
                    texts, walk_type=self.walk_type), dim=-1)
                scores = (visual @ text.T)[0]
            else:
                dense, valid = self.encoder.image_dense_features([str(path)])
                if not valid.any().item():
                    raise ValueError("Image has no valid patches")
                text = F.normalize(self.encoder.encode_texts_walk(
                    texts, walk_type=self.walk_type), dim=-1)
                scores = (F.normalize(dense[0], dim=-1) @ text.T)
                scores = scores.masked_fill(~valid[0, :, None], -torch.inf).max(dim=0).values
        return scores.float().cpu().tolist()


class LlavaBackend:
    """Adapter for this repository's patched LLaVA-1.5, one response at a time.

    Vision tower/projector and all original weights stay frozen. Only LLM LoRA
    weights are trainable. Multimodal labels are inspected after image expansion
    instead of assuming a fixed number of visual patches.
    """
    def __init__(self, model_name, *, device="cuda", precision="bf16",
                 train=False, adapter_path=None, lora_rank=16, lora_alpha=32):
        import torch
        try:
            from mvpg_run.llava_inference import (load_model_and_tokenizer, build_query,
                                                  process_image)
            model, self.tokenizer, self.image_processor, self.dtype = load_model_and_tokenizer(
                model_name, bits=16, fp16=precision == "fp16",
                bf16=precision == "bf16", device=device)
        except ImportError as exc:
            raise RuntimeError("LLaVA backend requires the repository's patched LLaVA and its environment; see root README model environment instructions") from exc
        self.torch, self.device = torch, device
        self.build_query, self.process_image = build_query, process_image
        # Trainability is limited to explicitly added language adapters below.
        model.requires_grad_(False)
        self.lora_targets = []
        self.adapter_config = {}
        if train or adapter_path:
            try:
                from peft import LoraConfig, PeftModel, get_peft_model
            except ImportError as exc:
                raise RuntimeError("LoRA requires peft in the LLaVA environment (see requirements-claim-feedback.txt)") from exc
            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=train)
                self.lora_targets = sorted({target for cfg in model.peft_config.values()
                    for target in ([cfg.target_modules] if isinstance(cfg.target_modules, str) else cfg.target_modules)})
            else:
                self.lora_targets = llm_lora_targets(model)
                cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha,
                                 target_modules=self.lora_targets, lora_dropout=0.0,
                                 bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(model, cfg)
            self.adapter_config = {name: {
                "r": cfg.r, "lora_alpha": cfg.lora_alpha,
                "target_modules": (cfg.target_modules if isinstance(cfg.target_modules, str)
                                   else sorted(cfg.target_modules)),
                "lora_dropout": cfg.lora_dropout, "bias": cfg.bias}
                for name, cfg in model.peft_config.items()}
        self.model = model
        self.base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if train:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "lora_" not in name or any(s in name for s in ("vision_tower", "mm_projector", "vision_resampler")):
                        raise ValueError(f"Unexpected trainable parameter: {name}")
                    # Keep small adapter updates representable with a frozen bf16 base.
                    param.data = param.data.float()
            if not any(p.requires_grad for p in model.parameters()):
                raise ValueError("No trainable language adapter parameters")
        else:
            model.requires_grad_(False)
        # Eval mode disables dropout; autograd still operates on LoRA during SGD.
        model.eval()
        model.config.use_cache = False

    def _inputs(self, record, image_root=None):
        path = Path(record["image"])
        path = path if path.is_absolute() else Path(image_root or ".") / path
        if not path.is_file():
            raise FileNotFoundError(path)
        query = self.build_query(self.tokenizer, record["question"]).to(self.device)[None]
        pixels = self.process_image(self.image_processor, str(path)).to(
            device=self.device, dtype=self.dtype)[None]
        return query, pixels

    def generate(self, record, *, image_root=None, max_new_tokens=256,
                 temperature=0.7, top_p=0.9):
        torch = self.torch
        query, pixels = self._inputs(record, image_root)
        with torch.no_grad():
            result = self.model.generate(
                inputs=query, images=pixels, attention_mask=torch.ones_like(query),
                do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p, max_new_tokens=max_new_tokens, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        seq = result[0]
        if seq.numel() >= query.shape[1] and torch.equal(seq[:query.shape[1]], query[0]):
            seq = seq[query.shape[1]:]
        else:
            # This adapter supports the repo's prompt-prefixed generation API.
            # Guessing whether a mismatched prefix is new-only can mislabel text.
            raise ValueError("Unexpected generation return format: patched LLaVA must return the original prompt prefix")
        ids = seq.cpu().tolist()
        return dict(record, response=decode_exact(self.tokenizer, ids), response_ids=ids,
                    claims=[], generation={"max_new_tokens": max_new_tokens,
                                            "temperature": temperature, "top_p": top_p})

    def token_nll(self, record, ids, *, image_root=None):
        torch = self.torch
        import torch.nn.functional as F
        if not ids:
            raise ValueError("Empty response")
        query, pixels = self._inputs(record, image_root)
        response = torch.tensor([ids], dtype=torch.long, device=self.device)
        full = torch.cat((query, response), dim=1)
        labels = torch.cat((torch.full_like(query, -100), response), dim=1)
        # Both cloud LLaVA-1.5 and the repository patch expand image labels here.
        with torch.no_grad():
            embeds, expanded, attention = prepare_multimodal(self.base,
                full, torch.ones_like(full), labels, pixels)
        if (not isinstance(embeds, torch.Tensor) or not isinstance(expanded, torch.Tensor)
                or embeds.ndim != 3 or expanded.shape != embeds.shape[:2]):
            raise ValueError("Invalid multimodal embedding/label shapes")
        if attention is not None and attention.shape != expanded.shape:
            raise ValueError("Invalid expanded multimodal attention shape")
        positions = (expanded[0] != -100).nonzero(as_tuple=True)[0]
        if positions.numel() != len(ids) or not torch.equal(expanded[0, positions], response[0]):
            raise ValueError("Response labels were truncated or changed during image expansion")
        if positions.numel() and (positions[0] == 0 or not torch.equal(
                positions, torch.arange(embeds.shape[1]-len(ids), embeds.shape[1], device=positions.device))):
            raise ValueError("Response is not the contiguous multimodal tail")
        context = getattr(self.base.config, "max_position_embeddings", None)
        if context and embeds.shape[1] > context:
            raise ValueError("Multimodal sequence exceeds model context; regenerate shorter responses")
        outputs = self.base.get_model()(inputs_embeds=embeds, attention_mask=attention,
                                        use_cache=False, return_dict=True)
        hidden = outputs[0][0, positions - 1].to(self.base.lm_head.weight.dtype)
        logits = self.base.lm_head(hidden).float()
        return F.cross_entropy(logits, response[0], reduction="none")

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(str(path), safe_serialization=False)
        self.tokenizer.save_pretrained(str(path))

    def load_checkpoint(self, path):
        """Restore the existing language adapter without rebuilding the base."""
        import torch
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        path = Path(path)
        if (path / "adapter_model.bin").is_file():
            state = torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file
            state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
        expected = get_peft_model_state_dict(self.model)
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
            raise ValueError("Checkpoint adapter keys/shapes differ from the loaded language adapter")
        set_peft_model_state_dict(self.model, state)
        restored = get_peft_model_state_dict(self.model)
        if any(not torch.equal(restored[k].detach().cpu(), state[k].to(restored[k].dtype)) for k in state):
            raise ValueError("Checkpoint adapter did not restore exactly")


class CharacterTokenizer:
    """ASCII-only fixture tokenizer for a dependency-light synthetic smoke test."""
    def encode(self, text, add_special_tokens=False):
        if any(ord(c) >= 128 for c in text):
            raise ValueError("Synthetic fixture supports ASCII only")
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class TinyBackend:
    """Tiny causal model; deliberately does not pretend to inspect an image."""
    def __init__(self):
        import torch
        self.tokenizer = CharacterTokenizer()
        self.model = torch.nn.Sequential(torch.nn.Embedding(128, 16),
                                         torch.nn.Linear(16, 128))

    def token_nll(self, record, ids, **kwargs):
        import torch
        import torch.nn.functional as F
        inputs = torch.tensor([0] + list(ids[:-1]), dtype=torch.long)
        return F.cross_entropy(self.model(inputs), torch.tensor(ids), reduction="none")

    def save(self, path):
        import torch
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save(self.model.state_dict(), path / "synthetic_model.pt")

    def load_checkpoint(self, path):
        import torch
        state = torch.load(Path(path) / "synthetic_model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
