"""Score source-aligned object mentions with raw, unthresholded Grounding DINO.

The official IDEA-Research/grounding-dino-tiny Hugging Face conversion is loaded
locally. No boxes are filtered before calibration, and no detector score is
turned into a factuality label. One image is a recoverable scoring unit.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from run.build_grounding_corpus import atomic_bytes, digest, sha, stage_lock, write_json
from mvpg_run.claim_feedback.diagnostics import image_path, load_cases, validate_cases

SCORE_KEYS = ("dino_max", "dino_mean", "dino_min")
PROTOCOL = {
    "backend": "hf_grounding_dino_unfiltered",
    "reference_model": "IDEA-Research/grounding-dino-tiny",
    "text_form": "single_raw_object_phrase_lowercase_with_final_period",
    "content_tokens": "non-special tokens whose source character span contains a letter or digit; punctuation excluded",
    "query_selection": "all decoder queries, no box/text threshold, no NMS",
    "dino_max": "max_query(max_content_token(sigmoid(logit)))",
    "dino_mean": "max_query(mean_content_token(sigmoid(logit)))",
    "dino_min": "max_query(min_content_token(sigmoid(logit)))",
    "batching": "one independent caption per image-text pair; no multi-object caption",
    "image_preprocessing": "frozen HF processor; whole original image, no crops",
    "precision": "float32, no autocast",
    "disable_custom_kernels": True,
    "interpretation": "uncalibrated detector matching evidence, not probability of factual correctness or proof of absence",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def caption_for(phrase):
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("An object mention must be nonempty text")
    caption = phrase.lower().strip()
    return caption if caption.endswith(".") else caption + "."


def content_positions(caption, token_ids, offsets, special_mask, attention_mask,
                      special_ids, max_text_len):
    """Use exact character offsets, never a guessed [1:-2] token slice."""
    if not (len(token_ids) == len(offsets) == len(special_mask) == len(attention_mask)):
        raise ValueError("Tokenizer alignment arrays have different lengths")
    if sum(bool(x) for x in attention_mask) > max_text_len:
        raise ValueError("Grounding DINO caption exceeds max_text_len; refusing silent truncation")
    positions = []
    for index, (tid, bounds, special, valid) in enumerate(
            zip(token_ids, offsets, special_mask, attention_mask)):
        start, end = bounds
        if not 0 <= start <= end <= len(caption):
            raise ValueError("Tokenizer returned invalid character offsets")
        if valid and not special and tid not in special_ids and any(c.isalnum() for c in caption[start:end]):
            positions.append(index)
    if not positions:
        raise ValueError("Object mention has no representable content tokens; it must not receive a zero/negative score")
    return positions


def summarize_predictions(logits, boxes, positions, *, top_k, width, height):
    """Summarize every query, retaining below-threshold evidence for audits."""
    import torch
    if (logits.ndim != 2 or boxes.ndim != 2 or boxes.shape[1] != 4
            or logits.shape[0] != boxes.shape[0] or logits.shape[0] == 0
            or not positions or max(positions) >= logits.shape[1]):
        raise ValueError("Unexpected Grounding DINO output shapes or content-token positions")
    # Padding logits can legitimately be -inf; selected content logits cannot.
    content = logits[:, positions].float()
    if not torch.isfinite(content).all() or not torch.isfinite(boxes).all():
        raise ValueError("Grounding DINO returned nonfinite content logits or boxes")
    probabilities = content.sigmoid()
    query_max = probabilities.max(dim=-1).values
    query_mean = probabilities.mean(dim=-1)
    query_min = probabilities.min(dim=-1).values
    score = {key: float(values.max().item()) for key, values in
             zip(SCORE_KEYS, (query_max, query_mean, query_min))}
    # Ties are ordered by original query index for stable evidence serialization.
    indices = sorted(range(len(query_max)), key=lambda i: (-float(query_max[i].item()), i))[:top_k]
    candidates = []
    for index in indices:
        cx, cy, box_width, box_height = [float(x) for x in boxes[index].tolist()]
        if not all(0 <= x <= 1 for x in (cx, cy, box_width, box_height)):
            raise ValueError("Predicted normalized box coordinates must be in [0, 1]")
        xyxy = [cx - box_width / 2, cy - box_height / 2,
                cx + box_width / 2, cy + box_height / 2]
        clipped = [min(1.0, max(0.0, x)) for x in xyxy]
        candidates.append({"query_index": index,
            "content_token_probabilities": [float(x) for x in probabilities[index].tolist()],
            "max_content_score": float(query_max[index].item()),
            "mean_content_score": float(query_mean[index].item()),
            "min_content_score": float(query_min[index].item()),
            "box_cxcywh_normalized": [cx, cy, box_width, box_height],
            "box_xyxy_normalized_unclipped": xyxy,
            "box_xyxy_pixels_clipped": [clipped[0] * width, clipped[1] * height,
                                         clipped[2] * width, clipped[3] * height]})
    return {"scores": score, "query_count": len(query_max), "top_candidates": candidates,
            "best_box_xyxy_pixels": candidates[0]["box_xyxy_pixels_clipped"],
            "best_query_index": candidates[0]["query_index"]}


class HFDetector:
    def __init__(self, model_root, *, device="cuda", batch_size=4, top_k=5):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
        if not self.processor.tokenizer.is_fast:
            raise ValueError("Grounding DINO requires a fast tokenizer for exact content offsets")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_root, local_files_only=True, disable_custom_kernels=True,
            torch_dtype=torch.float32).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.batch_size, self.top_k = device, batch_size, top_k

    def __call__(self, path, phrases):
        from PIL import Image
        torch = self.torch
        with Image.open(path) as source:
            image = source.convert("RGB")
        width, height = image.size
        # Reuse image preprocessing; joint vision/text features remain caption-conditioned.
        pixels = self.processor.image_processor(images=image, return_tensors="pt")
        pixels = {key: value.to(self.device) for key, value in pixels.items()}
        results = []
        with torch.inference_mode():
            for start in range(0, len(phrases), self.batch_size):
                captions = [caption_for(p) for p in phrases[start:start + self.batch_size]]
                tokenized = self.processor.tokenizer(captions, padding=True, truncation=False,
                    return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors="pt")
                offsets = tokenized.pop("offset_mapping").tolist()
                specials = tokenized.pop("special_tokens_mask").tolist()
                ids = tokenized["input_ids"].tolist()
                attention = tokenized["attention_mask"].tolist()
                positions = [content_positions(text, token_ids, offs, mask, valid,
                    set(self.processor.tokenizer.all_special_ids), self.model.config.max_text_len)
                    for text, token_ids, offs, mask, valid in zip(captions, ids, offsets, specials, attention)]
                count = len(captions)
                image_batch = {key: value.expand(count, *value.shape[1:]) for key, value in pixels.items()}
                outputs = self.model(**image_batch, **{key: value.to(self.device) for key, value in tokenized.items()})
                for i, caption in enumerate(captions):
                    summary = summarize_predictions(outputs.logits[i].detach().cpu(),
                        outputs.pred_boxes[i].detach().cpu(), positions[i], top_k=self.top_k,
                        width=width, height=height)
                    selected_ids = [ids[i][j] for j in positions[i]]
                    summary.update(caption=caption, content_token_positions=positions[i],
                        content_token_ids=selected_ids,
                        content_tokens=self.processor.tokenizer.convert_ids_to_tokens(selected_ids),
                        image_size={"width": width, "height": height})
                    results.append(summary)
        return results


def model_fingerprint(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    extensions = {".safetensors", ".bin", ".json", ".txt", ".model", ".py"}
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix in extensions and not any(
                 part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts)}
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("No frozen model weights found in model root")
    if "config.json" not in files or "preprocessor_config.json" not in files:
        raise ValueError("Model root requires config.json and preprocessor_config.json")
    return {"root": str(root), "files_sha256": files}


def runtime_versions():
    from importlib.metadata import version
    return {name: version(name) for name in ("torch", "transformers", "tokenizers", "Pillow")}


def group_cases(cases, image_root):
    groups = OrderedDict()
    for case in cases:
        path = str(image_path(case["image"], image_root))
        groups.setdefault(path, []).append(case)
    return list(groups.items())


class ImageJournal:
    """Atomic image units: a killed job cannot leave a partially scored image."""
    def __init__(self, root, manifest, groups, *, resume=False):
        self.root, self.groups, self.manifest = Path(root), groups, manifest
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        if resume:
            if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("DINO resume manifest differs or is missing: input/model/image/runtime/protocol changed")
        else:
            if any(p.name != ".run.lock" for p in self.root.iterdir()):
                raise FileExistsError("DINO journal already exists; use --resume")
            write_json(path, manifest)
        self.shards = self.root / "shards"
        self.shards.mkdir(exist_ok=True)
        self.completed_images, self.completed_cases = 0, 0
        self.rows = {}
        names = sorted(p for p in self.shards.iterdir() if p.is_file() and not p.name.startswith("."))
        for ordinal, path in enumerate(names):
            if path.name != f"image-{ordinal:06d}.json" or ordinal >= len(groups):
                raise ValueError("DINO image journal is non-contiguous or has extra records")
            rows = self._load(path, ordinal)
            self.rows.update({row["case_id"]: row for row in rows})
            self.completed_images += 1
            self.completed_cases += len(rows)

    def _load(self, path, index):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        image, source = self.groups[index]
        if (envelope.get("sha256") != digest(payload) or payload.get("image") != image
                or payload.get("image_sha256") != self.manifest["image_sha256"][image]
                or payload.get("input_sha256") != [digest(c) for c in source]):
            raise ValueError("DINO image journal checksum/input differs")
        rows = payload.get("records")
        if not isinstance(rows, list) or len(rows) != len(source):
            raise ValueError("DINO journal image is not fully scored")
        validate_cases(rows)
        for expected, row in zip(source, rows):
            if row.get("dino_input_sha256") != digest(expected):
                raise ValueError("DINO result differs from source claim")
            protected = set(expected) - {"scores", "score_provenance"}
            if any(row.get(key) != expected[key] for key in protected):
                raise ValueError("DINO scoring rewrote an input field")
            if any(row["scores"].get(key) != value for key, value in expected["scores"].items()):
                raise ValueError("DINO scoring overwrote an existing score")
            if any(key not in row["scores"] or not 0 <= row["scores"][key] <= 1 for key in SCORE_KEYS):
                raise ValueError("DINO result requires finite [0,1] matching scores")
        return rows

    def commit(self, rows):
        index = self.completed_images
        image, source = self.groups[index]
        payload = {"image": image, "image_sha256": self.manifest["image_sha256"][image],
                   "input_sha256": [digest(c) for c in source], "records": rows}
        target = self.shards / f"image-{index:06d}.json"
        if target.exists():
            raise FileExistsError(target)
        temporary = target.with_name(".validate-" + target.name)
        write_json(temporary, {"sha256": digest(payload), "payload": payload})
        try:
            self._load(temporary, index)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        self.rows.update({row["case_id"]: row for row in rows})
        self.completed_images += 1
        self.completed_cases += len(rows)


def run(args, *, backend_factory=HFDetector, versions=None):
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size and top-k must be positive")
    cases = load_cases(args.input)
    if not cases:
        raise ValueError("No eligible object claims to score")
    if any(key in c["scores"] for c in cases for key in SCORE_KEYS):
        raise ValueError("Input already contains a DINO score; refusing to overwrite")
    groups = group_cases(cases, args.image_root)
    fingerprints = {image: sha(image) for image, _ in groups}
    model = model_fingerprint(args.model_root)
    versions = versions if versions is not None else runtime_versions()
    manifest = {"schema": "grounding_dino_image_journal_v1", "input_sha256": sha(args.input),
        "ordered_case_ids": [c["case_id"] for c in cases], "image_sha256": fingerprints,
        "ordered_images": list(fingerprints), "model": model, "runtime": versions,
        "device": args.device, "batch_size": args.batch_size, "top_k": args.top_k,
        "protocol": PROTOCOL, "implementation_sha256": sha(__file__)}
    output = Path(args.output).expanduser().resolve()
    journal_root = Path(str(output) + ".journal")
    output.parent.mkdir(parents=True, exist_ok=True)
    journal_root.mkdir(exist_ok=True)
    with stage_lock(journal_root):
        if output.exists() and not args.resume:
            raise FileExistsError(output)
        journal = ImageJournal(journal_root, manifest, groups, resume=args.resume)
        status_path = journal_root / "status.json"
        start, previous = time.monotonic(), journal.completed_images
        def status(state, **extra):
            data = {"stage": "grounding_dino_score", "status": state,
                "total_images": len(groups), "total_cases": len(cases),
                "completed_images": journal.completed_images, "completed_cases": journal.completed_cases,
                "session_completed_images": journal.completed_images - previous,
                "session_elapsed_seconds": time.monotonic() - start, "updated_utc": now(),
                "output": str(output), **extra}
            write_json(status_path, data)
            return data
        status("running")
        try:
            if journal.completed_images < len(groups):
                backend = backend_factory(str(Path(args.model_root).resolve()), device=args.device,
                                          batch_size=args.batch_size, top_k=args.top_k)
                for image, image_cases in groups[journal.completed_images:]:
                    predictions = backend(image, [c["claim_text"] for c in image_cases])
                    if len(predictions) != len(image_cases):
                        raise ValueError("DINO did not return exactly one result per claim")
                    scored = []
                    for case, result in zip(image_cases, predictions):
                        row = copy.deepcopy(case)
                        row["scores"].update(result["scores"])
                        provenance = {"backend": PROTOCOL["backend"], "root": model["root"],
                            "score_mode": "all_queries_content_tokens", "text_form": "raw_phrase",
                            "input_sha256": manifest["input_sha256"], "model_sha256": digest(model),
                            "protocol_sha256": digest(PROTOCOL), "runtime": versions,
                            "image_sha256": fingerprints[image]}
                        row.setdefault("score_provenance", {}).update({key: dict(provenance,
                            aggregation=PROTOCOL[key]) for key in SCORE_KEYS})
                        row["dino_input_sha256"] = digest(case)
                        row["detection_evidence"] = {key: value for key, value in result.items() if key != "scores"}
                        scored.append(row)
                    journal.commit(scored)
                    progress = status("running")
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
            ordered = [journal.rows[c["case_id"]] for c in cases]
            validate_cases(ordered)
            atomic_bytes(output, b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n").encode()
                                         for row in ordered))
            return status("complete", output_sha256=sha(output))
        except BaseException as exc:
            status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                   error_type=type(exc).__name__)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Frozen cases.jsonl; all original fields are retained")
    parser.add_argument("--output", required=True, help="Final scored.jsonl, published atomically")
    parser.add_argument("--model-root", required=True, help="Local official HF grounding-dino-tiny snapshot")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4, help="Independent object captions paired with the same image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of unthresholded candidate boxes retained per claim")
    parser.add_argument("--resume", action="store_true")
    result = run(parser.parse_args(argv))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()
