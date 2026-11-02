"""Staged JSONL workflow; run ``python -m mvpg_run.claim_feedback --help``."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import math
from pathlib import Path

from .data import (classify_claims, import_claims, read_jsonl, sentence_claims,
                   validate_record, write_jsonl)


def build_parser():
    parser = argparse.ArgumentParser(description="Source-anchored visual feedback without response preference pairs")
    commands = parser.add_subparsers(dest="command", required=True)

    def io(name, help):
        cmd = commands.add_parser(name, help=help)
        cmd.add_argument("--input", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--overwrite", action="store_true")
        return cmd

    def model_args(cmd):
        cmd.add_argument("--model-name", required=True, help="Local patched LLaVA base checkpoint")
        cmd.add_argument("--adapter-path", help="Optional local LoRA adapter; starts a new run")
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
        cmd.add_argument("--image-root", default=".")

    gen = io("generate", "Generate and preserve exact response IDs (no reference answers)")
    model_args(gen)
    gen.add_argument("--max-new-tokens", type=int, default=256)
    gen.add_argument("--temperature", type=float, default=0.7)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--seed", type=int, default=42)

    dec = io("decompose", "DiscoSG object-existence prototype or explicit sentence ablation")
    dec.add_argument("--parser", choices=("disco", "sentences"), default="disco")
    dec.add_argument("--disco-root")
    dec.add_argument("--seed-model")
    dec.add_argument("--refiner-model")
    dec.add_argument("--device", default="cpu")
    dec.add_argument("--rounds", type=int, default=3)

    ver = io("verify", "Frozen FG-CLIP2 cosine proxy with a calibrated abstention band")
    ver.add_argument("--fgclip-root", required=True)
    ver.add_argument("--device", default="cpu")
    ver.add_argument("--image-root", default=".")
    ver.add_argument("--score-mode", choices=("global", "patch"), default="global")
    ver.add_argument("--walk-type", choices=("box", "short", "long"), default="short")
    ver.add_argument("--negative-threshold", required=True, type=float)
    ver.add_argument("--positive-threshold", required=True, type=float)
    ver.add_argument("--calibration-id", required=True, help="Independent calibration artifact/version, not evaluation benchmark")
    ver.add_argument("--accepted-kinds", nargs="+", default=["object"])
    ver.add_argument("--feedback-weight", type=float, default=1.0)
    ver.add_argument("--allow-whole-claim-negative", action="store_true")

    audit = commands.add_parser("audit", help="Report weak labels/anchors; no hallucination accuracy claim")
    audit.add_argument("--input", required=True)
    audit.add_argument("--output")

    train = commands.add_parser("train", help="Train LLM-only LoRA: one response = one manual SGD step")
    model_args(train)
    train.add_argument("--input", required=True)
    train.add_argument("--output-dir", required=True, help="Must not already exist")
    train.add_argument("--mode", choices=("modulated", "fixed", "positive", "negative", "project_only"), default="modulated")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--alpha", type=float, default=1.0)
    train.add_argument("--beta", type=float, default=1.0)
    train.add_argument("--max-update-norm", type=float, default=1e-3,
                       help="L2 cap on actual parameter delta, not raw gradient norm")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--max-steps", type=int, help="Cap attempted responses, including skipped feedback")
    train.add_argument("--max-updates", type=int,
                       help="Stop after this many successful parameter updates; fail if the data/attempt budget ends first")
    train.add_argument("--checkpoint-every", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--lora-rank", type=int, default=16)
    train.add_argument("--lora-alpha", type=int, default=32)
    train.add_argument("--allow-whole-claim-negative", action="store_true")
    train.add_argument("--audit-loss-change", action="store_true",
                       help="Extra forward to log selected positive/negative NLL after each step")

    smoke = commands.add_parser("smoke", help="Synthetic CPU mechanics only; no vision model or performance claim")
    smoke.add_argument("--output-dir", required=True, help="Must not already exist")
    smoke.add_argument("--mode", choices=("modulated", "fixed", "positive", "negative", "project_only"), default="modulated")
    smoke.add_argument("--seed", type=int, default=42)
    return parser


def _llava(args, train=False):
    from .backends import LlavaBackend
    kwargs = dict(device=args.device, precision=args.precision,
                  train=train, adapter_path=args.adapter_path)
    if train:
        kwargs.update(lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
    return LlavaBackend(args.model_name, **kwargs)


def _output_available(args):
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use a new name or --overwrite")


def _invalidate_feedback(record, *, response=False):
    """Invalidate derived annotations when an upstream stage is rerun."""
    out = copy.deepcopy(record)
    metadata = out.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("record.metadata must be an object")
    metadata.pop("verifier", None)
    if response:
        out["metadata"].pop("decomposer", None)
        for key in ("response_ids", "disco_graph", "generation"):
            out.pop(key, None)
        out.update(response="", claims=[])
    return out


def run(args):
    if args.command == "generate":
        from .training import seed_everything
        if (args.max_new_tokens <= 0 or not math.isfinite(args.temperature)
                or args.temperature < 0 or not 0 < args.top_p <= 1):
            raise ValueError("Require positive max-new-tokens, nonnegative temperature, and 0 < top-p <= 1")
        _output_available(args)
        records = [validate_record(_invalidate_feedback(r, response=True))
                   for r in read_jsonl(args.input, validate=False)]
        seed_everything(args.seed)
        backend = _llava(args)
        def generated():
            for record in records:
                result = backend.generate(record, image_root=args.image_root,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
                result["generation"].update(seed=args.seed, model_name=str(Path(args.model_name).resolve()),
                                            adapter_path=args.adapter_path)
                yield result
        return {"records": write_jsonl(args.output, generated(), overwrite=args.overwrite)}

    if args.command == "decompose":
        _output_available(args)
        records = [_invalidate_feedback(r) for r in read_jsonl(args.input)]
        if args.parser == "sentences":
            for record in records:
                record.pop("disco_graph", None)
                record["metadata"].pop("decomposer", None)
            output = (import_claims(r, sentence_claims(r["response"])) for r in records)
        else:
            if not all((args.disco_root, args.seed_model, args.refiner_model)):
                raise ValueError("DiscoSG requires --disco-root, --seed-model and --refiner-model (user-supplied official weights)")
            from .disco import DiscoSGDecomposer
            decomposer = DiscoSGDecomposer(args.disco_root, args.seed_model,
                args.refiner_model, device=args.device, rounds=args.rounds)
            output = (decomposer.decompose(r) for r in records)
        return {"records": write_jsonl(args.output, output, overwrite=args.overwrite)}

    if args.command == "verify":
        _output_available(args)
        # Check thresholds and schema before loading potentially large weights.
        if (not math.isfinite(args.negative_threshold) or not math.isfinite(args.positive_threshold)
                or args.negative_threshold >= args.positive_threshold):
            raise ValueError("Finite negative-threshold must be smaller than positive-threshold")
        records = list(read_jsonl(args.input))
        if not args.calibration_id.strip() or not math.isfinite(args.feedback_weight) or not 0 <= args.feedback_weight <= 1:
            raise ValueError("Require a nonempty calibration-id and finite feedback-weight in [0,1]")
        from .backends import FGClipVerifier
        scorer = FGClipVerifier(args.fgclip_root, args.device, args.score_mode,
                                args.image_root, args.walk_type)
        def verified():
            for record in records:
                result = classify_claims(record, scorer, args.negative_threshold,
                    args.positive_threshold, args.accepted_kinds, args.feedback_weight,
                    allow_whole_claim_negative=args.allow_whole_claim_negative)
                result.setdefault("metadata", {})["verifier"] = {
                    "backend": "fgclip2_cosine_weak_feedback", "root": str(Path(args.fgclip_root).resolve()),
                    "score_mode": args.score_mode, "walk_type": args.walk_type,
                    "negative_threshold": args.negative_threshold,
                    "positive_threshold": args.positive_threshold,
                    "calibration_id": args.calibration_id, "accepted_kinds": args.accepted_kinds,
                    "allow_whole_claim_negative": args.allow_whole_claim_negative}
                yield result
        return {"records": write_jsonl(args.output, verified(), overwrite=args.overwrite)}

    if args.command == "audit":
        labels, kinds, reasons = Counter(), Counter(), Counter()
        n_records, anchored, error_anchored = 0, 0, 0
        for record in read_jsonl(args.input):
            n_records += 1
            for claim in record["claims"]:
                labels[claim["label"]] += 1
                kinds[claim["kind"]] += 1
                reasons[claim.get("feedback_reason", claim.get("alignment_reason", "unspecified"))] += 1
                anchored += bool(claim["source_spans"])
                error_anchored += bool(claim.get("error_spans"))
        report = {"records": n_records, "labels": dict(labels), "kinds": dict(kinds),
                  "feedback_reasons": dict(reasons), "source_anchored_claims": anchored,
                  "error_anchored_claims": error_anchored,
                  "note": "Weak feedback counts only; token-level alignment is logged by train"}
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    if args.command == "train":
        from .training import seed_everything, train_pool, validate_training_options
        if Path(args.output_dir).exists():
            raise FileExistsError("Training output directory already exists; choose a new run directory")
        validate_training_options(mode=args.mode, lr=args.lr, alpha=args.alpha,
            beta=args.beta, max_update_norm=args.max_update_norm, epochs=args.epochs,
            max_steps=args.max_steps, checkpoint_every=args.checkpoint_every,
            max_updates=args.max_updates)
        if args.lora_rank <= 0 or args.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        # Fail invalid data before loading the full model.
        if not list(read_jsonl(args.input)):
            raise ValueError("Feedback pool is empty")
        seed_everything(args.seed)
        backend = _llava(args, train=True)
        config = vars(args).copy()
        config["lora_targets"] = backend.lora_targets
        config["actual_adapter_config"] = backend.adapter_config
        config["trainable_parameters"] = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
        return train_pool(backend, args.input, args.output_dir, mode=args.mode, lr=args.lr,
            alpha=args.alpha, beta=args.beta, max_update_norm=args.max_update_norm,
            epochs=args.epochs, max_steps=args.max_steps, max_updates=args.max_updates,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed, image_root=args.image_root,
            allow_whole_claim_negative=args.allow_whole_claim_negative,
            audit_loss_change=args.audit_loss_change, config=config)

    if args.command == "smoke":
        return smoke(args)
    raise ValueError("Unknown command")


def smoke(args):
    from .backends import TinyBackend, response_tokens
    from .data import align_claims
    from .optimization import masked_objectives
    from .training import seed_everything, train_pool, write_json
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    backend = TinyBackend()
    # Artificial scores isolate plumbing; no image is opened in this fixture.
    record = validate_record({"id": "synthetic-1", "image": "synthetic://no-image",
        "question": "Describe the synthetic fixture", "response": "A cat and a dog.",
        "claims": [
            {"id": "cat", "text": "a cat", "kind": "object", "source_spans": [[2, 5]], "error_spans": [[2, 5]]},
            {"id": "dog", "text": "a dog", "kind": "object", "source_spans": [[12, 15]], "error_spans": [[12, 15]]}]})
    record["response_ids"] = backend.tokenizer.encode(record["response"])
    record = classify_claims(record, lambda image, texts: [0.9, 0.1], 0.2, 0.8)
    write_jsonl(root / "synthetic_feedback.jsonl", [record])
    ids, offsets = response_tokens(backend.tokenizer, record)
    units, _ = align_claims(record, offsets)
    def losses():
        import torch
        with torch.no_grad():
            p, n, _ = masked_objectives(backend.token_nll(record, ids), units)
        return {"positive_nll": float(p), "negative_nll": float(n)}
    before = losses()
    manifest = train_pool(backend, root / "synthetic_feedback.jsonl", root / "run",
        mode=args.mode, lr=0.01, max_update_norm=0.01, seed=args.seed,
        config={"purpose": "Synthetic CPU mechanics only; no visual/hallucination benchmark"})
    report = {"synthetic": True, "mode": args.mode, "before": before, "after": losses(),
              "counts": manifest["counts"], "output_dir": str(root),
              "limitation": "Synthetic gradients, not evidence of hallucination reduction or capability preservation"}
    write_json(root / "smoke_report.json", report)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
