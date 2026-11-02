"""One-response, one-actual-SGD-step experiments on a frozen feedback pool."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

from .backends import response_tokens
from .data import align_claims, read_jsonl
from .optimization import masked_objectives, response_step


def validate_training_options(*, mode, lr, alpha, beta, max_update_norm,
                              epochs, max_steps, checkpoint_every):
    from .optimization import MODES
    if mode not in MODES:
        raise ValueError(f"Unknown training mode: {mode}")
    for name, value, positive in (("lr", lr, True), ("alpha", alpha, False),
                                  ("beta", beta, False), ("max_update_norm", max_update_norm, True)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (positive and value == 0):
            raise ValueError(f"Invalid {name}: {value}")
    for name, value, minimum in (("epochs", epochs, 1), ("max_steps", max_steps, 1),
                                  ("checkpoint_every", checkpoint_every, 0)):
        if name == "max_steps" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def seed_everything(seed):
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def provenance():
    versions = {}
    for name in ("torch", "transformers", "peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo,
                                            text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(), "packages": versions}


def train_pool(backend, input_path, output_dir, *, mode="modulated", lr=1e-3,
               alpha=1.0, beta=1.0, max_update_norm=1e-3, epochs=1,
               max_steps=None, checkpoint_every=0, seed=42, image_root=None,
               allow_whole_claim_negative=False, audit_loss_change=False, config=None):
    """Fixed ordered pool, no batches/momentum/implicit refreshed supervision.

    Alignment failures skip the response and are counted; missing images, invalid
    model interfaces and numerical errors fail the run with a manifest reason.
    ``max_steps`` counts attempted responses, including explicit skips.
    """
    validate_training_options(mode=mode, lr=lr, alpha=alpha, beta=beta,
        max_update_norm=max_update_norm, epochs=epochs, max_steps=max_steps,
        checkpoint_every=checkpoint_every)
    records = list(read_jsonl(input_path))
    if not records:
        raise ValueError("Feedback pool is empty")
    seed_everything(seed)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    settings = dict(mode=mode, lr=lr, alpha=alpha, beta=beta,
                    max_update_norm=max_update_norm, epochs=epochs,
                    max_steps=max_steps, checkpoint_every=checkpoint_every,
                    seed=seed, image_root=str(Path(image_root or ".").resolve()),
                    allow_whole_claim_negative=allow_whole_claim_negative,
                    audit_loss_change=audit_loss_change)
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "pool": str(Path(input_path).resolve()),
                "pool_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                "pool_records": len(records), "settings": settings,
                "backend": config or {}, "provenance": provenance(),
                "protocol": "fixed_feedback_pool_one_response_manual_sgd",
                "synthetic": backend.__class__.__name__ == "TinyBackend"}
    write_json(root / "manifest.json", manifest)
    counts = Counter(attempted=0, updated=0, skipped=0)
    reasons = Counter()
    try:
        with (root / "steps.jsonl").open("x", encoding="utf-8") as log:
            stop = False
            for epoch in range(epochs):
                for record in records:
                    if max_steps is not None and counts["attempted"] >= max_steps:
                        stop = True
                        break
                    counts["attempted"] += 1
                    started = time.perf_counter()
                    row = {"attempt": counts["attempted"], "epoch": epoch,
                           "record_id": record["id"]}
                    try:
                        ids, offsets = response_tokens(backend.tokenizer, record)
                    except ValueError as exc:
                        row.update(skipped=True, reason="unalignable_response",
                                   detail=str(exc))
                    else:
                        units, alignment = align_claims(record, offsets,
                            allow_whole_claim_negative=allow_whole_claim_negative)
                        row["alignment"] = alignment
                        # Make token credit assignment auditable without raw gradients.
                        row["units"] = units
                        if not units:
                            row.update(skipped=True, reason="no_aligned_feedback")
                        else:
                            nll = backend.token_nll(record, ids, image_root=image_root)
                            pos, neg, objective_stats = masked_objectives(nll, units)
                            row["objectives"] = objective_stats
                            row.update(response_step(backend.model, pos, neg, mode=mode,
                                lr=lr, alpha=alpha, beta=beta,
                                max_update_norm=max_update_norm))
                            del nll, pos, neg
                            if audit_loss_change:
                                import torch
                                with torch.no_grad():
                                    nll_after = backend.token_nll(record, ids, image_root=image_root)
                                    pos_after, neg_after, _ = masked_objectives(nll_after, units)
                                row["positive_loss_after"] = float(pos_after) if pos_after is not None else None
                                row["negative_loss_after"] = float(neg_after) if neg_after is not None else None
                                del nll_after, pos_after, neg_after
                    row["elapsed_seconds"] = time.perf_counter() - started
                    counts["skipped" if row["skipped"] else "updated"] += 1
                    reasons[row["reason"]] += 1
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    if checkpoint_every and counts["attempted"] % checkpoint_every == 0:
                        backend.save(root / f"checkpoint-{counts['attempted']:06d}")
                if stop:
                    break
        backend.save(root / "final")
        manifest.update(status="complete", counts=dict(counts), reasons=dict(reasons))
    except Exception as exc:
        manifest.update(status="failed", counts=dict(counts), reasons=dict(reasons),
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "manifest.json", manifest)
    return manifest
