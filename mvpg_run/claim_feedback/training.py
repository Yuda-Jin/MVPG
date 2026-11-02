"""One-response, one-actual-SGD-step experiments on a frozen feedback pool."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from .backends import CharacterTokenizer, response_tokens
from .data import align_claims, read_jsonl
from .optimization import masked_objectives, response_step


class TrainingTargetNotReached(RuntimeError):
    """The configured data/attempt budget ended before the update target."""


def validate_training_options(*, mode, lr, alpha, beta, max_update_norm,
                              epochs, max_steps, checkpoint_every, max_updates=None,
                              log_every=10):
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
                                  ("max_updates", max_updates, 1),
                                  ("log_every", log_every, 1),
                                  ("checkpoint_every", checkpoint_every, 0)):
        if name in ("max_steps", "max_updates") and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")


def seed_everything(seed):
    import torch
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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


TRAINING_DEFAULTS = dict(mode="modulated", lr=1e-3, alpha=1.0, beta=1.0,
    max_update_norm=1e-3, epochs=1, max_steps=None, max_updates=None,
    checkpoint_every=500, seed=42, log_every=10, tensorboard=False,
    allow_whole_claim_negative=False, audit_loss_change=False)


def train_pool(backend, input_path, output_dir, *, mode="modulated", lr=1e-3,
               alpha=1.0, beta=1.0, max_update_norm=1e-3, epochs=1,
               max_steps=None, max_updates=None, checkpoint_every=500, seed=42, image_root=None,
               allow_whole_claim_negative=False, audit_loss_change=False, config=None,
               resume_from=None, allow_legacy_resume=False, log_every=10, tensorboard=False):
    """Fixed ordered pool with cumulative budgets and recoverable checkpoints.

    Resume always writes a NEW run directory, preserving the parent experiment.
    Epochs, max_steps (attempts) and max_updates are cumulative, not additional.
    Periodic checkpoints count successful updates only; final is always saved.
    No partial/failed attempt is published as a recoverable checkpoint.
    """
    from .checkpointing import (backend_signature, load_resume, restore_rng_state,
                               save_checkpoint)
    from .metrics import TrainingMetrics
    import torch

    validate_training_options(mode=mode, lr=lr, alpha=alpha, beta=beta,
        max_update_norm=max_update_norm, epochs=epochs, max_steps=max_steps,
        checkpoint_every=checkpoint_every, max_updates=max_updates, log_every=log_every)
    if not isinstance(tensorboard, bool) or not isinstance(allow_legacy_resume, bool):
        raise ValueError("tensorboard and allow_legacy_resume must be boolean")
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError("Training needs a new output directory, including continuation runs")
    records = list(read_jsonl(input_path))
    if not records:
        raise ValueError("Feedback pool is empty")
    settings = dict(mode=mode, lr=lr, alpha=alpha, beta=beta,
        max_update_norm=max_update_norm, epochs=epochs, max_steps=max_steps,
        max_updates=max_updates, checkpoint_every=checkpoint_every, seed=seed,
        image_root=str(Path(image_root or ".").resolve()),
        allow_whole_claim_negative=allow_whole_claim_negative,
        audit_loss_change=audit_loss_change, log_every=log_every, tensorboard=tensorboard)
    signature = backend_signature(backend, config or {})
    pool_sha = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    resumed = None
    if resume_from:
        resumed = load_resume(resume_from, input_path=input_path, records=records,
                              settings=settings, signature=signature,
                              allow_legacy=allow_legacy_resume)
        if not callable(getattr(backend, "load_checkpoint", None)):
            raise ValueError("Backend does not implement checkpoint restoration")
        backend.load_checkpoint(resume_from)
    counts = Counter(attempted=0, updated=0, skipped=0)
    reasons = Counter()
    history = b""
    resume_info = None
    if resumed:
        counts.update(resumed["state"]["counts"])
        reasons.update(resumed["state"]["reasons"])
        history = resumed["history_bytes"]
        resume_info = resumed["resume_info"]
    counts_at_start = dict(counts)
    cursor = dict(epoch=counts["attempted"] // len(records),
                  record_index=counts["attempted"] % len(records))
    root.mkdir(parents=True, exist_ok=False)
    steps_path = root / "steps.jsonl"
    steps_path.write_bytes(history)  # Preserve the immutable source prefix byte-for-byte.
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "pool": str(Path(input_path).resolve()), "pool_sha256": pool_sha,
        "pool_records": len(records), "settings": settings, "backend": config or {},
        "provenance": provenance(), "signature": signature, "resume": resume_info,
        "counts_at_start": counts_at_start, "counts": dict(counts), "cursor": cursor,
        "protocol": "fixed_feedback_pool_one_response_manual_sgd",
        "checkpoint_step_unit": "successful_updates",
        "optimizer_state": "none: manual SGD without momentum or scheduler",
        "synthetic": isinstance(getattr(backend, "tokenizer", None), CharacterTokenizer)}
    write_json(root / "manifest.json", manifest)
    metrics = None
    cuda_devices = sorted({p.device.index for p in backend.model.parameters() if p.is_cuda})

    def checkpoint(path):
        save_checkpoint(backend, path, history_path=steps_path, counts=dict(counts),
            reasons=dict(reasons), cursor=cursor, settings=settings, pool_path=input_path,
            pool_sha256=pool_sha, pool_records=len(records), signature=signature,
            resume_info=resume_info)

    try:
        metrics = TrainingMetrics(root, tensorboard=tensorboard)
        for i, line in enumerate(history.splitlines()):
            if line.strip():
                metrics.record(json.loads(line), record_index=i % len(records))
        if resumed and resumed["rng_state"] is not None:
            restore_rng_state(resumed["rng_state"])
        else:
            # Legacy continuations have no saved random stream; this is explicit
            # in their provenance. The current eval-mode LLaVA SGD is deterministic.
            seed_everything(seed)
        start_epoch, start_index = cursor["epoch"], cursor["record_index"]
        with steps_path.open("a", encoding="utf-8", newline="\n") as log:
            stop = False
            for epoch in range(start_epoch, epochs):
                first_index = start_index if epoch == start_epoch else 0
                for record_index in range(first_index, len(records)):
                    if max_updates is not None and counts["updated"] >= max_updates:
                        stop = True
                        break
                    if max_steps is not None and counts["attempted"] >= max_steps:
                        stop = True
                        break
                    record = records[record_index]
                    started = time.perf_counter()
                    for device in cuda_devices:
                        torch.cuda.reset_peak_memory_stats(device)
                    row = {"attempt": counts["attempted"] + 1, "epoch": epoch,
                           "record_index": record_index, "record_id": record["id"], "lr": lr}
                    try:
                        ids, offsets = response_tokens(backend.tokenizer, record)
                    except ValueError as exc:
                        row.update(skipped=True, reason="unalignable_response", detail=str(exc))
                    else:
                        units, alignment = align_claims(record, offsets,
                            allow_whole_claim_negative=allow_whole_claim_negative)
                        row["alignment"], row["units"] = alignment, units
                        if not units:
                            row.update(skipped=True, reason="no_aligned_feedback")
                        else:
                            nll = backend.token_nll(record, ids, image_root=image_root)
                            pos, neg, objective_stats = masked_objectives(nll, units)
                            row["objectives"] = objective_stats
                            row.update(response_step(backend.model, pos, neg, mode=mode,
                                lr=lr, alpha=alpha, beta=beta, max_update_norm=max_update_norm))
                            del nll, pos, neg
                            if audit_loss_change:
                                with torch.no_grad():
                                    nll_after = backend.token_nll(record, ids, image_root=image_root)
                                    pos_after, neg_after, _ = masked_objectives(nll_after, units)
                                row["positive_loss_after"] = float(pos_after) if pos_after is not None else None
                                row["negative_loss_after"] = float(neg_after) if neg_after is not None else None
                                del nll_after, pos_after, neg_after
                    row["elapsed_seconds"] = time.perf_counter() - started
                    row["global_step"] = counts["updated"] + (not row["skipped"])
                    if cuda_devices:
                        row["gpu_peak_allocated_bytes"] = sum(torch.cuda.max_memory_allocated(d) for d in cuda_devices)
                        row["gpu_reserved_bytes"] = sum(torch.cuda.memory_reserved(d) for d in cuda_devices)
                    # Commit the row only after a complete attempt. An exception
                    # after a parameter mutation never creates an emergency ckpt.
                    log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    counts["attempted"] += 1
                    counts["skipped" if row["skipped"] else "updated"] += 1
                    reasons[row["reason"]] += 1
                    cursor = dict(epoch=counts["attempted"] // len(records),
                                  record_index=counts["attempted"] % len(records))
                    metrics.record(row, record_index=record_index)
                    manifest.update(counts=dict(counts), reasons=dict(reasons), cursor=cursor,
                                    last_record_id=record["id"])
                    write_json(root / "manifest.json", manifest)
                    write_json(root / "training_metrics_summary.json", metrics.summary())
                    if not row["skipped"] and checkpoint_every and counts["updated"] % checkpoint_every == 0:
                        checkpoint(root / f"checkpoint-{counts['updated']:06d}")
                        print(f"Saved checkpoint at global update {counts['updated']}", flush=True)
                    if counts["attempted"] % log_every == 0:
                        print(f"attempt={counts['attempted']} update={counts['updated']} "
                              f"skipped={counts['skipped']} positive_nll={row.get('positive_loss')} "
                              f"negative_nll={row.get('negative_loss')}", flush=True)
                if stop:
                    break
        checkpoint(root / "final")
        target_reached = None if max_updates is None else counts["updated"] >= max_updates
        stop_reason = ("max_updates" if target_reached else
            "max_steps" if max_steps is not None and counts["attempted"] >= max_steps else "pool_exhausted")
        manifest.update(status="complete", target_updates_reached=target_reached,
                        stop_reason=stop_reason,
                        new_updates=counts["updated"] - counts_at_start["updated"])
        if target_reached is False:
            raise TrainingTargetNotReached(f"Requested {max_updates} actual updates, completed "
                f"{counts['updated']} after {counts['attempted']} attempts ({stop_reason}); "
                f"partial resumable checkpoint saved in {root / 'final'}")
    except TrainingTargetNotReached as exc:
        manifest.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        raise
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                        counts=dict(counts), reasons=dict(reasons), cursor=cursor)
        write_json(root / "manifest.json", manifest)
        if metrics:
            write_json(root / "training_metrics_summary.json", metrics.summary())
            metrics.close()
    return manifest
