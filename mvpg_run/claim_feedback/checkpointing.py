"""Immutable training snapshots for the stateless, per-response SGD trainer.

The published directory is also a standard, flat adapter directory. Resume state
belongs to that precise adapter and history prefix, never to a mutable run log.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import uuid


SCHEMA_VERSION = 1
FROZEN_SETTINGS = (
    "mode", "lr", "alpha", "beta", "max_update_norm", "seed", "image_root",
    "allow_whole_claim_negative", "audit_loss_change",
)
_RESERVED = frozenset({"trainer_state.json", "history.jsonl", "rng_state.pt"})


def _json_value(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Checkpoint metadata keys must be strings")
        return {k: _json_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported or nonfinite checkpoint metadata: {type(value).__name__}")


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint metadata: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state():
    """Use tensors/primitives only so torch.load(weights_only=True) suffices."""
    import torch
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
             "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]
             if torch.cuda.is_available() else [], "numpy": None}
    try:
        import numpy as np
    except ImportError:
        return state
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    state["numpy"] = {"bit_generator": name,
                      "keys": torch.tensor(keys.astype("int64"), dtype=torch.int64),
                      "position": int(position), "has_gauss": int(has_gauss),
                      "cached_gaussian": float(cached_gaussian)}
    return state


def _check_rng_state(state):
    import torch
    if not isinstance(state, dict) or set(state) != {"python", "torch_cpu", "torch_cuda", "numpy"}:
        raise ValueError("Incomplete RNG state")
    try:
        random.Random().setstate(state["python"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Invalid Python or CPU Torch RNG state") from exc
    cuda = state["torch_cuda"]
    if not isinstance(cuda, list) or any(not isinstance(x, torch.Tensor) or x.dtype != torch.uint8
                                         or x.ndim != 1 for x in cuda):
        raise ValueError("Invalid CUDA RNG state")
    if cuda and (not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()):
        raise ValueError("Exact RNG resume requires the saved CUDA device count")
    numpy_state = state["numpy"]
    if numpy_state is not None:
        try:
            import numpy as np
            candidate = np.random.RandomState()
            candidate.set_state((numpy_state["bit_generator"],
                                 np.asarray(numpy_state["keys"].tolist(), dtype=np.uint32),
                                 numpy_state["position"], numpy_state["has_gauss"],
                                 numpy_state["cached_gaussian"]))
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid or unavailable NumPy RNG state") from exc


def restore_rng_state(state):
    import torch
    _check_rng_state(state)
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state["numpy"] is not None:
        import numpy as np
        s = state["numpy"]
        np.random.set_state((s["bit_generator"], np.asarray(s["keys"].tolist(), dtype=np.uint32),
                             s["position"], s["has_gauss"], s["cached_gaussian"]))


def backend_signature(backend, config=None):
    config = config or {}
    model_name = config.get("model_name", getattr(backend, "model_name", None))
    if model_name:
        model_name = str(Path(model_name).expanduser().resolve())
    parameters = [{"name": name, "shape": list(param.shape), "dtype": str(param.dtype)}
                  for name, param in backend.model.named_parameters() if param.requires_grad]
    return _json_value({
        "backend_class": type(backend).__module__ + "." + type(backend).__qualname__,
        "model_name": model_name,
        "precision": config.get("precision", getattr(backend, "precision", None)),
        "adapter_config": getattr(backend, "adapter_config", None)
        or config.get("actual_adapter_config", {}),
        "lora_targets": sorted(getattr(backend, "lora_targets", [])
                               or config.get("lora_targets", [])),
        "trainable_parameters": sorted(parameters, key=lambda x: x["name"]),
    })


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Checkpoint {name} must be a nonnegative integer")
    return value


def _history_check(history_bytes, counts, reasons, cursor, records):
    if not isinstance(counts, dict) or set(counts) != {"attempted", "updated", "skipped"}:
        raise ValueError("Invalid checkpoint counts")
    checked = {key: _integer(value, key) for key, value in counts.items()}
    if checked["attempted"] != checked["updated"] + checked["skipped"]:
        raise ValueError("Checkpoint update/skip counts do not add up")
    if not records:
        raise ValueError("Cannot resume an empty feedback pool")
    n = len(records)
    e, i = divmod(checked["attempted"], n)
    if cursor != {"epoch": e, "record_index": i}:
        raise ValueError("Checkpoint cursor disagrees with committed history")
    _integer(cursor["epoch"], "cursor epoch")
    _integer(cursor["record_index"], "cursor record_index")
    try:
        text = history_bytes.decode("utf-8")
        rows = [json.loads(line, parse_constant=_reject_constant)
                for line in text.splitlines()]
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Checkpoint history is malformed or torn") from exc
    if len(rows) != checked["attempted"]:
        raise ValueError("Checkpoint history length disagrees with counts")
    updated = 0
    actual_reasons = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Every checkpoint history row must be an object")
        epoch, record_index = divmod(index, n)
        if (_integer(row.get("attempt"), "history attempt") != index + 1
                or _integer(row.get("epoch"), "history epoch") != epoch
                or row.get("record_id") != records[record_index]["id"]):
            raise ValueError("Checkpoint history order or source IDs do not match the feedback pool")
        if "record_index" in row and _integer(row["record_index"], "history record_index") != record_index:
            raise ValueError("Checkpoint history record_index is inconsistent")
        if not isinstance(row.get("skipped"), bool) or not isinstance(row.get("reason"), str) or not row["reason"]:
            raise ValueError("Checkpoint history is missing skip/reason fields")
        updated += not row["skipped"]
        if "global_step" in row and _integer(row["global_step"], "global_step") != updated:
            raise ValueError("Checkpoint global_step is inconsistent")
        actual_reasons[row["reason"]] += 1
    if updated != checked["updated"]:
        raise ValueError("Checkpoint successful-update count disagrees with history")
    if not isinstance(reasons, dict) or any(not isinstance(k, str) for k in reasons):
        raise ValueError("Invalid checkpoint reason counters")
    checked_reasons = {k: _integer(v, "reason count") for k, v in reasons.items()}
    if dict(actual_reasons) != {k: v for k, v in checked_reasons.items() if v}:
        raise ValueError("Checkpoint reason counters disagree with history")


def _settings_check(saved, requested, counts, pool_records):
    if not isinstance(saved, dict) or not isinstance(requested, dict):
        raise ValueError("Missing training settings")
    for key in FROZEN_SETTINGS:
        if key not in saved or key not in requested or _json_value(saved[key]) != _json_value(requested[key]):
            raise ValueError(f"Cannot change {key} when resuming training")
    for value, label in ((saved, "saved"), (requested, "requested")):
        epochs = value.get("epochs")
        if epochs is not None and _integer(epochs, label + " epochs") * pool_records < counts["attempted"]:
            raise ValueError(f"{label} epoch budget is below the committed cursor")
        for key, counter in (("max_steps", "attempted"), ("max_updates", "updated")):
            limit = value.get(key)
            if limit is not None and _integer(limit, label + " " + key) < counts[counter]:
                raise ValueError(f"{label} {key} is below the committed count")


def _safe_file(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Unsafe checkpoint file path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise ValueError("Unsafe checkpoint file path")
    target = root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Checkpoint files must not escape through symlinks")
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Checkpoint files must not escape through symlinks")
    if not target.is_file():
        raise ValueError(f"Missing checkpoint file: {relative}")
    return target


def _inventory(root):
    files = {}
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError("Published checkpoints must be self-contained, without symlinks")
        if p.is_file() and p != root / "trainer_state.json":
            files[p.relative_to(root).as_posix()] = _sha(p)
    return files


def save_checkpoint(backend, path, *, history_path, counts, reasons, cursor, settings,
                    pool_path, pool_sha256, pool_records, signature, resume_info=None):
    """Publish a complete flat adapter snapshot without perturbing training RNG."""
    import torch
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = capture_rng_state()
    container = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    container.mkdir()
    payload = container / "adapter"
    try:
        backend.save(payload)
        if any((payload / name).exists() for name in _RESERVED):
            raise ValueError("Backend wrote reserved checkpoint filenames")
        history = Path(history_path).read_bytes()
        (payload / "history.jsonl").write_bytes(history)
        torch.save(rng, payload / "rng_state.pt")
        files = _inventory(payload)
        if not any(name not in _RESERVED for name in files):
            raise ValueError("Backend did not save adapter/model files")
        state = _json_value({"schema_version": SCHEMA_VERSION,
            "counts": dict(counts), "reasons": dict(reasons), "cursor": dict(cursor),
            "settings": settings, "pool": {"path": str(Path(pool_path).resolve()),
                                           "sha256": pool_sha256, "records": pool_records},
            "pool_sha256": pool_sha256, "pool_records": pool_records,
            "signature": signature, "resume_info": resume_info,
            "history_sha256": hashlib.sha256(history).hexdigest(), "checkpoint_files": files})
        (payload / "trainer_state.json").write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if path.exists():
            raise FileExistsError(path)
        payload.rename(path)
        container.rmdir()
        return state
    finally:
        restore_rng_state(rng)
        if container.exists():
            # This invocation exclusively created this nonce directory.
            if container.resolve().parent != path.parent.resolve() or not container.name.startswith(f".{path.name}.tmp-"):
                raise RuntimeError("Unsafe checkpoint staging cleanup")
            shutil.rmtree(container)


def _legacy_manifest(checkpoint):
    if checkpoint.name != "final":
        raise ValueError("Legacy resume supports only a final/ adapter, not old attempt checkpoints")
    if (checkpoint / "history.jsonl").exists() or (checkpoint / "rng_state.pt").exists():
        raise ValueError("Modern checkpoint is missing trainer_state.json")
    manifest = _read_json(checkpoint.parent / "manifest.json")
    if "checkpoint_step_unit" in manifest or "signature" in manifest:
        raise ValueError("Modern checkpoint is missing its trainer state; it is not a legacy adapter")
    if manifest.get("status") not in ("complete", "incomplete"):
        raise ValueError("Legacy run must be complete or explicitly incomplete")
    return manifest


def read_checkpoint_settings(checkpoint, allow_legacy=False):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if (checkpoint / "trainer_state.json").is_file():
        if (checkpoint / "trainer_state.json").is_symlink():
            raise ValueError("Checkpoint metadata must be self-contained")
        state = _read_json(checkpoint / "trainer_state.json")
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint state schema")
    elif allow_legacy:
        state = _legacy_manifest(checkpoint)
    else:
        raise ValueError("Checkpoint lacks resumable trainer_state.json; legacy inference requires explicit opt-in")
    settings = state.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Checkpoint has no training settings")
    return dict(settings)


def load_resume(checkpoint, *, input_path, records, settings, signature, allow_legacy=False):
    """Validate the whole snapshot before caller mutates a model or creates output."""
    import torch
    checkpoint = Path(checkpoint).expanduser().resolve()
    modern = (checkpoint / "trainer_state.json").is_file()
    current_sha = _sha(input_path)
    if modern:
        if (checkpoint / "trainer_state.json").is_symlink():
            raise ValueError("Checkpoint metadata must be self-contained")
        state = _read_json(checkpoint / "trainer_state.json")
        if state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint state schema")
        if not isinstance(state.get("signature"), dict):
            raise ValueError("Checkpoint model signature is missing or malformed")
        if state.get("resume_info") is not None and not isinstance(state["resume_info"], dict):
            raise ValueError("Checkpoint lineage metadata is malformed")
        pool = state.get("pool")
        if (not isinstance(pool, dict) or pool.get("sha256") != current_sha
                or pool.get("records") != len(records)
                or state.get("pool_sha256") != current_sha or state.get("pool_records") != len(records)):
            raise ValueError("Resume feedback pool SHA256 or record count differs")
        if signature is not None and _json_value(state.get("signature")) != _json_value(signature):
            raise ValueError("Resume model, precision or trainable parameter schema differs")
        files = state.get("checkpoint_files")
        if not isinstance(files, dict) or not {"history.jsonl", "rng_state.pt"}.issubset(files):
            raise ValueError("Incomplete checkpoint file manifest")
        for name, expected in files.items():
            if not isinstance(expected, str) or len(expected) != 64 or _sha(_safe_file(checkpoint, name)) != expected:
                raise ValueError("Checkpoint file checksum differs")
        if _inventory(checkpoint) != files:
            raise ValueError("Checkpoint file inventory differs")
        history = (checkpoint / "history.jsonl").read_bytes()
        if hashlib.sha256(history).hexdigest() != state.get("history_sha256"):
            raise ValueError("Checkpoint history SHA256 differs")
        try:
            rng = torch.load(checkpoint / "rng_state.pt", map_location="cpu", weights_only=True)
            _check_rng_state(rng)
        except Exception as exc:
            raise ValueError("Cannot load validated checkpoint RNG state") from exc
        info = {"checkpoint": str(checkpoint), "kind": "stateful_checkpoint",
                "exact_rng_state_available": True,
                "source_state_sha256": _sha(checkpoint / "trainer_state.json"),
                "source_checkpoint_files": dict(files),
                "lineage_contains_legacy": bool(state.get("legacy_inferred_no_rng")
                    or (state.get("resume_info") or {}).get("lineage_contains_legacy"))}
    else:
        if not allow_legacy:
            raise ValueError("Checkpoint lacks resumable state; explicit legacy opt-in is required")
        manifest = _legacy_manifest(checkpoint)
        if manifest.get("pool_sha256") != current_sha or manifest.get("pool_records") != len(records):
            raise ValueError("Legacy feedback pool SHA256 or record count differs")
        history = (checkpoint.parent / "steps.jsonl").read_bytes()
        counts = manifest.get("counts", {})
        attempted = _integer(counts.get("attempted"), "attempted")
        epoch, index = divmod(attempted, len(records)) if records else (0, 0)
        old_config = manifest.get("backend", {})
        if not isinstance(old_config, dict):
            raise ValueError("Legacy backend configuration is invalid")
        for old_key, new_key in (("model_name", "model_name"), ("precision", "precision"),
                                 ("actual_adapter_config", "adapter_config"), ("lora_targets", "lora_targets")):
            if signature is not None and old_key in old_config:
                old_value = old_config[old_key]
                if old_key == "model_name" and old_value:
                    old_value = str(Path(old_value).expanduser().resolve())
                if old_key == "lora_targets":
                    old_value = sorted(old_value)
                if _json_value(old_value) != _json_value(signature.get(new_key)):
                    raise ValueError(f"Legacy resume {old_key} differs")
        if signature is not None:
            parameter_count = sum(math.prod(p["shape"]) for p in signature.get("trainable_parameters", []))
            if "trainable_parameters" in old_config and old_config["trainable_parameters"] != parameter_count:
                raise ValueError("Legacy adapter parameter count differs")
        if not _inventory(checkpoint):
            raise ValueError("Legacy adapter directory is empty")
        state = {"schema_version": SCHEMA_VERSION, "counts": counts,
                 "reasons": manifest.get("reasons", {}), "cursor": {"epoch": epoch, "record_index": index},
                 "settings": manifest.get("settings"), "signature": signature,
                 "pool": {"path": str(Path(input_path).resolve()), "sha256": current_sha, "records": len(records)},
                 "pool_sha256": current_sha, "pool_records": len(records),
                 "legacy_inferred_no_rng": True}
        rng = None
        info = {"checkpoint": str(checkpoint), "kind": "legacy_inferred_no_rng",
                "exact_rng_state_available": False,
                "lineage_contains_legacy": True,
                "source_manifest_sha256": _sha(checkpoint.parent / "manifest.json"),
                "source_history_sha256": hashlib.sha256(history).hexdigest(),
                "source_checkpoint_files": _inventory(checkpoint),
                "limitation": "Legacy final saved no RNG/cursor snapshot; cursor inferred from fully validated run history"}
    _history_check(history, state.get("counts"), state.get("reasons"), state.get("cursor"), records)
    _settings_check(state.get("settings"), settings, state["counts"], len(records))
    return {"state": state, "history_bytes": history, "rng_state": rng, "resume_info": info}
