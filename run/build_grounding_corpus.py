"""Durable caption/DiscoSG corpus stages; no training or external judge calls.

Each generation batch or decomposition record is an atomic, checksummed shard.
The flat JSONL is a periodically refreshed view of that ordered shard journal.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix="." + path.name,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_records(path):
    from mvpg_run.claim_feedback.data import validate_record
    result, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row.setdefault("response", "")
        row.setdefault("claims", [])
        row = validate_record(row)
        if row["id"] in seen:
            raise ValueError("Input contains duplicated record IDs")
        seen.add(row["id"])
        result.append(row)
    if not result:
        raise ValueError("Input corpus is empty")
    return result


def resolve_image(record, image_root):
    p = Path(record["image"]).expanduser()
    return (p if p.is_absolute() else image_root / p).resolve()


def model_fingerprint(root):
    """Small configs are hashed; huge weights retain size/mtime drift checks."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Model directory is missing: {root}")
    configs = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
               if p.is_file() and p.suffix in {".json", ".model", ".py"}}
    weights = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
               for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".bin", ".pt", ".safetensors"}}
    return {"root": str(root), "configuration_sha256": configs, "weight_file_stats": weights,
            "weight_hash_limitation": "Large weights checked by path/size/mtime, not full content SHA256"}


def configuration(args, records):
    root = Path(args.image_root).expanduser().resolve()
    images = {str(resolve_image(r, root)): None for r in records}
    for p in images:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
        images[p] = sha(p)
    value = {"stage": args.command, "input": str(Path(args.input).resolve()),
             "input_sha256": sha(args.input), "record_sha256": [digest(r) for r in records],
             "ordered_ids": [r["id"] for r in records], "image_root": str(root),
             "image_sha256": images, "device": args.device}
    if args.command == "generate":
        value.update(model=model_fingerprint(args.model_name), precision=args.precision,
                     batch_size=args.batch_size, seed=args.seed, max_new_tokens=args.max_new_tokens,
                     temperature=args.temperature, top_p=args.top_p,
                     seed_policy="seed + generation batch index; fixed batching is part of resume identity",
                     adapter=None)
    else:
        value.update(disco_root=model_fingerprint(args.disco_root), seed_model=model_fingerprint(args.seed_model),
                     refiner_model=model_fingerprint(args.refiner_model), rounds=args.rounds,
                     parser_batch_size=args.parser_batch_size, shard_size=1, seed=args.seed)
    return value


class ShardStore:
    def __init__(self, root, config, records, unit_size, resume=False):
        self.root, self.config, self.records = Path(root), config, records
        self.unit_size = unit_size
        manifest = self.root / "manifest.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError("Corpus stage already exists; use --resume")
            saved = read_json(manifest)
            if saved.get("schema") != "grounding_corpus_shards_v1" or saved.get("configuration") != config:
                raise ValueError("Resume input, images, model or generation/parser configuration changed")
        else:
            if resume:
                raise FileNotFoundError("--resume requires an existing stage manifest")
            if self.root.exists() and any(p.name != ".run.lock" for p in self.root.iterdir()):
                raise FileExistsError("Use a new, empty corpus stage directory")
            self.root.mkdir(parents=True, exist_ok=True)
            write_json(manifest, {"schema": "grounding_corpus_shards_v1", "configuration": config,
                                  "created_utc": datetime.now(timezone.utc).isoformat()})
        self.shards = self.root / "shards"
        self.shards.mkdir(exist_ok=True)
        self.output = self.root / ("responses.jsonl" if config["stage"] == "generate" else "claims.jsonl")
        self.paths, self.completed = [], 0
        names = sorted(p for p in self.shards.iterdir() if p.is_file() and not p.name.startswith("."))
        for ordinal, path in enumerate(names):
            if path.name != f"shard-{ordinal:06d}.json":
                raise ValueError("Shard journal has an unexpected name or non-contiguous prefix")
            payload = self._load(path, self.completed)
            self.completed += len(payload["records"])
            self.paths.append(path)
        self.exported = 0

    def _load(self, path, expected_start):
        envelope = read_json(path)
        payload = envelope["payload"]
        if envelope.get("sha256") != digest(payload):
            raise ValueError("Shard payload checksum differs")
        rows = payload.get("records")
        end = min(expected_start + self.unit_size, len(self.records))
        expected = self.records[expected_start:end]
        if (payload.get("start") != expected_start or not isinstance(rows, list) or len(rows) != len(expected)
                or not expected or payload.get("input_sha256") != [digest(r) for r in expected]):
            raise ValueError("Shard does not match the committed input prefix")
        for row, source in zip(rows, expected):
            if any(row.get(k) != source[k] for k in ("id", "image", "question")):
                raise ValueError("Shard record identity differs from input")
            if self.config["stage"] == "decompose" and any(
                    row.get(k) != source.get(k) for k in ("response", "response_ids", "generation")):
                raise ValueError("Decomposition rewrote the sampled response")
            if not isinstance(row.get("response"), str) or not isinstance(row.get("claims"), list):
                raise ValueError("Malformed corpus record in shard")
        return payload

    def commit(self, rows):
        start = self.completed
        expected = self.records[start:start + self.unit_size]
        if len(rows) != len(expected) or not expected:
            raise ValueError("Stage output batch does not match its input unit")
        payload = {"start": start, "input_sha256": [digest(r) for r in expected], "records": rows}
        path = self.shards / f"shard-{len(self.paths):06d}.json"
        if path.exists():
            raise FileExistsError(path)
        # Validate before publishing; never replace an existing journal entry.
        temporary = self.shards / (".validate-" + path.name)
        write_json(temporary, {"sha256": digest(payload), "payload": payload})
        try:
            self._load(temporary, start)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.paths.append(path)
        self.completed += len(rows)

    def export(self):
        temporary = self.output.with_name("." + self.output.name + ".tmp")
        start = 0
        try:
            with temporary.open("wb") as stream:
                for path in self.paths:
                    payload = self._load(path, start)
                    for row in payload["records"]:
                        stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
                    start += len(payload["records"])
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output)
        finally:
            temporary.unlink(missing_ok=True)
        self.exported = self.completed

    def status(self, state, **extra):
        data = {"stage": self.config["stage"], "status": state, "total": len(self.records),
                "completed_records": self.completed, "exported_records": self.exported,
                "shards": len(self.paths), "output": str(self.output.resolve()),
                "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
        write_json(self.root / "status.json", data)
        return data


def trim_response_ids(ids, *, eos_token_id, pad_token_id):
    """Retain the first sampled EOS, remove only generation's subsequent pads."""
    ids = [int(x) for x in ids]
    if eos_token_id is not None and eos_token_id in ids:
        end = ids.index(eos_token_id) + 1
        if any(x not in {eos_token_id, pad_token_id} for x in ids[end:]):
            raise ValueError("Unexpected non-padding tokens after sampled EOS")
        return ids[:end]
    return ids


def generate_batch(backend, records, *, image_root, batch_seed, max_new_tokens, temperature, top_p):
    from mvpg_run.claim_feedback.backends import decode_exact
    from mvpg_run.claim_feedback.training import seed_everything
    torch = backend.torch
    seed_everything(batch_seed)
    queries = [backend.build_query(backend.tokenizer, r["question"]) for r in records]
    if len({q.numel() for q in queries}) != 1:
        raise ValueError("Caption batches require equal-length unpadded prompts")
    ids = torch.stack(queries).to(backend.device)
    images = torch.stack([backend.process_image(backend.image_processor, str(resolve_image(r, image_root)))
                          for r in records]).to(device=backend.device, dtype=backend.dtype)
    backend.model.config.use_cache = True
    backend.model.config.cache_shape = (2048,)
    with torch.inference_mode():
        output = backend.model.generate(input_ids=ids, images=images, attention_mask=torch.ones_like(ids),
                    do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p, max_new_tokens=max_new_tokens, use_cache=True,
                    pad_token_id=backend.tokenizer.pad_token_id)
    n = ids.shape[1]
    if (output.ndim != 2 or output.shape[0] != len(records) or output.shape[1] < n
            or not torch.equal(output[:, :n], ids)):
        raise ValueError("Generated sequence changed or omitted the original prompt prefix")
    rows = []
    for source, seq in zip(records, output[:, n:]):
        sampled = trim_response_ids(seq.cpu().tolist(), eos_token_id=backend.tokenizer.eos_token_id,
                                    pad_token_id=backend.tokenizer.pad_token_id)
        row = copy.deepcopy(source)
        for key in ("disco_graph", "generation", "response_ids"):
            row.pop(key, None)
        meta = row.setdefault("metadata", {})
        for key in ("decomposer", "verifier"):
            meta.pop(key, None)
        row.update(response=decode_exact(backend.tokenizer, sampled), response_ids=sampled, claims=[])
        rows.append(row)
    return rows


@contextmanager
def stage_lock(root):
    with (Path(root) / ".run.lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0"); handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another process is using this corpus stage") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _run_locked(args):
    records = read_records(args.input)
    config = configuration(args, records)
    size = args.batch_size if args.command == "generate" else 1
    store = ShardStore(args.output_dir, config, records, size, args.resume)
    store.export()
    store.status("running", resumed_completed_records=store.completed)
    try:
        if store.completed < len(records):
            if args.command == "generate":
                from mvpg_run.claim_feedback.backends import LlavaBackend
                backend = LlavaBackend(args.model_name, device=args.device, precision=args.precision)
            else:
                from mvpg_run.claim_feedback.disco import DiscoSGDecomposer
                from mvpg_run.claim_feedback.training import seed_everything
                seed_everything(args.seed)
                backend = DiscoSGDecomposer(args.disco_root, args.seed_model, args.refiner_model,
                            device=args.device, rounds=args.rounds, batch_size=args.parser_batch_size)
            while store.completed < len(records):
                start = store.completed
                selected = records[start:start + size]
                if args.command == "generate":
                    batch_seed = args.seed + start // size
                    output = generate_batch(backend, selected, image_root=Path(config["image_root"]),
                               batch_seed=batch_seed, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p)
                    for row in output:
                        row["generation"] = {"model_name": config["model"]["root"], "adapter_path": None,
                            "precision": args.precision, "max_new_tokens": args.max_new_tokens,
                            "temperature": args.temperature, "top_p": args.top_p,
                            "batch_size": args.batch_size, "batch_index": start // size,
                            "seed": args.seed, "batch_seed": batch_seed,
                            "response_serialization": "exact decode; first EOS retained; post-EOS pads removed"}
                else:
                    output = [backend.decompose(selected[0])]
                store.commit(output)
                if len(store.paths) % args.export_every == 0:
                    store.export()
                store.status("running")
                if args.command == "generate" or store.completed % 16 == 0:
                    print(f"{args.command} {store.completed}/{len(records)}", flush=True)
        store.export()
        return store.status("complete", output_sha256=sha(store.output))
    except BaseException as exc:
        store.export()
        store.status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        raise


def run(args):
    root = Path(args.output_dir).expanduser().resolve()
    if root.exists() and not (root / "manifest.json").exists():
        if any(p.name != ".run.lock" for p in root.iterdir()):
            raise FileExistsError("Use a new, empty corpus stage directory")
    root.mkdir(parents=True, exist_ok=True)
    with stage_lock(root):
        return _run_locked(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "decompose"):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--image-root", default=".")
        p.add_argument("--device", default="cuda")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--export-every", type=int, default=32)
        if name == "generate":
            p.add_argument("--model-name", required=True)
            p.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--max-new-tokens", type=int, default=192)
            p.add_argument("--temperature", type=float, default=.7)
            p.add_argument("--top-p", type=float, default=.9)
        else:
            p.add_argument("--disco-root", required=True)
            p.add_argument("--seed-model", required=True)
            p.add_argument("--refiner-model", required=True)
            p.add_argument("--rounds", type=int, default=3)
            p.add_argument("--parser-batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    for key in ("export_every", "batch_size", "max_new_tokens", "rounds", "parser_batch_size"):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(key.replace("_", "-") + " must be positive")
    if args.command == "generate" and (not math.isfinite(args.temperature) or args.temperature < 0
                                       or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        parser.error("Require nonnegative finite temperature and0<top-p<=1")
    try:
        result = run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"Corpus: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
