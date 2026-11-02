"""Resume contracts: committed cursor, RNG, global budgets and checkpoint cadence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import shutil

import pytest
import torch

from mvpg_run.claim_feedback.backends import TinyBackend
from mvpg_run.claim_feedback.data import write_jsonl
from mvpg_run.claim_feedback.training import train_pool


@pytest.fixture(autouse=True)
def _small_cpu_workloads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _record(record_id, word="cat", *, accepted=True):
    return {
        "id": record_id, "image": "unused", "question": "Describe.",
        "response": word,
        "claims": [{"id": "entity", "text": word, "kind": "object",
                    "label": "+" if accepted else "?",
                    "source_spans": [[0, len(word)]],
                    "weight": 1.0 if accepted else 0.0}],
    }


def _pool(path):
    write_jsonl(path, [_record("skip0", accepted=False), _record("cat"),
                      _record("dog", "dog"), _record("skip1", accepted=False),
                      _record("bird", "bird")])
    return path


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _state(path):
    return json.loads((Path(path) / "trainer_state.json").read_text(encoding="utf-8"))


class StochasticTinyBackend(TinyBackend):
    """Synthetic stochastic training makes missing RNG restoration observable."""
    def token_nll(self, record, ids, **kwargs):
        scale = 0.8 + 0.1 * random.random() + 0.1 * torch.rand(()).item()
        return super().token_nll(record, ids, **kwargs) * scale

    def save(self, path):
        super().save(path)
        # Checkpoint I/O must not change the training random stream or make
        # the trajectory depend on how often checkpoints are requested.
        random.random()
        torch.rand(3)


def _backend(initial):
    backend = StochasticTinyBackend()
    backend.model.load_state_dict(initial)
    return backend


def _settings(**overrides):
    # A high cap keeps the random scale from cancelling through norm clipping.
    return dict(lr=0.005, max_update_norm=10.0, seed=42, epochs=4,
                checkpoint_every=2, **overrides)


def test_exact_resume_restores_cursor_rng_weights_and_history(tmp_path):
    source = _pool(tmp_path / "pool.jsonl")
    initial = copy.deepcopy(StochasticTinyBackend().model.state_dict())
    full = _backend(initial)
    expected_manifest = train_pool(full, source, tmp_path / "full",
                                   max_updates=7, **_settings())
    expected_torch = torch.rand(7)
    expected_python = [random.random() for _ in range(7)]

    part = _backend(initial)
    train_pool(part, source, tmp_path / "part", max_updates=4, **_settings())
    before = _rows(tmp_path / "part" / "steps.jsonl")
    prefix_bytes = (tmp_path / "part" / "steps.jsonl").read_bytes()
    saved = _state(tmp_path / "part" / "final")
    assert saved["counts"] == {"attempted": 7, "updated": 4, "skipped": 3}
    assert saved["cursor"] == {"epoch": 1, "record_index": 2}

    # Neither unrelated work nor model initialization may shift the resumed RNG.
    random.seed(9182)
    torch.manual_seed(7364)
    torch.rand(31)
    resumed = StochasticTinyBackend()
    actual_manifest = train_pool(resumed, source, tmp_path / "resumed",
                                 resume_from=tmp_path / "part" / "final",
                                 max_updates=7, **_settings())
    assert actual_manifest["counts"] == expected_manifest["counts"] == {
        "attempted": 12, "updated": 7, "skipped": 5}
    for name, tensor in full.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], tensor), name
    assert torch.equal(torch.rand(7), expected_torch)
    assert [random.random() for _ in range(7)] == expected_python
    expected_rows = _rows(tmp_path / "full" / "steps.jsonl")
    actual_rows = _rows(tmp_path / "resumed" / "steps.jsonl")
    assert actual_rows[:len(before)] == before
    assert (tmp_path / "resumed" / "steps.jsonl").read_bytes().startswith(prefix_bytes)
    assert [(r["attempt"], r["epoch"], r["record_id"])
            for r in actual_rows] == [(r["attempt"], r["epoch"], r["record_id"])
                                     for r in expected_rows]
    assert [r.get("positive_loss") for r in actual_rows] == [
        r.get("positive_loss") for r in expected_rows]
    metrics = _rows(tmp_path / "resumed" / "metrics.jsonl")
    assert [r["attempt"] for r in metrics] == list(range(1, 13))
    expected_step = 0
    for index, (row, metric) in enumerate(zip(actual_rows, metrics)):
        expected_step += not row["skipped"]
        assert metric["global_step"] == expected_step
        assert metric["epoch"] == index // 5
        assert metric["record_index"] == index % 5
        assert metric["record_id"] == row["record_id"]


def test_checkpoint_frequency_does_not_shift_training_rng_or_trajectory(tmp_path):
    source = _pool(tmp_path / "pool.jsonl")
    initial = copy.deepcopy(StochasticTinyBackend().model.state_dict())
    frequent, sparse = _backend(initial), _backend(initial)
    train_pool(frequent, source, tmp_path / "frequent", max_updates=7, **_settings())
    expected_python, expected_torch = random.random(), torch.rand(7)
    settings = _settings()
    settings["checkpoint_every"] = 500
    train_pool(sparse, source, tmp_path / "sparse", max_updates=7, **settings)
    for name, tensor in frequent.model.state_dict().items():
        assert torch.equal(sparse.model.state_dict()[name], tensor), name
    assert random.random() == expected_python
    assert torch.equal(torch.rand(7), expected_torch)


def test_resume_uses_checkpoint_prefix_not_later_or_torn_run_log(tmp_path):
    source = _pool(tmp_path / "pool.jsonl")
    initial = copy.deepcopy(StochasticTinyBackend().model.state_dict())
    full = _backend(initial)
    train_pool(full, source, tmp_path / "full", max_updates=7, **_settings())
    part = _backend(initial)
    train_pool(part, source, tmp_path / "part", max_updates=5, **_settings())
    checkpoint = tmp_path / "part" / "checkpoint-000004"
    assert _state(checkpoint)["counts"]["attempted"] == 7
    with (tmp_path / "part" / "steps.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"attempt": 999, "torn":')
    resumed = StochasticTinyBackend()
    manifest = train_pool(resumed, source, tmp_path / "resumed",
                          resume_from=checkpoint, max_updates=7, **_settings())
    assert manifest["counts"]["updated"] == 7
    assert [r["attempt"] for r in _rows(tmp_path / "resumed" / "steps.jsonl")] == list(range(1, 13))
    for name, tensor in full.model.state_dict().items():
        assert torch.equal(resumed.model.state_dict()[name], tensor), name


def test_checkpoint_cadence_counts_updates_and_skips_do_not_republish(tmp_path):
    source = tmp_path / "pool.jsonl"
    write_jsonl(source, [_record("skip", accepted=False), _record("accepted")])
    manifest = train_pool(TinyBackend(), source, tmp_path / "run", epochs=3,
                          max_updates=3, checkpoint_every=2)
    assert manifest["counts"] == {"attempted": 6, "updated": 3, "skipped": 3}
    checkpoints = sorted((tmp_path / "run").glob("checkpoint-*"))
    assert [p.name for p in checkpoints] == ["checkpoint-000002"]
    assert _state(checkpoints[0])["counts"] == {"attempted": 4, "updated": 2, "skipped": 2}
    assert _state(tmp_path / "run" / "final")["counts"]["updated"] == 3


def test_default_checkpoint_is_global_update_500_not_attempt_500(tmp_path):
    source = tmp_path / "pool.jsonl"
    write_jsonl(source, [_record("skip", accepted=False), _record("accepted")])
    manifest = train_pool(TinyBackend(), source, tmp_path / "run", epochs=500,
                          max_updates=500)
    assert manifest["counts"] == {"attempted": 1000, "updated": 500, "skipped": 500}
    assert [p.name for p in (tmp_path / "run").glob("checkpoint-*")] == ["checkpoint-000500"]
    assert _state(tmp_path / "run" / "checkpoint-000500")["counts"] == manifest["counts"]


@pytest.mark.parametrize("change", ["pool", "lr", "mode", "history", "state_counts"])
def test_resume_rejects_incompatible_or_corrupt_state_before_output_or_load(tmp_path, change):
    source = _pool(tmp_path / "pool.jsonl")
    train_pool(TinyBackend(), source, tmp_path / "part", max_updates=2,
               epochs=2, checkpoint_every=2)
    checkpoint = tmp_path / "part" / "final"
    kwargs = dict(max_updates=3, epochs=2, checkpoint_every=2)
    if change == "pool":
        with source.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif change == "lr":
        kwargs["lr"] = 0.004
    elif change == "mode":
        kwargs["mode"] = "positive"
    elif change == "history":
        with (checkpoint / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
    elif change == "state_counts":
        state = _state(checkpoint)
        state["counts"]["updated"] += 1
        (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    backend = TinyBackend()
    initial = copy.deepcopy(backend.model.state_dict())
    backend.load_checkpoint = lambda path: pytest.fail("Invalid resume must be rejected before model load")
    with pytest.raises(ValueError):
        train_pool(backend, source, tmp_path / "rejected", resume_from=checkpoint, **kwargs)
    assert not (tmp_path / "rejected").exists()
    for name, tensor in initial.items():
        assert torch.equal(tensor, backend.model.state_dict()[name])


def test_resume_preserves_existing_destination_and_parent_checkpoint(tmp_path):
    source = _pool(tmp_path / "pool.jsonl")
    train_pool(TinyBackend(), source, tmp_path / "part", max_updates=2,
               epochs=2, checkpoint_every=2)
    checkpoint = tmp_path / "part" / "final"
    state_bytes = (checkpoint / "trainer_state.json").read_bytes()
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        train_pool(TinyBackend(), source, destination, resume_from=checkpoint,
                   max_updates=3, epochs=2)
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "user data"
    assert (checkpoint / "trainer_state.json").read_bytes() == state_bytes


def _legacy_snapshot(tmp_path, source):
    """Model the old final: adapter + parent manifest/log, no saved RNG/state."""
    backend = TinyBackend()
    manifest = train_pool(backend, source, tmp_path / "modern",
                          epochs=4, max_updates=4, checkpoint_every=2)
    legacy = tmp_path / "legacy"
    (legacy / "final").mkdir(parents=True)
    shutil.copyfile(tmp_path / "modern" / "final" / "synthetic_model.pt",
                    legacy / "final" / "synthetic_model.pt")
    shutil.copyfile(tmp_path / "modern" / "steps.jsonl", legacy / "steps.jsonl")
    old_keys = ("status", "started_utc", "finished_utc", "pool", "pool_sha256",
                "pool_records", "settings", "backend", "provenance", "protocol",
                "synthetic", "counts", "reasons", "target_updates_reached", "stop_reason")
    old = {key: manifest[key] for key in old_keys}
    old["settings"].pop("log_every", None)
    old["settings"].pop("tensorboard", None)
    (legacy / "manifest.json").write_text(json.dumps(old), encoding="utf-8")
    return legacy


def test_legacy_final_requires_opt_in_and_records_nonexact_lineage(tmp_path):
    source = _pool(tmp_path / "pool.jsonl")
    legacy = _legacy_snapshot(tmp_path, source)
    with pytest.raises(ValueError):
        train_pool(TinyBackend(), source, tmp_path / "rejected",
                   resume_from=legacy / "final", epochs=4, max_updates=7)
    assert not (tmp_path / "rejected").exists()
    manifest = train_pool(TinyBackend(), source, tmp_path / "continued",
                          resume_from=legacy / "final", allow_legacy_resume=True,
                          epochs=4, max_updates=7)
    assert manifest["counts"] == {"attempted": 12, "updated": 7, "skipped": 5}
    assert manifest["counts_at_start"]["updated"] == 4
    assert "legacy_inferred_no_rng" in json.dumps(manifest["resume"])
    # A migrated checkpoint now captures RNG and a sealed prefix for future runs.
    assert (tmp_path / "continued" / "final" / "trainer_state.json").is_file()


def test_legacy_periodic_checkpoint_cannot_borrow_final_manifest_cursor(tmp_path):
    source = _pool(tmp_path / "pool.jsonl")
    legacy = _legacy_snapshot(tmp_path, source)
    periodic = legacy / "checkpoint-000002"
    periodic.mkdir()
    shutil.copyfile(legacy / "final" / "synthetic_model.pt", periodic / "synthetic_model.pt")
    with pytest.raises(ValueError):
        train_pool(TinyBackend(), source, tmp_path / "rejected",
                   resume_from=periodic, allow_legacy_resume=True,
                   epochs=4, max_updates=7)
    assert not (tmp_path / "rejected").exists()


def test_failed_post_update_audit_does_not_publish_half_committed_checkpoint(tmp_path):
    class AuditFailureBackend(TinyBackend):
        def __init__(self, fail=True):
            super().__init__()
            self.fail = fail

        def token_nll(self, record, ids, **kwargs):
            if self.fail and record["id"] == "second" and not torch.is_grad_enabled():
                raise RuntimeError("Post-update audit failed")
            return super().token_nll(record, ids, **kwargs)

    source = tmp_path / "pool.jsonl"
    write_jsonl(source, [_record("first"), _record("second")])
    with pytest.raises(RuntimeError, match="Post-update audit failed"):
        train_pool(AuditFailureBackend(), source, tmp_path / "failed",
                   max_updates=2, checkpoint_every=1, audit_loss_change=True)
    assert (tmp_path / "failed" / "checkpoint-000001").is_dir()
    assert not (tmp_path / "failed" / "checkpoint-000002").exists()
    assert not (tmp_path / "failed" / "final").exists()
    assert len(_rows(tmp_path / "failed" / "steps.jsonl")) == 1
    manifest = train_pool(AuditFailureBackend(fail=False), source, tmp_path / "recovered",
                          resume_from=tmp_path / "failed" / "checkpoint-000001",
                          max_updates=2, checkpoint_every=1, audit_loss_change=True)
    assert manifest["counts"] == {"attempted": 2, "updated": 2, "skipped": 0}
