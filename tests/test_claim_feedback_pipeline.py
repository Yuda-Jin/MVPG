"""Offline contracts: exact token credit and the real training entry plumbing."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mvpg_run.claim_feedback.__main__ import build_parser, run, smoke
from mvpg_run.claim_feedback.backends import (CharacterTokenizer, LlavaBackend,
    FGClipVerifier, TinyBackend, llm_lora_targets, response_tokens)
from mvpg_run.claim_feedback.data import write_jsonl
from mvpg_run.claim_feedback.training import TrainingTargetNotReached, train_pool


def test_exact_ids_and_prefix_offsets():
    tok = CharacterTokenizer()
    ids, spans = response_tokens(tok, {"response": "A cat", "response_ids": [65, 32, 99, 97, 116]})
    assert ids == [65, 32, 99, 97, 116]
    assert spans == [(i, i+1) for i in range(5)]
    with pytest.raises(ValueError, match="decode exactly"):
        response_tokens(tok, {"response": "A cat", "response_ids": [65]})


def test_rewriting_tokenizer_abstains():
    class Rewriting:
        def decode(self, ids, **kwargs):
            return "\ufffd" if len(ids) == 1 else "猫"
    with pytest.raises(ValueError, match="rewrites"):
        response_tokens(Rewriting(), {"response": "猫", "response_ids": [1, 2]})


def test_lora_targets_exclude_visual_modules():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)
            self.v_proj = torch.nn.Linear(2, 2)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])
            self.vision_tower = Block()
            self.mm_projector = Block()
    assert llm_lora_targets(Model()) == ["layers.0.q_proj", "layers.0.v_proj"]


@pytest.mark.parametrize("mode", ["modulated", "fixed", "positive", "negative", "project_only"])
def test_smoke_all_controls_and_checkpoint(tmp_path, mode):
    out = tmp_path / mode
    report = smoke(SimpleNamespace(output_dir=str(out), mode=mode, seed=42))
    assert report["synthetic"]
    assert report["counts"]["attempted"] == 1
    row = json.loads((out / "run" / "steps.jsonl").read_text())
    assert row["mode"] == mode
    assert row["actual_update_norm"] <= 0.01
    assert (out / "run" / "final" / "synthetic_model.pt").is_file()
    if mode == "modulated":
        assert report["counts"]["updated"] == 1
        assert row["actual_correction_dot"] > 0
        assert row["actual_positive_dot"] <= 0
        assert report["after"]["positive_nll"] < report["before"]["positive_nll"]
        assert report["after"]["negative_nll"] > report["before"]["negative_nll"]
    with pytest.raises(FileExistsError):
        smoke(SimpleNamespace(output_dir=str(out), mode=mode, seed=42))


def record():
    return {"id": "x", "image": "unused", "question": "q", "response": "cat",
            "claims": [{"id": "cat", "text": "cat", "kind": "object",
                        "label": "?", "source_spans": [[0, 3]], "weight": 0}]}


def test_unresolved_feedback_never_invokes_model(tmp_path):
    backend = TinyBackend()
    def fail(*args, **kwargs):
        raise AssertionError("Must not invoke the model without accepted feedback")
    backend.token_nll = fail
    source = tmp_path / "pool.jsonl"
    write_jsonl(source, [record()])
    manifest = train_pool(backend, source, tmp_path / "run")
    assert manifest["counts"] == {"attempted": 1, "updated": 0, "skipped": 1}
    assert manifest["reasons"] == {"no_aligned_feedback": 1}
    assert len(manifest["pool_sha256"]) == 64


def test_successful_update_target_ignores_skips_and_stops_across_epochs(tmp_path):
    source = tmp_path / "pool.jsonl"
    unresolved, supported = record(), record()
    supported["id"] = "supported"
    supported["claims"][0].update(label="+", weight=1)
    write_jsonl(source, [unresolved, supported])
    manifest = train_pool(TinyBackend(), source, tmp_path / "run",
                          epochs=3, max_updates=2)
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"attempted": 4, "updated": 2, "skipped": 2}
    assert manifest["settings"]["max_updates"] == 2
    assert manifest["target_updates_reached"] is True
    assert manifest["stop_reason"] == "max_updates"
    rows = [json.loads(line) for line in (tmp_path / "run" / "steps.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [0, 0, 1, 1]
    assert sum(not row["skipped"] for row in rows) == 2
    assert all(row["actual_update_norm"] > 0 for row in rows if not row["skipped"])
    assert (tmp_path / "run" / "final" / "synthetic_model.pt").is_file()


@pytest.mark.parametrize("max_steps, expected_counts, stop_reason", [
    (None, {"attempted": 2, "updated": 1, "skipped": 1}, "pool_exhausted"),
    (1, {"attempted": 1, "updated": 0, "skipped": 1}, "max_steps"),
])
def test_unmet_update_target_is_explicit_and_preserves_partial_adapter(
        tmp_path, max_steps, expected_counts, stop_reason):
    source = tmp_path / "pool.jsonl"
    unresolved, supported = record(), record()
    supported["id"] = "supported"
    supported["claims"][0].update(label="+", weight=1)
    write_jsonl(source, [unresolved, supported])
    with pytest.raises(TrainingTargetNotReached, match="Requested 2 actual updates"):
        train_pool(TinyBackend(), source, tmp_path / "run",
                   max_updates=2, max_steps=max_steps)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["counts"] == expected_counts
    assert manifest["target_updates_reached"] is False
    assert manifest["stop_reason"] == stop_reason
    assert "finished_utc" in manifest
    assert (tmp_path / "run" / "final" / "synthetic_model.pt").is_file()


def test_attempt_cap_retains_original_semantics_without_update_target(tmp_path):
    source = tmp_path / "pool.jsonl"
    write_jsonl(source, [record()])
    manifest = train_pool(TinyBackend(), source, tmp_path / "run",
                          epochs=3, max_steps=2)
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"attempted": 2, "updated": 0, "skipped": 2}
    assert manifest["target_updates_reached"] is None
    assert manifest["stop_reason"] == "max_steps"


def test_cli_passes_successful_update_target_to_training(tmp_path, monkeypatch):
    import mvpg_run.claim_feedback.__main__ as cli
    source, output = tmp_path / "pool.jsonl", tmp_path / "run"
    item = record()
    item["claims"][0].update(label="+", weight=1)
    write_jsonl(source, [item])
    backend = TinyBackend()
    backend.lora_targets, backend.adapter_config = [], {}
    monkeypatch.setattr(cli, "_llava", lambda *args, **kwargs: backend)
    args = build_parser().parse_args(["train", "--input", str(source),
        "--output-dir", str(output), "--model-name", "synthetic", "--epochs", "3",
        "--max-updates", "2"])
    manifest = run(args)
    assert manifest["counts"]["updated"] == 2
    assert manifest["settings"]["max_updates"] == 2
    assert manifest["backend"]["max_updates"] == 2


def test_invalid_forward_records_failed_manifest(tmp_path):
    backend = TinyBackend()
    def fail(*args, **kwargs):
        raise RuntimeError("backend failed")
    backend.token_nll = fail
    item = record()
    item["claims"][0].update(label="+", weight=1)
    source = tmp_path / "pool.jsonl"
    write_jsonl(source, [item])
    with pytest.raises(RuntimeError, match="backend failed"):
        train_pool(backend, source, tmp_path / "run")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (tmp_path / "run" / "final").exists()


@pytest.mark.parametrize("kwargs", [{"lr": -1}, {"alpha": float("nan")},
    {"max_update_norm": 0}, {"mode": "typo"}, {"epochs": 1.5}, {"checkpoint_every": -1},
    {"max_updates": 0}, {"max_updates": -1}, {"max_updates": 1.5}, {"max_updates": True}])
def test_preflight_rejects_invalid_options_even_when_no_feedback(tmp_path, kwargs):
    source = tmp_path / "pool.jsonl"
    write_jsonl(source, [record()])
    with pytest.raises(ValueError):
        train_pool(TinyBackend(), source, tmp_path / "run", **kwargs)
    assert not (tmp_path / "run").exists()


def test_optional_finite_step_loss_audit(tmp_path):
    source = tmp_path / "pool.jsonl"
    item = record()
    item["claims"][0].update(label="+", weight=1)
    write_jsonl(source, [item])
    train_pool(TinyBackend(), source, tmp_path / "run", audit_loss_change=True)
    row = json.loads((tmp_path / "run" / "steps.jsonl").read_text())
    assert row["positive_loss_after"] < row["positive_loss"]
    assert row["negative_loss_after"] is None


@pytest.mark.parametrize("mode", ["global", "patch"])
def test_visual_score_walk_and_patch_padding_contract(tmp_path, mode):
    image = tmp_path / "fake.jpg"
    image.write_bytes(b"backend fixture; not opened by the fake encoder")
    verifier = FGClipVerifier.__new__(FGClipVerifier)
    verifier.mode, verifier.walk_type, verifier.image_root = mode, "box", tmp_path
    class Encoder:
        def encode_texts_walk(self, texts, walk_type):
            assert walk_type == "box"
            return torch.tensor([[2., 0.], [0., 3.]])
        def encode_image_paths(self, paths):
            return torch.tensor([[1., 0.]])
        def image_dense_features(self, paths):
            # Padded patch has a high second-class score and must be ignored.
            return torch.tensor([[[2., 0.], [0., 5.]]]), torch.tensor([[True, False]])
    verifier.encoder = Encoder()
    assert verifier("fake.jpg", ["cat", "dog"]) == [1.0, 0.0]


def test_cli_sentence_ablation_and_audit(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "claims.jsonl"
    write_jsonl(source, [record()])
    parser = build_parser()
    args = parser.parse_args(["decompose", "--input", str(source), "--output", str(output), "--parser", "sentences"])
    assert run(args) == {"records": 1}
    assert json.loads(output.read_text())["claims"][0]["parser"] == "punctuation_baseline"
    audit = run(parser.parse_args(["audit", "--input", str(output)]))
    assert audit["labels"] == {"?": 1}
    with pytest.raises(FileExistsError):
        run(args)


def test_stage_rerun_invalidates_downstream_annotations(tmp_path):
    from mvpg_run.claim_feedback.__main__ import _invalidate_feedback
    item = dict(record(), disco_graph={"old": True}, response_ids=[99, 97, 116],
                metadata={"verifier": {"stale": True}, "decomposer": {"old": True}, "dataset": "keep"})
    fresh = _invalidate_feedback(item, response=True)
    assert fresh["response"] == "" and fresh["claims"] == []
    assert "response_ids" not in fresh and "disco_graph" not in fresh
    assert fresh["metadata"] == {"dataset": "keep"}
    assert item["response"] == "cat"  # input is not mutated


@pytest.mark.parametrize("metadata", [None, [], "invalid"])
def test_verify_rejects_invalid_metadata_before_model_loading(tmp_path, metadata):
    source = tmp_path / "bad_metadata.jsonl"
    write_jsonl(source, [dict(record(), metadata=metadata)], validate=False)
    args = build_parser().parse_args(["verify", "--input", str(source),
        "--output", str(tmp_path / "output.jsonl"), "--fgclip-root", "nonexistent",
        "--negative-threshold", "0.1", "--positive-threshold", "0.5", "--calibration-id", "dev-v1"])
    with pytest.raises(ValueError, match="metadata must be an object"):
        run(args)


@pytest.mark.parametrize("reason_field", ["feedback_reason", "alignment_reason"])
def test_audit_rejects_malformed_reason(tmp_path, reason_field):
    item = record()
    item["claims"][0][reason_field] = []
    source = tmp_path / "bad_reason.jsonl"
    write_jsonl(source, [item], validate=False)
    args = build_parser().parse_args(["audit", "--input", str(source)])
    with pytest.raises(ValueError, match=reason_field + " must be a string"):
        run(args)


def make_fake_llava(truncate=False, legacy=False):
    backend = LlavaBackend.__new__(LlavaBackend)
    backend.torch, backend.device = torch, "cpu"
    embedding, head = torch.nn.Embedding(128, 4), torch.nn.Linear(4, 128)
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mix = torch.nn.Linear(4, 4)
        def forward(self, inputs_embeds, **kwargs):
            return (self.mix(inputs_embeds),)
    class Base:
        config = SimpleNamespace(max_position_embeddings=20)
        lm_head = head
        decoder = Decoder()
        def get_model(self):
            return self.decoder
        def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, labels, images):
            # Replace prompt's image placeholder with three feature positions.
            ids = input_ids
            ids = torch.cat((ids[:, :1], torch.zeros((1, 3), dtype=torch.long), ids[:, 2:]), dim=1)
            labels = torch.cat((labels[:, :1], torch.full((1, 3), -100), labels[:, 2:]), dim=1)
            if truncate:
                ids, labels = ids[:, :-1], labels[:, :-1]
            return embedding(ids), labels, torch.ones_like(labels)
    backend.base = Base()
    if legacy:
        original = backend.base.prepare_inputs_labels_for_multimodal
        def legacy_prepare(input_ids, attention_mask, past_key_values, labels, images):
            assert past_key_values is None
            embeds, expanded, mask = original(input_ids, attention_mask, labels, images)
            return None, mask, None, embeds, expanded
        backend.base.prepare_inputs_labels_for_multimodal = legacy_prepare
    backend._inputs = lambda r, root=None: (torch.tensor([[1, 0, 2]]), torch.zeros(1))
    return backend


@pytest.mark.parametrize("legacy", [False, True])
def test_expanded_multimodal_labels_and_shift_are_checked(legacy):
    backend = make_fake_llava(legacy=legacy)
    nll = backend.token_nll({}, [99, 97, 116])
    assert nll.shape == (3,)
    nll.sum().backward()
    assert backend.base.decoder.mix.weight.grad is not None
    with pytest.raises(ValueError, match="truncated"):
        make_fake_llava(truncate=True, legacy=legacy).token_nll({}, [99, 97, 116])
    backend.base.config.max_position_embeddings = 3
    with pytest.raises(ValueError, match="exceeds"):
        backend.token_nll({}, [99, 97, 116])


def test_model_internal_type_error_is_not_retried_or_hidden():
    from mvpg_run.claim_feedback.backends import prepare_multimodal
    class Broken:
        def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, labels, images):
            raise TypeError("failure inside model")
    with pytest.raises(TypeError, match="failure inside model"):
        prepare_multimodal(Broken(), None, None, None, None)
