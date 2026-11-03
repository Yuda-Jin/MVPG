"""Durable caption identity, exact sampled text, and resumable corpus shards."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from run.build_grounding_corpus import (ShardStore, digest, generate_batch,
                                      stage_lock, trim_response_ids)


def inputs(n=3):
    return [{"id": str(i), "image": f"image{i}.png", "question": "Describe.",
             "response": "", "claims": [], "metadata": {"image_group": str(i)}} for i in range(n)]


def output_rows(rows):
    return [dict(row, response=" A.", response_ids=[32, 65, 46, 2]) for row in rows]


def config(stage="generate"):
    return {"stage": stage, "input_sha256": "input", "batch_size": 2}


def test_atomic_prefix_resume_and_flat_export_preserve_order_and_source(tmp_path):
    rows = inputs(3)
    original = copy.deepcopy(rows)
    store = ShardStore(tmp_path, config(), rows, 2)
    store.commit(output_rows(rows[:2]))
    store.export()
    assert store.completed == 2
    restored = ShardStore(tmp_path, config(), rows, 2, resume=True)
    assert restored.completed == 2
    restored.commit(output_rows(rows[2:]))
    restored.export()
    flat = [json.loads(x) for x in restored.output.read_text().splitlines()]
    assert [r["id"] for r in flat] == ["0", "1", "2"]
    assert all(r["response"] == " A." and r["response_ids"][-1] == 2 for r in flat)
    assert rows == original
    assert restored.status("complete")["exported_records"] == 3


def test_manifest_drift_and_nonresume_refuse_to_reuse(tmp_path):
    rows = inputs()
    store = ShardStore(tmp_path, config(), rows, 2)
    store.commit(output_rows(rows[:2]))
    with pytest.raises(FileExistsError, match="--resume"):
        ShardStore(tmp_path, config(), rows, 2)
    with pytest.raises(ValueError, match="configuration changed"):
        ShardStore(tmp_path, dict(config(), batch_size=1), rows, 1, resume=True)
    changed = copy.deepcopy(rows)
    changed[0]["question"] = "Other."
    with pytest.raises(ValueError, match="input prefix"):
        ShardStore(tmp_path, config(), changed, 2, resume=True)


def test_corruption_and_noncontiguous_shards_fail_before_loading_model(tmp_path):
    rows = inputs()
    store = ShardStore(tmp_path, config(), rows, 2)
    store.commit(output_rows(rows[:2]))
    path = store.paths[0]
    data = json.loads(path.read_text())
    data["payload"]["records"][0]["response"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        ShardStore(tmp_path, config(), rows, 2, resume=True)
    data["sha256"] = digest(data["payload"])
    path.write_text(json.dumps(data))
    path.rename(path.with_name("shard-000001.json"))
    with pytest.raises(ValueError, match="non-contiguous"):
        ShardStore(tmp_path, config(), rows, 2, resume=True)


def test_decomposition_cannot_rewrite_response_or_sampled_ids(tmp_path):
    rows = output_rows(inputs(1))
    store = ShardStore(tmp_path, config("decompose"), rows, 1)
    changed = copy.deepcopy(rows)
    changed[0]["response"] = changed[0]["response"].strip()
    with pytest.raises(ValueError, match="rewrote"):
        store.commit(changed)
    assert not store.paths
    row = dict(rows[0], disco_graph={"triples": [["a", "is", "b"]]})
    store.commit([row])
    store.export()
    assert json.loads(store.output.read_text())["disco_graph"]["triples"] == [["a", "is", "b"]]


def test_eos_trimming_does_not_strip_spaces_or_non_eos_tokens():
    assert trim_response_ids([32, 65, 46, 2, 0, 0], eos_token_id=2, pad_token_id=0) == [32, 65, 46, 2]
    assert trim_response_ids([32, 65, 0], eos_token_id=2, pad_token_id=0) == [32, 65, 0]
    with pytest.raises(ValueError, match="non-padding"):
        trim_response_ids([65, 2, 66], eos_token_id=2, pad_token_id=0)


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, **kwargs):
        assert kwargs["clean_up_tokenization_spaces"] is False
        return "".join(chr(i) for i in ids if i not in (0, 2))


def backend(bad_prefix=False):
    def generate(**kwargs):
        ids = kwargs["input_ids"]
        prefix = ids.clone()
        if bad_prefix:
            prefix[0, 0] = 99
        suffix = torch.tensor([[32, 65, 46, 2, 0]] * len(ids))
        return torch.cat([prefix, suffix], 1)
    return SimpleNamespace(torch=torch, tokenizer=Tokenizer(), device="cpu", dtype=torch.float32,
        build_query=lambda tokenizer, text: torch.tensor([1, 3, 9]),
        process_image=lambda processor, path: torch.zeros(3, 4, 4), image_processor=None,
        model=SimpleNamespace(config=SimpleNamespace(), generate=generate))


def test_batched_generation_retains_exact_sampled_ids_and_rejects_prefix_guessing(tmp_path):
    rows = inputs(2)
    rows[0].update(disco_graph={"stale": True}, claims=[{"stale": True}])
    rows[0]["metadata"].update(verifier={"stale": True}, decomposer={"stale": True})
    kwargs = dict(image_root=tmp_path, batch_seed=42, max_new_tokens=192, temperature=.7, top_p=.9)
    actual = generate_batch(backend(), rows, **kwargs)
    assert actual[0]["response"] == " A."
    assert actual[0]["response_ids"] == [32, 65, 46, 2]
    assert not actual[0]["claims"] and "disco_graph" not in actual[0]
    assert actual[0]["metadata"] == {"image_group": "0"}
    with pytest.raises(ValueError, match="prompt prefix"):
        generate_batch(backend(True), rows, **kwargs)


def test_run_lock_rejects_concurrent_writer(tmp_path):
    with stage_lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            with stage_lock(tmp_path):
                pass
