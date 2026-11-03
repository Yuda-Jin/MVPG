"""Content-token scoring and interrupted image journals must not create negatives."""
from __future__ import annotations

import argparse
import copy
import json

import pytest

from run import score_grounding_dino as dino


def test_content_tokens_exclude_special_tokens_punctuation_and_padding():
    # CLS fire - hydrant . SEP PAD: the detector may give punctuation high scores.
    positions = dino.content_positions("fire-hydrant.",
        [101, 10, 11, 12, 13, 102, 0],
        [(0, 0), (0, 4), (4, 5), (5, 12), (12, 13), (0, 0), (0, 0)],
        [1, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 0], {101, 102, 0}, 256)
    assert positions == [1, 3]
    assert dino.caption_for(" Fire Hydrant ") == "fire hydrant."


def test_empty_unknown_or_truncated_mention_does_not_receive_zero_score():
    with pytest.raises(ValueError, match="no representable"):
        dino.content_positions("...", [101, 1, 102], [(0, 0), (0, 3), (0, 0)],
            [1, 0, 1], [1, 1, 1], {101, 102}, 256)
    with pytest.raises(ValueError, match="no representable"):
        dino.content_positions("unknown.", [101, 100, 102], [(0, 0), (0, 7), (0, 0)],
            [1, 0, 1], [1, 1, 1], {101, 102, 100}, 256)
    with pytest.raises(ValueError, match="truncation"):
        dino.content_positions("dog.", [101, 1, 2, 102], [(0, 0), (0, 3), (3, 4), (0, 0)],
            [1, 0, 0, 1], [1, 1, 1, 1], {101, 102}, 3)


def test_unfiltered_scores_ignore_extreme_punctuation_logits():
    torch = pytest.importorskip("torch")
    # All object scores are below common .25/.35 box thresholds: none may be dropped.
    probabilities = torch.tensor([[.9999, .05, .20, .9999], [.99, .10, .12, .99]])
    boxes = torch.tensor([[.1, .2, .4, .6], [.8, .6, .1, .2]])
    result = dino.summarize_predictions(torch.logit(probabilities), boxes, [1, 2],
        top_k=5, width=100, height=200)
    assert result["scores"]["dino_max"] == pytest.approx(.20)
    # min is computed for both terms within ONE query before maximizing queries.
    assert result["scores"]["dino_min"] == pytest.approx(.10)
    assert result["scores"]["dino_mean"] == pytest.approx(.125)
    assert len(result["top_candidates"]) == result["query_count"] == 2
    assert result["best_query_index"] == 0
    assert result["best_box_xyxy_pixels"][0] == 0  # normalized box can cross image boundary


def test_nonfinite_content_is_an_error_instead_of_a_negative():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="nonfinite"):
        dino.summarize_predictions(torch.tensor([[0., float("nan")]]),
            torch.tensor([[.5, .5, .1, .1]]), [1], top_k=1, width=1, height=1)


def case(image, cid, *, group=None):
    return {"case_id": cid, "record_id": image.stem, "claim_id": cid,
        "image": str(image), "image_group": group or str(image), "split": "dev",
        "question": "Describe the image.", "response": "a dog", "claim_text": "dog",
        "source_spans": [[2, 5]], "kind": "object", "judge_label": None, "scores": {}}


def fixtures(tmp_path):
    images = [tmp_path / "image1.png", tmp_path / "image2.png"]
    for i, path in enumerate(images):
        path.write_bytes(f"image-{i}".encode())
    # Two claims in the first image form one indivisible commit unit.
    rows = [case(images[0], "a"), case(images[0], "b"), case(images[1], "c")]
    source = tmp_path / "cases.jsonl"
    source.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"frozen-weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    return rows, argparse.Namespace(input=str(source), output=str(tmp_path / "scored.jsonl"),
        image_root=".", model_root=str(model), device="cpu", batch_size=2, top_k=5, resume=False)


def predictions(phrases):
    return [{"scores": {"dino_max": .2, "dino_mean": .1, "dino_min": .1},
             "caption": dino.caption_for(p), "top_candidates": []} for p in phrases]


def test_interruption_resumes_only_complete_images_and_restores_input_order(tmp_path):
    rows, args = fixtures(tmp_path)
    calls = []
    class Interrupted:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, image, phrases):
            calls.append(image)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return predictions(phrases)
    with pytest.raises(KeyboardInterrupt):
        dino.run(args, backend_factory=Interrupted, versions={"test": "frozen"})
    status = json.loads((tmp_path / "scored.jsonl.journal/status.json").read_text())
    assert status["completed_images"] == 1 and status["completed_cases"] == 2
    assert not (tmp_path / "scored.jsonl").exists()
    args.resume = True
    class Resumed:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, image, phrases):
            calls.append(image)
            return predictions(phrases)
    result = dino.run(args, backend_factory=Resumed, versions={"test": "frozen"})
    assert result["completed_cases"] == 3 and result["completed_images"] == 2
    assert calls == [rows[0]["image"], rows[2]["image"], rows[2]["image"]]
    out = dino.load_cases(args.output)
    assert [r["case_id"] for r in out] == ["a", "b", "c"]
    assert all(r["judge_label"] is None for r in out)
    assert all(r["response"] == "a dog" for r in out)
    # Completed resume must not reload the expensive detector.
    def never_load(*a, **kw):
        raise AssertionError("Completed journal must not re-score")
    assert dino.run(args, backend_factory=never_load, versions={"test": "frozen"})["status"] == "complete"


@pytest.mark.parametrize("mutation", ["image", "model", "input", "runtime", "protocol"])
def test_resume_rejects_changed_provenance(tmp_path, mutation):
    rows, args = fixtures(tmp_path)
    class Scorer:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, image, phrases):
            return predictions(phrases)
    dino.run(args, backend_factory=Scorer, versions={"test": "frozen"})
    args.resume = True
    versions = {"test": "frozen"}
    if mutation == "image":
        (tmp_path / "image1.png").write_bytes(b"changed")
    elif mutation == "model":
        (tmp_path / "model/model.safetensors").write_bytes(b"changed")
    elif mutation == "input":
        rows[0]["claim_text"] = "cat"
        (tmp_path / "cases.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    elif mutation == "runtime":
        versions = {"test": "changed"}
    else:
        args.top_k += 1
    with pytest.raises(ValueError, match="resume manifest"):
        dino.run(args, backend_factory=Scorer, versions=versions)


def test_journal_rejects_valid_json_with_modified_scored_content(tmp_path):
    rows, args = fixtures(tmp_path)
    class Scorer:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, image, phrases):
            return predictions(phrases)
    dino.run(args, backend_factory=Scorer, versions={"test": "frozen"})
    shard = tmp_path / "scored.jsonl.journal/shards/image-000000.json"
    envelope = json.loads(shard.read_text())
    envelope["payload"]["records"][0]["scores"]["dino_max"] = 0
    shard.write_text(json.dumps(envelope), encoding="utf-8")
    args.resume = True
    with pytest.raises(ValueError, match="checksum"):
        dino.run(args, backend_factory=Scorer, versions={"test": "frozen"})
