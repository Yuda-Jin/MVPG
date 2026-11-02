"""Feedback safety tests with natural mixed-content and tokenizer edge cases."""

import copy
import json
import math

import pytest

from mvpg_run.claim_feedback.data import (
    align_claims,
    classify_claims,
    import_claims,
    read_jsonl,
    sentence_claims,
    validate_record,
    write_jsonl,
)


def record(response="red dog.", claims=None, **extra):
    return {"id": "image-1", "image": "images/1.jpg", "question": "Describe it.",
            "response": response, "claims": claims or [], **extra}


def claim(cid, text, spans, label="?", weight=1.0, **extra):
    return {"id": cid, "text": text, "source_spans": spans, "kind": "object",
            "label": label, "weight": weight, **extra}


def test_validation_preserves_provenance_and_does_not_mutate_source():
    original = record(claims=[claim("c", "dog", [[4, 7]], "+")], model={"revision": "abc"})
    snapshot = copy.deepcopy(original)
    result = validate_record(original)
    result["model"]["revision"] = "other"
    result["claims"][0]["source_spans"][0][0] = 0
    assert original == snapshot


@pytest.mark.parametrize("bad", [
    {"source_spans": [[-1, 3]]}, {"source_spans": [[0, 99]]},
    {"source_spans": [[3, 3]]}, {"source_spans": [[True, 3]]},
    {"source_spans": [[0, 3]], "error_spans": [[4, 7]]},
    {"weight": math.nan}, {"weight": -0.1}, {"weight": 1.1}, {"label": "false"},
    {"label": []}, {"alignment_status": []},
])
def test_bad_feedback_records_fail_loudly(bad):
    c = claim("c", "dog", [[4, 7]])
    c.update(bad)
    with pytest.raises(ValueError):
        validate_record(record(claims=[c]))


def test_sentence_baseline_has_exact_source_spans_and_is_not_atomic_disco():
    text = "  A dog sits.\nA lamp appears!  Two chairs? "
    units = sentence_claims(text)
    assert [c["text"] for c in units] == ["A dog sits.", "A lamp appears!", "Two chairs?"]
    assert all(text[a:b] == c["text"] for c in units for a, b in c["source_spans"])
    assert all(c["kind"] == "sentence" and c["label"] == "?" for c in units)
    assert all(c["parser"] == "punctuation_baseline" for c in units)
    assert sentence_claims(" \n ") == []


def test_rewritten_or_repeated_text_is_never_guessed_as_an_anchor():
    out = import_claims(record("A dog sits. Another dog runs."), [
        {"id": "rewritten", "text": "There is a dog.", "kind": "object", "label": "+", "weight": 1.0},
    ])
    assert out["claims"][0]["source_spans"] == []
    assert out["claims"][0]["label"] == "?"
    assert out["claims"][0]["weight"] == 0


def test_two_thresholds_preserve_an_abstention_band_and_visual_score_provenance():
    r = record("red dog. blue dog. green dog.", [
        claim("bad", "red dog", [[0, 7]], error_spans=[[0, 3]]),
        claim("uncertain", "blue dog", [[9, 17]]),
        claim("good", "green dog", [[19, 28]]),
    ])
    calls = []

    def scorer(image, texts):
        calls.append((image, texts))
        return [-0.7, 0.2, 0.8]

    out = classify_claims(r, scorer, -0.3, 0.6, weight=0.75)
    assert [c["label"] for c in out["claims"]] == ["-", "?", "+"]
    assert [c["weight"] for c in out["claims"]] == [0.75, 0.0, 0.75]
    assert calls == [("images/1.jpg", ["red dog", "blue dog", "green dog"])]
    assert r["claims"][0]["label"] == "?"
    assert out["claims"][0]["feedback_reason"] == "accepted_weak_negative"


def test_low_score_without_reliable_error_span_is_unresolved_by_default():
    r = record(claims=[claim("c", "red dog", [[0, 7]])])
    out = classify_claims(r, lambda image, texts: [-0.9], -0.3, 0.6)
    assert out["claims"][0]["label"] == "?"
    assert out["claims"][0]["feedback_reason"] == "negative_error_span_unavailable"
    ablation = classify_claims(r, lambda image, texts: [-0.9], -0.3, 0.6,
                              allow_whole_claim_negative=True)
    assert ablation["claims"][0]["label"] == "-"


@pytest.mark.parametrize("low,high", [(0.5, 0.5), (0.8, 0.2), (math.nan, 0.5), (-0.5, math.inf)])
def test_unusable_calibration_thresholds_are_rejected(low, high):
    with pytest.raises(ValueError):
        classify_claims(record(), lambda image, texts: [], low, high)


@pytest.mark.parametrize("kinds", ["object", None, {"object": True}, ["object", None]])
def test_kind_configuration_requires_an_explicit_sequence(kinds):
    with pytest.raises(ValueError, match="accepted_kinds"):
        classify_claims(record(), lambda image, texts: [], -0.3, 0.6, accepted_kinds=kinds)


def test_unsupported_or_ambiguous_claims_are_not_sent_for_visual_scoring():
    r = record(claims=[
        claim("unsupported", "dog is red", [[0, 7]], kind="relation"),
        claim("ambiguous", "dog", [[4, 7]], ambiguous_binding=True),
    ])

    def do_not_call(*args):
        raise AssertionError("unusable claims were scored")

    out = classify_claims(r, do_not_call, -0.3, 0.6)
    assert [c["label"] for c in out["claims"]] == ["?", "?"]


def test_scorer_length_mismatch_fails_and_nonfinite_scores_abstain():
    r = record(claims=[claim("c", "dog", [[4, 7]])])
    with pytest.raises(ValueError, match="number of scores"):
        classify_claims(r, lambda image, texts: [], -0.3, 0.6)
    for bad in [math.nan, math.inf, True, None]:
        out = classify_claims(r, lambda image, texts: [bad], -0.3, 0.6)
        assert out["claims"][0]["label"] == "?"
        assert out["claims"][0]["support_score"] is None


def test_positive_and_negative_units_use_response_tokens_not_prompt_or_special_tokens():
    r = record(claims=[
        claim("red", "red dog", [[0, 7]], "-", error_spans=[[0, 3]]),
        claim("dog", "dog", [[4, 7]], "+"),
    ], response_ids=[99, 10, 11, 12, 100])
    units, stats = align_claims(r, [(0, 0), (0, 3), (3, 7), (7, 8), (0, 0)])
    assert {u["claim_id"]: u["token_indices"] for u in units} == {"red": [1], "dog": [2]}
    assert stats["n_positive_tokens"] == stats["n_negative_tokens"] == 1
    assert stats["coverage"] == pytest.approx(2 / 3)


def test_token_crossing_an_extra_content_boundary_is_not_partially_penalized():
    r = record(claims=[claim("red", "red dog", [[0, 7]], "-", error_spans=[[0, 3]])])
    units, stats = align_claims(r, [(0, 7), (7, 8)])
    assert units == []
    assert stats["n_boundary_crossing_claims"] == 1


def test_truncated_fact_is_not_silently_turned_into_a_shorter_training_span():
    r = record("dog runs", [claim("runs", "runs", [[4, 8]], "+")])
    units, stats = align_claims(r, [(0, 3), (4, 6)])
    assert units == []
    assert stats["n_truncated_claims"] == 1


def test_opposite_polarity_conflicts_are_removed_before_ownership():
    r = record("dog runs", [
        claim("positive", "dog runs", [[0, 8]], "+"),
        claim("negative", "dog", [[0, 3]], "-", error_spans=[[0, 3]]),
    ])
    units, stats = align_claims(r, [(0, 3), (4, 8)])
    assert units == [{"claim_id": "positive", "label": "+", "weight": 1.0, "token_indices": [1]}]
    assert stats["n_conflict_tokens"] == 1


def test_repeated_claims_are_counted_once_with_confidence_and_stable_id_ownership():
    r = record(claims=[
        claim("weak", "dog", [[4, 7]], "+", 0.1),
        claim("z", "dog", [[4, 7]], "+", 0.9),
        claim("a", "dog", [[4, 7]], "+", 0.9),
    ])
    units, stats = align_claims(r, [(0, 3), (4, 7), (7, 8)])
    assert len(units) == 1 and units[0]["claim_id"] == "a"
    assert units[0]["token_indices"] == [1]
    assert stats["n_duplicate_tokens"] == 2


def test_prelabeled_negatives_still_require_error_spans_during_alignment():
    r = record(claims=[claim("bad", "red dog", [[0, 7]], "-")])
    assert align_claims(r, [(0, 3), (4, 7), (7, 8)])[0] == []
    units, _ = align_claims(r, [(0, 3), (4, 7), (7, 8)], allow_whole_claim_negative=True)
    assert units[0]["token_indices"] == [0, 1]


def test_invalid_offset_order_or_stored_token_count_is_not_accepted():
    with pytest.raises(ValueError, match="source order"):
        align_claims(record(), [(4, 7), (0, 3)])
    with pytest.raises(ValueError, match="equal length"):
        align_claims(record(response_ids=[1]), [(0, 3), (4, 7)])


def test_empty_and_ambiguous_examples_safely_have_no_update_units():
    units, stats = align_claims(record(""), [])
    assert units == [] and stats["coverage"] == 0
    r = record(claims=[claim("unknown", "dog", [[4, 7]], "+", ambiguous=True)])
    assert align_claims(r, [(0, 3), (4, 7), (7, 8)])[0] == []


def test_jsonl_roundtrip_exclusive_creation_and_failed_replacement(tmp_path):
    path = tmp_path / "responses.jsonl"
    r = record("一只狗。", source={"model": "frozen-checkpoint"})
    assert write_jsonl(path, [r]) == 1
    assert list(read_jsonl(path)) == [validate_record(r)]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_jsonl(path, [record("something else")])
    with pytest.raises(ValueError):
        write_jsonl(path, [r, {"id": "invalid"}], overwrite=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_jsonl_error_identifies_the_input_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(record()) + "\n{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.jsonl:2"):
        list(read_jsonl(path))


def test_validated_jsonl_read_rejects_duplicate_normalized_record_ids(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    path.write_text(json.dumps(record(id=1)) + "\n" + json.dumps(record(id="1")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate.jsonl:2: duplicate record ID '1'"):
        list(read_jsonl(path))
    assert [r["id"] for r in read_jsonl(path, validate=False)] == [1, "1"]


def test_duplicate_record_ids_never_publish_or_replace_validated_output(tmp_path):
    path = tmp_path / "responses.jsonl"
    duplicates = [record(id=1), record(id="1")]
    with pytest.raises(ValueError, match="record 2: duplicate record ID '1'"):
        write_jsonl(path, duplicates)
    assert not path.exists()
    write_jsonl(path, [record(id="original")])
    original = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate record ID"):
        write_jsonl(path, duplicates, overwrite=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
