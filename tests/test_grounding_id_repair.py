"""ID transport repairs preserve every valid judgment and frozen source field."""
from __future__ import annotations

import copy

import pytest

from run.repair_grounding_judge_ids import merge_repaired_rows


def corpus():
    cases = []
    for index in range(4):
        group = str(index // 2)
        cases.append({"case_id": f"canonical-{index}", "record_id": group,
            "claim_id": str(index), "image": f"image-{group}.png", "image_group": group,
            "split": "dev" if group == "0" else "test", "question": "Describe.",
            "response": "A cat stands nearby.", "claim_text": "cat", "source_spans": [[2, 5]],
            "kind": "object", "judge_label": None, "scores": {},
            "source_provenance": {"input_sha256": "frozen-corpus", "generation": {"seed": 42}}})
    original = copy.deepcopy(cases)
    for index, row in enumerate(original):
        if index < 2:
            row.update(judge_label="supported" if index == 0 else "uncertain", judge_error=None,
                judge={"label": "supported" if index == 0 else "uncertain",
                       "reason": "Original reason", "evidence": "Original evidence"},
                judge_request_id=f"original-request-{index}",
                judge_image_journal_ref={"image_key": "original-image", "judge_attempt": 4})
        else:
            row.update(judge_error="ValueError", judge_request_id=f"failed-request-{index}",
                judge_image_journal_ref={"image_key": "failed-image", "judge_attempt": 4})
    repaired = copy.deepcopy(cases[2:])
    for index, row in enumerate(repaired):
        label = "supported" if index == 0 else "unsupported"
        row.update(judge_label=label, judge_error=None,
            judge={"label": label, "reason": "New actual judgment", "evidence": "Visible evidence"},
            judge_protocol="image_batch_claim_judge_v1/request_alias",
            judge_id_encoding="request_alias", judge_request_case_id=f"c{index + 1:04d}",
            judge_request_id="repair-request",
            judge_image_journal_ref={"image_key": "repair-image", "judge_attempt": 1})
    reference = {"protocol": "request_local_short_ids_v1", "repair_manifest_sha256": "repair-sha",
                 "repair_raw_journal": "repairs/journal.jsonl", "extra": {"source": "recorded"}}
    return cases, original, repaired, reference


def test_merge_restores_canonical_order_and_preserves_all_successful_judgments():
    cases, original, repaired, reference = corpus()
    merged, audit = merge_repaired_rows(cases, original[::-1], repaired[::-1], reference)
    assert [row["case_id"] for row in merged] == [row["case_id"] for row in cases]
    assert merged[:2] == original[:2]
    assert not any("judge_repair" in row for row in merged[:2])
    assert [row["judge_label"] for row in merged[2:]] == ["supported", "unsupported"]
    for row in merged[2:]:
        assert row["judge_repair"] == dict(reference, original_judge_error="ValueError")
        assert row["judge_request_id"] == "repair-request"
    assert audit == {"total_cases": 4, "preserved_successful_cases": 2, "repaired_cases": 2}


def test_merge_never_mutates_inputs_or_shares_nested_repair_metadata():
    cases, original, repaired, reference = corpus()
    before = copy.deepcopy((cases, original, repaired, reference))
    merged, _ = merge_repaired_rows(cases, original, repaired, reference)
    assert (cases, original, repaired, reference) == before
    merged[0]["judge"]["evidence"] = "Changed only in result"
    merged[2]["source_provenance"]["generation"]["seed"] = 999
    merged[2]["judge_repair"]["extra"]["source"] = "Changed only in result"
    assert (cases, original, repaired, reference) == before
    assert merged[3]["judge_repair"]["extra"]["source"] == "recorded"


@pytest.mark.parametrize("change", ["missing", "extra_success", "unknown_id", "duplicate"])
def test_only_the_exact_original_failed_id_set_can_be_repaired(change):
    cases, original, repaired, reference = corpus()
    if change == "missing":
        repaired.pop()
    elif change == "extra_success":
        repaired.append(copy.deepcopy(original[0]))
    elif change == "unknown_id":
        repaired[0]["case_id"] = "not-in-source"
    else:
        repaired.append(copy.deepcopy(repaired[0]))
    with pytest.raises(ValueError):
        merge_repaired_rows(cases, original, repaired, reference)


@pytest.mark.parametrize("field,value", [
    ("image", "another.png"), ("record_id", "another-record"), ("claim_id", "another-claim"),
    ("question", "Changed question"), ("response", "A dog stands nearby."),
    ("claim_text", "dog"), ("source_spans", [[0, 2]]), ("scores", {"global_raw": .5}),
    ("source_provenance", {"input_sha256": "changed"}),
])
def test_repair_cannot_change_any_frozen_source_field(field, value):
    cases, original, repaired, reference = corpus()
    repaired[0][field] = value
    with pytest.raises(ValueError):
        merge_repaired_rows(cases, original, repaired, reference)


def test_original_successful_rows_must_still_match_the_frozen_source():
    cases, original, repaired, reference = corpus()
    original[0]["claim_text"] = "dog"
    with pytest.raises(ValueError):
        merge_repaired_rows(cases, original, repaired, reference)


@pytest.mark.parametrize("change", ["missing_original", "extra_original", "duplicate_original"])
def test_original_judgments_must_cover_the_complete_corpus_exactly(change):
    cases, original, repaired, reference = corpus()
    if change == "missing_original":
        original.pop()
    elif change == "extra_original":
        extra = copy.deepcopy(original[0])
        extra["case_id"] = "extra-original"
        original.append(extra)
    else:
        original.append(copy.deepcopy(original[0]))
    with pytest.raises(ValueError):
        merge_repaired_rows(cases, original, repaired, reference)


@pytest.mark.parametrize("fields", [
    {"judge_label": None}, {"judge_label": "invalid"},
    {"judge_label": "supported", "judge_error": "HTTP 503"},
])
def test_invalid_or_unfinished_repair_labels_are_rejected(fields):
    cases, original, repaired, reference = corpus()
    repaired[0].update(fields)
    with pytest.raises(ValueError):
        merge_repaired_rows(cases, original, repaired, reference)
