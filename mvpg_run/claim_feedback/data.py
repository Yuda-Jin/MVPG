"""Source-preserving claim feedback and conservative token alignment.

This module deliberately has no model-framework dependencies.  A visual score is
weak supervision, not proof that an object is absent.  Unresolved claims never
become positive merely because they were not accepted as negative.

Record spans are half-open Python character offsets into ``record['response']``.
``sentence_claims`` is a punctuation ablation, *not* a DiscoSG implementation.
External parsers must supply source spans; rewritten text is never automatically
searched for and treated as a trustworthy source anchor.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from numbers import Integral, Real
from typing import Any


LABELS = frozenset({"+", "-", "?"})
_UNRESOLVED_ALIGNMENT = frozenset({"ambiguous", "unresolved", "failed", "unknown"})


def _identifier(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{name} must be a nonempty string or integer")
    value = str(value)
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _finite_number(value: Any, name: str, *, unit_interval: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if unit_interval and not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _spans(value: Any, length: int, name: str) -> list[list[int]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of [start, end] character spans")
    result = []
    for span in value:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError(f"{name} contains a malformed span")
        start, end = span
        if any(isinstance(x, bool) or not isinstance(x, Integral) for x in span):
            raise ValueError(f"{name} offsets must be integers")
        start, end = int(start), int(end)
        if not 0 <= start < end <= length:
            raise ValueError(f"{name} span [{start}, {end}] is outside the response")
        result.append([start, end])
    return _merge_spans(result)


def _merge_spans(spans: Sequence[Sequence[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _contained(start: int, end: int, spans: Sequence[Sequence[int]]) -> bool:
    return any(a <= start and end <= b for a, b in spans)


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and deep-copy a generated-response record.

    Extra provenance fields are retained. Empty anchors are allowed as unresolved
    parser outputs, but they cannot produce training units. Labels default to
    ``?`` and weights to zero; no truth labels or character offsets are invented.
    """
    if not isinstance(record, Mapping):
        raise ValueError("a response record must be an object")
    out = copy.deepcopy(dict(record))
    if "metadata" in out and not isinstance(out["metadata"], dict):
        raise ValueError("record.metadata must be an object")
    for key in ("id", "image", "question", "response"):
        if key not in out:
            raise ValueError(f"record is missing required field {key!r}")
    out["id"] = _identifier(out["id"], "record.id")
    for key in ("image", "question", "response"):
        if not isinstance(out[key], str):
            raise ValueError(f"record.{key} must be a string")
    if not out["image"].strip():
        raise ValueError("record.image must be nonempty")
    response = out["response"]
    if "response_ids" in out:
        ids = out["response_ids"]
        if not isinstance(ids, (list, tuple)) or any(
            isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in ids
        ):
            raise ValueError("response_ids must be a sequence of nonnegative token IDs")
        out["response_ids"] = [int(x) for x in ids]
    claims = out.setdefault("claims", [])
    if not isinstance(claims, (list, tuple)):
        raise ValueError("record.claims must be a list")
    validated = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(claims):
        if not isinstance(raw, Mapping):
            raise ValueError("each claim must be an object")
        claim = copy.deepcopy(dict(raw))
        cid = _identifier(claim.get("id", f"claim-{index:04d}"), "claim.id")
        if cid in seen_ids:
            raise ValueError(f"duplicate claim ID {cid!r}")
        seen_ids.add(cid)
        claim["id"] = cid
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            raise ValueError(f"claim {cid!r} needs nonempty text")
        kind = claim.setdefault("kind", "unknown")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"claim {cid!r} needs a nonempty kind")
        label = claim.setdefault("label", "?")
        if not isinstance(label, str) or label not in LABELS:
            raise ValueError(f"claim {cid!r} label must be '+', '-', or '?'")
        claim["weight"] = _finite_number(claim.get("weight", 0.0), "claim.weight", unit_interval=True)
        claim["source_spans"] = _spans(claim.get("source_spans", []), len(response), "source_spans")
        if "error_spans" in claim:
            errors = _spans(claim["error_spans"], len(response), "error_spans")
            if any(not _contained(a, b, claim["source_spans"]) for a, b in errors):
                raise ValueError(f"claim {cid!r} error_spans must be contained in source_spans")
            claim["error_spans"] = errors
        if claim.get("support_score") is not None:
            claim["support_score"] = _finite_number(claim["support_score"], "support_score")
        if "alignment_confidence" in claim:
            claim["alignment_confidence"] = _finite_number(
                claim["alignment_confidence"], "alignment_confidence", unit_interval=True
            )
        if "alignment_status" in claim and not isinstance(claim["alignment_status"], str):
            raise ValueError("claim.alignment_status must be a string")
        for reason_field in ("feedback_reason", "alignment_reason"):
            if reason_field in claim and not isinstance(claim[reason_field], str):
                raise ValueError(f"claim.{reason_field} must be a string")
        for flag in ("ambiguous", "ambiguous_binding"):
            if flag in claim and not isinstance(claim[flag], bool):
                raise ValueError(f"claim.{flag} must be boolean")
        validated.append(claim)
    out["claims"] = validated
    return out


def sentence_claims(response: str) -> list[dict[str, Any]]:
    """Return exact-source punctuation units for an explicitly named ablation.

    These are not atomic factual claims and carry kind ``sentence``. The visual
    scorer must explicitly allow that kind to use this coarse baseline.
    """
    if not isinstance(response, str):
        raise ValueError("response must be a string")
    claims = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|[\r\n]+", response):
        end = match.start()
        left = start + len(response[start:end]) - len(response[start:end].lstrip())
        right = end - (len(response[start:end]) - len(response[start:end].rstrip()))
        if left < right:
            claims.append(_sentence_unit(response, left, right, len(claims)))
        start = match.end()
    end = len(response)
    left = start + len(response[start:end]) - len(response[start:end].lstrip())
    right = end - (len(response[start:end]) - len(response[start:end].rstrip()))
    if left < right:
        claims.append(_sentence_unit(response, left, right, len(claims)))
    return claims


def _sentence_unit(response: str, start: int, end: int, index: int) -> dict[str, Any]:
    return {
        "id": f"sentence-{index:04d}", "text": response[start:end],
        "source_spans": [[start, end]], "kind": "sentence", "label": "?",
        "weight": 0.0, "parser": "punctuation_baseline",
    }


def _alignment_unresolved(claim: Mapping[str, Any]) -> bool:
    return bool(
        claim.get("ambiguous") or claim.get("ambiguous_binding")
        or claim.get("alignment_status") in _UNRESOLVED_ALIGNMENT
        or claim.get("alignment_confidence", 1.0) <= 0
    )


def import_claims(record: Mapping[str, Any], claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Attach externally parsed claims without guessing anchors from claim text."""
    out = copy.deepcopy(dict(record))
    out["claims"] = copy.deepcopy(list(claims))
    out = validate_record(out)
    for claim in out["claims"]:
        if not claim["source_spans"] or _alignment_unresolved(claim):
            claim.update(label="?", weight=0.0, feedback_reason="source_alignment_unresolved")
    return out


def classify_claims(
    record: Mapping[str, Any],
    score_claims_callable: Callable[[str, Sequence[str]], Iterable[float]],
    negative_threshold: float,
    positive_threshold: float,
    accepted_kinds: Sequence[str] = ("object",),
    weight: float = 1.0,
    *,
    allow_whole_claim_negative: bool = False,
) -> dict[str, Any]:
    """Assign calibrated-proxy feedback using two thresholds and abstention.

    The scorer is called as ``scorer(image_path, [claim_text, ...])`` and must
    return one score per eligible claim. Thresholds must come from an independent
    calibration set; this function supplies no published or inherited defaults.
    Low-score negatives are *weak* feedback. Without an identified error span,
    they remain unresolved unless the whole-claim-negative ablation is enabled.
    """
    low = _finite_number(negative_threshold, "negative_threshold")
    high = _finite_number(positive_threshold, "positive_threshold")
    if low >= high:
        raise ValueError("negative_threshold must be smaller than positive_threshold")
    confidence = _finite_number(weight, "weight", unit_interval=True)
    if isinstance(accepted_kinds, (str, bytes, Mapping)):
        raise ValueError("accepted_kinds must be a sequence of nonempty kind names")
    try:
        kinds = tuple(accepted_kinds)
    except TypeError as error:
        raise ValueError("accepted_kinds must be a sequence of nonempty kind names") from error
    if any(not isinstance(x, str) or not x for x in kinds):
        raise ValueError("accepted_kinds must be a sequence of nonempty kind names")
    if not isinstance(allow_whole_claim_negative, bool):
        raise ValueError("allow_whole_claim_negative must be boolean")
    out = validate_record(record)
    eligible = []
    for claim in out["claims"]:
        claim.update(label="?", weight=0.0, support_score=None)
        if claim["kind"] not in kinds:
            claim["feedback_reason"] = "unsupported_kind"
        elif not claim["source_spans"] or _alignment_unresolved(claim):
            claim["feedback_reason"] = "source_alignment_unresolved"
        else:
            eligible.append(claim)
    if not eligible:
        return out
    raw_scores = score_claims_callable(out["image"], [c["text"] for c in eligible])
    if isinstance(raw_scores, (str, bytes, Mapping)):
        raise ValueError("scorer must return a sequence of numeric scores")
    scores = list(raw_scores)
    if len(scores) != len(eligible):
        raise ValueError("scorer returned a different number of scores than claims")
    for claim, raw_score in zip(eligible, scores):
        try:
            if isinstance(raw_score, bool):
                raise ValueError("a boolean is not a support score")
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError):
            score = math.nan
        if not math.isfinite(score):
            claim["feedback_reason"] = "invalid_support_score"
            continue
        claim["support_score"] = score
        if score >= high:
            claim.update(label="+", weight=confidence, feedback_reason="accepted_visual_support")
        elif score <= low:
            if claim.get("error_spans") or allow_whole_claim_negative:
                claim.update(label="-", weight=confidence, feedback_reason="accepted_weak_negative")
            else:
                claim["feedback_reason"] = "negative_error_span_unavailable"
        else:
            claim["feedback_reason"] = "score_in_abstention_band"
    return out


def _validate_offsets(offsets: Sequence[Sequence[int]], response: str) -> list[tuple[int, int]]:
    result = []
    previous_start = -1
    previous_end = -1
    for pair in offsets:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(
            isinstance(x, bool) or not isinstance(x, Integral) for x in pair
        ):
            raise ValueError("token offsets must be integer [start, end] pairs")
        start, end = map(int, pair)
        if not 0 <= start <= end <= len(response):
            raise ValueError("a token offset is outside the response")
        if start != end:
            if start < previous_start or end < previous_end:
                raise ValueError("nonempty token offsets must follow source order")
            previous_start, previous_end = start, end
        result.append((start, end))
    return result


def align_claims(
    record: Mapping[str, Any],
    offsets: Sequence[Sequence[int]],
    *,
    allow_whole_claim_negative: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map accepted claims to response-token indices, never to prompt positions.

    All selected non-whitespace characters must be covered before a unit is
    accepted. A token crossing an anchor into other *content* excludes the whole
    unit; outside leading/trailing whitespace alone is safe. This supports common
    whitespace-prefixed subwords without penalizing an additional fact. Special
    tokens have empty offsets and receive no feedback. Truncation is not silently
    interpreted as a shorter claim. Opposite polarities lose their shared tokens;
    same-polarity ownership uses highest weight, then stable claim ID.
    """
    if not isinstance(allow_whole_claim_negative, bool):
        raise ValueError("allow_whole_claim_negative must be boolean")
    item = validate_record(record)
    response = item["response"]
    token_offsets = _validate_offsets(offsets, response)
    if "response_ids" in item and len(item["response_ids"]) != len(token_offsets):
        raise ValueError("token offsets and stored response_ids must have equal length")
    stats: dict[str, Any] = {
        "n_claims": len(item["claims"]), "n_units": 0,
        "n_response_tokens": sum(a != b for a, b in token_offsets),
        "n_positive_tokens": 0, "n_negative_tokens": 0, "n_conflict_tokens": 0,
        "n_duplicate_tokens": 0, "n_boundary_crossing_claims": 0,
        "n_truncated_claims": 0, "n_unresolved_claims": 0,
        "n_zero_weight_claims": 0, "coverage": 0.0, "excluded_claims": {},
    }
    candidates = []

    def exclude(claim: Mapping[str, Any], reason: str) -> None:
        stats["excluded_claims"][claim["id"]] = reason

    for claim in item["claims"]:
        if claim["label"] == "?" or _alignment_unresolved(claim):
            stats["n_unresolved_claims"] += 1
            exclude(claim, "unresolved_feedback_or_alignment")
            continue
        if claim["weight"] <= 0:
            stats["n_zero_weight_claims"] += 1
            exclude(claim, "zero_weight")
            continue
        spans = claim["source_spans"]
        if claim["label"] == "-":
            if claim.get("error_spans"):
                spans = claim["error_spans"]
            elif not allow_whole_claim_negative:
                exclude(claim, "negative_error_span_unavailable")
                continue
        selected = {i for a, b in spans for i in range(a, b) if not response[i].isspace()}
        if not selected:
            exclude(claim, "no_source_content")
            continue
        indices = []
        covered: set[int] = set()
        crossing = False
        for index, (start, end) in enumerate(token_offsets):
            if start == end or not any(start < b and a < end for a, b in spans):
                continue
            content = {i for i in range(start, end) if not response[i].isspace()}
            if not content & selected:
                continue
            if content - selected:
                crossing = True
                break
            indices.append(index)
            covered.update(content)
        if crossing:
            stats["n_boundary_crossing_claims"] += 1
            exclude(claim, "token_crosses_content_boundary")
        elif selected - covered:
            stats["n_truncated_claims"] += 1
            exclude(claim, "incomplete_source_token_coverage")
        elif indices:
            candidates.append({
                "claim_id": claim["id"], "label": claim["label"],
                "weight": claim["weight"], "token_indices": indices,
            })

    positive = {t for c in candidates if c["label"] == "+" for t in c["token_indices"]}
    negative = {t for c in candidates if c["label"] == "-" for t in c["token_indices"]}
    conflicts = positive & negative
    stats["n_conflict_tokens"] = len(conflicts)
    owned: set[int] = set()
    units = []
    for unit in sorted(candidates, key=lambda c: (-c["weight"], c["claim_id"])):
        unique = [t for t in unit["token_indices"] if t not in conflicts and t not in owned]
        stats["n_duplicate_tokens"] += sum(t in owned for t in unit["token_indices"] if t not in conflicts)
        if not unique:
            stats["excluded_claims"][unit["claim_id"]] = "only_conflicting_or_duplicate_tokens"
            continue
        unit["token_indices"] = unique
        owned.update(unique)
        units.append(unit)
        stats["n_positive_tokens" if unit["label"] == "+" else "n_negative_tokens"] += len(unique)
    stats["n_units"] = len(units)
    stats["coverage"] = len(owned) / stats["n_response_tokens"] if stats["n_response_tokens"] else 0.0
    return units, stats


def read_jsonl(path: str | os.PathLike[str], *, validate: bool = True) -> Iterator[dict[str, Any]]:
    """Stream JSON objects; include source path and line in validation errors."""
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSONL entries must be objects")
                if validate:
                    record = validate_record(record)
                    if record["id"] in seen_ids:
                        raise ValueError(f"duplicate record ID {record['id']!r}")
                    seen_ids.add(record["id"])
                yield record
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def write_jsonl(
    path: str | os.PathLike[str],
    records: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
    validate: bool = True,
) -> int:
    """Write a complete JSONL file, refusing accidental replacement by default.

    Serialization/validation finishes in a temporary file before the destination
    is published. With overwrite=True, a failed record leaves the old file intact.
    Returns the written record count.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary: Path | None = None
    count = 0
    seen_ids: set[str] = set()
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for record in records:
                item = validate_record(record) if validate else dict(record)
                if validate:
                    if item["id"] in seen_ids:
                        raise ValueError(f"{destination}: record {count + 1}: duplicate record ID {item['id']!r}")
                    seen_ids.add(item["id"])
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise
            except OSError:
                # Filesystems without hard links still use exclusive creation.
                created = False
                try:
                    with destination.open("xb") as target:
                        created = True
                        with temporary.open("rb") as source:
                            shutil.copyfileobj(source, target)
                except BaseException:
                    if created:
                        destination.unlink(missing_ok=True)
                    raise
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
