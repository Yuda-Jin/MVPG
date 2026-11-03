"""Offline diagnostics for blind visual-judge labels and FG-CLIP scores.

This module uses only the standard library. Judges are a silver reference, not
ground truth. ``uncertain`` is excluded from ranking metrics but stays in the
denominator of conservative feedback precision and all-claim coverage. Threshold
search uses development images only; the score variant must be chosen before
looking at the test set.
"""
from __future__ import annotations

import bisect
import math
from collections import Counter, defaultdict
from numbers import Integral, Real
import random
from statistics import NormalDist
from typing import Any, Iterable, Mapping


_LABELS = ("supported", "unsupported", "uncertain")
_WILSON_Z = NormalDist().inv_cdf(0.975)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _wilson_lower(correct: int, selected: int) -> float | None:
    if not selected:
        return None
    p = correct / selected
    z2 = _WILSON_Z ** 2
    return (p + z2 / (2 * selected) - _WILSON_Z * math.sqrt(
        p * (1 - p) / selected + z2 / (4 * selected ** 2)
    )) / (1 + z2 / selected)


def _validate_cases(cases: Iterable[Mapping[str, Any]], score_key: str):
    if not isinstance(score_key, str) or not score_key:
        raise ValueError("score_key must be a nonempty string")
    rows = []
    seen = set()
    group_splits = {}
    record_groups = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Each case must be an object")
        for key in ("case_id", "record_id", "image_group"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Each case requires a nonempty {key}")
        case_id, record_id, group = (case[k] for k in ("case_id", "record_id", "image_group"))
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        split = case.get("split")
        if split not in ("dev", "test"):
            raise ValueError("Every case split must be dev or test")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Image group crosses dev/test: {group}")
        group_splits[group] = split
        if record_id in record_groups and record_groups[record_id] != group:
            raise ValueError(f"Record assigned to inconsistent image groups: {record_id}")
        record_groups[record_id] = group
        label = case.get("judge_label")
        if label not in _LABELS:
            raise ValueError("judge_label must be supported, unsupported or uncertain")
        scores = case.get("scores")
        if not isinstance(scores, Mapping) or score_key not in scores:
            raise ValueError(f"Case {case_id} lacks score variant {score_key}")
        rows.append({"case_id": case_id, "record_id": record_id, "image_group": group,
                     "split": split, "label": label,
                     "score": _finite(scores[score_key], f"{case_id}.{score_key}")})
    if set(row["split"] for row in rows) != {"dev", "test"}:
        raise ValueError("Require nonempty dev and test cases")
    return rows


def _ranking(rows):
    decided = [r for r in rows if r["label"] != "uncertain"]
    positives = sum(r["label"] == "supported" for r in decided)
    negatives = len(decided) - positives
    # Group exact score ties before accumulating pairwise wins / recall steps.
    groups = defaultdict(Counter)
    for row in decided:
        groups[row["score"]][row["label"]] += 1
    wins, unsupported_below = 0.0, 0
    # Low support scores rank unsupported claims first, as required for AP.
    ap, seen, seen_unsupported = 0.0, 0, 0
    for score in sorted(groups):
        n_support = groups[score]["supported"]
        n_unsupported = groups[score]["unsupported"]
        wins += n_support * (unsupported_below + n_unsupported / 2)
        unsupported_below += n_unsupported
        seen += n_support + n_unsupported
        seen_unsupported += n_unsupported
        if negatives:
            ap += (n_unsupported / negatives) * (seen_unsupported / seen)
    return {"n_decided": len(decided), "supported": positives, "unsupported": negatives,
            "auroc_supported": wins / (positives * negatives) if positives and negatives else None,
            "ap_unsupported": ap if negatives else None,
            "unsupported_prevalence": _ratio(negatives, len(decided)),
            "ap_definition": "noninterpolated average precision at distinct -score thresholds"}


def _counts(rows):
    labels = Counter(r["label"] for r in rows)
    return {"n_cases": len(rows), "n_images": len({r["image_group"] for r in rows}),
            **{label: labels[label] for label in _LABELS}}


def _tail_candidates(rows, polarity, target_precision, min_selected, strict):
    groups = defaultdict(Counter)
    for row in rows:
        groups[row["score"]][row["label"]] += 1
    selected, correct, uncertain = 0, 0, 0
    correct_label = "unsupported" if polarity == "negative" else "supported"
    result = []
    for threshold in sorted(groups, reverse=polarity == "positive"):
        counts = groups[threshold]
        selected += sum(counts.values())
        correct += counts[correct_label]
        uncertain += counts["uncertain"]
        if selected < min_selected:
            continue
        estimate = _wilson_lower(correct, selected) if strict else correct / selected
        if estimate >= target_precision:
            result.append({"threshold": threshold, "n_selected": selected,
                           "n_correct": correct, "n_uncertain": uncertain,
                           "conservative_precision": correct / selected,
                           "conservative_precision_wilson_lower": _wilson_lower(correct, selected)})
    return result


def _select_policy(rows, target_precision, min_selected, strict):
    negatives = _tail_candidates(rows, "negative", target_precision, min_selected, strict)
    positives = sorted(_tail_candidates(rows, "positive", target_precision, min_selected, strict),
                       key=lambda c: c["threshold"])
    positive_thresholds = [c["threshold"] for c in positives]
    best, best_utility = (None, None), (-1, -1, -1)
    # For a fixed negative cutoff, the lowest eligible positive cutoff maximizes
    # positive-tail coverage. bisect_right excludes equal/overlapping cutoffs.
    for negative in [None, *negatives]:
        i = bisect.bisect_right(positive_thresholds, negative["threshold"]) if negative else 0
        positive = positives[i] if i < len(positives) else None
        counts = [c["n_selected"] for c in (negative, positive) if c is not None]
        # This experiment prioritizes retaining both feedback polarities when a
        # feasible nonoverlapping pair exists. Coverage is optimized within that
        # choice; a one-sided policy is not substituted solely for more coverage.
        utility = (len(counts), sum(counts), min(counts) if counts else 0)
        if utility > best_utility:
            best, best_utility = (negative, positive), utility
    negative, positive = best
    return {
        "thresholds": {"negative": negative["threshold"] if negative else None,
                       "positive": positive["threshold"] if positive else None},
        "selection": {"split": "dev", "target_precision": target_precision,
                      "min_selected_per_polarity": min_selected,
                      "criterion": "conservative_precision_wilson_lower" if strict else "conservative_precision",
                      "confidence_level": 0.95 if strict else None,
                      "objective": "prefer both feasible nonoverlapping polarities, then maximum all-case coverage; ties prefer larger smaller tail",
                      "feasible_negative_candidates": len(negatives),
                      "feasible_positive_candidates": len(positives),
                      "negative_disabled": negative is None, "positive_disabled": positive is None,
                      "negative_disabled_reason": (None if negative else
                          "no_compatible_nonoverlapping_pair" if negatives else "no_feasible_dev_threshold"),
                      "positive_disabled_reason": (None if positive else
                          "no_compatible_nonoverlapping_pair" if positives else "no_feasible_dev_threshold"),
                      "note": "Development threshold search is a selection heuristic, not a post-selection precision guarantee. Disabled tails abstain."},
    }


def _policy_metrics(rows, thresholds):
    total = len(rows)
    n_decided = sum(r["label"] != "uncertain" for r in rows)
    tails = {"negative": [], "positive": []}
    abstained = []
    for row in rows:
        if thresholds["negative"] is not None and row["score"] <= thresholds["negative"]:
            tails["negative"].append(row)
        elif thresholds["positive"] is not None and row["score"] >= thresholds["positive"]:
            tails["positive"].append(row)
        else:
            abstained.append(row)
    result = {"n_cases": total, "n_decided": n_decided}
    for polarity, selected in tails.items():
        labels = Counter(r["label"] for r in selected)
        target = "unsupported" if polarity == "negative" else "supported"
        other = "supported" if polarity == "negative" else "unsupported"
        correct, wrong, uncertain = labels[target], labels[other], labels["uncertain"]
        result[polarity] = {
            "threshold": thresholds[polarity], "n_selected": len(selected),
            "n_images": len({row["image_group"] for row in selected}),
            "n_correct": correct, "n_wrong": wrong, "n_uncertain": uncertain,
            "precision_on_decided": _ratio(correct, correct + wrong),
            "conservative_precision": _ratio(correct, len(selected)),
            "conservative_precision_wilson_lower": _wilson_lower(correct, len(selected)),
            "selected_uncertain_rate": _ratio(uncertain, len(selected)),
            "coverage": _ratio(len(selected), total),
            "decided_coverage": _ratio(correct + wrong, n_decided),
            "recall_on_decided_class": _ratio(correct, sum(r["label"] == target for r in rows)),
        }
    selected = tails["negative"] + tails["positive"]
    result.update(
        coverage=_ratio(len(selected), total),
        decided_coverage=_ratio(sum(r["label"] != "uncertain" for r in selected), n_decided),
        selected_uncertain_rate=_ratio(sum(r["label"] == "uncertain" for r in selected), len(selected)),
        abstention={"n_cases": len(abstained), **{label: sum(r["label"] == label for r in abstained) for label in _LABELS}},
    )
    return result


def _quantile(sorted_values, q):
    pos = q * (len(sorted_values) - 1)
    low, high = math.floor(pos), math.ceil(pos)
    return sorted_values[low] + (pos - low) * (sorted_values[high] - sorted_values[low])


def _bootstrap(test, policies, samples, seed):
    grouped = defaultdict(list)
    for row in test:
        grouped[row["image_group"]].append(row)
    groups = sorted(grouped)
    rng = random.Random(seed)
    intervals = {name: defaultdict(list) for name in policies}
    for _ in range(samples):
        # Each repeated group contributes all its cases again; do not deduplicate
        # sampled rows or case IDs, which would invalidate the cluster bootstrap.
        rows = [row for _ in groups for row in grouped[rng.choice(groups)]]
        ranking = _ranking(rows)
        for name, policy in policies.items():
            metrics = _policy_metrics(rows, policy["thresholds"])
            values = {"auroc_supported": ranking["auroc_supported"],
                      "ap_unsupported": ranking["ap_unsupported"],
                      "coverage": metrics["coverage"],
                      "decided_coverage": metrics["decided_coverage"],
                      "selected_uncertain_rate": metrics["selected_uncertain_rate"]}
            for polarity in ("negative", "positive"):
                for key in ("precision_on_decided", "conservative_precision", "coverage", "selected_uncertain_rate", "recall_on_decided_class"):
                    values[f"{polarity}.{key}"] = metrics[polarity][key]
            for key, value in values.items():
                if value is not None:
                    intervals[name][key].append(value)
                else:
                    intervals[name][key]  # retain an explicitly undefined metric
    for name, policy in policies.items():
        policy["test_bootstrap"] = {
            "method": "image_cluster_percentile", "confidence_level": 0.95,
            "samples": samples, "seed": seed, "n_image_groups": len(groups),
            "thresholds_refit": False,
            "intervals": {key: {
                "lower": _quantile(sorted(values), 0.025) if values else None,
                "upper": _quantile(sorted(values), 0.975) if values else None,
                "valid_replicates": len(values), "undefined_replicates": samples - len(values),
            } for key, values in intervals[name].items()},
            "note": "Intervals use valid replicates only and are conditional on frozen judge labels and dev-selected thresholds; they do not include judge error or threshold-selection uncertainty. Zero observed errors can produce degenerate percentile intervals and are not evidence of 100% true precision.",
        }


def analyze_cases(cases, *, score_key="global_raw", target_precision=0.95,
                  min_selected=30, bootstrap_samples=500, seed=42,
                  current_negative_threshold=None, current_positive_threshold=None):
    """Compare fixed and dev-selected feedback policies on disjoint test images.

    Each input case requires case_id, record_id, image_group, split (dev/test),
    judge_label (supported/unsupported/uncertain), and scores[score_key]. A None
    cutoff disables that feedback polarity, allowing asymmetric abstention.
    Output is JSON serializable with undefined values represented as None.
    """
    target_precision = _finite(target_precision, "target_precision")
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must lie in (0,1]")
    for key, value, lower in (("min_selected", min_selected, 1), ("bootstrap_samples", bootstrap_samples, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{key} must be an integer >= {lower}")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    thresholds = {}
    for polarity, value in (("negative", current_negative_threshold), ("positive", current_positive_threshold)):
        thresholds[polarity] = _finite(value, f"current_{polarity}_threshold") if value is not None else None
    if all(value is not None for value in thresholds.values()) and thresholds["negative"] >= thresholds["positive"]:
        raise ValueError("Current negative threshold must be smaller than positive threshold")
    rows = _validate_cases(cases, score_key)
    dev, test = ([r for r in rows if r["split"] == split] for split in ("dev", "test"))
    policies = {
        "strict_policy": _select_policy(dev, target_precision, min_selected, True),
        "exploratory_policy": _select_policy(dev, target_precision, min_selected, False),
    }
    if any(value is not None for value in thresholds.values()):
        policies["current_policy"] = {"thresholds": thresholds, "selection": {
            "split": "external_fixed", "criterion": "caller_supplied", "note": "Baseline thresholds were not selected here."}}
    for policy in policies.values():
        policy["dev"] = _policy_metrics(dev, policy["thresholds"])
        policy["test"] = _policy_metrics(test, policy["thresholds"])
    _bootstrap(test, policies, int(bootstrap_samples), int(seed))
    return {
        "schema": "feedback_diagnostic_metrics_v1", "score_key": score_key,
        "score_orientation": "higher_is_more_supported", "reference": "visual_judge_silver_labels",
        "definitions": {
            "ranking": "Determinate labels only; supported is positive for AUROC, unsupported is positive for AP using -score.",
            "conservative_precision": "correct / all selected, including uncertain judgments in the denominator",
            "precision_on_decided": "correct / (correct + wrong), excludes uncertain judgments",
            "coverage": "all selected / all cases, including uncertain judgments",
            "decided_coverage": "selected determinate cases / all determinate cases",
            "selection": "One prespecified score variant; threshold search on dev images only; None disables a tail.",
            "strict": "95% two-sided Wilson lower bound on conservative precision reaches target, with minimum selected count. A selection heuristic, not a post-search guarantee.",
            "exploratory": "Conservative point precision reaches target; lacks the strict sampling-uncertainty screen.",
        },
        "splits": {"dev": {"counts": _counts(dev), "ranking": _ranking(dev)},
                   "test": {"counts": _counts(test), "ranking": _ranking(test)}},
        **policies,
        "current_policy": policies.get("current_policy"),
        "limitations": ["Judge labels are not human ground truth.",
                        "Image grouping must identify duplicate images/crops before this analysis; paths alone cannot prove independence.",
                        "Bootstrap intervals condition on frozen judge labels and thresholds.",
                        "Score variant and target precision must not be chosen by looking at test performance."],
    }
