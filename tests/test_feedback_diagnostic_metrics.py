"""Statistical correctness and leakage checks for feedback quality diagnostics."""
import copy
import json
import math

import pytest

from mvpg_run.claim_feedback.diagnostic_metrics import analyze_cases


def case(i, score, label, split="dev", group=None):
    return {"case_id": f"{split}-{i}", "record_id": f"{split}-{i}",
            "image_group": group or f"image-{split}-{i}", "split": split,
            "judge_label": label, "scores": {"global_raw": score}}


def pool(dev, test=None):
    return [case(i, score, label) for i, (score, label) in enumerate(dev)] + [
        case(i, score, label, "test") for i, (score, label) in enumerate(test or dev)]


def analyze(rows, **kwargs):
    return analyze_cases(rows, bootstrap_samples=0, min_selected=1, **kwargs)


@pytest.mark.parametrize("values,auc,ap", [
    ([(0.1, "unsupported"), (0.9, "supported")], 1.0, 1.0),
    ([(0.9, "unsupported"), (0.1, "supported")], 0.0, 0.5),
    ([(0.5, "unsupported"), (0.5, "supported")], 0.5, 0.5),
    ([(0.1, "unsupported"), (0.5, "unsupported"), (0.5, "supported")], 0.75, 5 / 6),
])
def test_ranking_direction_and_score_ties(values, auc, ap):
    ranking = analyze(pool(values))["splits"]["test"]["ranking"]
    assert ranking["auroc_supported"] == pytest.approx(auc)
    assert ranking["ap_unsupported"] == pytest.approx(ap)


@pytest.mark.parametrize("label,ap", [("supported", None), ("unsupported", 1.0), ("uncertain", None)])
def test_missing_classes_are_explicitly_undefined(label, ap):
    result = analyze(pool([(0.1, label)]))
    ranking = result["splits"]["test"]["ranking"]
    assert ranking["auroc_supported"] is None
    assert ranking["ap_unsupported"] == ap
    json.dumps(result, allow_nan=False)


def test_strict_can_abstain_while_point_estimate_policy_selects():
    result = analyze(pool([(0.1, "unsupported"), (0.9, "supported")]))
    assert result["strict_policy"]["thresholds"] == {"negative": None, "positive": None}
    assert result["strict_policy"]["test"]["coverage"] == 0
    assert result["strict_policy"]["selection"]["negative_disabled_reason"] == "no_feasible_dev_threshold"
    assert result["exploratory_policy"]["thresholds"] == {"negative": 0.1, "positive": 0.9}


def test_dev_only_thresholds_and_nonmutating_analysis():
    dev = [(i / 1000, "unsupported") for i in range(80)] + [
        (0.9 + i / 1000, "supported") for i in range(80)]
    original = pool(dev)
    snapshot = copy.deepcopy(original)
    changed_test = pool(dev, [(0.999, "unsupported"), (-0.5, "supported")])
    first, second = analyze(original), analyze(changed_test)
    for name in ("strict_policy", "exploratory_policy"):
        assert first[name]["thresholds"] == second[name]["thresholds"]
        assert first[name]["test"]["coverage"] != second[name]["test"]["coverage"] or first[name]["test"]["positive"]["conservative_precision"] != second[name]["test"]["positive"]["conservative_precision"]
    assert first["strict_policy"]["thresholds"] == {"negative": 0.079, "positive": 0.9}
    assert original == snapshot


def test_uncertain_is_not_a_negative_and_stays_in_precision_and_coverage_denominators():
    result = analyze(pool([(0.1, "unsupported"), (0.1, "uncertain"), (0.9, "supported")]),
                     current_negative_threshold=0.2, current_positive_threshold=0.8)
    metrics = result["current_policy"]["test"]
    assert metrics["negative"]["precision_on_decided"] == 1
    assert metrics["negative"]["conservative_precision"] == 0.5
    assert metrics["negative"]["selected_uncertain_rate"] == 0.5
    assert metrics["negative"]["coverage"] == 2 / 3
    assert metrics["negative"]["decided_coverage"] == 0.5
    assert result["exploratory_policy"]["thresholds"]["negative"] is None
    assert result["splits"]["test"]["ranking"]["n_decided"] == 2


def test_candidates_include_ties_together_and_require_strict_order():
    rows = pool([(0.5, "unsupported"), (0.5, "supported")])
    result = analyze(rows, target_precision=0.5)
    thresholds = result["exploratory_policy"]["thresholds"]
    assert sum(x is not None for x in thresholds.values()) == 1
    assert result["exploratory_policy"]["test"]["coverage"] == 1
    selection = result["exploratory_policy"]["selection"]
    assert selection["feasible_negative_candidates"] == selection["feasible_positive_candidates"] == 1
    disabled = "positive" if thresholds["positive"] is None else "negative"
    assert selection[f"{disabled}_disabled_reason"] == "no_compatible_nonoverlapping_pair"


def test_class_imbalance_retains_feasible_nonoverlapping_feedback_polarities():
    rows = pool([(0.1, "unsupported")] + [(0.9, "supported")] * 100)
    result = analyze(rows, target_precision=0.95)
    policy = result["exploratory_policy"]
    assert policy["thresholds"] == {"negative": 0.1, "positive": 0.9}
    assert policy["test"]["negative"]["n_selected"] == 1
    assert policy["test"]["positive"]["n_selected"] == 100
    assert policy["test"]["negative"]["n_images"] == 1
    assert policy["test"]["positive"]["n_images"] == 100
    assert policy["selection"]["negative_disabled_reason"] is None
    assert policy["selection"]["positive_disabled_reason"] is None


def test_asymmetric_policy_and_minimum_sample_count():
    rows = pool([(0.1, "unsupported")] + [(0.9, "supported")] * 100)
    result = analyze_cases(rows, bootstrap_samples=0, min_selected=30)
    assert result["strict_policy"]["thresholds"] == {"negative": None, "positive": 0.9}
    assert result["strict_policy"]["test"]["negative"]["n_selected"] == 0
    assert result["strict_policy"]["test"]["negative"]["conservative_precision"] is None


def test_image_leakage_and_duplicate_case_ids_rejected():
    rows = pool([(0.1, "unsupported")])
    rows[1]["image_group"] = rows[0]["image_group"]
    with pytest.raises(ValueError, match="crosses"):
        analyze(rows)
    rows = pool([(0.1, "unsupported")])
    rows[1]["case_id"] = rows[0]["case_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(rows)


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, "0.1", None])
def test_nonfinite_or_invalid_scores_rejected(bad):
    rows = pool([(0.1, "unsupported")])
    rows[0]["scores"]["global_raw"] = bad
    with pytest.raises(ValueError, match="finite real"):
        analyze(rows)


def test_bad_threshold_order_and_empty_split_rejected():
    rows = pool([(0.1, "unsupported")])
    with pytest.raises(ValueError, match="smaller"):
        analyze(rows, current_negative_threshold=0.5, current_positive_threshold=0.5)
    with pytest.raises(ValueError, match="dev and test"):
        analyze(rows[:1])


def test_cluster_bootstrap_is_reproducible_and_repeats_all_cases_in_a_group():
    # All test cases in one group: every bootstrap resample must reproduce the
    # exact same distribution, retaining repeated cases rather than dropping IDs.
    rows = pool([(0.1, "unsupported"), (0.9, "supported"), (0.9, "uncertain")])
    for row in rows:
        if row["split"] == "test":
            row["image_group"] = "one-test-image"
    first = analyze_cases(rows, bootstrap_samples=25, min_selected=1, seed=17,
                          current_negative_threshold=0.2, current_positive_threshold=0.8)
    second = analyze_cases(rows, bootstrap_samples=25, min_selected=1, seed=17,
                           current_negative_threshold=0.2, current_positive_threshold=0.8)
    assert first == second
    ci = first["current_policy"]["test_bootstrap"]["intervals"]["positive.conservative_precision"]
    assert ci == {"lower": 0.5, "upper": 0.5, "valid_replicates": 25, "undefined_replicates": 0}
    assert first["current_policy"]["test_bootstrap"]["thresholds_refit"] is False
    assert "degenerate" in first["current_policy"]["test_bootstrap"]["note"]


def test_cluster_bootstrap_reports_undefined_replicates_without_nan():
    result = analyze_cases(pool([(0.1, "unsupported"), (0.9, "supported")]),
                           bootstrap_samples=30, min_selected=1, seed=1)
    ci = result["strict_policy"]["test_bootstrap"]["intervals"]["auroc_supported"]
    assert 0 < ci["valid_replicates"] < 30
    assert ci["undefined_replicates"] == 30 - ci["valid_replicates"]
    json.dumps(result, allow_nan=False)
