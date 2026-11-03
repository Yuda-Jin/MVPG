"""Dev-only calibration, uncertain-label accounting and atomic result publication."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from run import analyze_grounding_calibration as a


def row(cid, label, score, *, split="dev", role="formal", group=None):
    return {"case_id": cid, "record_id": cid, "image_group": group or cid,
            "split": split, "judge_label": label, "judge_error": None,
            "scores": {"global_raw": score}, "analysis_role": role}


def cases():
    return [row("dn", "unsupported", .1), row("dp", "supported", .9),
            row("du", "uncertain", .1),
            row("tn", "unsupported", .2, split="test"),
            row("tp", "supported", .8, split="test"),
            row("tu", "uncertain", .2, split="test")]


def build(rows, **kwargs):
    return a.build_report(rows, min_selected=1, bootstrap_samples=0, **kwargs)


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def test_tied_scores_are_selected_as_a_whole():
    rows = [row("n1", "unsupported", .1), row("p1", "supported", .1),
            row("n2", "unsupported", .2), row("p2", "supported", .8)]
    policy = a.select_negative_f1(rows, "global_raw")
    assert policy["threshold"] == .2
    metrics = a.negative_metrics(rows, "global_raw", policy["threshold"])
    assert metrics["confusion_determinate_only"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert metrics["f1_unsupported"] == pytest.approx(.8)


def test_f1_ties_prefer_precision_then_smaller_all_case_selection():
    rows = [row("n1", "unsupported", .1), row("n2", "unsupported", .2),
            *[row(f"p{i}", "supported", .2) for i in range(3)],
            row("n3", "unsupported", .9),
            *[row(f"x{i}", "supported", .9) for i in range(10)]]
    # F1=.5 at both .1 and .2; .1 has higher precision.
    assert a.select_negative_f1(rows, "global_raw")["threshold"] == .1
    rows = [row("n", "unsupported", .1), row("u", "uncertain", .2), row("p", "supported", .9)]
    # .2 adds only an uncertain case and cannot improve determinate F1.
    assert a.select_negative_f1(rows, "global_raw")["threshold"] == .1


def test_uncertain_is_not_a_false_positive_in_f1():
    rows = cases()
    policy = a.select_negative_f1(rows[:3], "global_raw")
    assert policy["threshold"] == .1
    metrics = a.negative_metrics(rows[:3], "global_raw", .1)
    assert metrics["f1_unsupported"] == 1
    assert metrics["precision_on_decided"] == 1
    assert metrics["confusion_determinate_only"]["fp"] == 0
    assert metrics["conservative_precision"] == .5
    assert metrics["n_selected_uncertain"] == 1
    assert metrics["selected_uncertain_rate"] == .5


def test_test_labels_and_scores_cannot_change_dev_threshold():
    original = cases()
    changed = copy.deepcopy(original)
    for item in changed:
        if item["split"] == "test":
            item["scores"]["global_raw"] = -9
            if item["judge_label"] != "uncertain":
                item["judge_label"] = "supported" if item["judge_label"] == "unsupported" else "unsupported"
    before, after = build(original), build(changed)
    assert before["best_f1_negative"]["selection"] == after["best_f1_negative"]["selection"]
    assert before["best_f1_negative"]["threshold"] == after["best_f1_negative"]["threshold"] == .1
    assert before["high_precision_analysis"]["strict_policy"]["thresholds"] == after["high_precision_analysis"]["strict_policy"]["thresholds"]


def test_old_thresholds_are_not_enabled_by_default():
    report = build(cases())
    assert report["high_precision_analysis"]["current_policy"] is None
    assert report["corpus"]["all"]["n_cases"] == len(cases())
    assert report["best_f1_negative"]["threshold"] == .1


def test_short_id_transport_repairs_are_disclosed_in_results(tmp_path):
    rows = cases()
    rows[0]["judge_id_encoding"] = "request_alias"
    source, output = tmp_path / "joined.jsonl", tmp_path / "run"
    dump(source, rows)
    result = a.analyze_file(source, output, bootstrap_samples=0)
    assert result["judge_id_encoding_counts"] == {"request_alias": 1, "canonical": 5}
    assert "1条声明因长编号复制错误" in (output / "results.md").read_text(encoding="utf-8")


def test_missing_dev_class_disables_f1_threshold_and_test_missing_class_only_disables_metric():
    missing_dev = cases()
    missing_dev[0]["judge_label"] = "supported"
    report = build(missing_dev)
    assert report["best_f1_negative"]["threshold"] is None
    assert report["best_f1_negative"]["selection"]["undefined_reason"] == "missing_unsupported"
    assert report["best_f1_negative"]["test"]["f1_unsupported"] is None
    missing_test = cases()
    missing_test[3]["judge_label"] = "supported"
    report = build(missing_test)
    assert report["best_f1_negative"]["threshold"] == .1
    assert report["best_f1_negative"]["test"]["f1_unsupported"] is None
    assert report["best_f1_negative"]["test"]["f1_undefined_reason"] == "missing_unsupported"


def test_retrospective_cases_are_counted_but_never_selected():
    rows = cases() + [row("rn", "unsupported", -100, role="retrospective"),
                      row("rp", "supported", -100, role="retrospective")]
    report = build(rows)
    assert report["corpus"]["all"]["n_cases"] == 8
    assert report["corpus"]["formal"]["n_cases"] == 6
    assert report["corpus"]["retrospective"]["n_cases"] == 2
    assert report["best_f1_negative"]["threshold"] == .1
    assert report["high_precision_analysis"]["splits"]["dev"]["counts"]["n_cases"] == 3


@pytest.mark.parametrize("change", ["missing_label", "request_error", "missing_score", "bad_score", "missing_role"])
def test_bad_formal_cases_are_not_silently_dropped(change):
    rows = cases()
    if change == "missing_label": rows[0]["judge_label"] = None
    elif change == "request_error": rows[0]["judge_error"] = "HTTP 503"
    elif change == "missing_score": rows[0]["scores"] = {}
    elif change == "bad_score": rows[0]["scores"]["global_raw"] = float("nan")
    else: rows[0].pop("analysis_role")
    with pytest.raises(ValueError):
        build(rows)


def test_image_leakage_and_duplicate_cases_are_rejected():
    rows = cases()
    rows[3]["image_group"] = rows[0]["image_group"]
    with pytest.raises(ValueError, match="crosses dev/test"):
        build(rows)
    rows = cases() + [row("r", "supported", .4, role="retrospective", group="dn")]
    with pytest.raises(ValueError, match="crosses formal and retrospective"):
        build(rows)
    rows = cases() + [copy.deepcopy(cases()[0])]
    with pytest.raises(ValueError, match="Duplicate"):
        build(rows)


def test_zero_selected_test_cases_have_f1_zero_with_both_classes_present():
    report = build(cases())
    metrics = report["best_f1_negative"]["test"]
    assert metrics["n_selected"] == 0
    assert metrics["precision_on_decided"] is None
    assert metrics["f1_unsupported"] == 0


def test_invalid_formal_input_leaves_existing_progress_results_unchanged(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "results.json").write_text('{"status":"running"}\n')
    (output / "results.md").write_text("Running\n")
    source = tmp_path / "joined.jsonl"
    rows = cases(); rows[0]["judge_label"] = None
    dump(source, rows)
    with pytest.raises(ValueError):
        a.analyze_file(source, output, bootstrap_samples=0)
    assert (output / "results.json").read_text() == '{"status":"running"}\n'
    assert (output / "results.md").read_text() == "Running\n"
    assert not (output / "details/analysis_report.json").exists()


def test_cli_publishes_clean_three_entry_layout_and_explicit_retrospective_errors(tmp_path):
    source, output = tmp_path / "joined.jsonl", tmp_path / "run"
    rows = cases() + [row("r", None, .3, role="retrospective")]
    rows[-1]["judge_error"] = "HTTP 503"
    dump(source, rows)
    assert a.main(["--input", str(source), "--output-dir", str(output), "--bootstrap-samples", "0"]) == 1
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "formal_complete_retrospective_incomplete"
    assert result["corpus"]["retrospective"]["unlabeled_or_invalid"] == 1
    assert sorted(p.name for p in output.iterdir()) == ["details", "results.json", "results.md"]
    assert "银标" in (output / "results.md").read_text(encoding="utf-8")
    assert result["plots"]["status"] == "not_requested"
