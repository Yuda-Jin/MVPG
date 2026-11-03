"""Frozen sample boundaries, lossless joins and bounded judge recovery."""
import copy
import json
from pathlib import Path

import pytest

from run import run_coco_grounding_calibration as pipeline


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pipeline.rows_bytes(rows))


def sample(tmp_path):
    rows = []
    for i, split in enumerate(("dev", "test")):
        path = tmp_path / f"image{i}.jpg"
        path.write_bytes(f"image-{i}".encode())
        rows.append(dict(id=f"coco-{i}", image=str(path), question="Describe", response="", claims=[],
                         metadata=dict(image_group="sha256:" + pipeline.sha(path), split=split, analysis_role="formal")))
    return rows


def cases():
    return [dict(case_id=f"c{i}", record_id=f"r{i}", claim_id="noun-0", image=f"image{i}.jpg",
                 image_group=f"g{i}", split=split, analysis_role="formal", question="Describe", response="A car.",
                 claim_text="car", source_spans=[[2, 5]], kind="object", judge_label=None, scores={})
            for i, split in enumerate(("dev", "test"))]


def test_sample_hash_and_formal_boundary(tmp_path):
    rows = sample(tmp_path)
    assert pipeline.validate_input(rows, 2)["split_images"] == {"dev": 1, "test": 1}
    rows[0]["metadata"]["analysis_role"] = "retrospective"
    with pytest.raises(ValueError, match="formal"):
        pipeline.validate_input(rows, 2)
    rows[0]["metadata"]["analysis_role"] = "formal"
    Path(rows[0]["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        pipeline.validate_input(rows, 2)


def test_input_cannot_already_contain_generated_feedback(tmp_path):
    rows = sample(tmp_path)
    rows[0]["claims"] = [{"text": "car"}]
    with pytest.raises(ValueError, match="precede generation"):
        pipeline.validate_input(rows, 2)
    with pytest.raises(ValueError, match="exactly"):
        pipeline.validate_input(rows, 1000)


def test_initialize_resume_rejects_changed_options_and_input(tmp_path):
    path = tmp_path / "sample.jsonl"
    write_rows(path, sample(tmp_path))
    parser = pipeline.build_parser()
    args = parser.parse_args(["--run", str(tmp_path / "run"), "--input", str(path),
                              "--expected-images", "2", "--judge-model", "test-judge"])
    root, plan = pipeline.initialize(args)
    assert plan["judge_id_encoding"] == "request_alias"
    resumed, saved = pipeline.initialize(parser.parse_args(["--run", str(root)]))
    assert resumed == root and saved == plan
    with pytest.raises(ValueError, match="frozen option"):
        pipeline.initialize(parser.parse_args(["--run", str(root), "--seed", "99"]))
    (root / "details/inputs.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="input changed"):
        pipeline.initialize(parser.parse_args(["--run", str(root)]))


def test_environment_values_are_literal_and_existing_values_win(tmp_path, monkeypatch):
    path = tmp_path / "settings.env"
    path.write_text("# comment\nexport COCO_RUNNER_TEST_A='$(do-not-execute)' # note\nCOCO_RUNNER_TEST_B=keep # note\n")
    monkeypatch.delenv("COCO_RUNNER_TEST_A", raising=False)
    monkeypatch.setenv("COCO_RUNNER_TEST_B", "already")
    pipeline.load_environment(path)
    import os
    assert os.environ["COCO_RUNNER_TEST_A"] == "$(do-not-execute)"
    assert os.environ["COCO_RUNNER_TEST_B"] == "already"
    monkeypatch.delenv("COCO_RUNNER_TEST_A")


def prepare_join(tmp_path):
    source = cases()
    judged = [dict(row, judge_label=label, judge={"label": label}, judge_error=None)
              for row, label in zip(source, ("supported", "unsupported"))]
    scored = [dict(row, scores={"dino_max": .8 - i * .6},
                   score_provenance={"dino_max": {"backend": "test"}}, detection_evidence={"box": [0, 0, 4, 4]})
              for i, row in enumerate(source)]
    details = tmp_path / "details"
    for name, rows in (("cases", source), ("judged", judged), ("scored", scored)):
        write_rows(details / f"{name}.jsonl", rows)
    return source, judged, scored


def test_join_keeps_region_evidence_and_independent_judge(tmp_path):
    _, judged, _ = prepare_join(tmp_path)
    audit = pipeline.join_scored(tmp_path)
    joined = pipeline.read_rows(tmp_path / "details/joined.jsonl")
    assert audit["cases"] == 2
    assert [r["judge_label"] for r in joined] == [r["judge_label"] for r in judged]
    assert joined[0]["detection_evidence"]["box"] == [0, 0, 4, 4]
    assert joined[1]["scores"]["dino_max"] == pytest.approx(.2)


@pytest.mark.parametrize("failure", ["missing_judge", "failed_judge", "changed_source", "missing_score"])
def test_join_refuses_silent_case_filtering_or_mutation(tmp_path, failure):
    source, judged, scored = prepare_join(tmp_path)
    if failure == "missing_judge":
        judged.pop()
    elif failure == "failed_judge":
        judged[0].update(judge_label=None, judge_error="TimeoutError")
    elif failure == "changed_source":
        judged[0]["response"] = "A dog."
    else:
        scored.pop()
    write_rows(tmp_path / "details/judged.jsonl", judged)
    write_rows(tmp_path / "details/scored.jsonl", scored)
    with pytest.raises(ValueError):
        pipeline.join_scored(tmp_path)
    assert not (tmp_path / "details/joined.jsonl").exists()


class JudgeRunner:
    def __init__(self, details, *, succeeds=True, fatal=False):
        self.details = details
        self.plan = dict(pipeline.DEFAULTS, judge_model="test")
        self.calls, self.state = [], {"stages": {}}
        self.succeeds, self.fatal = succeeds, fatal

    def command(self, name, args):
        self.calls.append((name, [str(x) for x in args]))
        self.state["stages"][name] = {}
        return name

    def await_jobs(self, jobs):
        complete = len(self.calls) > 1 and self.succeeds
        output = self.details / "judged.jsonl"
        Path(str(output) + ".manifest.json").write_text("{}")
        Path(str(output) + ".status.json").write_text(json.dumps(
            dict(status="complete" if complete else "incomplete", cases_total=2, cases_judged=2 if complete else 0)))
        write_rows(str(output) + ".images.progress.jsonl", [dict(image_key="image", judge_fatal_error=self.fatal)])
        if not complete:
            raise RuntimeError("incomplete")

    def save(self):
        pass


def test_judge_retry_uses_resume_and_short_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    runner = JudgeRunner(tmp_path)
    pipeline.run_judge(runner)
    assert len(runner.calls) == 2
    assert "--resume" not in runner.calls[0][1]
    assert "--resume" in runner.calls[1][1]
    assert all(args[args.index("--id-encoding") + 1] == "request_alias" for _, args in runner.calls)
    assert runner.state["stages"]["judge"]["status"] == "complete"


@pytest.mark.parametrize("fatal, expected_calls", [(False, 3), (True, 1)])
def test_judge_failure_is_bounded_and_does_not_succeed(tmp_path, monkeypatch, fatal, expected_calls):
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    runner = JudgeRunner(tmp_path, succeeds=False, fatal=fatal)
    with pytest.raises(RuntimeError, match="fatal error" if fatal else "exhausted"):
        pipeline.run_judge(runner)
    assert len(runner.calls) == expected_calls


def test_full_automatic_stage_wiring_keeps_same_cases_for_baseline(tmp_path, monkeypatch):
    """Exercise orchestration/publication with deterministic stand-ins for model jobs."""
    import io
    from run import analyze_grounding_calibration as analysis
    rows = sample(tmp_path)
    source = tmp_path / "sample.jsonl"
    write_rows(source, rows)
    fgclip = tmp_path / "fgclip"
    fgclip.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.model"):
        (fgclip / name).write_bytes(name.encode())
    args = pipeline.build_parser().parse_args(["--run", str(tmp_path / "run"), "--input", str(source),
        "--expected-images", "2", "--judge-model", "test", "--fgclip-root", str(fgclip)])
    root, plan = pipeline.initialize(args)
    order = []
    frozen_cases = cases()

    class Done:
        pid = 1
        def poll(self):
            return 0

    def corpus_stage(self, name, arguments):
        order.append(name)

    def prepare(root, plan):
        write_rows(root / "details/cases.jsonl", frozen_cases)
        return {"cases": 2}

    def command(self, name, arguments, *, python=None):
        order.append(name)
        arguments = [str(x) for x in arguments]
        self.state["stages"][name] = {}
        self.state["stage"] = name
        if name == "score":
            assert python == plan["score_python"]
            scored = [dict(row, scores={"dino_max": value}, score_provenance={"dino_max": {"backend": "test"}})
                      for row, value in zip(frozen_cases, (.8, .2))]
            write_rows(self.details / "scored.jsonl", scored)
        elif name == "judge":
            judged = [dict(row, judge_label=label, judge={"label": label}, judge_error=None)
                      for row, label in zip(frozen_cases, ("supported", "unsupported"))]
            write_rows(self.details / "judged.jsonl", judged)
            pipeline.write_json(str(self.details / "judged.jsonl") + ".status.json",
                                {"status": "complete", "cases_judged": 2, "cases_total": 2})
        elif name == "fgclip_score":
            path = Path(arguments[arguments.index("--input") + 1])
            scored = [dict(row, scores={"global_raw": value}, score_provenance={"global_raw": {
                "root": str(fgclip.resolve()), "score_mode": "global", "text_form": "raw", "walk_type": "short",
                "input_sha256": pipeline.sha(path), "weights_sha256": pipeline.sha(fgclip / "model.safetensors"),
                "config_sha256": pipeline.sha(fgclip / "config.json"), "tokenizer_sha256": pipeline.sha(fgclip / "tokenizer.model")}})
                for row, value in zip(frozen_cases, (.07, .01))]
            write_rows(arguments[arguments.index("--output") + 1], scored)
        elif name in ("analysis", "fgclip_analysis"):
            analysis.analyze_file(arguments[arguments.index("--input") + 1],
                arguments[arguments.index("--output-dir") + 1], bootstrap_samples=0,
                score_key=arguments[arguments.index("--score-key") + 1])
        else:
            raise AssertionError(name)
        return Done(), io.StringIO()

    monkeypatch.setattr(pipeline.CocoRunner, "corpus_stage", corpus_stage)
    monkeypatch.setattr(pipeline.CocoRunner, "command", command)
    monkeypatch.setattr(pipeline, "prepare", prepare)
    result = pipeline.run(root, plan)
    assert result["status"] == "complete"
    assert order == ["generation", "decomposition", "score", "judge", "analysis", "fgclip_score", "fgclip_analysis"]
    main = pipeline.read_json(root / "results.json")
    baseline = pipeline.read_json(root / "details/fgclip_baseline/results.json")
    assert main["score_key"] == "dino_max" and baseline["score_key"] == "global_raw"
    assert main["baseline"]["same_cases_and_judge"] is True
    joined = pipeline.read_rows(root / "details/joined.jsonl")
    base_joined = pipeline.read_rows(root / "details/fgclip_joined.jsonl")
    assert [(r["case_id"], r["split"], r["judge_label"]) for r in joined] == [
        (r["case_id"], r["split"], r["judge_label"]) for r in base_joined]
