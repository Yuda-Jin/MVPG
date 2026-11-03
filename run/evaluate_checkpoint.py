"""Evaluate one adapter and publish results.md/results.json beside details/.

The retained shell runner owns generation, scoring and stage recovery. This
entry point supplies its configuration and checks the final artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit


REPO = Path(__file__).resolve().parents[1]
EXPECTED = {"mmhal": 96, "qa": 90, "pope_random": 3000,
            "pope_popular": 3000, "pope_adversarial": 3000, "amber": 15220, "object": 300}
GROUPS = {
    "mmhal": ["mmhal_generate", "mmhal_score"],
    "qa": ["llava_generate", "llava_review", "llava_score"],
    **{f"pope_{c}": [f"pope_{c}_generate", f"pope_{c}_score"] for c in ("random", "popular", "adversarial")},
    "amber": ["amber_generate", "amber_score"],
    "object": ["object_generate", "object_score_local", "object_score_gpt"],
}
STAGES = {s for group in GROUPS.values() for s in group}
PAPER_COLUMNS = [("mmhal_avg_score", "MMHal Score↑", 1),
                 ("mmhal_hallucination_rate", "MMHal Hal↓ (%)", 100),
                 ("amber_chair", "AMBER CHAIR↓ (%)", 1), ("amber_cover", "AMBER Cover↑ (%)", 1),
                 ("amber_hal", "AMBER Hal↓ (%)", 1), ("amber_cog", "AMBER Cog↓ (%)", 1),
                 ("object_hal_chairs", "ObjectHal CHAIRs↓ (%)", 1),
                 ("object_hal_chairi", "ObjectHal CHAIRi↓ (%)", 1),
                 ("pope_adv_accuracy", "POPE adversarial Accuracy↑ (%)", 100),
                 ("pope_adv_precision", "POPE adversarial Precision↑ (%)", 100)]
ENV_KEYS = ("MODEL_BASE", "EVAL_DATA_ROOT", "GPT_MODEL", "OPENAI_ENDPOINT", "OPENAI_BASE_URL", "JUDGE_THINKING")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def require(value, message):
    if not value:
        raise ValueError(message)


def checkpoint_info(path, model_base=None):
    checkpoint = Path(path).expanduser().resolve()
    require(checkpoint.is_dir(), f"Checkpoint directory does not exist: {checkpoint}")
    weights = next((checkpoint / n for n in ("adapter_model.safetensors", "adapter_model.bin")
                    if (checkpoint / n).is_file()), None)
    require(weights and (checkpoint / "adapter_config.json").is_file(), "Checkpoint needs adapter weights and adapter_config.json")
    state_path = checkpoint / "trainer_state.json"
    if state_path.is_file():
        state = read(state_path)
        inferred = (state.get("signature") or {}).get("model_name")
    elif (checkpoint.parent / "manifest.json").is_file():
        require(checkpoint.name == "final", "Legacy intermediate checkpoints have no verified update count; use legacy final/ or a checkpoint with trainer_state.json")
        state_path = checkpoint.parent / "manifest.json"
        state = read(state_path)
        inferred = (state.get("backend") or {}).get("model_name")
    else:
        state, inferred, state_path = {}, None, None
    base = model_base or inferred
    require(base, "Cannot infer the base model; provide --model-base")
    step = (state.get("counts") or {}).get("updated")
    require(step is None or (isinstance(step, int) and not isinstance(step, bool) and step >= 0), "Invalid actual checkpoint update count")
    return {"checkpoint": str(checkpoint), "model_base": str(Path(base).expanduser().resolve()),
            "step": step, "adapter_sha256": {"adapter_config.json": sha(checkpoint / "adapter_config.json"),
                                               weights.name: sha(weights)},
            "training_manifest": str(state_path) if state_path else None}


def env_settings(env_file):
    # Bash interprets the existing trusted .env. Only this nonsecret whitelist
    # and a key-presence boolean are captured; the environment is never dumped.
    code = "import os,json;print(json.dumps({k:os.getenv(k) for k in " + repr(ENV_KEYS) + "} | {'key_present':bool(os.getenv('OPENAI_API_KEY'))}))"
    command = 'set -a; if [[ -f "$1" ]]; then source "$1"; fi; "$2" -c "$3"'
    result = subprocess.run(["bash", "-c", command, "evaluation-config", str(env_file), sys.executable, code],
                            cwd=REPO, capture_output=True, text=True)
    require(result.returncode == 0, "Could not read evaluation configuration from .env")
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError("The .env file must contain assignments without console output") from exc


def sources(data_root):
    base = REPO / "eval_llava_rlhf_coco"
    return {"mmhal": data_root / "mmhal/response_template.json", "qa": base / "llava/qa90_questions.jsonl",
            "amber": base / "AMBER_data/query/query_all.json", "object": base / "object_hal/obj_halbench_300_with_image.jsonl",
            **{f"pope_{c}": base / f"pope/coco_pope_{c}.jsonl" for c in ("random", "popular", "adversarial")}}


def make_identity(info, settings, data_root, batch_size):
    endpoint = settings.get("OPENAI_ENDPOINT") or settings.get("OPENAI_BASE_URL") or None
    if endpoint:
        uri = urlsplit(endpoint)
        require(not (uri.username or uri.password or uri.query or uri.fragment), "Judge endpoint must not contain credentials or query parameters")
    paths = sources(data_root)
    require(all(p.is_file() for p in paths.values()), "Benchmark question files are missing; check --data-root")
    return dict(info, data_root=str(data_root), batch_size=batch_size,
                judge={"model": settings.get("GPT_MODEL") or "gpt-4", "endpoint": endpoint,
                       "thinking": settings.get("JUDGE_THINKING") or None},
                sources_sha256={str(p.resolve()): sha(p) for p in paths.values()})


def judge_options(identity):
    judge = identity["judge"]
    if judge["model"].lower().startswith("deepseek-") and judge["thinking"]:
        require(judge["thinking"] == "disabled", "DeepSeek thinking must be unset or disabled")
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def qa_fingerprint(answer):
    base = REPO / "eval_llava_rlhf_coco"
    paths = [base / "llava/qa90_questions.jsonl", base / "table/caps_boxes_coco2014_val_80.jsonl",
             base / "llava/qa90_gpt4_answer.jsonl", answer, base / "table/rule.json"]
    digest = hashlib.sha256()
    for path in paths:
        data = path.read_bytes(); digest.update(len(data).to_bytes(8, "big")); digest.update(data)
    return digest.hexdigest()


def name_for(identity):
    weight_sha = next(v for k, v in identity["adapter_sha256"].items() if k != "adapter_config.json")
    model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", Path(identity["model_base"]).name)
    return f"{model}_step{identity['step'] if identity['step'] is not None else 'unknown'}_{weight_sha[:8]}"


def default_output(name):
    root = Path("/root/autodl-tmp/output") if Path("/root/autodl-tmp").is_dir() else REPO / "output"
    return root / "benchmark_results" / name


def check_output(root):
    if root.exists():
        require(not any(root.iterdir()) or (root / "details/run.json").is_file(),
                "Output directory is not an evaluation owned by this entry point; choose a new directory")
        foreign = {p.name for p in root.iterdir()} - {"results.md", "results.json", "details"}
        require(not foreign, "Output directory contains unrelated files; choose a new --output-dir")
        if (root / "details").exists():
            require((root / "details/run.json").is_file(), "Existing details/ is not owned by this entry point; choose a new output directory")


@contextmanager
def run_lock(details):
    # Locks are released by the OS after a crash. Leave the lock inode in place
    # so concurrent waiters cannot lock two different files after an unlink.
    with (details / ".run.lock").open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0"); handle.flush()
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another evaluation is using this output directory") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def import_run(source, output, info, args):
    source = source.resolve()
    require(source.is_dir(), "--import-existing must be an evaluation directory")
    require(source != output and source not in output.parents and output not in source.parents,
            "Import source and output must be separate directory trees")
    summaries = list(source.glob("summary_*.json"))
    require(len(summaries) == 1, "Import requires exactly one summary_*.json; choose one evaluation run")
    suffix = summaries[0].stem.removeprefix("summary_")
    manifest = read(source / "logs" / suffix / "manifest.json")
    require(str(Path(manifest["adapter"]).resolve()) == info["checkpoint"], "Imported evaluation belongs to another checkpoint")
    require(manifest["adapter_sha256"] == info["adapter_sha256"], "Imported adapter hashes differ")
    require(str(Path(manifest["model"]).resolve()) == info["model_base"], "Imported base model differs")
    require(args.name is None or args.name == suffix, "Imported run name is fixed by its original files")
    batch = int(manifest["batch_size"])
    require(args.batch_size is None or args.batch_size == batch, "Imported batch size differs")
    mmhal = next(Path(p) for p in manifest["sources_sha256"] if Path(p).name == "response_template.json")
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else mmhal.parent.parent
    settings = {"GPT_MODEL": manifest["judge_model"], "OPENAI_ENDPOINT": manifest.get("judge_endpoint"),
                "JUDGE_THINKING": manifest.get("judge_thinking")}
    identity = make_identity(info, settings, data_root, batch)
    require(identity["sources_sha256"] == manifest["sources_sha256"], "Imported question sources differ from current files")
    metadata = {"schema_version": 1, "name": suffix, "identity": identity, "imported_from": str(source)}
    details = output / "details"
    if details.exists():
        require(read(details / "run.json") == metadata, "Existing output has another checkpoint, protocol or import source")
        return metadata
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".import-", dir=output))
    try:
        require(not any(p.is_symlink() for p in source.rglob("*")), "Import source contains symbolic links; use the original raw output directory")
        shutil.copytree(source, temporary, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".env", ".env.*", ".ipynb_checkpoints", "__pycache__", "*.pyc"))
        write(temporary / "run.json", json.dumps(metadata, indent=2) + "\n")
        temporary.rename(details)
    finally:
        if temporary.exists():
            require(temporary.resolve().parent == output and temporary.name.startswith(".import-"), "Unsafe temporary cleanup path")
            shutil.rmtree(temporary)
    return metadata


def engine_environment(metadata, details):
    identity = metadata["identity"]
    data = Path(identity["data_root"])
    env = {"MODEL_BASE": identity["model_base"], "MODEL_LORA_BASE": identity["checkpoint"],
           "MODEL_SUFFIX": metadata["name"], "EVAL_ROOT": str(details), "PYTHON_BIN": sys.executable,
           "IMAGE_FOLDER_LB": str(data / "coco/train2017"), "IMAGE_FOLDER_POPE": str(data / "coco/val2014"),
           "IMAGE_DIR_AMBER": str(data / "AMBER/image"), "MMHAL_DATA_ROOT": str(data / "mmhal"),
           "ANNOTATION_FILE": str(data / "coco/annotations"), "POPE_CAT": "all", "RUN_GPT": "1",
           "EVAL_BATCH_SIZE": str(identity["batch_size"]), "GPT_MODEL": identity["judge"]["model"],
           "OPENAI_ENDPOINT": identity["judge"]["endpoint"] or "", "OPENAI_BASE_URL": "",
           "JUDGE_THINKING": identity["judge"]["thinking"] or "", "ENV_FILE": "/dev/null",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "8",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    if identity["training_manifest"]:
        env["TRAINING_MANIFEST"] = identity["training_manifest"]
    return env


def run_engine(metadata, details, env_file):
    env = engine_environment(metadata, details)
    command = 'set -a; if [[ -f "$1" ]]; then source "$1"; fi; shift; exec env "$@"'
    args = ["bash", "-c", command, "checkpoint-evaluation", str(env_file),
            *[f"{k}={v}" for k, v in env.items()], "bash", str(REPO / "run/eval_all_metrics.sh")]
    with (details / "runner.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end="", flush=True)
        return process.wait()


def publish(output, metadata):
    sys.path.insert(0, str(REPO))
    from run.summarize_eval_results import collect_stages, validate_artifact, validate_score
    details, suffix, identity = output / "details", metadata["name"], metadata["identity"]
    logs = details / "logs" / suffix
    issues, metrics, stages = [], {}, {}
    try:
        metrics, stages = collect_stages(logs, logs / "stages.tsv")
    except (OSError, ValueError, KeyError) as exc:
        issues.append(f"Stage summary unavailable: {type(exc).__name__}")
    protocol_valid = False
    try:
        manifest = read(logs / "manifest.json")
        expected = {"model": identity["model_base"], "adapter": identity["checkpoint"],
                    "adapter_sha256": identity["adapter_sha256"], "judge_model": identity["judge"]["model"],
                    "judge_endpoint": identity["judge"]["endpoint"], "judge_thinking": identity["judge"]["thinking"],
                    "batch_size": str(identity["batch_size"]), "sources_sha256": identity["sources_sha256"]}
        require(manifest == expected, "Saved evaluation manifest differs from checkpoint or protocol")
        require(all(sha(Path(identity["checkpoint"]) / n) == h for n, h in identity["adapter_sha256"].items()), "Adapter changed")
        require(all(sha(p) == h for p, h in identity["sources_sha256"].items()), "Question sources changed")
        protocol_valid = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(f"Protocol: {exc}")
    valid_groups = set()
    paths = sources(Path(identity["data_root"]))
    answer_paths = {"mmhal": details / f"mmhal_bench/answer/answer-file-{suffix}.jsonl",
                    "qa": details / f"llava_bench/answer/answer-file-{suffix}.jsonl",
                    "amber": details / f"AMBER_bench/answer/answer-file-{suffix}-all.jsonl",
                    "object": details / f"objhal_bench/answer/answer-file-{suffix}.jsonl",
                    **{f"pope_{c}": details / f"pop_bench/answer/answer-file-{suffix}_{c}.jsonl"
                       for c in ("random", "popular", "adversarial")}}
    counts = {}
    for group, required in GROUPS.items():
        try:
            require(all(stages.get(s, {}).get("status") == "completed" and str(stages[s].get("exit_code")) == "0"
                        for s in required), "stages are incomplete")
            for stage in required:
                if not stage.endswith("_generate"):
                    require(validate_score(stage, logs / f"{stage}.log"), "score log is invalid")
            counts[group] = validate_artifact("pope" if group.startswith("pope_") else group,
                                               answer_paths[group], paths[group])
            if group == "qa":
                review = details / f"llava_bench/review/review-file-{suffix}.jsonl"
                validate_artifact("qa_review", review, paths[group])
                require(read(str(review) + ".meta.json") == {"inputs_sha256": qa_fingerprint(answer_paths[group]),
                        "judge_model": identity["judge"]["model"], "max_tokens": 1024,
                        "schema": "llava_bench_reviews_v1", "judge_request_options": judge_options(identity)},
                        "qa90 review inputs or judge differ")
            elif group == "mmhal":
                review = details / f"mmhal_bench/review/review-file-{suffix}.jsonl"
                scores = []
                for text in read(review):
                    values = {int(x) for x in re.findall(r"\brating\s*:\s*([0-6])\b", text, re.I)}
                    require(len(values) == 1, "ambiguous MMHal rating")
                    scores.append(values.pop())
                require(len(scores) == 96, "MMHal needs96ratings")
                meta = read(str(review) + ".meta.json")
                require(meta == {"response_sha256": sha(answer_paths[group]), "judge_model": identity["judge"]["model"],
                        "schema": "mmhal_ordered_reviews_v1", "judge_request_options": judge_options(identity)},
                        "MMHal review metadata differs")
                metrics.update(mmhal_avg_score=sum(scores) / 96, mmhal_hallucination_rate=sum(x < 3 for x in scores) / 96)
            elif group == "object":
                answers = [json.loads(x) for x in answer_paths[group].read_text(encoding="utf-8").splitlines() if x.strip()]
                for mode in ("local", "gpt"):
                    review = read(details / f"objhal_bench/review-{mode}/review-file-{suffix}.jsonl")
                    require(len(review["sentences"]) == 300 and review["overall_metrics"]["sentence_num"] == 300,
                            "ObjectHal needs300reviewed answers")
                    require(all(s["image_id"] == int(a["image_id"]) and s["caption"] == a["text"].replace("Assistant:", "").strip()
                                for s, a in zip(review["sentences"], answers)), "ObjectHal reviews belong to other answers")
            valid_groups.add(group)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append(f"{group}: {exc}")
    object_mode = metrics.get("object_hal_scoring")
    if not protocol_valid:
        valid_groups.clear()
    for key in list(metrics):
        group = ("qa" if key.startswith("llava_") else "object" if key.startswith("object_") else
                 "pope_adversarial" if key.startswith("pope_adv_") else
                 next((g for g in GROUPS if key.startswith(g + "_")), None))
        if group not in valid_groups or isinstance(metrics[key], bool) or not isinstance(metrics[key], (int, float)):
            del metrics[key]
    metrics = {key: metrics[key] for key, _, _ in PAPER_COLUMNS if key in metrics}
    complete = not issues and set(stages) == STAGES and counts == EXPECTED
    result = {"status": "complete" if complete else "incomplete", "model": Path(identity["model_base"]).name,
              "checkpoint": identity["checkpoint"], "step": identity["step"], "judge": identity["judge"],
              "batch_size": identity["batch_size"], "counts": counts,
              "completed_stages": sum(s.get("status") == "completed" for s in stages.values()), "total_stages": 16,
              "metrics": metrics, "object_scoring": object_mode,
              "details": "details/", "run_id": suffix, "issues": issues}
    write(output / "results.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# 检查点评测结果", "", f"状态：{'完整通过' if complete else '未完成，请查看 details/ 中的日志'}；阶段：{result['completed_stages']}/16。",
             f"模型：{result['model']}；实际训练更新：{identity['step']}；裁判：{identity['judge']['model']}。", "",
             f"ObjectHal对象提取：{'裁判 ' + identity['judge']['model'] if object_mode == 'gpt' else object_mode or '未完成'}。仅汇报参考论文Table 2的10项指标。", "",
             "| 指标 | 数值 |", "|---|---:|"]
    for key, label, scale in PAPER_COLUMNS:
        lines.append(f"| {label} | {metrics[key] * scale:.2f} |" if key in metrics else f"| {label} | - |")
    lines += ["", "原始回答、裁判记录、配置及断点状态统一保存在 details/。重复同一命令可继续未完成阶段。",
              "裁判替代GPT-4时，相关分数不能直接作为原论文GPT-4成绩比较。", ""]
    write(output / "results.md", "\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-base")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path, default=REPO / ".env")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--name")
    parser.add_argument("--import-existing", type=Path, help="Copy and summarize an existing evaluation without model or judge calls")
    parser.add_argument("--refresh-only", action="store_true", help="Refresh reports from existing details without running evaluation")
    args = parser.parse_args(argv)
    try:
        require(args.batch_size is None or args.batch_size > 0, "Batch size must be positive")
        require(args.name is None or re.fullmatch(r"[a-zA-Z0-9_.-]+", args.name), "Invalid run name")
        info = checkpoint_info(args.checkpoint, args.model_base)
        output = (args.output_dir or default_output(args.name or name_for(info))).expanduser().resolve()
        check_output(output)
        previous = read(output / "details/run.json") if (output / "details/run.json").is_file() else None
        if args.import_existing:
            metadata = import_run(args.import_existing, output, info, args)
        elif args.refresh_only:
            require((output / "details/run.json").is_file(), "No evaluation exists to refresh")
            metadata = read(output / "details/run.json")
            require(all(metadata["identity"][k] == v for k, v in info.items()), "Output belongs to another checkpoint")
            require(args.batch_size is None or args.batch_size == metadata["identity"]["batch_size"], "Batch size differs")
        else:
            settings = env_settings(args.env_file.resolve())
            require(settings.get("key_present"), "Configure OPENAI_API_KEY in .env before running the full evaluation")
            stored = previous["identity"] if previous else {}
            data = Path(args.data_root or settings.get("EVAL_DATA_ROOT") or stored.get("data_root") or "/root/autodl-tmp/data/eval").expanduser().resolve()
            identity = make_identity(info, settings, data, args.batch_size or stored.get("batch_size") or 16)
            require(Path(info["model_base"]).is_dir(), "Base model directory is missing; check --model-base")
            require(all((data / p).is_dir() for p in ("coco/train2017", "coco/val2014", "AMBER/image", "coco/annotations")),
                    "Benchmark image/annotation directories are missing; check --data-root")
            metadata = {"schema_version": 1, "name": args.name or (previous or {}).get("name") or name_for(identity), "identity": identity}
        details = output / "details"
        details.mkdir(parents=True, exist_ok=True)
        with run_lock(details):
            path = details / "run.json"
            if path.exists():
                old = read(path)
                require(old["identity"] == metadata["identity"] and old["name"] == metadata["name"],
                        "Checkpoint or evaluation protocol changed; choose a new --output-dir")
                metadata = old
            else:
                write(path, json.dumps(metadata, indent=2) + "\n")
            code = 0
            if not args.import_existing and not args.refresh_only:
                print(f"Evaluating {info['checkpoint']}\nResults: {output}", flush=True)
                code = run_engine(metadata, details, args.env_file.resolve())
            result = publish(output, metadata)
            print(f"{result['status']}: {output / 'results.md'}")
            return code if code != 0 else 0 if result["status"] == "complete" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evaluation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
