"""DiscoSG's public API contract and conservative source projection, offline."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from mvpg_run.claim_feedback import disco
from mvpg_run.claim_feedback.disco import DiscoSGDecomposer, build_object_claims


def response_record(text="A dog rests beside a chair."):
    return {
        "id": "example-1", "image": "image.jpg", "question": "Describe this image.",
        "response": text, "response_ids": [1, 2, 3], "metadata": {"origin": "test"},
    }


def by_entity(claims):
    return {claim["entity"].casefold(): claim for claim in claims}


def test_objects_have_exact_source_anchors_but_no_visual_truth_labels():
    text = "A RED fire hydrant stands beside a Dog."
    claims = build_object_claims(text, [
        ["fire hydrant", "beside", "dog"], ["dog", "is", "red"],
    ])
    assert set(by_entity(claims)) == {"fire hydrant", "dog"}
    for claim in claims:
        assert claim["kind"] == "object" and claim["parser"] == "discosg"
        assert claim["label"] == "?" and claim["weight"] == 0.0
        assert claim["source_spans"] == claim["error_spans"]
        a, b = claim["source_spans"][0]
        assert text[a:b] == claim["text"]
        assert claim["alignment_status"] == "exact_unique"
    assert by_entity(claims)["dog"]["text"] == "Dog"


def test_attribute_quantity_and_taxonomy_objects_do_not_become_entities():
    claims = build_object_claims("A dog and a chair are present.", [
        ["dog", "is", "red"], ["dog", "is", "2"],
        ["dog", "is a", "animal"], ["chair", "made of", "wood"],
    ])
    assert set(by_entity(claims)) == {"dog", "chair"}


@pytest.mark.parametrize("text,entity,reason", [
    ("A dog rests. Another dog runs.", "dog", "repeated_entity_mention"),
    ("A dog rests.", "canine", "entity_not_in_source"),
    ("It rests.", "it", "non_entity_or_coreference_slot"),
    ("No dog is visible.", "dog", "negated_source_sentence"),
    ("A dog isn't visible.", "dog", "negated_source_sentence"),
    ("A dog isn’t visible.", "dog", "negated_source_sentence"),
    ("There might be a dog.", "dog", "hedged_source_sentence"),
    ("An object looks like a dog.", "dog", "hedged_source_sentence"),
    ("Is there a dog?", "dog", "question_source_sentence"),
    ("Some dogs rest.", "dog", "entity_not_in_source"),
    ("A dog-like statue stands here.", "dog", "entity_not_in_source"),
])
def test_ambiguous_nonasserted_and_nonexact_mentions_abstain(text, entity, reason):
    claim = build_object_claims(text, [[entity, "is", "brown"]])[0]
    assert claim["source_spans"] == claim["error_spans"] == []
    assert claim["alignment_reason"] == reason
    assert claim["alignment_confidence"] == 0.0
    assert claim["ambiguous"] is True and claim["label"] == "?"


def test_casefold_offsets_remain_in_original_text_with_expanding_unicode_fold():
    text = "A Straße is visible."
    claim = build_object_claims(text, [["STRASSE", "is", "wide"]])[0]
    assert claim["source_spans"] == [[2, 8]]
    assert claim["text"] == text[2:8] == "Straße"


def test_negation_is_scoped_conservatively_to_the_source_sentence():
    claims = by_entity(build_object_claims(
        "A dog rests. No chair is present.", [["dog", "beside", "chair"]],
    ))
    assert claims["dog"]["source_spans"] == [[2, 5]]
    assert claims["chair"]["source_spans"] == []
    assert claims["chair"]["alignment_reason"] == "negated_source_sentence"


def test_overlapping_entity_names_do_not_create_competing_training_units():
    claims = build_object_claims("A brown dog rests.", [
        ["brown dog", "is", "large"], ["dog", "is", "brown"],
    ])
    assert len(claims) == 2
    assert all(claim["alignment_reason"] == "overlapping_entity_mentions" for claim in claims)
    assert all(claim["source_spans"] == [] for claim in claims)


def test_triple_order_and_duplicates_do_not_change_claim_ids():
    triples = [["dog", "beside", "chair"], ["dog", "is", "brown"]]
    assert build_object_claims("A dog is beside a chair.", triples) == build_object_claims(
        "A dog is beside a chair.", list(reversed(triples)) + [triples[0]],
    )


@pytest.mark.parametrize("triples", ["dog", [["dog", "is"]], [["dog", "is", None]], [["", "is", "red"]]])
def test_malformed_triples_are_not_silently_interpreted(triples):
    with pytest.raises(ValueError, match="triple"):
        build_object_claims("A dog.", triples)


@pytest.fixture
def paths(tmp_path):
    repo = tmp_path / "DiscoSG"
    for relative in (
        "src/factual_scene_graph/parser/scene_graph_parser_insert_delete.py",
        "src/factual_scene_graph/parser/DualTaskSceneGraphParser.py",
        "src/factual_scene_graph/triple_utils.py", "src/arch/T5ForDualTask.py",
    ):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    checkpoint = tmp_path / "trained-refiner"
    checkpoint.mkdir()
    (checkpoint / "model_info.json").write_text(json.dumps({
        "base_model_name": "local-tokenizer", "use_lora": False,
    }), encoding="utf-8")
    # These are never deserialized; all model calls in these tests are fake.
    (checkpoint / "classifier.pt").touch()
    (checkpoint / "model.safetensors").touch()
    return repo, checkpoint


@pytest.fixture
def official_fakes(monkeypatch):
    calls = {"load": [], "seed_init": [], "seed": [], "refiner_init": [], "refine": [], "merge": []}

    class Seed:
        def __init__(self, checkpoint, **kwargs):
            calls["seed_init"].append((checkpoint, kwargs))

        def parse(self, descriptions, **kwargs):
            calls["seed"].append((descriptions, kwargs))
            return ["( dog , beside , chair )"]

    class Refiner:
        def __init__(self, **kwargs):
            calls["refiner_init"].append(kwargs)

        def parse(self, **kwargs):
            calls["refine"].append(copy.deepcopy(kwargs))
            text = kwargs["descriptions"][0]
            return {"delete": {text: "retained"}, "insert": {text: "inserted"}}

    def merge(**kwargs):
        calls["merge"].append(kwargs)
        return [
            ("dog", "is", "brown"), ("dog", "beside", "chair"),
        ], "merged-" + str(len(calls["merge"]))

    modules = (
        types.SimpleNamespace(SceneGraphParser=Seed),
        types.SimpleNamespace(DualTaskSceneGraphParser=Refiner),
        types.SimpleNamespace(merge_delete_insert_results=merge, extract_triples=lambda graph: []),
    )

    def load(repo):
        calls["load"].append(repo)
        return modules

    monkeypatch.setattr(disco, "_load_official_modules", load)
    monkeypatch.setattr(disco, "_git_revision", lambda repo: (disco.REFERENCE_COMMIT, False))
    return calls, modules


def test_real_public_api_seed_refine_merge_contract_is_used_lazily(paths, official_fakes):
    repo, checkpoint = paths
    calls, _ = official_fakes
    parser = DiscoSGDecomposer(repo, "seed-checkpoint", checkpoint, rounds=3)
    assert calls["load"] == []
    record = response_record()
    before = copy.deepcopy(record)
    result = parser.decompose(record)
    assert record == before
    assert calls["load"] == [repo]
    assert calls["seed_init"] == [("seed-checkpoint", {
        "device": "cpu", "lemmatize": False, "lowercase": True,
    })]
    assert calls["seed"][0][0] == [record["response"]]
    assert calls["seed"][0][1]["sub_sentence"] is True
    assert calls["seed"][0][1]["return_text"] is True
    assert len(calls["refine"]) == len(calls["merge"]) == 3
    assert [call["graph_to_fix"][record["response"]] for call in calls["refine"]] == [
        "( dog , beside , chair )", "merged-1", "merged-2",
    ]
    assert all(call["task"] == "delete_before_insert" and call["do_sample"] is False
               for call in calls["refine"])
    assert calls["merge"][0] == {"insert_res": "inserted", "delete_res": "retained"}
    assert result["response"] == record["response"]
    assert result["response_ids"] == [1, 2, 3]
    assert result["metadata"]["origin"] == "test"
    provenance = result["metadata"]["decomposer"]
    assert provenance["parser"] == "discosg"
    assert provenance["checkout_commit"] == disco.REFERENCE_COMMIT
    assert provenance["external_models"]["refiner_tokenizer"] == "local-tokenizer"
    assert len(result["claims"]) == 2
    assert all(claim["label"] == "?" for claim in result["claims"])
    assert len(result["disco_graph"]["unresolved_facts"]) == 2
    assert result["disco_graph"]["refined_graph"] == "merged-3"
    parser.decompose(record)
    assert calls["load"] == [repo]


def test_missing_checkpoint_classifier_fails_before_loading_models(paths, monkeypatch):
    repo, checkpoint = paths
    (checkpoint / "classifier.pt").unlink()
    monkeypatch.setattr(disco, "_load_official_modules", lambda repo: pytest.fail("loaded a model"))
    with pytest.raises(FileNotFoundError, match="classifier.pt"):
        DiscoSGDecomposer(repo, "seed", checkpoint)


def test_missing_model_weights_has_actionable_failure(paths):
    repo, checkpoint = paths
    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="trained model weights"):
        DiscoSGDecomposer(repo, "seed", checkpoint)


def test_non_checkout_is_not_replaced_with_a_fallback_parser(paths):
    _, checkpoint = paths
    with pytest.raises(FileNotFoundError, match="Incomplete DiscoSG checkout"):
        DiscoSGDecomposer(checkpoint, "seed", checkpoint)


@pytest.mark.parametrize("revision,dirty", [(None, None), ("a" * 40, False), (disco.REFERENCE_COMMIT, True)])
def test_optional_revision_pin_requires_matching_clean_source(paths, monkeypatch, revision, dirty):
    repo, checkpoint = paths
    monkeypatch.setattr(disco, "_git_revision", lambda repo: (revision, dirty))
    with pytest.raises(ValueError, match="required clean revision"):
        DiscoSGDecomposer(repo, "seed", checkpoint, expected_commit=disco.REFERENCE_COMMIT)


def test_exported_parser_does_not_inherit_enclosing_repositories_revision(paths, monkeypatch):
    repo, _ = paths
    monkeypatch.setattr(disco.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(
        returncode=0, stdout=str(repo.parent) + "\n",
    ))
    assert disco._git_revision(repo) == (None, None)


def test_empty_responses_are_auditable_and_do_not_load_models(paths, official_fakes):
    repo, checkpoint = paths
    calls, _ = official_fakes
    result = DiscoSGDecomposer(repo, "seed", checkpoint).decompose(response_record(" \n "))
    assert result["claims"] == []
    assert result["disco_graph"]["skip_reason"] == "empty_response"
    assert calls["load"] == []


def test_redecomposition_removes_stale_visual_feedback_provenance(paths, official_fakes):
    repo, checkpoint = paths
    record = response_record()
    record["metadata"]["verifier"] = {"calibration_id": "old-claims"}
    result = DiscoSGDecomposer(repo, "seed", checkpoint).decompose(record)
    assert "verifier" not in result["metadata"]
    assert record["metadata"]["verifier"] == {"calibration_id": "old-claims"}
    assert result["metadata"]["origin"] == "test"
    assert all(claim["label"] == "?" for claim in result["claims"])


def test_input_truncation_and_graph_cap_fail_explicitly(paths, official_fakes):
    repo, checkpoint = paths
    calls, modules = official_fakes
    parser = DiscoSGDecomposer(repo, "seed", checkpoint, seed_max_input_len=2)
    parser._load()
    parser._seed_parser.tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: [1, 2, 3])
    with pytest.raises(ValueError, match="Silent source truncation"):
        parser.decompose(response_record())
    assert calls["seed"] == []
    parser = DiscoSGDecomposer(repo, "seed", checkpoint, max_triples_num=1)
    modules[2].extract_triples = lambda graph: [["dog", "is", "brown"], ["chair", "is", "red"]]
    with pytest.raises(ValueError, match="silent triple truncation"):
        parser.decompose(response_record())
    assert calls["refine"] == []


def test_released_hardcoded_import_namespace_is_supported_without_model_dependencies(paths):
    repo, _ = paths
    (repo / "src/factual_scene_graph/parser/scene_graph_parser_insert_delete.py").write_text(
        "class SceneGraphParser: pass\n", encoding="utf-8",
    )
    (repo / "src/arch/T5ForDualTask.py").write_text("class T5ForDualTask: pass\n", encoding="utf-8")
    (repo / "src/factual_scene_graph/parser/DualTaskSceneGraphParser.py").write_text(
        "from disco_sg_camera.src.arch.T5ForDualTask import T5ForDualTask\n"
        "class DualTaskSceneGraphParser: pass\n", encoding="utf-8",
    )
    code = (
        "import sys; from pathlib import Path; "
        "from mvpg_run.claim_feedback.disco import _load_official_modules; "
        "modules = _load_official_modules(Path(sys.argv[1])); "
        "assert hasattr(modules[0], 'SceneGraphParser'); "
        "assert hasattr(modules[1], 'DualTaskSceneGraphParser'); "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code, str(repo)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
