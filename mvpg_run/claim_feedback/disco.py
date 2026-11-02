"""Optional, source-preserving bridge to the official DiscoSG implementation.

The released parser produces scene-graph strings, not source spans or visual
truth labels.  This first implementation projects only graph entities to
object-existence units.  A unit receives an anchor only when its complete name
occurs exactly once in an affirmative, unhedged source sentence.  Attributes,
relations, unresolved coreference and ambiguous mentions remain untrained.

No model framework is imported until ``decompose`` needs the external parser.
Use a separate environment when DiscoSG and the LLaVA training installation
need different framework versions.  The caller supplies the official checkout
and trained refiner weights; this module never substitutes a sentence splitter
for DiscoSG.
"""

from __future__ import annotations

import copy
import importlib
import importlib.machinery
import json
from pathlib import Path
import re
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from .data import validate_record


REFERENCE_COMMIT = "d2be2768ddc80199ab56f38cf50c132351a8d248"
OFFICIAL_REPOSITORY = "https://github.com/ShaoqLin/DiscoSG"
_PARSER_MODULE = "src.factual_scene_graph.parser.scene_graph_parser_insert_delete"
_REFINER_MODULE = "src.factual_scene_graph.parser.DualTaskSceneGraphParser"
_TRIPLE_MODULE = "src.factual_scene_graph.triple_utils"

# In the official graph representation, ``(entity, is, value)`` represents
# attributes or quantities, not a second entity (triple_utils.py).  Other
# property/taxonomy aliases are excluded conservatively as well.
_ATTRIBUTE_RELATIONS = frozenset({
    "is", "are", "was", "were", "be", "is a", "is an", "type of",
    "attribute", "has attribute", "has color", "has colour", "color",
    "colour", "has size", "size", "has quantity", "quantity", "count",
    "number", "has number", "is made of", "made of", "material",
})
_NON_ENTITIES = frozenset({
    "i", "me", "my", "we", "us", "you", "he", "him", "she", "her",
    "it", "its", "they", "them", "their", "there", "this", "that",
    "these", "those", "who", "which", "someone", "somebody", "anyone",
    "anybody", "something", "anything", "everything", "nothing", "none",
    "one", "another", "unknown", "n/a",
})
_NEGATION = re.compile(
    r"\b(?:no|not|never|neither|nor|without|absent|missing|"
    r"lack(?:s|ed|ing)?|cannot|can['\u2019]t|isn['\u2019]t|aren['\u2019]t|"
    r"wasn['\u2019]t|weren['\u2019]t|doesn['\u2019]t|don['\u2019]t|didn['\u2019]t|"
    r"hasn['\u2019]t|haven['\u2019]t|won['\u2019]t|wouldn['\u2019]t|"
    r"shouldn['\u2019]t|couldn['\u2019]t)\b", re.IGNORECASE,
)
_HEDGING = re.compile(
    r"\b(?:may|might|could|would|perhaps|possibly|probably|likely|unlikely|"
    r"apparently|potentially|conceivably|unclear|uncertain|whether|if|"
    r"seems?|appears?|resembles?|suggests?|imagine[ds]?)\b|\blooks?\s+like\b",
    re.IGNORECASE,
)
_SOURCE_SENTENCE = re.compile(r"[^.!?\r\n]+(?:[.!?]+|$)")


def _normalize_triples(triples: Sequence[Sequence[str]]) -> list[list[str]]:
    if isinstance(triples, (str, bytes)) or not isinstance(triples, Sequence):
        raise ValueError("DiscoSG triples must be a sequence of three-string triples")
    normalized = set()
    for triple in triples:
        if (isinstance(triple, (str, bytes)) or not isinstance(triple, Sequence)
                or len(triple) != 3
                or any(not isinstance(part, str) or not part.strip() for part in triple)):
            raise ValueError("each DiscoSG triple must contain three nonempty strings")
        normalized.add(tuple(part.strip() for part in triple))
    # The official merger uses a set, so canonicalize its otherwise unstable order.
    return [list(triple) for triple in sorted(normalized)]


def _word_character(char: str) -> bool:
    return char.isalnum() or char == "_" or char in "-\u2010\u2011"


def _exact_mentions(response: str, entity: str) -> list[list[int]]:
    """Casefold matches with original offsets, including length-changing folds."""
    folded_chars: list[str] = []
    source_indices: list[int] = []
    for index, char in enumerate(response):
        folded = char.casefold()
        folded_chars.append(folded)
        source_indices.extend([index] * len(folded))
    haystack, needle = "".join(folded_chars), entity.casefold()
    matches = []
    cursor = 0
    while needle:
        start = haystack.find(needle, cursor)
        if start < 0:
            break
        stop = start + len(needle)
        a, b = source_indices[start], source_indices[stop - 1] + 1
        cursor = start + 1
        # A match may not begin/end halfway through a Unicode casefold expansion.
        if response[a:b].casefold() != needle:
            continue
        if ((a and _word_character(response[a - 1]) and _word_character(entity[0]))
                or (b < len(response) and _word_character(response[b])
                    and _word_character(entity[-1]))):
            continue
        if not matches or matches[-1] != [a, b]:
            matches.append([a, b])
    return matches


def _source_status(response: str, span: Sequence[int]) -> str | None:
    a, b = span
    containing = [m for m in _SOURCE_SENTENCE.finditer(response)
                  if m.start() <= a and b <= m.end()]
    if len(containing) != 1:
        return "source_sentence_unresolved"
    sentence = containing[0].group()
    if _NEGATION.search(sentence):
        return "negated_source_sentence"
    if _HEDGING.search(sentence):
        return "hedged_source_sentence"
    if sentence.rstrip().endswith("?"):
        return "question_source_sentence"
    return None


def build_object_claims(response: str, triples: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """Project graph entities to conservative, initially unlabeled source units.

    This is an explicit lexical projection after the actual DiscoSG parser, not
    a claim that DiscoSG itself supplies anchors. Repeated, absent, pronominal,
    negated, hedged and overlapping mentions receive no trainable source/error
    spans. All original graph facts should also be retained by the caller.
    """
    if not isinstance(response, str):
        raise ValueError("response must be a string")
    graph = _normalize_triples(triples)
    entities: dict[str, dict[str, Any]] = {}
    for index, (subject, relation, object_) in enumerate(graph):
        slots = [("subject", subject)]
        if " ".join(relation.casefold().split()) not in _ATTRIBUTE_RELATIONS:
            slots.append(("object", object_))
        for slot, entity in slots:
            key = entity.casefold()
            candidate = entities.setdefault(key, {"entity": entity, "graph_references": []})
            candidate["graph_references"].append({"triple_index": index, "slot": slot})

    claims = []
    for candidate in entities.values():
        entity = candidate["entity"]
        claim: dict[str, Any] = {
            "id": f"discosg-object-{len(claims):04d}", "text": entity,
            "entity": entity, "kind": "object", "parser": "discosg",
            "label": "?", "weight": 0.0, "source_spans": [], "error_spans": [],
            "graph_references": candidate["graph_references"],
            "alignment_status": "unresolved", "alignment_confidence": 0.0,
        }
        if entity.casefold() in _NON_ENTITIES or not any(c.isalpha() for c in entity):
            reason = "non_entity_or_coreference_slot"
        else:
            mentions = _exact_mentions(response, entity)
            if not mentions:
                reason = "entity_not_in_source"
            elif len(mentions) != 1:
                reason = "repeated_entity_mention"
            else:
                reason = _source_status(response, mentions[0])
                if reason is None:
                    a, b = mentions[0]
                    claim.update(
                        text=response[a:b], source_spans=[[a, b]], error_spans=[[a, b]],
                        alignment_status="exact_unique", alignment_confidence=1.0,
                    )
        if reason is not None:
            claim.update(alignment_reason=reason, ambiguous=True)
        claims.append(claim)

    # Alternative graph names for the same source region must not produce
    # competing updates on overlapping words.
    overlapping: set[int] = set()
    for index, claim in enumerate(claims):
        if not claim["source_spans"]:
            continue
        a, b = claim["source_spans"][0]
        for other_index in range(index):
            other = claims[other_index]
            if other["source_spans"]:
                c, d = other["source_spans"][0]
                if max(a, c) < min(b, d):
                    overlapping.update((index, other_index))
    for index in overlapping:
        claims[index].update(
            source_spans=[], error_spans=[], alignment_status="unresolved",
            alignment_confidence=0.0, alignment_reason="overlapping_entity_mentions",
            ambiguous=True,
        )
    return claims


def _git_revision(repo: Path) -> tuple[str | None, bool | None]:
    try:
        root = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        # An exported parser directory nested inside MVPG must not inherit
        # MVPG's commit and be mislabeled as a versioned DiscoSG checkout.
        if root.returncode != 0 or Path(root.stdout.strip()).resolve() != repo.resolve():
            return None, None
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        revision = result.stdout.strip() if result.returncode == 0 else ""
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
            return None, None
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _package_points_to(module: Any, directory: Path) -> bool:
    paths = getattr(module, "__path__", ())
    return any(Path(path).resolve() == directory.resolve() for path in paths)


def _load_official_modules(repo: Path) -> tuple[Any, Any, Any]:
    """Accommodate the released ``disco_sg_camera.src`` import without editing it."""
    for name, directory in (("src", repo / "src"), ("disco_sg_camera", repo)):
        existing = sys.modules.get(name)
        if existing is not None and not _package_points_to(existing, directory):
            raise RuntimeError(
                f"Python package {name!r} is already loaded from another location. "
                "Run the DiscoSG decomposition command in a separate process/environment."
            )
    if "disco_sg_camera" not in sys.modules:
        package = types.ModuleType("disco_sg_camera")
        package.__path__ = [str(repo)]
        package.__package__ = "disco_sg_camera"
        package.__spec__ = importlib.machinery.ModuleSpec(
            "disco_sg_camera", loader=None, is_package=True,
        )
        sys.modules["disco_sg_camera"] = package
    repo_string = str(repo)
    sys.path.insert(0, repo_string)
    try:
        seed = importlib.import_module(_PARSER_MODULE)
        refiner = importlib.import_module(_REFINER_MODULE)
        triples = importlib.import_module(_TRIPLE_MODULE)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot import the official DiscoSG parser. Install its model dependencies "
            "(including transformers, torch, peft, nltk and spacy) in a compatible "
            f"environment and check the supplied checkout {repo}. Original error: {exc}"
        ) from exc
    finally:
        sys.path.remove(repo_string)
    return seed, refiner, triples


class DiscoSGDecomposer:
    """Use the official seed/delete/insert/merge pipeline on generated responses.

    ``refiner_model`` must be a local trained checkpoint, including
    ``model_info.json``, ``classifier.pt`` and model weights. A random deletion
    classifier is never accepted. ``expected_commit`` optionally enforces a clean
    checkout at a particular revision; otherwise the actual revision is recorded
    without asserting compatibility with the documented reference revision.
    """

    def __init__(
        self, repo: str | Path, seed_model: str | Path, refiner_model: str | Path,
        device: str = "cpu", rounds: int = 3, *, expected_commit: str | None = None,
        batch_size: int = 4, max_input_len: int = 2048, max_output_len: int = 512,
        max_triples_num: int = 256, num_beams: int = 1,
        seed_num_beams: int = 3, seed_max_input_len: int = 512,
        seed_max_output_len: int = 512,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.seed_model = str(seed_model)
        self.refiner_model = Path(refiner_model).expanduser().resolve()
        if not self.seed_model.strip() or not isinstance(device, str) or not device.strip():
            raise ValueError("seed_model and device must be nonempty")
        limits = {
            "rounds": rounds, "batch_size": batch_size, "max_input_len": max_input_len,
            "max_output_len": max_output_len, "max_triples_num": max_triples_num,
            "num_beams": num_beams, "seed_num_beams": seed_num_beams,
            "seed_max_input_len": seed_max_input_len,
            "seed_max_output_len": seed_max_output_len,
        }
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
               for value in limits.values()):
            raise ValueError("DiscoSG rounds, batch sizes, lengths and beam counts must be positive integers")
        self.device, self.limits = device, {key: int(value) for key, value in limits.items()}
        required_sources = [
            self.repo / "src/factual_scene_graph/parser/scene_graph_parser_insert_delete.py",
            self.repo / "src/factual_scene_graph/parser/DualTaskSceneGraphParser.py",
            self.repo / "src/factual_scene_graph/triple_utils.py",
            self.repo / "src/arch/T5ForDualTask.py",
        ]
        if any(not path.is_file() for path in required_sources):
            raise FileNotFoundError(
                f"Incomplete DiscoSG checkout: {self.repo}. Supply a checkout of "
                f"{OFFICIAL_REPOSITORY}; the reference implementation is {REFERENCE_COMMIT}."
            )
        missing = [name for name in ("model_info.json", "classifier.pt")
                   if not (self.refiner_model / name).is_file()]
        weight_names = (
            "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
            "pytorch_model.bin.index.json", "adapter_model.safetensors", "adapter_model.bin",
        )
        if not any((self.refiner_model / name).is_file() for name in weight_names):
            missing.append("trained model weights")
        if missing:
            raise FileNotFoundError(
                f"DiscoSG refiner checkpoint {self.refiner_model} is missing: "
                f"{', '.join(missing)}. Supply the trained official refiner, including its "
                "deletion classifier; the public repository does not bundle these weights. "
                "No fallback parser or randomly initialized classifier will be used."
            )
        try:
            self.model_info = json.loads((self.refiner_model / "model_info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("DiscoSG refiner model_info.json must contain valid JSON") from exc
        if not isinstance(self.model_info, dict):
            raise ValueError("DiscoSG refiner model_info.json must contain an object")
        revision, dirty = _git_revision(self.repo)
        if expected_commit is not None:
            if (not isinstance(expected_commit, str)
                    or not re.fullmatch(r"[0-9a-fA-F]{40,64}", expected_commit)):
                raise ValueError("expected_commit must be a full Git commit hash")
            if revision != expected_commit or dirty is not False:
                raise ValueError(
                    "DiscoSG checkout does not match the required clean revision "
                    f"{expected_commit}; actual revision={revision!r}, dirty={dirty!r}."
                )
        self.provenance = {
            "parser": "discosg", "repository": OFFICIAL_REPOSITORY,
            "reference_commit": REFERENCE_COMMIT, "checkout": str(self.repo),
            "checkout_commit": revision, "checkout_dirty": dirty,
            "expected_commit": expected_commit,
            "external_models": {
                "seed_model": self.seed_model, "refiner_model": str(self.refiner_model),
                "refiner_tokenizer": self.model_info.get("base_model_name", "google/flan-t5-base"),
                "deletion_classifier": str(self.refiner_model / "classifier.pt"),
            },
            "projection": "exact_unique_object_existence_only",
            "task": "delete_before_insert", "decoding": "deterministic",
            **self.limits,
        }
        self._seed_parser = self._refiner_parser = self._triple_tools = None

    def _load(self) -> None:
        if self._seed_parser is not None:
            return
        seed, refiner, triples = _load_official_modules(self.repo)
        try:
            seed_parser = seed.SceneGraphParser(
                self.seed_model, device=self.device, lemmatize=False, lowercase=True,
            )
            refiner_parser = refiner.DualTaskSceneGraphParser(
                model_path=str(self.refiner_model), device=self.device,
                lemmatize=False, lowercase=True,
            )
            if not callable(triples.merge_delete_insert_results):
                raise AttributeError("merge_delete_insert_results is not callable")
        except Exception as exc:
            raise RuntimeError(
                "Cannot load the official DiscoSG models. Check the seed checkpoint, trained "
                f"refiner weights and compatible dependencies. Original error: {exc}"
            ) from exc
        self._seed_parser, self._refiner_parser, self._triple_tools = seed_parser, refiner_parser, triples

    @staticmethod
    def _check_budget(parser: Any, prompts: Sequence[str], limit: int, name: str) -> None:
        tokenizer = getattr(parser, "tokenizer", None)
        if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
            # Real released parsers expose a tokenizer. Keeping the helper
            # optional also permits dependency-free contract tests with fakes.
            return
        for prompt in prompts:
            if len(tokenizer.encode(prompt, add_special_tokens=True)) > limit:
                raise ValueError(
                    f"DiscoSG {name} input exceeds its configured token limit {limit}; "
                    "increase the corresponding input limit or exclude this response. "
                    "Silent source truncation is not allowed."
                )

    def decompose(self, record: Mapping[str, Any]) -> dict[str, Any]:
        out = validate_record(record)
        response = out["response"]
        metadata = out.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("record.metadata must be an object")
        # New claims invalidate all scores and calibration provenance attached
        # to the previous decomposition, even when the response is unchanged.
        metadata.pop("verifier", None)
        metadata["decomposer"] = copy.deepcopy(self.provenance)
        if not response.strip():
            out["claims"] = []
            out["disco_graph"] = {
                "triples": [], "seed_graph": "", "refined_graph": "",
                "unresolved_facts": [], "skip_reason": "empty_response",
            }
            return out
        self._load()
        seed_sentences = [sentence.strip() + ("" if sentence.strip().endswith(".") else ".")
                          for sentence in response.lower().split(". ") if sentence]
        self._check_budget(
            self._seed_parser, ["Generate Scene Graph: " + s for s in seed_sentences],
            self.limits["seed_max_input_len"], "seed parser",
        )
        seeds = self._seed_parser.parse(
            [response], sub_sentence=True, return_text=True,
            batch_size=self.limits["batch_size"], num_beams=self.limits["seed_num_beams"],
            max_input_len=self.limits["seed_max_input_len"],
            max_output_len=self.limits["seed_max_output_len"], do_sample=False,
        )
        if not isinstance(seeds, (list, tuple)) or len(seeds) != 1 or not isinstance(seeds[0], str):
            raise RuntimeError("Official DiscoSG seed parser must return one graph string per response")
        seed_graph = graph = seeds[0]
        triples: list[list[str]] = []
        for _ in range(self.limits["rounds"]):
            if callable(getattr(self._triple_tools, "extract_triples", None)):
                if len(self._triple_tools.extract_triples(graph)) > self.limits["max_triples_num"]:
                    raise ValueError("DiscoSG graph exceeds max_triples_num; refusing silent triple truncation")
            self._check_budget(
                self._refiner_parser,
                [f"Delete Task:\nCaption: {response}\nCandidate Graph: {graph}",
                 f"Insert Task:\nCaption: {response}\nCorrupted Graph: {graph}"],
                self.limits["max_input_len"], "refiner",
            )
            result = self._refiner_parser.parse(
                descriptions=[response], graph_to_fix={response: graph},
                task="delete_before_insert", batch_size=self.limits["batch_size"],
                max_input_len=self.limits["max_input_len"],
                max_output_len=self.limits["max_output_len"],
                max_triples_num=self.limits["max_triples_num"],
                skip_toolong=False, skip_len=-1, num_beams=self.limits["num_beams"],
                do_sample=False,
            )
            try:
                delete_result, insert_result = result["delete"][response], result["insert"][response]
            except (TypeError, KeyError) as exc:
                raise RuntimeError("DiscoSG refiner returned an incompatible delete/insert result") from exc
            if not isinstance(delete_result, str) or not isinstance(insert_result, str):
                raise RuntimeError("DiscoSG delete/insert results must be graph strings")
            merged, graph = self._triple_tools.merge_delete_insert_results(
                insert_res=insert_result, delete_res=delete_result,
            )
            triples = _normalize_triples(merged)
            if not isinstance(graph, str):
                raise RuntimeError("DiscoSG graph merger must return (triples, graph_string)")
        out["claims"] = build_object_claims(response, triples)
        out["disco_graph"] = {
            "triples": triples, "seed_graph": seed_graph, "refined_graph": graph,
            "unresolved_facts": [
                {"triple": triple, "reason": "attribute_and_relation_training_not_implemented"}
                for triple in triples
            ],
        }
        return validate_record(out)
