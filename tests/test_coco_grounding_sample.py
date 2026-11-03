"""Frozen image sampling and official byte checks, independent of model APIs."""
import hashlib
import zlib

import pytest

from run.prepare_coco_grounding_sample import inventory, select_and_split


def candidate(index, content=None):
    return dict(image_id=index, filename=f"COCO_val2014_{index:012d}.jpg",
                sha256=hashlib.sha256(str(index if content is None else content).encode()).hexdigest())


def test_frozen_sampling_has_disjoint_content_splits_and_expected_counts():
    pool = [candidate(index) for index in range(1500)]
    selected = select_and_split(pool, size=1000, seed=42, dev_fraction=.4)
    repeated = select_and_split(pool, size=1000, seed=42, dev_fraction=.4)
    assert selected == repeated
    assert len({row["filename"] for row in selected}) == 1000
    assert sum(row["split"] == "dev" for row in selected) == 400
    assert sum(row["split"] == "test" for row in selected) == 600
    assert {row["sha256"] for row in selected if row["split"] == "dev"}.isdisjoint(
        row["sha256"] for row in selected if row["split"] == "test")


def test_identical_image_contents_cannot_leak_across_splits():
    pool = [candidate(1, "same"), candidate(2, "same"), candidate(3), candidate(4), candidate(5)]
    selected = select_and_split(pool, size=5, seed=42, dev_fraction=.4)
    assert selected[0]["split"] == selected[1]["split"]


def test_renaming_train_image_is_not_enough_to_pass_official_byte_check(tmp_path):
    path = tmp_path / "COCO_val2014_000000000001.jpg"
    original = b"official bytes"
    expected = {"files": {path.name: {"bytes": len(original), "crc32": f"{zlib.crc32(original):08x}"}}}
    path.write_bytes(original)
    verified, ignored = inventory(tmp_path, expected)
    assert not ignored
    assert verified[0]["sha256"] == hashlib.sha256(original).hexdigest()
    path.write_bytes(b"modified bytes")
    with pytest.raises(ValueError, match="differ from official"):
        inventory(tmp_path, expected)


def test_nonvalidation_image_filename_is_rejected(tmp_path):
    (tmp_path / "COCO_train2014_000000000001.jpg").write_bytes(b"jpeg")
    with pytest.raises(ValueError, match="not a canonical val2014"):
        inventory(tmp_path, {"files": {}})


@pytest.mark.parametrize("size,fraction", [(2000, .4), (1, .4), (10, 1.0), (10, 0.0)])
def test_invalid_sampling_configuration_is_rejected(size, fraction):
    with pytest.raises(ValueError):
        select_and_split([candidate(i) for i in range(100)], size=size, seed=42, dev_fraction=fraction)
