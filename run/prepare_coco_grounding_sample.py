"""Freeze a random sample from the locally available COCO val2014 image pool.

The source folder need not be the complete COCO validation set. Membership and
unmodified file bytes are checked against the official archive central directory
without downloading its images. No benchmark-overlap exclusion or training is
performed. Example:

  python run/prepare_coco_grounding_sample.py --source C:/data/coco_raw \
    --output output/grounding_calibration/coco1000/details/sample \
    --remote-image-root /root/autodl-tmp/data/coco_calibration_1000 --tar
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import tarfile
import urllib.request
import zipfile
import zlib


REPO = Path(__file__).resolve().parents[1]
OFFICIAL_ARCHIVE = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2014.zip"
CANONICAL_ARCHIVE = "http://images.cocodataset.org/zips/val2014.zip"
NAME = re.compile(r"COCO_val2014_(\d{12})\.jpg\Z")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def immutable_write(path, data):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen artifact would change: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class HttpRangeReader(io.RawIOBase):
    """A seekable metadata-only reader; reject non-range or huge responses."""
    def __init__(self, url):
        self.url, self.position, self.downloaded = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative archive seek")
        return self.position

    def read(self, size=-1):
        count = max(0, min(self.size - self.position, self.size if size < 0 else size))
        if not count:
            return b""
        if count > 32 * 1024 * 1024:
            raise ValueError("Refusing to download image payload; only ZIP metadata is needed")
        start, end = self.position, self.position + count - 1
        headers = {"Range": f"bytes={start}-{end}"}
        if self.etag:
            headers["If-Match"] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=30) as response:
            expected_range = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError("Server did not honor the bounded metadata range")
            data = response.read(count + 1)
        if len(data) != count:
            raise ValueError("Truncated or oversized archive metadata range")
        self.position += count
        self.downloaded += count
        return data


def load_official_index(path):
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "official_coco_val2014_zip_index_v1" or value.get("archive_url") != OFFICIAL_ARCHIVE:
            raise ValueError("Official index has an unexpected schema or source")
        return value
    reader = HttpRangeReader(OFFICIAL_ARCHIVE)
    with zipfile.ZipFile(reader) as archive:
        files = {Path(info.filename).name: {"bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
                 for info in archive.infolist()
                 if info.filename.startswith("val2014/") and NAME.fullmatch(Path(info.filename).name)}
    if len(files) != 40504:
        raise ValueError(f"Official val2014 archive has unexpected image count: {len(files)}")
    value = dict(schema="official_coco_val2014_zip_index_v1", archive_url=OFFICIAL_ARCHIVE,
                 canonical_url=CANONICAL_ARCHIVE, archive_bytes=reader.size, etag=reader.etag,
                 metadata_downloaded_bytes=reader.downloaded, images=len(files), files=files)
    immutable_write(path, json_bytes(value))
    return value


def inventory(source, official):
    paths = sorted(p for p in source.iterdir() if p.is_file())
    candidates, ignored = [], []
    for path in paths:
        match = NAME.fullmatch(path.name)
        if not match:
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                raise ValueError(f"Image is not a canonical val2014 filename: {path.name}")
            ignored.append(path.name)
            continue
        expected = official["files"].get(path.name)
        if expected is None:
            raise ValueError(f"Image is absent from official val2014 archive: {path.name}")
        h, checksum, size = hashlib.sha256(), 0, 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
                checksum = zlib.crc32(chunk, checksum)
                size += len(chunk)
        actual_crc = f"{checksum & 0xffffffff:08x}"
        if size != expected["bytes"] or actual_crc != expected["crc32"]:
            raise ValueError(f"Image bytes differ from official archive: {path.name}")
        candidates.append(dict(image_id=int(match.group(1)), filename=path.name, bytes=size,
                               sha256=h.hexdigest(), crc32=actual_crc))
    if not candidates:
        raise ValueError("No canonical COCO val2014 images found")
    return candidates, ignored


def select_and_split(candidates, *, size, seed, dev_fraction):
    if size < 2 or size > len(candidates):
        raise ValueError("Sample size must be between 2 and the available image count")
    if not 0 < dev_fraction < 1:
        raise ValueError("Development fraction must lie strictly between zero and one")
    selected = sorted(random.Random(seed).sample(candidates, size), key=lambda row: row["filename"])
    groups = sorted({r["sha256"] for r in selected})
    if len(groups) < 2:
        raise ValueError("At least two distinct image contents are needed for disjoint splits")
    random.Random(seed + 1).shuffle(groups)
    n_dev = min(len(groups) - 1, max(1, math.floor(len(groups) * dev_fraction)))
    dev_groups = set(groups[:n_dev])
    for row in selected:
        row["split"] = "dev" if row["sha256"] in dev_groups else "test"
    return selected


def evaluation_overlaps(selected, repo=REPO):
    """Audit declared COCO image references without changing random selection."""
    selected_ids = {row["image_id"] for row in selected}
    paths = list((repo / "eval_llava_rlhf_coco" / "pope").glob("*.jsonl"))
    paths += list((repo / "eval_llava_rlhf_coco" / "object_hal").glob("*.jsonl"))
    result = []
    image_reference = re.compile(r"(?:COCO_(?:val|train)201[47]_)?(\d{12})\.(?:jpg|png)", re.I)
    for path in sorted(paths):
        ids = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for key, value in row.items():
                if "image" not in key.lower():
                    continue
                if isinstance(value, str):
                    ids.update(int(match) for match in image_reference.findall(value))
                if key == "image_id" and isinstance(value, (int, str)) and str(value).isdigit():
                    ids.add(int(value))
        intersection = sorted(selected_ids & ids)
        result.append(dict(source=str(path.relative_to(repo)), source_sha256=sha256(path),
                           referenced_coco_images=len(ids), selected_overlap_count=len(intersection),
                           selected_overlap_image_ids=intersection))
    return dict(policy="Audit only; no overlap exclusion. This sample is for verifier calibration, not training.",
                scope="Repository POPE and object_hal declarations; other external evaluation sets are not certified disjoint.",
                sets=result)


def make_archive(path, source, selected):
    expected = {row["filename"]: row for row in selected}
    if not path.exists():
        temporary = path.with_name("." + path.name + ".tmp")
        with tarfile.open(temporary, "x") as archive:
            for row in selected:
                info = archive.gettarinfo(str(source / row["filename"]), arcname=row["filename"])
                info.uid, info.gid, info.uname, info.gname, info.mtime = 0, 0, "", "", 0
                with (source / row["filename"]).open("rb") as handle:
                    archive.addfile(info, handle)
        os.replace(temporary, path)
    with tarfile.open(path, "r") as archive:
        entries = archive.getmembers()
        if len(entries) != len(expected) or {p.name for p in entries} != set(expected):
            raise ValueError("Existing tar does not contain exactly the frozen sample")
        for entry in entries:
            if not entry.isfile() or entry.size != expected[entry.name]["bytes"]:
                raise ValueError("Archive member differs from the frozen image")
            h = hashlib.sha256()
            with archive.extractfile(entry) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected[entry.name]["sha256"]:
                raise ValueError("Archive image bytes differ from source")
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path),
                layout=f"{len(selected)} selected image filenames at archive root; no absolute paths or links")


def prepare(args):
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    official_path = Path(args.official_index or output / "official_val2014_index.json").resolve()
    official = load_official_index(official_path)
    candidates, ignored = inventory(source, official)
    selected = select_and_split(candidates, size=args.size, seed=args.seed, dev_fraction=args.dev_fraction)
    records = []
    for row in selected:
        records.append(dict(id=f"coco_val2014_{row['image_id']:012d}",
            image=str(PurePosixPath(args.remote_image_root) / row["filename"]),
            question="Please describe this image in detail.", response="", claims=[],
            metadata=dict(image_group="sha256:"+row["sha256"], split=row["split"], analysis_role="formal",
                          dataset="COCO", official_split="val2014", coco_image_id=row["image_id"],
                          image_sha256=row["sha256"], official_archive_crc32=row["crc32"],
                          sample_seed=args.seed, use="verifier_calibration_only_not_training")))
    payload = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in records)
    immutable_write(output / "images.jsonl", payload)
    selected_names = {row["filename"] for row in selected}
    inventory_rows = [dict(row, selected=row["filename"] in selected_names) for row in candidates]
    manifest = dict(schema="coco_grounding_sample_v1", source=str(source),
                    official_dataset_split="COCO val2014", official_split_size=official["images"],
                    official_index_path=str(official_path), official_index_sha256=sha256(official_path),
                    available_images=len(candidates), selected_images=len(selected),
                    population_limitation="Uniform random sample from local available val2014 subset, not all 40504 validation images.",
                    selection=dict(seed=args.seed, algorithm="Python random.Random(seed).sample over lexicographically sorted canonical filenames; output sorted by filename",
                                   dev_fraction=args.dev_fraction, split_seed=args.seed+1,
                                   split_algorithm="Shuffle sorted image SHA256 groups; floor(group_count * dev_fraction) dev groups, remaining test groups; before any generation or labels"),
                    split_images=dict(Counter(r["split"] for r in selected)),
                    unique_selected_content=len({r["sha256"] for r in selected}),
                    verified_against_official_archive="All candidate filenames, file sizes and CRC32 match official ZIP central directory",
                    remote_image_root=args.remote_image_root, question=records[0]["question"],
                    images_jsonl_sha256=hashlib.sha256(payload).hexdigest(),
                    selected=selected, input_candidates=inventory_rows, ignored_nonimage_files=ignored,
                    evaluation_overlaps=evaluation_overlaps(selected),
                    intended_use="Verifier calibration only. Do not train on this validation sample or report it as held-out model evaluation.")
    if args.tar:
        manifest["archive"] = make_archive(output / "images.tar", source, selected)
    immutable_write(output / "sample_manifest.json", json_bytes(manifest))
    print(json.dumps(dict(status="complete", output=str(output), available=len(candidates), selected=len(selected),
                         splits=manifest["split_images"], archive=manifest.get("archive"),
                         overlaps=[dict(source=s["source"], count=s["selected_overlap_count"])
                                   for s in manifest["evaluation_overlaps"]["sets"]]), ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--remote-image-root", required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=.4)
    parser.add_argument("--official-index")
    parser.add_argument("--tar", action="store_true", help="Package only selected images for cloud upload")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
