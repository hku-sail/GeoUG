#!/usr/bin/env python3
"""Validate a frozen 3DCueBench v1.0 release parquet."""

from __future__ import annotations

import argparse
import hashlib
import io
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


REQUIRED = {
    "question_id", "image_id", "image", "image_sha256", "source", "domain",
    "split", "difficulty_bin", "relative_gap", "question", "option_a",
    "option_b", "answer", "point_a", "point_b", "point_anchor",
    "image_width", "image_height", "grid_width", "grid_height", "distance_a",
    "distance_b",
}
INTERNAL_MARKERS = (
    "/mnt/bn/", "/mnt/hdfs/", "hdfs://", "haruna", "code.byted.org",
)
POINT_RE = re.compile(r"<point>\s*(-?\d+)\s+(-?\d+)\s*</point>")


def fail(message):
    raise AssertionError(message)


def _check_point(point, width, height, label):
    if point is None:
        return
    if len(point) != 2:
        fail(f"{label}: expected two coordinates, got {point!r}")
    x, y = point
    if not (0 <= x <= width and 0 <= y <= height):
        fail(f"{label}: point {point!r} is outside {width}x{height}")


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _strings(nested)


def validate(path: Path, expected_questions: int, expected_images: int):
    table = pq.read_table(path)
    missing = REQUIRED - set(table.column_names)
    if missing:
        fail(f"missing columns: {sorted(missing)}")
    rows = table.to_pylist()

    if len(rows) != expected_questions:
        fail(f"expected {expected_questions} questions, got {len(rows)}")
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        fail("question_id is not unique")
    if len({row["image_id"] for row in rows}) != expected_images:
        fail(f"expected {expected_images} unique images")

    split_counts = {}
    answer_counts = {}
    bin_counts = {}
    image_hashes = {}

    for row in rows:
        qid = row["question_id"]
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        answer_counts[row["answer"]] = answer_counts.get(row["answer"], 0) + 1
        key = (row["split"], row["difficulty_bin"])
        bin_counts[key] = bin_counts.get(key, 0) + 1

        public_text = " ".join(_strings(row))
        marker = next((x for x in INTERNAL_MARKERS if x in public_text), None)
        if marker:
            fail(f"{qid}: internal marker {marker!r} leaked into public fields")

        width, height = int(row["grid_width"]), int(row["grid_height"])
        for name in ("point_a", "point_b", "point_anchor"):
            _check_point(row[name], width, height, f"{qid}.{name}")

        prompt_points = [list(map(int, point)) for point in POINT_RE.findall(row["question"])]
        structured = [row["point_a"], row["point_b"]]
        if row["point_anchor"] is not None:
            structured.append(row["point_anchor"])
        for point in structured:
            if point not in prompt_points:
                fail(f"{qid}: structured point {point} is absent from question")

        expected_answer = "A" if row["distance_a"] < row["distance_b"] else "B"
        if row["answer"] != expected_answer:
            fail(
                f"{qid}: answer={row['answer']} disagrees with distances "
                f"{row['distance_a']} and {row['distance_b']}"
            )

        image = row["image"]
        content = image.get("bytes") if image else None
        if content is not None:
            digest = hashlib.sha256(content).hexdigest()
            if digest != row["image_sha256"]:
                fail(f"{qid}: embedded image checksum mismatch")
            with Image.open(io.BytesIO(content)) as decoded:
                decoded.verify()
        else:
            image_path = image.get("path") if image else None
            if not image_path:
                fail(f"{qid}: image has neither bytes nor a manifest path")
            relative_path = Path(image_path)
            if relative_path.is_absolute():
                fail(f"{qid}: manifest image path must be relative: {image_path}")
            release_root = path.parent.resolve()
            resolved_path = (release_root / relative_path).resolve()
            if not resolved_path.is_relative_to(release_root):
                fail(f"{qid}: manifest image escapes the release directory: {image_path}")
            if not resolved_path.is_file():
                fail(f"{qid}: manifest image is unavailable: {resolved_path}")
            content = resolved_path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != row["image_sha256"]:
                fail(f"{qid}: manifest image checksum mismatch")
            with Image.open(io.BytesIO(content)) as decoded:
                decoded.verify()

        previous = image_hashes.setdefault(row["image_id"], row["image_sha256"])
        if previous != row["image_sha256"]:
            fail(f"{qid}: one image_id maps to multiple image hashes")

    expected_split = expected_questions // 2
    if split_counts != {"depth": expected_split, "distance": expected_split}:
        fail(f"unexpected split counts: {split_counts}")
    if answer_counts != {"A": expected_split, "B": expected_split}:
        fail(f"unexpected answer counts: {answer_counts}")
    if len(bin_counts) != 8 or set(bin_counts.values()) != {expected_questions // 8}:
        fail(f"unexpected split x difficulty counts: {bin_counts}")

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"OK: {len(rows)} questions, {len(image_hashes)} images")
    print(f"splits: {split_counts}; answers: {answer_counts}")
    print(f"SHA256 {file_hash}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--expected-questions", type=int, default=800)
    parser.add_argument("--expected-images", type=int, default=757)
    args = parser.parse_args()
    validate(args.parquet, args.expected_questions, args.expected_images)


if __name__ == "__main__":
    main()

