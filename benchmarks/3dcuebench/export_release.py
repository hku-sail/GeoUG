#!/usr/bin/env python3
"""Convert the frozen internal 3DCueBench parquet into a portable release shard."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


REQUIRED_COLUMNS = {
    "question_id", "image_id", "image", "source", "domain", "split", "bin",
    "relgap", "question", "option_a", "option_b", "answer", "pt_a", "pt_b",
    "pt_anchor", "image_w", "image_h", "grid_w", "grid_h", "d_option_a",
    "d_option_b", "sev",
}


def _point(value):
    if value is None:
        return None
    try:
        if pa.scalar(value).is_valid is False:
            return None
    except (TypeError, pa.ArrowInvalid):
        pass
    return [int(x) for x in value]


def _encode_image(path: Path, max_side: int, quality: int) -> bytes:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export(args):
    source = pq.read_table(args.questions)
    missing = REQUIRED_COLUMNS - set(source.column_names)
    if missing:
        raise ValueError(f"source parquet is missing columns: {sorted(missing)}")

    rows = source.to_pylist()
    image_cache: dict[str, tuple[dict, str]] = {}
    released = []

    for row in rows:
        image_id = str(row["image_id"])
        source_path = Path(row["image"])
        if image_id not in image_cache:
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if args.embed_images:
                content = _encode_image(source_path, args.max_side, args.jpeg_quality)
                image = {"bytes": content, "path": f"{image_id}.jpg"}
                digest = _sha256(content)
            else:
                # The manifest path is deliberately public and relative. A source-specific
                # preparation script must materialize it before evaluation.
                suffix = source_path.suffix.lower() or ".jpg"
                image = {
                    "bytes": None,
                    "path": f"images/{row['source']}/{image_id}{suffix}",
                }
                digest = _sha256(source_path.read_bytes())
            image_cache[image_id] = image, digest

        image, digest = image_cache[image_id]
        released.append({
            "question_id": str(row["question_id"]),
            "image_id": image_id,
            "image": image,
            "image_sha256": digest,
            "source": str(row["source"]),
            "domain": str(row["domain"]),
            "split": str(row["split"]),
            "difficulty_bin": str(row["bin"]),
            "relative_gap": float(row["relgap"]),
            "question": str(row["question"]),
            "option_a": str(row["option_a"]),
            "option_b": str(row["option_b"]),
            "answer": str(row["answer"]).strip().upper(),
            "point_a": _point(row["pt_a"]),
            "point_b": _point(row["pt_b"]),
            "anchor": None if row.get("anchor") is None else str(row["anchor"]),
            "point_anchor": _point(row["pt_anchor"]),
            "image_width": int(row["image_w"]),
            "image_height": int(row["image_h"]),
            "grid_width": int(row["grid_w"]),
            "grid_height": int(row["grid_h"]),
            "distance_a": float(row["d_option_a"]),
            "distance_b": float(row["d_option_b"]),
            "quality_severity": int(row["sev"]),
            "shortcut_point_correct": (
                None if row.get("point_rule_ok") is None else bool(row["point_rule_ok"])
            ),
            "shortcut_second_correct": (
                None if row.get("second_rule_ok") is None else bool(row["second_rule_ok"])
            ),
            "shortcut_third_correct": (
                None if row.get("third_rule_ok") is None else bool(row["third_rule_ok"])
            ),
            "shortcut_class_prior_correct": (
                None if row.get("class_prior_ok") is None else bool(row["class_prior_ok"])
            ),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(released)
    pq.write_table(table, output, compression="zstd", use_dictionary=True)
    digest = _sha256(output.read_bytes())
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(released)} questions and {len(image_cache)} images to {output} "
        f"({output.stat().st_size / 1e6:.1f} MB)"
    )
    print(f"SHA256 {digest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed resized JPEG bytes for the canonical public parquet.",
    )
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    export(parser.parse_args())


if __name__ == "__main__":
    main()

