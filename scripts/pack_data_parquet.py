#!/usr/bin/env python3
"""Build self-contained release parquets without changing source artifacts.

RL preserves the original columns and adds camera (intrinsics, image_wh) and
depth_npy (original .npy bytes; decode with np.load(BytesIO(...), allow_pickle=False)).
Textual evaluation uses the same image {bytes, path} struct as 3DCueBench.
Only rows in --input are exported; unrelated camera/cache entries are excluded.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def unpack_rl(args):
    """Restore the existing training layout from a single release parquet."""
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent))
    try:
        depth_dir = temporary / "depth-cache"
        depth_dir.mkdir()
        source = pq.ParquetFile(args.input)
        schema = pa.schema([field for field in source.schema_arrow
                            if field.name not in {"camera", "depth_npy"}])
        cameras = {}
        with pq.ParquetWriter(temporary / "train.parquet", schema, compression="zstd") as writer:
            for batch in source.iter_batches(batch_size=16):
                rows = batch.to_pylist()
                for row in rows:
                    key = row["image_key"]
                    if not key or Path(key).name != key:
                        raise ValueError(f"Invalid image key: {key}")
                    camera = row.pop("camera")
                    raw = row.pop("depth_npy")
                    path = depth_dir / f"{key}.npy"
                    if key in cameras:
                        if cameras[key] != camera or path.read_bytes() != raw:
                            raise ValueError(f"Conflicting data for image key: {key}")
                    else:
                        np.load(io.BytesIO(raw), allow_pickle=False)
                        path.write_bytes(raw)
                        cameras[key] = camera
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        (temporary / "camera_table.json").write_text(json.dumps(cameras))
        temporary.rename(args.output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Restored train.parquet, camera_table.json and depth-cache/ under {args.output}")


def pack(args):
    source = pq.ParquetFile(args.input)
    schema = source.schema_arrow.remove_metadata()
    cameras = None
    if args.kind == "rl":
        cameras = json.loads(args.cameras.read_text())
        schema = schema.append(pa.field("camera", pa.struct([
            ("intrinsics", pa.list_(pa.list_(pa.float64()))),
            ("image_wh", pa.list_(pa.int64())),
        ]))).append(pa.field("depth_npy", pa.binary()))
    else:
        index = schema.get_field_index("image")
        if index < 0:
            raise ValueError("Missing image column")
        schema = schema.set(index, pa.field("image", pa.struct([
            ("bytes", pa.binary()), ("path", pa.string()),
        ])))
    schema = schema.with_metadata({
        b"geoug_format": b"rl-embedded-v1" if cameras is not None else b"textual-embedded-v1",
        b"depth_encoding": b"npy; numpy.load(io.BytesIO(depth_npy), allow_pickle=False)",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    fd, temporary = tempfile.mkstemp(dir=args.output.parent, suffix=".parquet.tmp")
    os.close(fd)
    count = 0
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for batch in source.iter_batches(batch_size=16):
                rows = batch.to_pylist()
                for row in rows:
                    if cameras is not None:
                        key = row["image_key"]
                        if Path(key).name != key:
                            raise ValueError(f"Invalid image key: {key}")
                        camera = cameras[key]
                        intrinsics = np.asarray(camera["intrinsics"])
                        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                            raise ValueError(f"Invalid intrinsics: {key}")
                        if len(camera["image_wh"]) != 2 or min(camera["image_wh"]) <= 0:
                            raise ValueError(f"Invalid image_wh: {key}")
                        depth_bytes = (args.depth_cache / f"{key}.npy").read_bytes()
                        depth = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
                        if depth.ndim != 2 or not np.isfinite(depth).all():
                            raise ValueError(f"Invalid depth: {key}")
                        row["camera"] = camera
                        row["depth_npy"] = depth_bytes
                        image_bytes = row["image"]
                    else:
                        path = Path(row["image"])
                        if not path.is_absolute():
                            path = (args.image_root or args.input.parent) / path
                        image_bytes = path.read_bytes()
                        # No local paths are needed to read this release.
                        row["image"] = {"bytes": image_bytes, "path": None}
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                count += len(rows)
        if pq.ParquetFile(temporary).metadata.num_rows != source.metadata.num_rows:
            raise ValueError("Row count mismatch")
        os.replace(temporary, args.output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"{args.output}: {count} rows, {args.output.stat().st_size:,} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["rl", "textual", "unpack-rl"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--depth-cache", type=Path)
    parser.add_argument("--image-root", type=Path)
    args = parser.parse_args()
    if args.kind == "rl" and (args.cameras is None or args.depth_cache is None):
        parser.error("rl requires --cameras and --depth-cache")
    if args.kind == "unpack-rl":
        unpack_rl(args)
    else:
        pack(args)


if __name__ == "__main__":
    main()
