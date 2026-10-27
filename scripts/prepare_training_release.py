#!/usr/bin/env python3
"""Materialize upload-ready GeoUG training artifacts outside the code repo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def _link_or_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _materialize(source: Path, destination: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        _link_or_copy(source, destination)
        return [destination]

    outputs = []
    for path in sorted(source.rglob("*")):
        if not (path.is_file() or path.is_symlink()):
            continue
        output = destination / path.relative_to(source)
        _link_or_copy(path, output)
        outputs.append(output)
    return outputs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--v5-init", type=Path, required=True)
    parser.add_argument("--sft-init", type=Path, required=True)
    parser.add_argument("--rl-dataset", type=Path, required=True)
    parser.add_argument("--train-depth-cache", type=Path, required=True)
    parser.add_argument("--eval-questions", type=Path, required=True)
    parser.add_argument("--eval-depth-cache", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    artifacts = {
        "models/geoug-v5-step-1000": args.v5_init,
        "models/geoug-v6-sft-1500": args.sft_init,
        "training/rl-train-bal800": args.rl_dataset,
        "training/depth-cache-train": args.train_depth_cache,
        "training/d3-eval-questions.parquet": args.eval_questions,
        "training/depth-cache-eval": args.eval_depth_cache,
    }

    staged = {}
    all_files = []
    for relative, source in artifacts.items():
        destination = output_root / relative
        files = _materialize(source, destination)
        all_files.extend(files)
        staged[relative] = {
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
        print(f"{relative}: {len(files)} files")

    checksum_lines = []
    for path in sorted(all_files):
        relative = path.relative_to(output_root)
        checksum_lines.append(f"{_sha256(path)}  {relative.as_posix()}")
    (output_root / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "format_version": 1,
        "artifacts": staged,
        "total_files": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Release staging ready: {output_root}")


if __name__ == "__main__":
    main()
