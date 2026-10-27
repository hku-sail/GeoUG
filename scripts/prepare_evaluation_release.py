#!/usr/bin/env python3
"""Collect metrics and predictions into an upload-ready evaluation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


INCLUDE_NAMES = {"metrics.json", "summary.txt"}
INCLUDE_SUFFIXES = (
    "predictions.tsv",
    "_acc.csv",
    "_acc.xlsx",
    "_circular.pkl",
    "_result.pkl",
)


def _selected(path: Path) -> bool:
    if path.suffix == ".pkl" and "_supp" not in path.name:
        return True
    return path.name in INCLUDE_NAMES or path.name.endswith(INCLUDE_SUFFIXES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--command", default="")
    parser.add_argument("--protocol", choices=("current", "legacy"), default="current")
    args = parser.parse_args()

    destination_root = args.output_root.resolve() / args.label
    files = [path for path in args.source_root.rglob("*") if path.is_file() and _selected(path)]
    if not files:
        raise FileNotFoundError(f"No metrics or predictions found below {args.source_root}")

    manifest_files = []
    for source in sorted(files):
        relative = source.relative_to(args.source_root)
        destination = destination_root / relative
        _link_or_copy(source, destination)
        manifest_files.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )

    manifest = {
        "format_version": 1,
        "label": args.label,
        "protocol": args.protocol,
        "command": args.command,
        "files": manifest_files,
    }
    (destination_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Collected {len(files)} files -> {destination_root}")


if __name__ == "__main__":
    main()
