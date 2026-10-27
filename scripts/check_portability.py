#!/usr/bin/env python3
"""Reject broken or external symlinks in distributable GeoUG source."""

import os
from pathlib import Path

EXCLUDED = {".git", "outputs", "artifacts", "release", "release-notes", "__pycache__", "build", "dist"}


def check(root):
    root = Path(root).resolve()
    errors = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in EXCLUDED and not d.startswith((".venv", ".pytest", ".ruff"))]
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                target = path.resolve()
                if not target.is_relative_to(root) or not target.exists():
                    errors.append(str(path.relative_to(root)))
    if errors:
        raise ValueError("Broken or external symlinks in source: " + ", ".join(errors))


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[1])
    print("Source portability: no broken or external symlinks")
