#!/usr/bin/env python3
"""Copy the public source without weights, then run checks outside the checkout."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from check_portability import check

SOURCE_DIRS = {
    ".github", "benchmarks", "ckpt", "configs", "data", "docs", "evaluation",
    "inference", "licenses", "modeling", "release_tests", "scripts",
    "third_party", "train",
}
SOURCE_FILES = {
    ".gitattributes", ".gitignore", "CITATION.cff", "CONTRIBUTING.md",
    "LICENSE", "NOTICE", "README.md", "SECURITY.md",
    "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements.txt",
    "requirements-dev.txt", "requirements-eval.txt", "requirements-rl.txt",
}
IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", ".venv*", "*.egg-info",
    "*.safetensors", "*.parquet", "*.pt", "*.pth", "*.bin", "*.ckpt",
    ".coverage", ".coverage.*", "htmlcov", ".ipynb_checkpoints", "*.swp", "*.swo", "*~",
)


def run(destination):
    source = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("The standalone copy must be outside the source checkout")
    check(source)
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        target = destination / path.name
        if path.is_dir() and path.name in SOURCE_DIRS:
            shutil.copytree(path, target, ignore=IGNORE)
        elif path.is_file() and path.name in SOURCE_FILES:
            shutil.copy2(path, target)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHON"] = sys.executable
    subprocess.run(["bash", "scripts/check_release.sh"], cwd=destination, env=env, check=True)
    for module in ("inference.depth", "inference.spatial", "evaluation.depth.eval_depth"):
        subprocess.run([sys.executable, "-m", module, "--help"], cwd=destination,
                       env=env, check=True, stdout=subprocess.DEVNULL)
    print(f"Standalone source checks passed: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, help="Keep the validated source at this new directory")
    args = parser.parse_args()
    if args.destination:
        run(args.destination)
    else:
        with tempfile.TemporaryDirectory(prefix="geoug-standalone-") as temporary:
            run(Path(temporary) / "source")
