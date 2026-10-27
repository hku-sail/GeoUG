#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/check_portability.py

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

import yaml

excluded = {".venv-sft-smoke", "outputs", "release", "__pycache__", ".git"}
for path in Path(".").rglob("*.py"):
    if excluded.intersection(path.parts):
        continue
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

for path in Path("configs").rglob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))
for path in Path("configs").rglob("*.yaml"):
    yaml.safe_load(path.read_text(encoding="utf-8"))
yaml.safe_load(Path("CITATION.cff").read_text(encoding="utf-8"))
PY

while IFS= read -r script; do
  bash -n "${script}"
done < <(rg --files -g '*.sh' -g '!outputs/**' -g '!.venv*/**')

if rg -n \
  -g '!train/rl/data_pipeline/build_depth_cache.py' \
  -g '!train/rl/tests/**' \
  -g '!benchmarks/3dcuebench/validate.py' \
  -g '!scripts/check_release.sh' \
  '(/mnt/bn/|/mnt/hdfs/|/mnt/petrelfs/|/mnt/shared-storage-user/|/home/[^/]+/|hdfs://|code\.byted\.org|projects/rl/|work_dirs/)' \
  data modeling train inference evaluation scripts configs benchmarks/3dcuebench \
  third_party/VLMEvalKit/vlmeval/config.py \
  third_party/VLMEvalKit/vlmeval/tools.py \
  third_party/VLMEvalKit/vlmeval/vlm/minigpt4.py; then
  echo "Internal path marker found in public source" >&2
  exit 1
fi

test -f LICENSE
test -f NOTICE
test -f THIRD_PARTY_NOTICES.md
test -f licenses/CC-BY-NC-4.0.txt
test -f licenses/GPL-3.0.txt
test -f licenses/InternVL-MIT.txt

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${PYTHON}" -m pytest -q release_tests
