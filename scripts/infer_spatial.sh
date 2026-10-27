#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BAGEL_REASONING_TOKENS=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

exec "${PYTHON:-python}" -m inference.spatial "$@"
