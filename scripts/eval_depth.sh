#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
  exec "${PYTHON}" -m evaluation.depth.eval_depth "$@"
fi

exec "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NPROC_PER_NODE}" \
  -m evaluation.depth.eval_depth \
  "$@"
