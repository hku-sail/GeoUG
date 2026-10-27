#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MODEL="${MODEL:-BAGEL-7B-MoT}"
export DATASETS="${DATASETS:-CV-Bench-3D CV-Bench-3D_circular}"
export WORK_DIR="${WORK_DIR:-${ROOT}/outputs/vlmevalkit/bagel-base-cv3d}"

exec bash "${ROOT}/evaluation/vlmevalkit/run.sh" "$@"
