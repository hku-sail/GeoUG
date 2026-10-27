#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Change these two values for another model family.
export MODEL="${MODEL:-Qwen2.5-VL-7B-Instruct}"
COORD_STYLE="${COORD_STYLE:-qwen25}"

# Required textual benchmark files. Use absolute paths.
: "${CVBENCH_3D_TEXTUAL_PATH:?Set CVBENCH_3D_TEXTUAL_PATH}"
: "${DA2K_TEXTUAL_PARQUET:?Set DA2K_TEXTUAL_PARQUET}"
export CVBENCH_3D_TEXTUAL_PATH DA2K_TEXTUAL_PARQUET

# Styles: qwen25, qwen3, internvl, bagel, or vst.
DATASETS="3DCueBench_circular"
DATASETS+=" CVBench3DTexutal_circular_${COORD_STYLE}"
DATASETS+=" DA2KTextual_circular_${COORD_STYLE}"
export DATASETS

export WORK_DIR="${WORK_DIR:-${ROOT}/outputs/vlmevalkit/spatial-suite}"
exec bash "${ROOT}/evaluation/vlmevalkit/run.sh" \
  --judge "${JUDGE:-exact_matching}" \
  "$@"
