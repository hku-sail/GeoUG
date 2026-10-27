#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLMEVAL_ROOT="${ROOT}/third_party/VLMEvalKit"
PYTHON="${PYTHON:-python}"
WORK_DIR="${WORK_DIR:-${ROOT}/outputs/vlmevalkit}"
WORK_DIR="$(realpath -m "${WORK_DIR}")"
MODE="${MODE:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29521}"

export PYTHONPATH="${VLMEVAL_ROOT}:${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

args=(--work-dir "${WORK_DIR}" --mode "${MODE}")
if [[ -n "${VLMEVAL_CONFIG:-}" ]]; then
  args+=(--config "$(realpath -m "${VLMEVAL_CONFIG}")")
else
  : "${MODEL:?Set MODEL to a VLMEvalKit model name}"
  : "${DATASETS:?Set DATASETS to a space-separated dataset list}"
  read -r -a model_list <<< "${MODEL}"
  read -r -a dataset_list <<< "${DATASETS}"
  args+=(--model "${model_list[@]}" --data "${dataset_list[@]}")
fi

if [[ "${REUSE:-1}" == "1" ]]; then
  args+=(--reuse)
fi

cd "${VLMEVAL_ROOT}"
if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
  exec "${PYTHON}" run.py "${args[@]}" "$@"
fi

exec "${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  run.py "${args[@]}" "$@"
