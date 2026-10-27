#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-spatial_gate_draft_v3}"
LAMBDA_COST="${LAMBDA_COST:-0.1}"

required=(MODEL_PATH RL_DATASET DEPTH_CACHE_DIR EVAL_QUESTIONS EVAL_DEPTH_CACHE)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29517}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

if (( WORLD_SIZE < 2 )); then
  echo "Warning: one process is only a wiring smoke test; GRPO needs a group." >&2
fi

FORCE_DEPTH_PER_GROUP="${FORCE_DEPTH_PER_GROUP:-4}"
if (( FORCE_DEPTH_PER_GROUP >= WORLD_SIZE )); then
  FORCE_DEPTH_PER_GROUP=1
fi

RUN_NAME="${RUN_NAME:-${CONFIG}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/rl/${RUN_NAME}}"
mkdir -p "${OUTPUT_DIR}"

export BAGEL_REASONING_TOKENS=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

exec "${PYTHON}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --module train.rl.spatial_train \
  --config "${ROOT}/train/rl/config_spatial.py:${CONFIG}" \
  --config.num_gpus="${WORLD_SIZE}" \
  --config.sample.num_image_per_prompt="${WORLD_SIZE}" \
  --config.train.force_depth_per_group="${FORCE_DEPTH_PER_GROUP}" \
  --config.train.lambda_cost="${LAMBDA_COST}" \
  --config.logdir="${OUTPUT_DIR}/logs" \
  --config.save_dir="${OUTPUT_DIR}/checkpoints" \
  "$@"

