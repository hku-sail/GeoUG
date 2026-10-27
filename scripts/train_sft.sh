#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
: "${MODEL_PATH:?Set MODEL_PATH to the v5 step-1000 checkpoint}"
: "${GEOUG_DATASET_MANIFEST:?Set GEOUG_DATASET_MANIFEST to your dataset manifest; see configs/sft/datasets.example.json}"

DATASET_CONFIG="${DATASET_CONFIG:-${ROOT}/configs/sft/interleave_reasoning_v6_moge3.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/sft/v6}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29516}"
WANDB_OFFLINE="${WANDB_OFFLINE:-True}"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
NUM_SHARD="${NUM_SHARD:-${WORLD_SIZE}}"
NUM_REPLICATE="${NUM_REPLICATE:-1}"

mkdir -p "${OUTPUT_DIR}"
export BAGEL_REASONING_TOKENS=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

exec "${PYTHON}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --module train.sft.train \
  --dataset_config_file "${DATASET_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume_from "${MODEL_PATH}" \
  --finetune_from_hf True \
  --auto_resume False \
  --resume_model_only True \
  --finetune_from_ema True \
  --seed_new_tokens False \
  --lr 3e-5 \
  --num_workers 1 \
  --expected_num_tokens 32768 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 32768 \
  --num_shard "${NUM_SHARD}" \
  --num_replicate "${NUM_REPLICATE}" \
  --sharding_strategy HYBRID_SHARD \
  --checkpoint_dir "${OUTPUT_DIR}" \
  --results_dir "${OUTPUT_DIR}" \
  --visual_und True \
  --visual_gen True \
  --total_steps 3000 \
  --save_every 500 \
  --warmup_steps 290 \
  --ema 0.995 \
  --freeze_vit False \
  --freeze_und False \
  --vit_cond_dropout_prob 0 \
  --vae_cond_dropout_prob 0 \
  --text_cond_dropout_prob 0 \
  --use_flex True \
  --lr_scheduler cosine \
  --ce_loss_reweighting True \
  --wandb_offline "${WANDB_OFFLINE}" \
  "$@"

