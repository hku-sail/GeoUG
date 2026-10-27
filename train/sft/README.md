# SFT training

The supported Python entry point is:

```bash
python -m train.sft.train --help
```

For distributed training, use `scripts/train_sft.sh`. It invokes the same
module through `torch.distributed.run` and supplies the released SFT recipe.

Required inputs:

```bash
MODEL_PATH=/path/to/v5-step-1000
GEOUG_DATASET_MANIFEST=/path/to/datasets.json
```

`MODEL_PATH` must contain the BAGEL configs, tokenizer, `ae.safetensors`, and
`ema.safetensors`. The dataset manifest format is documented by
`configs/sft/datasets.example.json`.

The paper recipe uses 16 GPUs:

```bash
NNODES=2 NPROC_PER_NODE=8 bash scripts/train_sft.sh
```

`NUM_SHARD` defaults to the total world size. WandB defaults to offline mode;
set `WANDB_OFFLINE=False` only after configuring an account.

Full-parameter AdamW plus EMA exceeded 4×A100-80GB in validation. For a
memory-bounded wiring test that still runs a real forward, backward, and
optimizer step, freeze the two large backbones:

```bash
bash scripts/train_sft.sh \
  --total_steps 1 \
  --freeze_llm True \
  --freeze_vit True
```

## Files

- `train.py`: argument definitions, model/data setup, and the SFT loop.
- `fsdp_utils.py`: FSDP wrapping, checkpointing, and EMA updates.
- `train_utils.py`: logging, checkpoint discovery, and diagnostic records.
- `model_utils.py`: model-parameter reporting helpers.

`train/pretrain_unified_navit.py` is a compatibility wrapper for the former
file-based entry point. The SFT implementation is maintained only in this
package. RL remains independently implemented under `train/rl/`.
