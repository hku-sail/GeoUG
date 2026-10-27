# RL Training

[Training data card and setup](TRAINING.md)

The public RL recipe starts from GeoUG-ColdStart and produces the adaptive
spatial reasoning policy:

```text
GeoUG-ColdStart
  -> spatial-policy RL
  -> merge the LoRA adapter into the initialization
  -> GeoUG-Spatial
```

## Method

The policy chooses between answering directly and generating a depth map. For each rollout:

```text
reward = answer_correct - lambda_cost * used_depth
```

The paper configuration uses `lambda_cost=0.1`. `scripts/train_rl.sh` sets this
through `LAMBDA_COST=0.1` by default; setting it to zero is an explicit
always-depth smoke baseline, not the released policy.

The final `spatial_gate_draft_v3` configuration differs from ordinary trajectory-level GRPO:

- the draft answer segment receives zero advantage;
- grounding receives a depth-branch advantage without the generation-cost term;
- the gate uses a two-branch counterfactual advantage;
- part of every group is forced through the depth branch to estimate its value;
- loss from multiple generated text segments is weighted by token count;
- empty loss segments are padded with zero-gradient backward calls so all distributed workers perform the same number of collectives.

The default optimization path uses LoRA rank 32, GSPO sequence-level importance sampling, and one rollout per process. A GRPO group is formed across all processes, so:

```text
group_size = world_size
```

Meaningful training therefore requires multiple GPUs. A one-GPU run is only a wiring smoke test.

## Source map

The public RL package is under `train/rl/`:

- `spatial_train.py`: integration entry point;
- `config_spatial.py`: final and smoke-test configurations;
- `spatial_dataset.py`: Arrow-backed parquet dataset;
- `spatial_inferencer.py`: depth-aware rollout generation;
- `spatial_task.py`: rollout, reward, metrics, and online evaluation;
- `spatial_draft.py`: segmented and two-branch advantages;
- `spatial_algorithm.py`: padded backward and token-weighted loss;
- `spatial_advantage.py`: fixed-scale advantage normalization;
- `spatial_eval.py`: cached-depth online evaluation;
- `reward.py`: deterministic multiple-choice verifier;
- `merge_lora.py`: merge a trained adapter into the base checkpoint.

`train/rl/data_pipeline/` contains portable data conversion and sampling tools.
Depth-cache generation used the paper training infrastructure and is not part
of the public runtime; the generated training caches are embedded in
[train_geoug_rl.parquet](https://huggingface.co/datasets/rayruiyang/GeoUG_data/blob/main/train_geoug_rl.parquet). Offline probes and experiment scheduling scripts are intentionally
not part of the runtime package.

## Required dataset layout

The public training download is a **single parquet** with 800 training rows.
It preserves the original question and image columns and adds `camera`
(`intrinsics`, `image_wh`) and `depth_npy` (the original NumPy file bytes).
Read a depth array with `np.load(io.BytesIO(row["depth_npy"]), allow_pickle=False)`.
No separate camera file or depth-cache download is needed.

To restore the directory layout used by the training runtime:

```bash
python scripts/pack_data_parquet.py unpack-rl \
  --input /path/to/geoug-spatial-rl/train_geoug_rl.parquet \
  --output /path/to/rl_train
```

Set `RL_DATASET=/path/to/rl_train` and
`DEPTH_CACHE_DIR=/path/to/rl_train/depth-cache`. This restores only the public
training split. Prepare your own validation/evaluation data for training;
the internal evaluation subset is not included in the release.

`RL_DATASET` points to:

```text
rl_train/
├── train.parquet
├── validate.parquet
└── camera_table.json
```

Required parquet columns:

```text
id
image
question
answer
options
category
source
image_key
```

`image` contains encoded image bytes. `camera_table.json` maps every `image_key` to:

```json
{
  "intrinsics": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "image_wh": [0, 0]
}
```

The actual values must come from the source dataset. Missing intrinsics are treated as an error rather than silently replaced with an approximation.

The unpacked depth cache directory contains the generated depth representation
expected by `spatial_inferencer.py`.

## Bundled training engine

The project-owned engine required by GeoUG is maintained under
`train/rl/engine`: the
optimizer loop, generation replay, GRPO base algorithm, task and dataset
registries, BAGEL RL hooks, DeepSpeed configurations, and checkpointing. The
engine is released under Apache-2.0 and has no external runtime checkout.

## Bundled geometry tool runtime

Depth rollouts execute the same geometry tools used to generate the SFT traces. The runtime must provide `SelfContainedToolSession` with the v3 inline-coordinate protocol.

The parity-critical runtime, geometry math, and result formatting are bundled
under `train/rl/geometry_tools`. Replacing them with a superficially similar
implementation would change the tool-result text and therefore the policy
trajectory. `GEOMETRY_TOOL_ROOT` remains available only as an explicit
development override.

## Launch contract

The public launcher uses environment variables rather than cluster-specific paths:

```bash
MODEL_PATH=/path/to/geoug-coldstart \
RL_DATASET=/path/to/rl_train \
DEPTH_CACHE_DIR=/path/to/depth_cache \
EVAL_QUESTIONS=/path/to/d3-eval.parquet \
EVAL_DEPTH_CACHE=/path/to/d3-depth-cache \
LAMBDA_COST=0.1 \
NPROC_PER_NODE=8 \
bash scripts/train_rl.sh
```

Multi-node launch additionally sets standard torchrun variables:

```text
NNODES
NODE_RANK
MASTER_ADDR
MASTER_PORT
```

The paper recipe used 16 GPUs. Model initialization and the public RL training
parquet are described in the [training guide](TRAINING.md). Validation/evaluation
data must be prepared separately.
