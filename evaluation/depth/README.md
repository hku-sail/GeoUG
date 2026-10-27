# Metric-depth evaluation

This directory contains the public BAGEL depth-generation and evaluation
pipeline. It supports one process on one GPU or one process per GPU with
`torchrun`.

Install the base environment and BAGEL's CUDA attention dependency:

```bash
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

## Dataset format

Pass a dataset JSON with `--dataset-config`. The top-level keys are benchmark
names and each value configures one benchmark:

Start from `configs/eval/depth_datasets.example.json`.

```json
{
  "example_metric_depth": {
    "path": "datasets/example",
    "split": ".index.txt",
    "width": 640,
    "height": 480,
    "depth_unit": 1.0,
    "drop_max_depth": 1000.0,
    "subset": null,
    "eigen_valid_mask": false
  }
}
```

No dataset path is built into the evaluator. Relative paths are resolved from
the current working directory. `split` is resolved relative to `path` unless
it is absolute. Each non-empty split-file line names a sample directory:

```text
scene_000/frame_000
scene_000/frame_001
```

Every sample directory must contain:

- `image.jpg`: RGB input.
- `depth.png`: 16-bit logarithmically encoded depth. PNG text metadata must
  contain `near` and `far`; encoded values 0 and 65535 mean invalid and
  infinite depth respectively.
- `meta.json`: an object containing normalized OpenCV intrinsics under
  `"intrinsics"` as a 3x3 matrix.

`depth_unit` converts decoded values to meters and marks the benchmark as
metric. Omit it or set it to `null` for relative-depth data. Set
`eigen_valid_mask` only for NYU evaluation at 640x480. `subset: N` evaluates
every Nth split entry and is intended for smoke tests.

## Usage

Run commands from the repository root. The checkpoint directory must contain
the BAGEL model configs, tokenizer, `ae.safetensors`, and either
`ema.safetensors` or `model.safetensors`.

Single GPU:

```bash
python -m evaluation.depth.eval_depth \
  --dataset-config configs/eval/my_depth_datasets.json \
  --seed 0 \
  --depth-type metric
```

Multiple GPUs:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m evaluation.depth.eval_depth \
  --dataset-config configs/eval/my_depth_datasets.json \
  --depth-type metric
```

The convenience wrapper accepts the same evaluator arguments:

```bash
NPROC_PER_NODE=8 scripts/eval_depth.sh \
  --dataset-config configs/eval/my_depth_datasets.json \
  --seed 0 \
  --depth-type metric
```

The default checkpoint is `ckpt/geoug-metric-depth`; use `--model-path` for a
different model.

The default transform is `configs/inference/metric_depth.yaml`, matching the
released metric-depth expert. Public
outputs default to `outputs/depth_eval`. Use `--save-depth` to retain NPZ
predictions and `--resume` to reuse previously saved predictions.
Each sample derives its generation seed from `--seed` and the sample filename,
so results are stable across process counts.

## Metrics

Each metric reports:

- `rel`: mean absolute relative depth error (lower is better).
- `delta1`: fraction with max(gt/pred, pred/gt) below 1.25 (higher is better).

`--depth-type affine_invariant` aligns generated depth to ground truth with a
robust scale and shift, then reports `depth_affine_invariant`.

`--depth-type metric` treats generated values as meters. On benchmarks with
`depth_unit`, it reports the direct `depth_metric` score. It also reports:

- `depth_scale_invariant`: robust scale-aligned depth.
- `depth_affine_invariant`: robust scale-and-shift-aligned depth.
- `disparity_affine_invariant`: least-squares scale-and-shift alignment in
  inverse-depth space.

Metric mode additionally records the median log ground-truth/prediction scale.
Aggregate metrics are written under `OUTPUT_DIR/metrics`, including
per-benchmark summaries, per-sample values, and `all.json`.
