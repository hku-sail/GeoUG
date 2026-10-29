# Inference

GeoUG provides standalone inference independently of VLMEvalKit.

## Metric depth

Generate a metric depth array and an 8-bit visualization:

```bash
python -m inference.depth \
  --image /path/to/image.jpg \
  --seed 0 \
  --output-dir outputs/inference/depth
```

The `.npy` file is the model output used for quantitative work. The PNG is
percentile-normalized for visualization and must not be treated as metric data.

## Adaptive spatial reasoning

Run the released geometry-aware policy:

```bash
python -m inference.spatial \
  --image /path/to/image.jpg \
  --seed 0 \
  --question 'Which is closer, <point>120 400</point> A or <point>730 420</point> B?'
```

In `adaptive` mode, the model decides at `<DECIDE>` whether to answer directly
or generate metric depth. Depth generation is lazy: it runs only when the
depth branch is selected. The generated depth is then supplied to the bundled
geometry runtime for tool calls.

Use `--mode force_depth` and `--mode force_answer` to inspect the two policy
endpoints. These modes override the action only after the model emits `<DECIDE>`;
if it answers without that marker, `reached_decide` and `forced` remain false.
Inspect those fields together with `used_depth` before interpreting a forced run.
`--intrinsics camera.json` optionally supplies a 3x3 normalized
OpenCV intrinsic matrix:

```json
{"intrinsics": [[0.8, 0.0, 0.5], [0.0, 1.1, 0.5], [0.0, 0.0, 1.0]]}
```

Without intrinsics, the geometry runtime uses its documented centered-camera
approximation. Results are written as JSON; generated depth is saved as
float32 `.npy`.

By default, depth inference loads `ckpt/geoug-metric-depth`, while spatial
inference loads `ckpt/geoug-spatial-rl-lambda0.1`. Pass `--model-path` to
override either checkpoint. See the [model download links and local layout](../README.md#models).

The transform defaults to `transform_config.yaml` inside the selected model
directory, including when `--model-path` is overridden. Use
`--transform-config /path/to/custom.yaml` only for an intentional override.
The shell launchers in `scripts/` can be called from any working directory;
relative image, model, and output paths are resolved from the caller's directory.

Both entry points currently require CUDA and the SFT environment, including
FlashAttention where required by the BAGEL model. `--seed` controls both text
sampling and the initial depth-generation noise.
