# Running GeoUG outside the research workspace

The `geoug` directory is the repository root. Copy or clone it into any directory;
no parent workspace, private Python package, or external source symlink is needed.
Use Linux, Python 3.10 or 3.11, and a CUDA GPU for model inference/training.
Model weights and datasets remain separate artifacts, supplied through explicit
local paths. Their public upload status is tracked in the [README release status](../README.md#release-status).

## Inference and SFT environment

From the new repository directory:

```bash
python -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install -r requirements.txt
python -m pip install flash_attn==2.5.8 --no-build-isolation
python -m pip install -e . --no-deps
```

FlashAttention installation needs a compatible CUDA build environment. The
editable install is optional when running modules from the repository root;
it makes those modules available from other working directories too.
`pyproject.toml` reads its dependencies from `requirements.txt`.

Put a complete model package in `ckpt/geoug-metric-depth` or
`ckpt/geoug-spatial-rl-lambda0.1`, or pass its actual directory:

```bash
python -m inference.depth \
  --model-path /path/to/geoug-metric-depth \
  --image /path/to/image.jpg --output-dir /path/to/depth-output

python -m inference.spatial \
  --model-path /path/to/geoug-spatial-rl-lambda0.1 \
  --image /path/to/image.jpg \
  --question 'Which is closer, <point>120 400</point> A or <point>730 420</point> B?'
```

Each model package supplies its own `transform_config.yaml`; no config from a
different default model is silently substituted. Missing artifacts are reported
before loading the GPU model. Spatial and standalone depth inference use
different BAGEL implementations: run them in separate Python processes.

The loader materializes model parameters from the checkpoint rather than first
allocating a random FP32 copy. The inference seed is applied after model loading,
so initialization does not consume the sampling RNG stream. Outputs for an old
seed may differ from earlier releases that seeded before random initialization.

Without an editable install, use the absolute path to `scripts/infer_depth.sh`
or `scripts/infer_spatial.sh`. These launchers preserve the caller's working
directory, so relative input/output paths keep their normal meaning. SFT, RL,
and depth-evaluation launchers behave the same way. `PYTHON` selects the interpreter.

## RL and VLMEvalKit

Use separate environments following the README's RL and evaluation installation
commands. Do not install the SFT package dependencies into the RL environment:
its Transformers and Arrow versions differ. The RL launcher sets the source
import path itself and uses the bundled training engine and geometry tools.
Explicit model/data/cache inputs are listed in `docs/RL_TRAINING.md`.

## CPU release and relocation checks

```bash
python -m pip install -r requirements-dev.txt
bash scripts/check_release.sh
python scripts/check_standalone.py
```

The relocation check creates a temporary copy of the source without weights,
datasets, outputs, virtual environments, or Git metadata. It clears `PYTHONPATH`,
runs the release tests from that copy, and checks all three inference/evaluation
CLI help commands. CI runs this check on Python 3.10 and 3.11.
To retain a checked copy, pass a new directory outside the checkout:

```bash
python scripts/check_standalone.py --destination /tmp/geoug-source-check
```

`scripts/check_portability.py` rejects broken symlinks and links pointing outside
the source tree. Regression tests cover model/config selection, final-answer
parsing, RL package selection, and launcher working-directory behavior.

These CPU checks do not prove model quality or training convergence. Before a
public release, run real GPU inference and the benchmark using published, fixed
model/data revisions; the legacy paper scores are not current-protocol validation.
