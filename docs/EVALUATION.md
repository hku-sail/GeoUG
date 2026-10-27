# Evaluation

GeoUG vendors a fixed VLMEvalKit snapshot under `third_party/VLMEvalKit`.
This preserves the project-specific dataset adapters and scoring changes that
are not available in upstream VLMEvalKit. Provenance and licensing are recorded
in `third_party/VLMEvalKit/GEOUG_PROVENANCE.md`.

Public launchers for BAGEL base, arbitrary registered open-source models, and
the three-benchmark circular paper suite are documented in
`evaluation/vlmevalkit/README.md`.

## Environment

Use a separate environment from SFT and RL:

```bash
python -m venv .venv-eval
. .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

## 3DCueBench

The adapter accepts either the public Hugging Face release or a local canonical
parquet:

```bash
export THREEDCUEBENCH_PATH=/path/to/3dcuebench-v1.0.0/data/test-00000-of-00001.parquet
python third_party/VLMEvalKit/run.py \
  --data 3DCueBench \
  --model YOUR_MODEL_NAME \
  --work-dir outputs/vlmeval
```

If `THREEDCUEBENCH_PATH` is unset, the adapter downloads
`ByteDance-Seed/3DCueBench` at revision `v1.0.0`.

The historical `SpatialDepth3D` dataset name remains an alias so existing result
files can still be evaluated.

## Snapshot policy

Do not replace this directory with an arbitrary current VLMEvalKit checkout.
For the first release, commit the vendored snapshot as-is. Once the GeoUG
changes are pushed to a public VLMEvalKit fork, replace it with a submodule
pinned to the fork's full commit SHA. The pin must move only through reviewed
GeoUG changes, never through a floating branch.

Optional local-only datasets use explicit paths:

- `SAT_DATASET_PATH`
- `CVBENCH_3D_TEXTUAL_PATH`
- `DA2K_DATASET_PATH`
- `DA2K_CIRCULAR_DATASET_PATH`
- `THREEDSRBENCH_CLOSER_PATH`

These variables are not required for 3DCueBench.
