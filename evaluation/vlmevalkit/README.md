# VLMEvalKit evaluation

GeoUG vendors a fixed VLMEvalKit snapshot so the project-specific spatial
datasets, coordinate rendering, circular scoring, and BAGEL adapter are
reproducible.

Install the evaluation environment, then run commands from the GeoUG root:

```bash
pip install 'setuptools==69.5.1'
pip install -r requirements-eval.txt
```

The setuptools pin is required by VLMEvalKit's legacy dependencies, including
`timeout-decorator` and `openai-clip`.

## BAGEL base model on CV-Bench-3D

The visual CV-Bench split is downloaded by VLMEvalKit:

```bash
NPROC_PER_NODE=8 \
bash evaluation/vlmevalkit/run_bagel_base_cv3d.sh
```

This evaluates both:

- `CV-Bench-3D`: ordinary VLMEvalKit accuracy;
- `CV-Bench-3D_circular`: every binary question under both option orders.

Override `MODEL` to use another registered model.

## Open-source models on the paper suite

The paper suite is:

```text
3DCueBench_circular
CVBench3DTexutal_circular_<coordinate-style>
DA2KTextual_circular_<coordinate-style>
```

Example:

```bash
MODEL=Qwen2.5-VL-7B-Instruct \
NPROC_PER_NODE=8 \
THREEDCUEBENCH_PATH=/path/to/3dcuebench.parquet \
CVBENCH_3D_TEXTUAL_PATH=/path/to/CV-Bench-3D-textual.tsv \
DA2K_TEXTUAL_PARQUET=/path/to/da2k-textual.parquet \
bash evaluation/vlmevalkit/run_spatial_suite.sh
```

Supported built-in examples include:

- `BAGEL-7B-MoT`
- `Qwen2.5-VL-7B-Instruct`
- `Qwen3-VL-8B-Instruct`
- `InternVL3_5-8B`

Run `vlmutil mlist all` in the evaluation environment for the full list.

The paper-suite launcher defaults to `JUDGE=exact_matching`, so answer
extraction does not depend on an API key. Override `JUDGE` only for an
intentional alternate protocol. Any model/dataset failure makes the launcher
exit non-zero after reporting all failed combinations.

## Coordinate styles

Textual coordinates must match the model family's training convention.
`COORD_STYLE` defaults to `qwen25`; set it when evaluating another family:

| Style | Model family | Coordinate convention |
|---|---|---|
| `bagel` | BAGEL | 0–1000 per axis with point/bbox tags |
| `qwen25` | Qwen2.5-VL | pixels after model smart-resize |
| `qwen3` | Qwen3-VL | 0–1000 per axis |
| `internvl` | InternVL / SenseNova-SI | 0–1000 per axis |
| `vst` | VST | pixels after VST resize |

For example:

```bash
MODEL=InternVL3_5-8B COORD_STYLE=internvl \
CVBENCH_3D_TEXTUAL_PATH=/abs/path/cv.tsv \
DA2K_TEXTUAL_PARQUET=/abs/path/da2k.parquet \
bash evaluation/vlmevalkit/run_spatial_suite.sh
```

## Generic launcher

Evaluate any registered model/dataset combination:

```bash
MODEL="Qwen3-VL-8B-Instruct" \
DATASETS="3DCueBench CV-Bench-3D DA2KDataset" \
WORK_DIR=outputs/vlmevalkit/qwen3 \
NPROC_PER_NODE=8 \
bash evaluation/vlmevalkit/run.sh
```

Set `MODE=infer` or `MODE=eval` to run only one phase. `REUSE=1` is the
default and resumes compatible existing outputs.

For a local checkpoint or a custom model class, copy
`configs/eval/vlmevalkit.local.example.json`, replace its paths, then run:

```bash
VLMEVAL_CONFIG=/path/to/eval.json \
bash evaluation/vlmevalkit/run.sh
```

## Dataset paths

- `THREEDCUEBENCH_PATH`: canonical parquet with embedded or portable images.
  If unset, the adapter downloads the pinned public dataset revision.
- `CVBENCH_3D_TEXTUAL_PATH`: textual-bbox TSV.
- `DA2K_TEXTUAL_PARQUET`: textual-point parquet.
- `DA2K_DATASET_PATH`: optional visual-marker DA-2K TSV.

`CVBENCH_3D_TEXTUAL_PATH` and `DA2K_TEXTUAL_PARQUET` are required only for the
textual paper suite. The public visual CV-Bench split uses its normal
VLMEvalKit source.

## Metrics

Ordinary `Overall` is per rendered row. `Overall_circular` credits a question
only when every tested option ordering is correct. Use circular accuracy for
the paper's main comparison.

Outputs are written below `WORK_DIR/<model>/<evaluation-id>/`, including raw
predictions, scored rows, and `*_acc.csv`.
