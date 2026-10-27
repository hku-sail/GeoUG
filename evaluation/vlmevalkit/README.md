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

### Reproduce DA-2K Textual and Visual

Both variants are generated locally from the upstream DA-2K images and
`annotations.json`; a GeoUG-hosted Visual TSV download is not required.
Run the following commands from the GeoUG root after installing the evaluation
environment. Set the source directory explicitly:

```bash
export DA2K_ROOT=/absolute/path/to/DA-2K
```

The directory must contain `annotations.json` and the image paths referenced
by its keys (typically `images/<category>/<filename>`).

**Textual:** keep the images unmarked and put point coordinates in the question.

```bash
python scripts/prepare_da2k_textual.py \
  --root "$DA2K_ROOT" --out_dir outputs/da2k_textual --seed 0
```

This produces `outputs/da2k_textual/questions.parquet` with 2,068 questions
for the full upstream annotations. Source points are `[row, col]` pixels;
the exported points are `[x, y]`, normalized independently to 0–1000 on each
axis. Seed 0 randomizes the base A/B order reproducibly, without guaranteeing
an exactly balanced answer count. Keep the original images accessible, or
optionally embed them in a portable parquet:

```bash
python scripts/pack_data_parquet.py textual \
  --input outputs/da2k_textual/questions.parquet \
  --output outputs/da2k_textual/DA2KTextual.parquet

DA2K_TEXTUAL_PARQUET="$PWD/outputs/da2k_textual/DA2KTextual.parquet" \
MODEL=Qwen2.5-VL-7B-Instruct \
DATASETS=DA2KTextual_circular_qwen25 \
WORK_DIR=outputs/vlmevalkit/da2k_textual \
bash evaluation/vlmevalkit/run.sh --judge exact_matching
```

Choose the coordinate-style suffix for your model using the table above.

**Visual:** draw green A / red B markers and encode the marked images in TSV.

```bash
# Legacy closer/farther protocol: fixed marker colours, two rows per pair.
python scripts/prepare_da2k.py \
  --datadir "$DA2K_ROOT" --axes polarity --num-workers 4 \
  --output outputs/da2k/DA2KDataset.tsv

# Colour-swap circular protocol: swap marker positions, two rows per pair.
python scripts/prepare_da2k.py \
  --datadir "$DA2K_ROOT" --axes colour --num-workers 4 \
  --output outputs/da2k/DA2KDataset_circular.tsv

DA2K_DATASET_PATH="$PWD/outputs/da2k/DA2KDataset.tsv" \
DA2K_CIRCULAR_DATASET_PATH="$PWD/outputs/da2k/DA2KDataset_circular.tsv" \
MODEL=Qwen2.5-VL-7B-Instruct \
DATASETS="DA2KDataset DA2KDataset_circular" \
WORK_DIR=outputs/vlmevalkit/da2k_visual \
bash evaluation/vlmevalkit/run.sh --judge exact_matching
```

Each Visual TSV contains 4,136 rendered rows for the full 2,068 point pairs.
`--axes colour+polarity` generates all four combinations (8,272 rows) and
constitutes a different protocol. Do not mix its scores with colour-only results.
Original DA-2K annotations label point1 as closer, so fixed-colour closer/farther
questions admit a colour/wording shortcut; colour swapping is needed to test
both assignments. Circular accuracy requires both permutations to be correct.

For reproducibility, keep the rendering defaults: `--marker-radius-scale 10`,
`--marker-outline-scale 3`, and `--marker-font-scale 0.9`. TSVs embed full-size
PNG images as base64 and can be about 21 GiB each. The current generator collects
rows in memory; use a host with sufficient RAM and disk. A small preview can be
generated with `--max-images 2 --debug --num-workers 1` before a full run.

### Environment variables

- `THREEDCUEBENCH_PATH`: canonical parquet with embedded or portable images.
  If unset, the adapter downloads the pinned public dataset revision.
- `CVBENCH_3D_TEXTUAL_PATH`: textual-bbox TSV.
- `DA2K_TEXTUAL_PARQUET`: textual-point parquet.
- `DA2K_DATASET_PATH`: optional visual-marker DA-2K TSV.
- `DA2K_CIRCULAR_DATASET_PATH`: colour-swapped visual-marker TSV.

`CVBENCH_3D_TEXTUAL_PATH` and `DA2K_TEXTUAL_PARQUET` are required only for the
textual paper suite. The public visual CV-Bench split uses its normal
VLMEvalKit source.

## Metrics

Ordinary `Overall` is per rendered row. `Overall_circular` credits a question
only when every tested option ordering is correct. Use circular accuracy for
the paper's main comparison.

Outputs are written below `WORK_DIR/<model>/<evaluation-id>/`, including raw
predictions, scored rows, and `*_acc.csv`.
