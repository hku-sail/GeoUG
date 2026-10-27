<h1 align="center">GeoUG: Learning and Reasoning with Geometry — How Generation Helps Spatial Understanding?</h1>

<p align="center">
  <a href="https://yangr116.github.io/geoug-project-page/"><img src="https://img.shields.io/badge/Project-Page-2563eb" alt="Project Page"></a>
  <a href="https://yangr116.github.io/geoug-project-page/assets/geoug-paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b" alt="Paper PDF"></a>
  <a href="#model-card"><img src="https://img.shields.io/badge/Model-Card-f2c94c" alt="Model Card"></a>
  <a href="#data-card"><img src="https://img.shields.io/badge/Data-Card-22a699" alt="Data Card"></a>
</p>

<p align="center">
  <b>Generate geometry. Measure in 3D. Rethink the answer.</b><br>
  A unified model for geometric generation and adaptive spatial reasoning.
</p>

<p align="center">
  <a href="https://yangr116.github.io/geoug-project-page/assets/video/geoug-direct-fusion-hd.mp4">
    <img src="docs/assets/teaser.gif" alt="GeoUG project-page teaser: depth generation, 3D measurement and answer revision, generated first-person view, and illustrative two-view fusion" width="100%">
  </a>
</p>
<p align="center">
  <a href="#highlights">Highlights</a> ·
  <a href="#see-geoug-in-action">Demos</a> ·
  <a href="#model-card">Models</a> ·
  <a href="#data-card">Datasets</a> ·
  <a href="#main-results">Results</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#training">Training</a>
</p>

## Highlights

- **Learn with geometry.** Geometric co-training uses metric depth and view
  generation to improve spatial understanding.
- **Reason with geometry.** The model drafts an answer, decides whether depth
  is useful, and uses generated geometry and measurement tools to confirm or
  revise its answer. The paper's RL policy skips depth generation on
  27.6–49.7% of questions across three reasoning benchmarks.
- **Multi-scale Metric Depth Codec (MMDC).** Encode metric depth at 2 m, 8 m,
  and 32 m pivots into the three channels of an RGB VAE.
- **3DCueBench.** 800 balanced depth and 3D-distance questions, with controlled
  shortcut agreement and four difficulty intervals.

## See GeoUG in action

### Image → metric depth

Generate depth in meters with MMDC.

<p align="center">
  <a href="https://yangr116.github.io/geoug-project-page/assets/video/gen-depth.mp4"><img src="docs/assets/gen-depth.gif" alt="Input image, generated metric depth, and raw three-channel MMDC output" width="720"></a>
</p>

[▶ Watch demo](https://yangr116.github.io/geoug-project-page/assets/video/gen-depth.mp4)

### Explore another viewpoint

Geometric generation across views.

<p align="center">
  <a href="https://yangr116.github.io/geoug-project-page/assets/video/gen-multiview.mp4"><img src="docs/assets/gen-multiview.gif" alt="GeoUG multi-view generation demonstration" width="720"></a>
</p>

[▶ Watch demo](https://yangr116.github.io/geoug-project-page/assets/video/gen-multiview.mp4)

### See through another perspective

Third-person → first-person generation.

<p align="center">
  <a href="https://yangr116.github.io/geoug-project-page/assets/video/gen-exo-ego.mp4"><img src="docs/assets/gen-exo-ego.gif" alt="Third-person input and generated first-person view" width="720"></a>
</p>

[▶ Watch demo](https://yangr116.github.io/geoug-project-page/assets/video/gen-exo-ego.mp4)

These research demos use GeoUG task configurations; they do not imply that all
capabilities are available in each planned inference checkpoint. Video transitions
are presentation effects, not denoising trajectories.

<details>
<summary><b>More evidence: spatial reasoning on 3DCueBench</b></summary>

<img src="docs/assets/qual_d3_depth.png" alt="Grounded objects, generated metric depth, and a top-view point cloud on 3DCueBench" width="100%">

Explore [answer corrections, direct answers, and failure cases](https://yangr116.github.io/geoug-project-page/#reasoning-cases).

</details>

## Release status

Public model weights and GeoUG data packages are **coming soon**. Download links
will be added here as versioned artifacts become available. The usage guides below
describe the planned code release; model weights and datasets are separate downloads.

## Model Card

The planned model release includes two inference models and one SFT checkpoint
for initializing spatial-policy RL training.

| Model | Download | Notes / capabilities | Documentation |
|---|---|---|---|
| **GeoUG Spatial RL (λ=0.1)** | [Coming soon][model-spatial-rl] | Adaptive spatial reasoning: draft an answer, optionally generate metric depth, call geometry tools, and revise the answer. Final merged EMA policy. | [Inference guide](inference/README.md) |
| **GeoUG Metric Depth** | [Coming soon][model-metric-depth] | Standalone metric-depth generation with MMDC; outputs depth in meters for measurement and 3D reconstruction. | [Inference guide](inference/README.md) |
| **GeoUG v6 · SFT step 1500** | [Coming soon][model-sft-init] | SFT initialization for spatial-policy RL training. | [RL recipe](docs/RL_TRAINING.md) |

**Local use.** Place the two inference packages in
`ckpt/geoug-spatial-rl-lambda0.1/` and `ckpt/geoug-metric-depth/`, respectively,
or pass `--model-path /path/to/model`. Each package contains model configs,
tokenizer files, `transform_config.yaml`, `ae.safetensors`, `ema.safetensors`,
and SHA256 checksums. Public package download links will be added above.

**Download placeholders.** All **Coming soon** links in the model and data
tables currently point to the release status above. They will be replaced with
versioned Hugging Face download URLs as each artifact is published.


## Data Card

### Evaluation datasets

Textual variants identify objects with coordinates in the question; visual
variants mark the objects or points on the image. Keep the variant and
ordinary/circular scoring protocol fixed when comparing models.

| Dataset | Format / task | Download | Notes / protocol |
|---|---|---|---|
| **3DCueBench** | Image-embedded parquet; depth ordering and 3D-distance ordering | [Coming soon][data-3dcuebench] | 800 questions on 757 images; 400 depth + 400 distance questions, four difficulty bins, balanced A/B answers. [Data card and validation](benchmarks/3dcuebench/README.md). |
| **DA-2K Textual** | Parquet; relative depth with textual point coordinates | [Coming soon][data-da2k-textual] | Set `DA2K_TEXTUAL_PARQUET`; model-specific coordinate rendering and circular evaluation are supported. [Evaluation guide](evaluation/vlmevalkit/README.md). |
| **DA-2K Visual** | TSV; relative depth with colored point markers | [Coming soon][data-da2k-visual] | Set `DA2K_DATASET_PATH`; circular evaluation uses a separate marker-swapped TSV via `DA2K_CIRCULAR_DATASET_PATH`. [Evaluation guide](evaluation/vlmevalkit/README.md). |
| **CV-Bench-3D Textual** | TSV; depth and 3D-distance questions with textual boxes | [Coming soon][data-cvbench-textual] | Set `CVBENCH_3D_TEXTUAL_PATH`. Includes the circular paper-suite protocol. [Evaluation guide](evaluation/vlmevalkit/README.md). |
| **CV-Bench-3D Visual** | Images with visual annotations; depth and 3D-distance questions | [Via VLMEvalKit](evaluation/vlmevalkit/README.md#bagel-base-model-on-cv-bench-3d) | The bundled evaluator downloads the upstream visual split; no separate GeoUG download is needed. |
| **Metric-depth evaluation suite** | NYU-v2, KITTI, ETH3D, and other configured benchmarks | [Dataset setup](evaluation/depth/README.md) | Obtain the source datasets separately and configure their local paths. [Config template](configs/eval/depth_datasets.example.json). |

### Training datasets and supporting artifacts

| Dataset / artifact | Stage | Download | Notes / contents |
|---|---|---|---|
| **GeoUG cold start data** | SFT | [Coming soon][data-cold-start] | COCO- and ADE20K-derived interleaved spatial reasoning samples. [Manifest format](configs/sft/datasets.example.json). |
| **GeoUG Spatial RL data** | RL | [Coming soon][data-rl] | Spatial-policy training data, camera metadata, and generated-depth caches packaged together, with manifests and checksums. [RL guide](docs/RL_TRAINING.md). |

Dataset repositories will provide the final revisions, schemas, preparation
instructions, and applicable source-dataset terms when published. The tables
above describe the planned release; they do not imply that upstream raw data
will be redistributed with GeoUG.

## Main results

Circular accuracy requires both answer-option orders to be correct.

| Model | CV-Bench-3D | 3DCueBench | DA-2K | Average |
|---|---:|---:|---:|---:|
| GeoUG (RL policy, λ=0.1) | **97.9** | **90.8** | **94.2** | **94.3** |

These are **legacy-paper protocol** results. The current dataset-owned prompt
router has only been rerun for the text-only endpoint; an adaptive-policy
rerun is still required before claiming reproduction with the current code.
The table reports paper results, not a verified reproduction using the current release.

The standalone GeoUG-Depth model obtains affine-aligned AbsRel / δ1 of
3.32 / 98.46 on NYU-v2, 4.84 / 97.27 on KITTI, and 3.75 / 97.85 on ETH3D.
See the [Project Page](https://yangr116.github.io/geoug-project-page/) for the
full tables, visualizations, and reasoning traces.

## Installation

SFT, RL, and VLMEvalKit use separately tested dependency sets. Python 3.10 or
3.11 is required.

```bash
# SFT, standalone inference, and depth evaluation
python -m venv .venv-sft
source .venv-sft/bin/activate
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

Run these commands from the `geoug` checkout. Its source is self-contained:
the parent research workspace is not required. For use outside the checkout,
also run `pip install -e . --no-deps` in this environment. Package installation
and `requirements.txt` share the same dependency list. CUDA/FlashAttention
must be installed separately as above. See the
[standalone setup and checks](docs/STANDALONE.md).

```bash
# RL
python -m venv .venv-rl
source .venv-rl/bin/activate
pip install -r requirements-rl.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

```bash
# VLMEvalKit
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install 'setuptools==69.5.1'
pip install -r requirements-eval.txt
```

## Inference

The commands below use the default local model directories listed in the
[Model Card](#model-card). Pass `--model-path` to use a different location.

Generate metric depth:

```bash
python -m inference.depth \
  --image /path/to/image.jpg \
  --output-dir outputs/depth
```

Run adaptive spatial reasoning:

```bash
python -m inference.spatial \
  --image /path/to/image.jpg \
  --question 'Which object is closer to the camera?'
```

Use `--mode force_depth` or `--mode force_answer` to inspect the two policy
endpoints. See `inference/README.md` for intrinsic-matrix input and output
formats.

## Evaluation

### Open-source VLMs and BAGEL base

VLMEvalKit launchers are included under `evaluation/vlmevalkit/`.

```bash
# BAGEL base on visual CV-Bench-3D, ordinary and circular
NPROC_PER_NODE=8 \
bash evaluation/vlmevalkit/run_bagel_base_cv3d.sh

# Any registered open-source model on the three textual paper benchmarks
MODEL=Qwen2.5-VL-7B-Instruct \
THREEDCUEBENCH_PATH=/path/to/3dcuebench.parquet \
CVBENCH_3D_TEXTUAL_PATH=/path/to/cvbench3d-textual.tsv \
DA2K_TEXTUAL_PARQUET=/path/to/da2k-textual.parquet \
NPROC_PER_NODE=8 \
bash evaluation/vlmevalkit/run_spatial_suite.sh
```

Set `COORD_STYLE` for non-Qwen2.5 models. The launcher reports circular
accuracy for 3DCueBench, CV-Bench-3D Textual, and DA-2K Textual. See
`evaluation/vlmevalkit/README.md`. The paper suite uses deterministic exact
matching by default; set `JUDGE` only when intentionally testing another
answer extractor.

### Metric depth

Start from `configs/eval/depth_datasets.example.json`, then run:

```bash
bash scripts/eval_depth.sh \
  --dataset-config /path/to/depth_datasets.json \
  --depth-type metric
```

The evaluator reports metric, scale-invariant, affine-invariant, and
disparity-aligned depth metrics. Details are in `evaluation/depth/README.md`.

### 3DCueBench

Validate a release parquet:

```bash
python benchmarks/3dcuebench/validate.py \
  /path/to/3dcuebench-v1.0.0.parquet
```

Score predictions containing `question_id` and `prediction`:

```bash
python benchmarks/3dcuebench/evaluate.py \
  --dataset /path/to/3dcuebench-v1.0.0.parquet \
  --predictions /path/to/predictions.jsonl \
  --output outputs/3dcuebench/metrics.json
```

The local release candidate contains 800 questions on 757 images and is staged
for publication as an image-embedded parquet.

## Training

### SFT

Copy `configs/sft/datasets.example.json`, replace all dataset and parquet-index
paths, and launch:

```bash
MODEL_PATH=/path/to/v5-step-1000 \
GEOUG_DATASET_MANIFEST=/path/to/datasets.json \
NNODES=2 \
NPROC_PER_NODE=8 \
bash scripts/train_sft.sh
```

`NUM_SHARD` defaults to `NNODES × NPROC_PER_NODE`, so the launcher works for
both single-node smoke tests and the 16-GPU paper recipe. Full-parameter SFT
does not fit on 4×A100-80GB with AdamW and EMA; the released recipe was trained
with 16 GPUs. A memory-bounded 4-GPU smoke test can freeze the large backbones:

```bash
bash scripts/train_sft.sh \
  --total_steps 1 \
  --freeze_llm True \
  --freeze_vit True
```

Paper-exact reproduction starts from the v5 step-1000 checkpoint. That
initialization and the public SFT data manifests must be published separately;
using either final checkpoint as the initialization is only a wiring test.
See `train/sft/README.md`.

### RL

RL requires the SFT-1500 initialization, portable train/validation parquets,
camera metadata, generated-depth caches, and an evaluation subset:

```bash
MODEL_PATH=/path/to/v6-sft-1500 \
RL_DATASET=/path/to/rl_train \
DEPTH_CACHE_DIR=/path/to/train_depth_cache \
EVAL_QUESTIONS=/path/to/eval_questions.parquet \
EVAL_DEPTH_CACHE=/path/to/eval_depth_cache \
LAMBDA_COST=0.1 \
NPROC_PER_NODE=8 \
bash scripts/train_rl.sh
```

The final method is `spatial_gate_draft_v3`: draft tokens receive zero
advantage, grounding uses depth-branch credit, and the gate uses a forced
two-branch counterfactual estimate. See `docs/RL_TRAINING.md`.

## Repository layout

```text
configs/                 SFT, RL, inference, and evaluation configs
data/                    Dataset readers, packing, transforms, and MMDC
modeling/                BAGEL-based unified model
train/sft/               FSDP SFT implementation
train/rl/                GeoUG RL implementation and bundled training engine
inference/               Metric-depth and adaptive spatial inference
evaluation/depth/        Metric-depth evaluation
evaluation/vlmevalkit/   Open-model and circular spatial evaluation
benchmarks/3dcuebench/   3DCueBench export, validation, and scoring
third_party/VLMEvalKit/  Fixed evaluation snapshot with GeoUG adapters
ckpt/                    Local model packages and upload instructions
scripts/                 Public launchers
```

<!-- Replace each placeholder independently with the published, versioned download URL. -->
[model-spatial-rl]: #release-status
[model-metric-depth]: #release-status
[model-sft-init]: #release-status
[data-3dcuebench]: #release-status
[data-da2k-textual]: #release-status
[data-da2k-visual]: #release-status
[data-cvbench-textual]: #release-status
[data-cold-start]: #release-status
[data-rl]: #release-status
