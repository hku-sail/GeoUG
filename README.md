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

**Available now:** GeoUG-Spatial, GeoUG-Depth, GeoUG-ColdStart, 3DCueBench,
CV-Bench-3D Textual, and GeoUG Spatial RL training data. Download links are in
the model and data cards below. DA-2K Textual / Visual can be reproduced locally.
GeoUG cold start **data** is coming soon; the ColdStart **model** is already available.

## Model Card

The model release includes two inference models and one SFT checkpoint
for initializing spatial-policy RL training.

| Model | Download | Notes / capabilities | Documentation |
|---|---|---|---|
| **GeoUG-Spatial** | [Hugging Face][model-spatial-rl] | Adaptive spatial reasoning: draft an answer, optionally generate metric depth, call geometry tools, and revise the answer. Final merged EMA policy (λ=0.1). | [Inference guide](inference/README.md) |
| **GeoUG-Depth** | [Hugging Face][model-metric-depth] | Standalone metric-depth generation with MMDC; outputs depth in meters for measurement and 3D reconstruction. | [Inference guide](inference/README.md) |
| **GeoUG-ColdStart** | [Hugging Face][model-sft-init] | v6 SFT step 1500; initialization for spatial-policy RL training. | [RL recipe](docs/RL_TRAINING.md) |

**Local use.** Place the two inference packages in
`ckpt/geoug-spatial-rl-lambda0.1/` and `ckpt/geoug-metric-depth/`, respectively,
or pass `--model-path /path/to/model`. Each package contains model configs,
tokenizer files, `transform_config.yaml`, `ae.safetensors`, `ema.safetensors`,
and SHA256 checksums. Download the complete package from the model links above.

**Download availability.** The three model packages are available on Hugging Face.
See the [training data card](docs/TRAINING.md#training-data-card) for training-data availability.


## Data Card

### Evaluation datasets

Textual variants identify objects with coordinates in the question; visual
variants mark the objects or points on the image. Keep the variant and
ordinary/circular scoring protocol fixed when comparing models.

| Dataset | Format / task | Download | Notes / protocol |
|---|---|---|---|
| **3DCueBench** | Image-embedded parquet; depth ordering and 3D-distance ordering | [Hugging Face][data-3dcuebench] | One self-contained parquet: 800 questions on 757 images; 400 depth + 400 distance questions, four difficulty bins, balanced A/B answers. [Data card and validation](benchmarks/3dcuebench/README.md). |
| **DA-2K Textual** | Single image-embedded parquet; relative depth with textual point coordinates | [Reproduce locally][data-da2k-textual] | 2,068 questions. Set `DA2K_TEXTUAL_PARQUET`; model-specific coordinate rendering and circular evaluation are supported. [Evaluation guide](evaluation/vlmevalkit/README.md). |
| **DA-2K Visual** | TSV; relative depth with colored point markers | [Reproduce locally][data-da2k-visual] | Set `DA2K_DATASET_PATH`; circular evaluation uses a separate marker-swapped TSV via `DA2K_CIRCULAR_DATASET_PATH`. [Evaluation guide](evaluation/vlmevalkit/README.md). |
| **CV-Bench-3D Textual** | TSV; depth and 3D-distance questions with textual boxes | [Hugging Face][data-cvbench-textual] | Download `CVBench3DTexutal.tsv` and set `CVBENCH_3D_TEXTUAL_PATH`. Includes the circular paper-suite protocol. [Evaluation guide](evaluation/vlmevalkit/README.md). |
| **CV-Bench-3D Visual** | Images with visual annotations; depth and 3D-distance questions | [Via VLMEvalKit](evaluation/vlmevalkit/README.md#bagel-base-model-on-cv-bench-3d) | The bundled evaluator downloads the upstream visual split; no separate GeoUG download is needed. |
| **Metric-depth evaluation suite** | NYU-v2, KITTI, ETH3D, and other configured benchmarks | [Dataset setup](evaluation/depth/README.md) | Obtain the source datasets separately and configure their local paths. [Config template](configs/eval/depth_datasets.example.json). |

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

For the separate RL environment, see [training setup](docs/TRAINING.md#environment-setup).

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

Download the [released parquet][data-3dcuebench], then validate it:

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

The published release contains 800 questions on 757 images in one image-embedded
parquet. No separate image directory or metadata download is required. See the
[download and evaluation guide](benchmarks/3dcuebench/README.md).

## Training

See [Training data card and recipes](docs/TRAINING.md) for data downloads,
environment setup, and SFT / RL training instructions.

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
ckpt/                    Model download guide (local packages are ignored)
scripts/                 Public launchers
```

<!-- Published repositories and local reproduction guides. -->
[model-spatial-rl]: https://huggingface.co/rayruiyang/GeoUG-Spatial
[model-metric-depth]: https://huggingface.co/rayruiyang/GeoUG-Depth
[model-sft-init]: https://huggingface.co/rayruiyang/GeoUG-ColdStart
[data-3dcuebench]: https://huggingface.co/datasets/rayruiyang/3DCueBench
[data-da2k-textual]: evaluation/vlmevalkit/README.md#reproduce-da-2k-textual-and-visual
[data-da2k-visual]: evaluation/vlmevalkit/README.md#reproduce-da-2k-textual-and-visual
[data-cvbench-textual]: https://huggingface.co/datasets/rayruiyang/CVBench3DTexutal
