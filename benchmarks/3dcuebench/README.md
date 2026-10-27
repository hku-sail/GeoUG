# 3DCueBench v1.0

3DCueBench evaluates image-space depth ordering and object-to-object 3D
distance ordering across controlled relative-gap difficulty bins.

The frozen v1.0 source contains:

- 800 questions on 757 images;
- 400 depth and 400 distance questions;
- four difficulty bins per split, 100 questions per bin;
- exactly 400 A and 400 B answers;
- one `closer` formulation per item.

## Download

The public release is available at
[rayruiyang/3DCueBench](https://huggingface.co/datasets/rayruiyang/3DCueBench).
Download **only the parquet**; it already contains the images, questions,
answers, and analysis fields. No separate image directory or metadata files
are required.

Run from the GeoUG repository root:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="rayruiyang/3DCueBench",
    repo_type="dataset",
    filename="data/test-00000-of-00001.parquet",
    revision="36ba0ed4fedc29e95acae9557b28f9aa201ca31c",
    local_dir="artifacts/3dcuebench-v1.0.0",
)
print(path)
```

File size: **115,541,247 bytes** (110.2 MiB).

```text
SHA256: 5299fbcb85a9c0a2adf661bd89f7828b7fb332ac1c3951ba691379c78430e4b4
```

The Hugging Face split is `test`. The parquet's `split` column identifies the
task (`depth` or `distance`), not a train/test partition. Provide only the
image and question to the model; labels, reference distances, and diagnostic
columns are for scoring and analysis.

See the [Hugging Face data card](https://huggingface.co/datasets/rayruiyang/3DCueBench/blob/main/README.md)
for the schema and image-loading examples.

## Export from source (optional)

Users of the published parquet can skip this step. The internal source
parquet stores absolute image paths and is not portable.
Convert it into the canonical release schema:

```bash
python benchmarks/3dcuebench/export_release.py \
  --questions /path/to/benchmark_800/questions.parquet \
  --output artifacts/3dcuebench-v1.0.0/data/test-00000-of-00001.parquet \
  --embed-images
```

`--embed-images` re-encodes images as JPEG with a maximum side of 1600 pixels
and stores Hugging Face Image-compatible `{bytes, path}` records. This is the
canonical public release mode.

Without `--embed-images`, the exporter writes public relative manifest paths
and hashes. Paths are resolved relative to the release parquet, and validation
requires every image to exist and match its SHA256. Source-specific preparation
scripts must materialize those paths.

## Validate

```bash
python benchmarks/3dcuebench/validate.py \
  artifacts/3dcuebench-v1.0.0/data/test-00000-of-00001.parquet
```

Validation covers schema, counts, balance, points, answers, image decoding,
hashes, and internal-path leakage.

## Predictions and scoring

Prediction files may be parquet, JSONL, TSV, or CSV and must contain:

```text
question_id
prediction
```

Score them with:

```bash
python benchmarks/3dcuebench/evaluate.py \
  --dataset artifacts/3dcuebench-v1.0.0/data/test-00000-of-00001.parquet \
  --predictions predictions.parquet \
  --output outputs/3dcuebench/metrics.json
```

Official v1.0 reports overall accuracy and breakdowns by split, difficulty,
domain, and source. Unparsed answers count as incorrect.

The source set does not contain paired closer/farther formulations. Therefore
the old closer/farther consistency metric is not an official v1.0 metric.
Option-order circular evaluation may be reported separately.

## Coordinate convention

The release stores points on an aspect-preserving grid whose long side is
1000. The short side is scaled by the image aspect ratio. The origin is at the
top-left, x increases rightward, and y increases downward.

Model-specific adapters perform runtime conversion. In particular, BAGEL's
per-axis 0–1000 convention must not be published as a second canonical dataset.

