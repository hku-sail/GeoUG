# 3DCueBench v1.0

3DCueBench evaluates image-space depth ordering and object-to-object 3D
distance ordering across controlled relative-gap difficulty bins.

The frozen v1.0 source contains:

- 800 questions on 757 images;
- 400 depth and 400 distance questions;
- four difficulty bins per split, 100 questions per bin;
- exactly 400 A and 400 B answers;
- one `closer` formulation per item.

## Export

The internal source parquet stores absolute image paths and is not portable.
Convert it into the canonical release schema:

```bash
python benchmarks/3dcuebench/export_release.py \
  --questions /path/to/benchmark_800/questions.parquet \
  --output artifacts/3dcuebench-v1.0.0.parquet \
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
  artifacts/3dcuebench-v1.0.0.parquet
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
  --dataset artifacts/3dcuebench-v1.0.0.parquet \
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

