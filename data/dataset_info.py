"""Dataset registry for the public GeoUG SFT recipe.

Dataset locations are intentionally kept out of source control. Set
``GEOUG_DATASET_MANIFEST`` to a JSON file following
``configs/sft/datasets.example.json`` before launching training.
"""

import json
import os
from pathlib import Path

from .interleave_datasets.depth_metric_dataset import MetricDepthIterableDataset
from .interleave_datasets.interleave_reasoning_v6_dataset import InterleaveReasoningV6Dataset
from .vlm_dataset import VLMParquetIterableDataset


DATASET_REGISTRY = {}
DATASET_INFO = {}

_DATASET_TYPES = {
    "metric_depth": MetricDepthIterableDataset,
    "reasoning_v6": InterleaveReasoningV6Dataset,
    "vlm_parquet": VLMParquetIterableDataset,
}


def _load_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest must be a JSON object: {manifest_path}")

    for group_name, group in manifest.items():
        try:
            dataset_type = group["dataset_type"]
            datasets = group["datasets"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{group_name!r} must define 'dataset_type' and 'datasets'"
            ) from exc

        if dataset_type not in _DATASET_TYPES:
            choices = ", ".join(sorted(_DATASET_TYPES))
            raise ValueError(
                f"Unknown dataset_type {dataset_type!r} for {group_name!r}; "
                f"expected one of: {choices}"
            )
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError(f"{group_name!r}.datasets must be a non-empty object")

        DATASET_REGISTRY[group_name] = _DATASET_TYPES[dataset_type]
        DATASET_INFO[group_name] = datasets


_MANIFEST_PATH = os.environ.get("GEOUG_DATASET_MANIFEST")
if _MANIFEST_PATH:
    _load_manifest(_MANIFEST_PATH)
