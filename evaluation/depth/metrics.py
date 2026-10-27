"""Depth-only metrics used by the public GeoUG evaluator."""

import math
from numbers import Number
from typing import Any, Dict, Iterable, Tuple

import torch

from .alignment import (
    align_affine_lstsq,
    align_depth_affine,
    align_depth_scale,
)
from .geometry import masked_nearest_resize_torch


def _relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return (
        (prediction - target).abs() / (target + 1e-6)
    ).mean().item()


def _delta1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    ratio = torch.maximum(target / prediction, prediction / target)
    return (ratio < 1.25).float().mean().item()


def _depth_scores(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, float]:
    return {
        "rel": _relative_error(prediction[mask], target[mask]),
        "delta1": _delta1(prediction[mask], target[mask]),
    }


def compute_depth_metrics(
    prediction: Dict[str, torch.Tensor],
    ground_truth: Dict[str, Any],
) -> Dict[str, Dict[str, Number]]:
    """Compute metric and invariant depth metrics.

    Supported prediction keys are ``depth_metric``,
    ``depth_scale_invariant``, ``depth_affine_invariant``, and
    ``disparity_affine_invariant``. More constrained predictions are also
    evaluated with every compatible invariant metric.
    """
    metrics: Dict[str, Dict[str, Number]] = {}
    mask = ground_truth["depth_mask"]
    target = ground_truth["depth"]
    if not mask.any():
        raise ValueError("Ground-truth depth mask is empty")
    low_resolution_mask, low_resolution_index = (
        masked_nearest_resize_torch(mask, size=(64, 64))
    )

    if "depth_metric" in prediction and ground_truth["is_metric"]:
        metrics["depth_metric"] = _depth_scores(
            prediction["depth_metric"], target, mask
        )

    scale_prediction = prediction.get(
        "depth_scale_invariant", prediction.get("depth_metric")
    )
    if scale_prediction is not None:
        prediction_low = scale_prediction[low_resolution_index][
            low_resolution_mask
        ]
        target_low = target[low_resolution_index][low_resolution_mask]
        scale = align_depth_scale(
            prediction_low, target_low, 1.0 / target_low
        )
        aligned = scale_prediction * scale
        metrics["depth_scale_invariant"] = _depth_scores(
            aligned, target, mask
        )

    affine_prediction = prediction.get(
        "depth_affine_invariant",
        prediction.get(
            "depth_scale_invariant", prediction.get("depth_metric")
        ),
    )
    if affine_prediction is not None:
        prediction_low = affine_prediction[low_resolution_index][
            low_resolution_mask
        ]
        target_low = target[low_resolution_index][low_resolution_mask]
        scale, shift = align_depth_affine(
            prediction_low, target_low, 1.0 / target_low
        )
        aligned = affine_prediction * scale + shift
        metrics["depth_affine_invariant"] = _depth_scores(
            aligned, target, mask
        )

    disparity_prediction = prediction.get("disparity_affine_invariant")
    if disparity_prediction is None:
        source_depth = prediction.get(
            "depth_scale_invariant", prediction.get("depth_metric")
        )
        if source_depth is not None:
            disparity_prediction = 1.0 / source_depth
    if disparity_prediction is not None:
        scale, shift = align_affine_lstsq(
            disparity_prediction[mask], 1.0 / target[mask]
        )
        aligned_disparity = disparity_prediction * scale + shift
        aligned_depth = 1.0 / aligned_disparity.clamp_min(
            1.0 / target[mask].max().item()
        )
        metrics["disparity_affine_invariant"] = _depth_scores(
            aligned_depth, target, mask
        )

    return metrics


def _nested_items(
    mapping: Dict[str, Any], prefix: Tuple[str, ...] = ()
) -> Iterable[Tuple[Tuple[str, ...], Number]]:
    for key, value in mapping.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _nested_items(value, path)
        else:
            yield path, value


def average_metrics(
    metric_list: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    metrics = list(metric_list)
    all_paths = sorted(
        {path for metric in metrics for path, _ in _nested_items(metric)}
    )
    result: Dict[str, Any] = {}
    for path in all_paths:
        values = []
        for metric in metrics:
            value: Any = metric
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if value is not None and not math.isnan(value):
                values.append(value)
        cursor = result
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = (
            sum(values) / len(values) if values else float("nan")
        )
    return result
