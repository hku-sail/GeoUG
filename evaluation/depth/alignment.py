"""Robust alignment routines used by the public depth metrics."""

import math
from typing import Optional, Tuple

import torch


def _scatter_min(
    size: int, index: torch.LongTensor, src: torch.Tensor
) -> torch.return_types.min:
    minimum = torch.full(
        (size,), float("inf"), dtype=src.dtype, device=src.device
    ).scatter_reduce(0, index, src, reduce="amin", include_self=False)
    locations = torch.where(src == minimum[index])[0]
    indices = torch.full((size,), -1, dtype=torch.long, device=src.device)
    indices[index[locations]] = locations
    return torch.return_types.min((minimum, indices))


def _align_scale(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """Minimize ``sum(weight * abs(scale * source - target))``."""
    source, target, weight = torch.broadcast_tensors(source, target, weight)
    sign = torch.sign(source)
    source, target = source * sign, target * sign
    target_over_source, order = (target / source.clamp_min(eps)).sort(dim=-1)
    weighted_source = torch.gather(source * weight, dim=-1, index=order)
    derivatives = (
        2 * weighted_source.cumsum(dim=-1)
        - weighted_source.sum(dim=-1, keepdim=True)
    )
    location = torch.searchsorted(
        derivatives, torch.zeros_like(derivatives[..., :1]), side="left"
    ).clamp_max(derivatives.shape[-1] - 1)
    scale = target_over_source.gather(dim=-1, index=location).squeeze(-1)
    index = order.gather(dim=-1, index=location).squeeze(-1)
    loss = (weight * (scale[..., None] * source - target).abs()).sum(dim=-1)
    return scale, loss, index


def align_depth_scale(
    source: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    scale, _, _ = _align_scale(source, target, weight)
    return scale


def align_depth_affine(
    source: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robust L1 depth alignment with a shared scale and shift."""
    batch_shape, count = source.shape[:-1], source.shape[-1]
    batch_size = math.prod(batch_shape)
    source = source.reshape(batch_size, count)
    target = target.reshape(batch_size, count)
    weight = weight.reshape(batch_size, count)

    anchor_batch, anchor_index = torch.where(weight > 0)
    if anchor_index.numel() == 0:
        raise ValueError("Cannot align depth maps without valid pixels")

    with torch.no_grad():
        source_anchor = source[anchor_batch, anchor_index]
        target_anchor = target[anchor_batch, anchor_index]
        source_centered = source[anchor_batch] - source_anchor[:, None]
        target_centered = target[anchor_batch] - target_anchor[:, None]
        anchor_weight = weight[anchor_batch]
        scale, loss, second_index = _align_scale(
            source_centered, target_centered, anchor_weight
        )
        _, best_anchor = _scatter_min(batch_size, anchor_batch, loss)

    first_index = anchor_index[best_anchor]
    second_index = second_index[best_anchor]
    source_first = source.gather(1, first_index[:, None]).squeeze(1)
    target_first = target.gather(1, first_index[:, None]).squeeze(1)
    source_second = source.gather(1, second_index[:, None]).squeeze(1)
    target_second = target.gather(1, second_index[:, None]).squeeze(1)
    scale = (target_second - target_first) / torch.where(
        source_second != source_first,
        source_second - source_first,
        torch.full_like(source_second, 1e-7),
    )
    shift = target_first - scale * source_first
    return scale.reshape(batch_shape), shift.reshape(batch_shape)


def align_affine_lstsq(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve weighted least-squares affine alignment."""
    weight_sqrt = torch.ones_like(source) if weight is None else weight.sqrt()
    design = torch.stack(
        [weight_sqrt * source, weight_sqrt], dim=-1
    )
    response = (weight_sqrt * target)[..., None]
    scale, shift = (
        torch.linalg.lstsq(design, response)[0].squeeze(-1).unbind(-1)
    )
    return scale, shift
