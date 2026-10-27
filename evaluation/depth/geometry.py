"""Geometry helpers needed by depth dataset warping and evaluation."""

import math
from numbers import Integral
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as torch_functional


def uv_map_np(height: int, width: int) -> np.ndarray:
    u = np.linspace(0.5 / width, 1.0 - 0.5 / width, width, dtype=np.float32)
    v = np.linspace(0.5 / height, 1.0 - 0.5 / height, height, dtype=np.float32)
    u, v = np.meshgrid(u, v, indexing="xy")
    return np.stack([u, v], axis=-1)


def unproject_cv_np(
    uv: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    homogeneous = np.concatenate(
        [uv, np.ones((*uv.shape[:-1], 1), dtype=uv.dtype)], axis=-1
    )
    return (homogeneous @ np.linalg.inv(intrinsics).T) * depth[..., None]


def depth_map_to_point_map_np(
    depth: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    return unproject_cv_np(uv_map_np(*depth.shape[-2:]), depth, intrinsics)


def rotation_matrix_from_vectors(
    source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = np.sum(source * target)
    x, y, z = cross
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=source.dtype
    )
    return (
        np.eye(3, dtype=source.dtype)
        + skew
        + skew @ skew / np.asarray(1.0 + cosine, dtype=source.dtype)
    )


def ray_intersection(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray:
    origin_a, direction_a, origin_b, direction_b = np.broadcast_arrays(
        origin_a, direction_a, origin_b, direction_b
    )
    dimensions = origin_a.shape[-1]
    directions = np.stack([direction_a, direction_b], axis=-2)
    origins = np.stack([origin_a, origin_b], axis=-2)
    matrix = np.concatenate(
        [
            (
                np.eye(dimensions, dtype=origin_a.dtype)
                * np.ones((*origins.shape[:-2], 2, 1, 1))
            ).reshape(*directions.shape[:-2], 2 * dimensions, dimensions),
            -(
                np.eye(2, dtype=origin_a.dtype)[..., None]
                * directions[..., None, :]
            )
            .swapaxes(-2, -1)
            .reshape(*directions.shape[:-2], 2 * dimensions, 2),
        ],
        axis=-1,
    )
    vector = origins.reshape(*origins.shape[:-2], 2 * dimensions)
    solution = np.linalg.solve(
        matrix.swapaxes(-1, -2) @ matrix
        + 1e-12 * np.eye(dimensions + 2, dtype=origin_a.dtype),
        (matrix.swapaxes(-1, -2) @ vector[..., :, None]),
    )[..., 0]
    return solution[..., :dimensions]


def intrinsics_from_focal_center(
    fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def uv_to_pixel(uv: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return uv * np.flip(np.asarray(size), axis=-1) - 0.5


def _sliding_window_np(
    array: np.ndarray,
    window_size: Tuple[int, int],
    padding: Tuple[Tuple[int, int], Tuple[int, int]],
    axes: Tuple[int, int],
) -> np.ndarray:
    axes = tuple(axis % array.ndim for axis in axes)
    full_padding = [
        (0, 0) if index not in axes else padding[axes.index(index)]
        for index in range(array.ndim)
    ]
    array = np.pad(array, full_padding, mode="constant")
    return np.lib.stride_tricks.sliding_window_view(
        array, window_size, axis=axes
    )


def masked_nearest_resize_np(
    image: np.ndarray, mask: np.ndarray, size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_height = height / target_height
    filter_width = width / target_width
    window_shape = (math.ceil(filter_height), math.ceil(filter_width))
    filter_size = math.prod(window_shape)
    padding_height = window_shape[0] // 2 + 1
    padding_width = window_shape[1] // 2 + 1
    padding = (
        (padding_height, padding_height),
        (padding_width, padding_width),
    )

    pixel_u, pixel_v = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
        indexing="xy",
    )
    pixels = np.stack([pixel_u, pixel_v], axis=-1)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    pixel_windows = _sliding_window_np(
        pixels, window_shape, padding, (0, 1)
    )
    index_windows = _sliding_window_np(
        indices, window_shape, padding, (0, 1)
    )
    mask_windows = _sliding_window_np(mask, window_shape, padding, (-2, -1))

    target_centers = uv_map_np(target_height, target_width) * np.array(
        [width, height], dtype=np.float32
    )
    target_left_top = target_centers - np.array(
        [filter_width / 2, filter_height / 2], dtype=np.float32
    )
    target_windows = np.round(target_left_top).astype(np.int32) + np.array(
        [padding_width, padding_height], dtype=np.int32
    )
    window_pixels = pixel_windows[
        target_windows[..., 1], target_windows[..., 0]
    ].reshape(target_height, target_width, 2, filter_size)
    window_indices = index_windows[
        target_windows[..., 1], target_windows[..., 0]
    ].reshape(target_height, target_width, filter_size)
    window_mask = mask_windows[
        target_windows[..., 1], target_windows[..., 0]
    ].reshape(target_height, target_width, filter_size)

    delta = window_pixels - target_centers[..., None]
    epsilon = np.finfo(np.float32).eps * max(height, width)
    window_mask &= (
        (-filter_width / 2 + epsilon < delta[..., 0, :])
        & (delta[..., 0, :] <= filter_width / 2 + epsilon)
        & (-filter_height / 2 + epsilon < delta[..., 1, :])
        & (delta[..., 1, :] <= filter_height / 2 + epsilon)
    )
    distance = np.where(
        window_mask,
        np.square(delta[..., 0, :]) + np.square(delta[..., 1, :]),
        np.inf,
    )
    nearest = np.argmin(distance, axis=-1)
    nearest_flat = np.take_along_axis(
        window_indices, nearest[..., None], axis=-1
    ).squeeze(-1)
    resized = image[nearest_flat // width, nearest_flat % width]
    return resized, np.any(window_mask, axis=-1)


def _sliding_window_torch(
    tensor: torch.Tensor,
    window_size: Tuple[int, int],
    padding: Tuple[Tuple[int, int], Tuple[int, int]],
) -> torch.Tensor:
    full_padding = tuple(value for pair in reversed(padding) for value in pair)
    tensor = torch_functional.pad(tensor, full_padding)
    return tensor.unfold(-2, window_size[0], 1).unfold(
        -2, window_size[1], 1
    )


def masked_nearest_resize_torch(
    mask: torch.Tensor, size: Tuple[int, int]
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_height = height / target_height
    filter_width = width / target_width
    window_shape = (math.ceil(filter_height), math.ceil(filter_width))
    padding_height = window_shape[0] // 2 + 1
    padding_width = window_shape[1] // 2 + 1
    padding = (
        (padding_height, padding_height),
        (padding_width, padding_width),
    )
    device = mask.device

    pixel_u, pixel_v = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        torch.arange(height, dtype=torch.float32, device=device) + 0.5,
        indexing="xy",
    )
    pixels = torch.stack([pixel_u, pixel_v], dim=-1)
    indices = torch.arange(height * width, device=device).reshape(height, width)
    pixel_windows = _sliding_window_torch(
        pixels.permute(2, 0, 1), window_shape, padding
    ).permute(1, 2, 0, 3, 4)
    index_windows = _sliding_window_torch(indices, window_shape, padding)
    mask_windows = _sliding_window_torch(mask, window_shape, padding)

    u = torch.linspace(
        0.5 / target_width,
        1.0 - 0.5 / target_width,
        target_width,
        device=device,
    )
    v = torch.linspace(
        0.5 / target_height,
        1.0 - 0.5 / target_height,
        target_height,
        device=device,
    )
    target_centers = torch.stack(
        torch.meshgrid(u, v, indexing="xy"), dim=-1
    ) * torch.tensor([width, height], device=device)
    target_left_top = target_centers - torch.tensor(
        [filter_width / 2, filter_height / 2], device=device
    )
    target_windows = target_left_top.round().long() + torch.tensor(
        [padding_width, padding_height], device=device
    )
    window_pixels = pixel_windows[
        target_windows[..., 1], target_windows[..., 0]
    ].flatten(-2)
    window_indices = index_windows[
        target_windows[..., 1], target_windows[..., 0]
    ].flatten(-2)
    window_mask = mask_windows[
        target_windows[..., 1], target_windows[..., 0]
    ].flatten(-2)

    delta = window_pixels - target_centers[..., None]
    epsilon = torch.finfo(torch.float32).eps * max(height, width)
    window_mask &= (
        (-filter_width / 2 + epsilon < delta[..., 0, :])
        & (delta[..., 0, :] <= filter_width / 2 + epsilon)
        & (-filter_height / 2 + epsilon < delta[..., 1, :])
        & (delta[..., 1, :] <= filter_height / 2 + epsilon)
    )
    distance = torch.where(
        window_mask,
        delta[..., 0, :].square() + delta[..., 1, :].square(),
        torch.inf,
    )
    nearest = distance.argmin(dim=-1, keepdim=True)
    nearest_flat = window_indices.gather(-1, nearest).squeeze(-1)
    return window_mask.any(dim=-1), (
        nearest_flat // width,
        nearest_flat % width,
    )
