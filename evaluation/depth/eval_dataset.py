"""Dataset loader for public metric-depth evaluation."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import (
    depth_map_to_point_map_np,
    intrinsics_from_focal_center,
    masked_nearest_resize_np,
    ray_intersection,
    rotation_matrix_from_vectors,
    unproject_cv_np,
    uv_map_np,
    uv_to_pixel,
)
from .io import read_depth, read_image, read_json


class DepthEvalDataset(Dataset):
    """Load and geometrically normalize a depth benchmark."""

    def __init__(
        self,
        benchmark_name: str,
        path: str,
        width: int,
        height: int,
        split: str = ".index.txt",
        drop_max_depth: float = 1000.0,
        depth_unit: Optional[float] = None,
        subset: Optional[int] = None,
        eigen_valid_mask: bool = False,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.root = Path(path).expanduser()
        split_path = Path(split).expanduser()
        if not split_path.is_absolute():
            split_path = self.root / split_path
        filenames = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if subset is not None:
            if subset < 1:
                raise ValueError("subset must be a positive integer")
            filenames = filenames[::subset]
        self.filenames = filenames
        self.width = width
        self.height = height
        self.drop_max_depth = drop_max_depth
        self.depth_unit = depth_unit
        self.eigen_valid_mask = eigen_valid_mask

    def __len__(self) -> int:
        return len(self.filenames)

    def _load(self, index: int):
        relative_path = self.filenames[index]
        sample_path = self.root / relative_path
        image = read_image(sample_path / "image.jpg")
        raw_depth = read_depth(sample_path / "depth.png")
        metadata = read_json(sample_path / "meta.json")
        intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected a 3x3 intrinsics matrix in {sample_path / 'meta.json'}"
            )
        return relative_path, image, raw_depth, intrinsics

    def _warp(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        depth_mask: np.ndarray,
        intrinsics: np.ndarray,
    ):
        raw_height, raw_width = image.shape[:2]
        raw_horizontal = abs(1.0 / intrinsics[0, 0])
        raw_vertical = abs(1.0 / intrinsics[1, 1])
        raw_pixel_width = raw_horizontal / raw_width
        raw_pixel_height = raw_vertical / raw_height
        target_aspect = self.width / self.height
        target_horizontal = min(
            raw_horizontal, raw_vertical * target_aspect
        )
        target_vertical = target_horizontal / target_aspect

        direction = unproject_cv_np(
            np.array([[0.5, 0.5]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            intrinsics,
        )[0]
        rotation = rotation_matrix_from_vectors(
            direction, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )

        corners = np.array(
            [[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32
        )
        corners = np.concatenate(
            [corners, np.ones((4, 1), dtype=np.float32)], axis=1
        ) @ (np.linalg.inv(intrinsics).T @ rotation.T)
        corners = corners[:, :2] / corners[:, 2:3]
        warp_horizontal, warp_vertical = raw_horizontal, raw_vertical
        ray_directions = np.array(
            [[target_aspect, 1.0], [target_aspect, -1.0]],
            dtype=np.float32,
        )
        for corner_index in range(4):
            intersection = ray_intersection(
                np.array([0.0, 0.0], dtype=np.float32),
                ray_directions,
                corners[corner_index - 1],
                corners[corner_index] - corners[corner_index - 1],
            )
            warp_horizontal = min(
                warp_horizontal, 2 * np.abs(intersection[:, 0]).min()
            )
            warp_vertical = min(
                warp_vertical, 2 * np.abs(intersection[:, 1]).min()
            )
        target_horizontal = min(target_horizontal, warp_horizontal)
        target_vertical = min(target_vertical, warp_vertical)
        target_intrinsics = intrinsics_from_focal_center(
            1.0 / target_horizontal,
            1.0 / target_vertical,
            0.5,
            0.5,
        )

        target_pixel_width = target_horizontal / self.width
        target_pixel_height = target_vertical / self.height
        rescaled_width = int(
            raw_width * raw_pixel_width / target_pixel_width
        )
        rescaled_height = int(
            raw_height * raw_pixel_height / target_pixel_height
        )
        image = np.asarray(
            Image.fromarray(image).resize(
                (rescaled_width, rescaled_height), Image.Resampling.LANCZOS
            )
        )
        depth, depth_mask = masked_nearest_resize_np(
            depth,
            depth_mask,
            (rescaled_height, rescaled_width),
        )
        points = depth_map_to_point_map_np(depth, intrinsics)
        distance = np.sqrt(np.square(points).sum(axis=-1))

        transform = intrinsics @ np.linalg.inv(rotation) @ np.linalg.inv(
            target_intrinsics
        )
        target_uv = uv_map_np(self.height, self.width)
        homogeneous_uv = np.concatenate(
            [
                target_uv,
                np.ones((self.height, self.width, 1), dtype=np.float32),
            ],
            axis=-1,
        )
        source_points = homogeneous_uv @ transform.T
        source_uv = source_points[..., :2] / (
            source_points[..., 2:3] + 1e-12
        )
        pixel_remap = uv_to_pixel(
            source_uv, (rescaled_height, rescaled_width)
        ).astype(np.float32)
        target_image = cv2.remap(
            image,
            pixel_remap[..., 0],
            pixel_remap[..., 1],
            cv2.INTER_LINEAR,
        )
        target_distance = cv2.remap(
            distance,
            pixel_remap[..., 0],
            pixel_remap[..., 1],
            cv2.INTER_NEAREST,
        )
        target_mask = (
            cv2.remap(
                depth_mask.astype(np.uint8),
                pixel_remap[..., 0],
                pixel_remap[..., 1],
                cv2.INTER_NEAREST,
            )
            > 0
        )
        target_rays = unproject_cv_np(
            target_uv,
            np.ones((self.height, self.width), dtype=np.float32),
            target_intrinsics,
        )
        target_depth = target_distance / (
            np.sqrt(np.square(target_rays).sum(axis=-1)) + 1e-12
        )
        return target_image, target_depth, target_mask, target_intrinsics

    def __getitem__(self, index: int):
        filename, image, depth, intrinsics = self._load(index)
        depth_mask = np.isfinite(depth)
        depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=1.0)
        image, depth, depth_mask, intrinsics = self._warp(
            image, depth, depth_mask, intrinsics
        )

        if depth_mask.any():
            minimum = np.nanquantile(
                np.where(depth_mask, depth, np.nan), 0.01
            )
            depth_mask &= depth <= minimum * self.drop_max_depth
        if self.depth_unit is not None:
            depth *= self.depth_unit
        if not depth_mask.any():
            raise ValueError(f"No valid depth pixels remain in sample {filename}")

        depth_mask_tensor = torch.from_numpy(depth_mask).bool()
        if self.eigen_valid_mask and "nyu" in self.benchmark_name.lower():
            if (self.height, self.width) != (480, 640):
                raise ValueError(
                    "NYU Eigen crop requires evaluation size 640x480"
                )
            eigen_mask = torch.zeros_like(depth_mask_tensor)
            eigen_mask[45:471, 41:601] = True
            depth_mask_tensor &= eigen_mask

        return {
            "filename": filename,
            "image": torch.from_numpy(
                image.astype(np.float32) / 255.0
            ).permute(2, 0, 1),
            "depth": torch.from_numpy(depth).float(),
            "depth_mask": depth_mask_tensor,
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "is_metric": self.depth_unit is not None,
        }
