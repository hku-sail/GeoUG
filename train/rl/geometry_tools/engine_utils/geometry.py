# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Geometry utilities for the 3DSR distance-comparison data engine (v2).

v2 focuses on two categories that only need a per-object 3D position:

  * ``location_closer_to_camera``     (compare Z)
  * ``multi_object_closer_to``         (compare 3D Euclidean distance)

There is no ground-plane / "up" requirement, so no RANSAC. The per-object 3D
position is computed by:

  1. Picking a single representative *interior* point on the instance mask
     (distance-transform argmax -- the pixel farthest from the mask boundary).
  2. Optionally jittering this point by a few pixels (used only at data-gen
     time, to make the depth reading robust to small inference-time errors in
     the predicted point).
  3. Reading depth values from a circular window around the point with
     adaptive radius, applying MAD outlier rejection (the "flat region"
     filter), and taking the median.
  4. Unprojecting (u, v, depth) once to give the object's 3D centroid in the
     camera frame.

The same compute path is used at offline data generation and at the inference
runtime, so train/infer agree by construction.

Conventions
-----------
* Camera frame: +X right, +Y down, +Z forward (OpenCV / DA3 style).
* Image pixel coords: (u, v) with u=column (x), v=row (y), origin top-left.
* All metric quantities are in meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Intrinsics handling
# ---------------------------------------------------------------------------


def scale_intrinsics_to_depth(K: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    """Rescale DA3 intrinsics to the depth map's pixel grid.

    DA3 stores K at its internal processing resolution. The principal point is
    at the center of that resolution, so we can recover the processing (H_p,
    W_p) as roughly (2*cy, 2*cx) and rescale K linearly. Different sx/sy are
    used to handle slight aspect-ratio rounding inside DA3.
    """
    Hd, Wd = depth_shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    Wp = max(1.0, 2.0 * cx)
    Hp = max(1.0, 2.0 * cy)
    sx = Wd / Wp
    sy = Hd / Hp

    Ks = K.copy().astype(np.float64)
    Ks[0, 0] = fx * sx
    Ks[1, 1] = fy * sy
    Ks[0, 2] = cx * sx
    Ks[1, 2] = cy * sy
    return Ks


# ---------------------------------------------------------------------------
# Single-pixel unprojection
# ---------------------------------------------------------------------------


def unproject_uv_to_cam(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Unproject a single (u, v, z) to a 3D point in the camera frame."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([
        (float(u) - cx) * float(z) / fx,
        (float(v) - cy) * float(z) / fy,
        float(z),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Interior-point selection
# ---------------------------------------------------------------------------


def pick_interior_point(
    mask: np.ndarray,
    jitter_px: int = 0,
    rng: Optional[np.random.Generator] = None,
    min_interior_distance: float = 3.0,
) -> Optional[tuple[int, int, float]]:
    """Return the most-interior pixel of a binary mask.

    "Most interior" = argmax of the distance-transform of the mask, i.e. the
    pixel that is farthest from the boundary. This guarantees:

    * the point lies on the object (it's a mask pixel by construction)
    * a small neighbourhood around it is also on the object, so the flat-region
      depth reading is not contaminated by background

    If the maximum interior distance is below ``min_interior_distance``, the
    object is too thin to support a stable point + window read and we return
    None.

    ``jitter_px`` (>0) randomly offsets the point by up to that many pixels in
    each axis, retrying until the offset lands inside the mask. Use this only
    at training-data generation time; tools at inference don't need it.
    """
    if not mask.any():
        return None
    dt = distance_transform_edt(mask)
    max_d = float(dt.max())
    if max_d < float(min_interior_distance):
        return None
    v_star, u_star = np.unravel_index(int(np.argmax(dt)), dt.shape)
    u_star, v_star = int(u_star), int(v_star)

    if jitter_px > 0 and rng is not None:
        H, W = mask.shape
        for _ in range(16):
            du = int(rng.integers(-jitter_px, jitter_px + 1))
            dv = int(rng.integers(-jitter_px, jitter_px + 1))
            u = u_star + du
            v = v_star + dv
            if 0 <= u < W and 0 <= v < H and mask[v, u]:
                return u, v, max_d
        # fallback to unjittered if every attempt left the mask
    return u_star, v_star, max_d


# ---------------------------------------------------------------------------
# Per-point 3D stats (the v2 "tool input")
# ---------------------------------------------------------------------------


@dataclass
class PointStats:
    """3D summary for one instance, derived from a single point + its flat
    neighbourhood. All metric quantities are in meters; pixel coords in pixels.
    """

    point_xy_pixel: tuple[int, int]
    depth_m: float
    depth_std_m: float                 # std of the kept depths in the region
    n_pixels_used: int                 # how many pixels contributed to depth_m
    radius_px: int                     # the window radius used
    centroid_3d_cam: np.ndarray        # (3,) unprojected from (point, depth)
    bbox_xyxy_pixel: tuple[int, int, int, int]
    mask_area_pixels: int
    interior_distance_px: float        # max distance-transform value of the mask


def _adaptive_radius_px(area: int, lo: int = 5, hi: int = 25) -> int:
    """Heuristic radius: ~10% of sqrt(area), clamped to [lo, hi]."""
    return int(max(lo, min(hi, int(0.1 * np.sqrt(max(1, area))))))


def read_depth_at_point(
    depth: np.ndarray,                 # (H, W) float, meters
    conf: np.ndarray,                  # (H, W) float
    K: np.ndarray,                     # 3x3, already rescaled to depth shape
    point_uv: tuple[int, int],         # (u, v) pixel to read at
    radius_px: int,                    # window radius (in depth-grid pixels)
    conf_thresh: float = 1.05,
    min_pixels: int = 20,
    mad_clip: float = 3.0,
) -> Optional[tuple[float, float, int, np.ndarray]]:
    """Mask-free flat-region depth read.

    Returns ``(depth_m, depth_std_m, n_pixels_used, centroid_3d_cam)``, or
    None if the window does not have enough confident pixels.

    This is the core primitive used by both the offline data engine and the
    inference runtime. The data engine has a precise mask so it computes an
    adaptive radius from the mask area; the runtime has only the model's
    predicted ``<bbox>`` and ``<point>`` and uses bbox area as the size proxy.
    """
    u, v = int(point_uv[0]), int(point_uv[1])
    H, W = depth.shape
    if not (0 <= u < W and 0 <= v < H):
        return None

    u0 = max(0, u - radius_px)
    u1 = min(W, u + radius_px + 1)
    v0 = max(0, v - radius_px)
    v1 = min(H, v + radius_px + 1)
    d_crop = depth[v0:v1, u0:u1]
    c_crop = conf[v0:v1, u0:u1]
    Hc, Wc = d_crop.shape
    yy, xx = np.indices((Hc, Wc))
    cu = u - u0
    cv = v - v0
    in_circle = (xx - cu) ** 2 + (yy - cv) ** 2 <= radius_px ** 2

    valid = (
        in_circle
        & np.isfinite(d_crop)
        & (d_crop > 0)
        & (c_crop > conf_thresh)
    )
    if int(valid.sum()) < min_pixels:
        return None

    zs = d_crop[valid].astype(np.float64)
    med = float(np.median(zs))
    mad = float(np.median(np.abs(zs - med)))
    if mad > 0:
        keep = np.abs(zs - med) < mad_clip * 1.4826 * mad
        if int(keep.sum()) >= min_pixels:
            zs = zs[keep]
    depth_m = float(np.median(zs))
    depth_std = float(np.std(zs))
    centroid = unproject_uv_to_cam(u, v, depth_m, K)
    return depth_m, depth_std, int(len(zs)), centroid


def compute_point_stats(
    mask: np.ndarray,                  # (H, W) bool -- precise instance mask
    depth: np.ndarray,                 # (H, W) float, meters
    conf: np.ndarray,                  # (H, W) float
    K: np.ndarray,                     # 3x3, already rescaled to depth shape
    point_uv: tuple[int, int],         # the (u, v) pixel to read at
    interior_distance_px: float,       # output of pick_interior_point
    radius_px: Optional[int] = None,   # if None, adaptive based on mask area
    conf_thresh: float = 1.05,
    min_pixels: int = 20,
    mad_clip: float = 3.0,
) -> Optional[PointStats]:
    """Data-engine wrapper around :func:`read_depth_at_point` that also keeps
    the mask-derived bbox / area metadata in the output dataclass."""
    mask_b = mask.astype(bool)
    mask_area = int(mask_b.sum())
    if mask_area < 1:
        return None
    if radius_px is None:
        radius_px = _adaptive_radius_px(mask_area)

    out = read_depth_at_point(
        depth, conf, K, point_uv, radius_px,
        conf_thresh=conf_thresh, min_pixels=min_pixels, mad_clip=mad_clip,
    )
    if out is None:
        return None
    depth_m, depth_std, n_used, centroid = out

    ys_all, xs_all = np.where(mask_b)
    x0, x1 = int(xs_all.min()), int(xs_all.max())
    y0, y1 = int(ys_all.min()), int(ys_all.max())

    u, v = int(point_uv[0]), int(point_uv[1])
    return PointStats(
        point_xy_pixel=(u, v),
        depth_m=depth_m,
        depth_std_m=depth_std,
        n_pixels_used=n_used,
        radius_px=int(radius_px),
        centroid_3d_cam=centroid,
        bbox_xyxy_pixel=(x0, y0, x1, y1),
        mask_area_pixels=mask_area,
        interior_distance_px=float(interior_distance_px),
    )
