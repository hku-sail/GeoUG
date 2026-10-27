# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tools exposed to the model in its reasoning trace.

The model emits a tool_call (JSON) and the runtime executes it, replacing the
tool's args with these implementations and feeding the result back into the
model. The same code is used at offline data generation, so the numerical
content of every training trace is *exactly* what the inference runtime will
later produce.

Two tools:

* ``query_point_depth`` -- given one or more (name, point) pairs, returns the
  per-object metric depth (median over a flat region around the point), the
  3D centroid in the camera frame, and a couple of confidence proxies.
* ``query_3d_distance`` -- given pairs of names (resolved by an earlier call),
  returns 3D Euclidean distances.

All numbers are rounded to 2 decimals (centimetre precision); finer than that
is pretend precision given monocular depth uncertainty.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import PointStats


_DECIMALS = 2


def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None else round(float(x), _DECIMALS)


def query_point_depth(stats_by_name: dict[str, PointStats]) -> dict:
    """Per-object depth + 3D centroid that the model is allowed to read."""
    out: dict[str, dict] = {}
    for name, s in stats_by_name.items():
        out[name] = {
            "depth_m": _round(s.depth_m),
            "depth_std_m": _round(s.depth_std_m),
            "region_px": int(s.n_pixels_used),
            "centroid_3d_cam": [_round(float(c)) for c in s.centroid_3d_cam],
        }
    return out


def query_3d_distance(
    pairs: list[tuple[str, str]],
    stats_by_name: dict[str, PointStats],
) -> dict:
    """Pairwise 3D Euclidean distance between unprojected point centroids."""
    out: dict[str, dict] = {}
    for a, b in pairs:
        sa = stats_by_name[a]
        sb = stats_by_name[b]
        delta = sa.centroid_3d_cam - sb.centroid_3d_cam
        euclid = float(np.linalg.norm(delta))
        out[f"{a} <-> {b}"] = {
            "euclidean_3d_m": _round(euclid),
        }
    return out
