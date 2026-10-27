# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Inference-time tool dispatch for the **v3 self-contained** protocol.

This is the v3 counterpart to ``engine_utils/runtime.py`` (which is the v2
runtime). It is intentionally small: in v3 every tool call carries each
object's ``point`` (+ optional ``bbox``) inline in its own ``args``, so the
runtime reads depth straight from the call and **never parses the free-text
grounding lines**. That removes the entire grounding-parser / name-matching /
quote-recovery machinery that makes the v2 runtime large.

Two design rules the v3 protocol buys us:

  * **No grounding text.** ``run_tool_call`` looks only at the JSON args; the
    assistant's prose before the call is irrelevant to execution.
  * **One call per question.** A whole multi-object question is answered by a
    single ``query_3d_distance`` (or ``query_point_depth``) call, so there is
    no cross-call cache and no "names must agree across two JSON blobs" bug.

Plumbing (same interleave as v2)
================================
At inference the model emits, one segment at a time::

    <text> ...
    <GENERATE_DEPTH></GENERATE_DEPTH>
    <|tool_call_start|>{...self-contained call...}<|tool_call_end|>
    <|tool_result_start|>{...}<|tool_result_end|>
    <text> ... <eos>

On ``<|tool_call_end|>`` the runtime parses the JSON, runs it against the
just-generated depth map via :class:`SelfContainedToolSession`, and returns the
``<|tool_result_start|>{...}<|tool_result_end|>`` string to append.

Usage
=====
::

    from engine_utils.runtime_v3 import SelfContainedToolSession

    # Once per question, when the depth map for THIS question is ready:
    session = SelfContainedToolSession(
        depth=predicted_depth_HW,    # (Hd, Wd) float32, meters
        conf=predicted_conf_HW,      # (Hd, Wd) float32 (or np.ones_like(depth))
        intrinsics=K_3x3,            # camera intrinsics (rescaled lazily)
        image_wh=(W_img, H_img),     # original image size, for 0-1000 denorm
    )

    # Run a tool call and get back the <|tool_result_*|> string. Unlike v2
    # there is NO ingest_grounding step -- the call is self-contained.
    tool_result_str = session.run_tool_call(tool_call_str)

Schema (all coords are 0-1000 normalised image coords)
======================================================
``query_point_depth`` (location task, 1 call)::

    {"name":"query_point_depth","args":{"objects":[
        {"name":<str>,"point":[x,y],"bbox":[x0,y0,x1,y1]}, ...]}}
    -> {"<name>":{"depth_m","depth_std_m","region_px","centroid_3d_cam"}, ...}

``query_3d_distance`` (multi-object task, 1 self-contained call)::

    {"name":"query_3d_distance","args":{
        "objects":[{"name","point","bbox"}, ...],
        "pairs":[["a","b"], ...]}}
    -> {"objects":  {"<name>":{...depth fields...}},
        "distances":{"<a> <-> <b>":{"euclidean_3d_m":<float>}}}

``bbox`` is optional; its area sets the depth-sampling window radius (bigger
object -> bigger window, clamped to [5,25]px). Missing bbox -> a fixed default
radius around the point.

Parity
======
The depth math is *imported*, never reimplemented, so a trace produced by the
data engine and a trace produced here at inference use the exact same numbers:

  * ``geometry.read_depth_at_point`` -- median depth in a circular window + 3D
    unproject.
  * ``geometry._adaptive_radius_px`` -- window radius from bbox area.
  * ``geometry.scale_intrinsics_to_depth`` / ``PointStats``.
  * ``tools.query_point_depth`` / ``tools.query_3d_distance`` -- the pure
    PointStats -> JSON result formatters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


# ===========================================================================
# SECTION 1 -- imports & module constants
# ---------------------------------------------------------------------------
# Review note: nothing here computes anything; it only (a) imports the depth
# math we MUST share with data-gen for train==infer parity, and (b) defines the
# string/regex constants used throughout the file.
# ===========================================================================

# --- parity-critical depth math: imported (never reimplemented) so a trace
#     made by the data engine and one made here at inference use identical
#     numbers. If you touch any of these, both paths change together. ---------
from .geometry import (
    PointStats,                 # the per-object 3D summary dataclass
    _adaptive_radius_px,        # bbox area (px) -> depth-window radius, clamped
    read_depth_at_point,        # median depth in a circular window + unproject
    scale_intrinsics_to_depth,  # rescale K to the depth-map pixel grid
)
# --- pure result formatters: PointStats dict -> JSON-able result dict. Reused
#     verbatim from data-gen so the result JSON is byte-for-byte the same. -----
from .tools import query_3d_distance as _distances_from_stats
from .tools import query_point_depth as _depths_from_stats
# --- modality / tool boundary tokens (single source of truth lives in
#     templates.py; we only read them here). ---------------------------------
from .templates import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    TOOL_RESULT_END,
    TOOL_RESULT_START,
)

# json_repair is optional. We only reach for it to salvage malformed JSON the
# model occasionally emits; if it isn't installed we silently fall back to
# strict parsing only (see _loads_tolerant).
try:
    from json_repair import repair_json as _repair_json
except Exception:  # pragma: no cover
    _repair_json = None


# Tool names as constants so callers / dispatch never hard-code the literals.
TOOL_POINT_DEPTH = "query_point_depth"
TOOL_3D_DISTANCE = "query_3d_distance"

# Depth-window radius (in depth-grid px) for an object that gives no bbox. It
# matches the lower clamp of _adaptive_radius_px, so a point-only object still
# reads a small patch instead of a single (noisy) pixel.
DEFAULT_RADIUS_PX = 5

# Extracts the JSON object out of "<|tool_call_start|>{...}<|tool_call_end|>".
# DOTALL so the body can span newlines. The greedy \{.*\} deliberately grabs
# the largest brace span (the whole call), not the first nested object.
_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_START) + r"\s*(\{.*\})\s*" + re.escape(TOOL_CALL_END),
    flags=re.DOTALL,
)


# ===========================================================================
# SECTION 2 -- parsing & name helpers (raw model JSON -> clean python values)
# ---------------------------------------------------------------------------
# Review note: these are small, pure, side-effect-free functions. They never
# touch the depth map. Their only job is to be tolerant of the messy JSON a
# model emits (stray quotes, [x,y] vs (x,y), pair as list vs dict, ...).
# ===========================================================================


def _loads_tolerant(blob: str) -> Optional[dict]:
    """Parse a tool-call JSON object. Strict ``json.loads`` first (the common
    case), then ``json_repair`` if available. Returns None if nothing parses."""
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    if _repair_json is not None:
        try:
            obj = json.loads(_repair_json(blob))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _clean_name(raw: Any) -> str:
    """Strip quote chars + collapse whitespace, so ``"851" sign`` -> ``851 sign``.
    Names here are only output labels / local pair references."""
    s = str(raw).strip()
    for q in ('"', "'", "\u201c", "\u201d", "\u2018", "\u2019", "`"):
        s = s.replace(q, "")
    return re.sub(r"\s+", " ", s).strip()


def _normalize_name(s: str) -> str:
    """Canonical form for tolerant *local* pair-name matching."""
    return _clean_name(s).lower().strip(" .,:;-")


def _match_local(name: str, keys: list[str]) -> Optional[str]:
    """Resolve a pair endpoint name to one of the object names *in the same
    call*. Purely local (no grounding text, no cross-call cache), so it's cheap
    and safe. Order: exact -> normalized-exact -> unique normalized containment.
    Returns None if nothing matches or a containment match is ambiguous."""
    if name in keys:
        return name
    target = _normalize_name(name)
    if not target:
        return None
    norm = [(k, _normalize_name(k)) for k in keys]
    exact = [k for k, kn in norm if kn == target]
    if exact:
        return exact[0]
    cand = [
        k for k, kn in norm
        if kn and min(len(kn), len(target)) >= 2 and (kn in target or target in kn)
    ]
    return cand[0] if len(cand) == 1 else None


def _coerce_point(value: Any) -> Optional[tuple[int, int]]:
    """Accept ``[x, y]`` / ``(x, y)`` -> integer ``(x, y)``, else None."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(round(float(value[0]))), int(round(float(value[1])))
        except (TypeError, ValueError):
            return None
    return None


def _coerce_bbox(value: Any) -> Optional[tuple[int, int, int, int]]:
    """Accept ``[x0, y0, x1, y1]`` -> integer 4-tuple, else None."""
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(round(float(c))) for c in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    return None


def _pair_endpoints(pair: Any) -> tuple[Optional[str], Optional[str]]:
    """Accept a pair as ``["a","b"]`` or ``{"a":...,"b":...}`` -> (a, b)."""
    if isinstance(pair, dict):
        return pair.get("a"), pair.get("b")
    if isinstance(pair, (list, tuple)) and len(pair) == 2:
        return pair[0], pair[1]
    return None, None


# ===========================================================================
# SECTION 3 -- result wrappers (dict -> the <|tool_result_*|> string)
# ---------------------------------------------------------------------------
# Review note: every successful or failed call returns ONE of these strings.
# separators=(",", ":") => compact JSON, byte-identical to the data-gen render.
# An error is still a valid result so the model can read it and self-correct
# instead of the run aborting.
# ===========================================================================


def _tool_result(payload: dict) -> str:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_RESULT_START}{body}{TOOL_RESULT_END}"


def _tool_result_error(msg: str) -> str:
    return _tool_result({"error": msg})


# ===========================================================================
# SECTION 4 -- the inference runtime (SelfContainedToolSession)
# ---------------------------------------------------------------------------
# Review note: this is the only stateful piece. Read it in three layers:
#   4a. coordinate helpers   -- 0-1000 norm coords <-> depth-grid pixels
#   4b. the core depth read  -- one object's inline coords -> PointStats
#   4c. run_tool_call        -- parse -> read all objects -> dispatch by tool
# ===========================================================================


@dataclass
class SelfContainedToolSession:
    """Per-question runtime state for the v3 protocol.

    Built once after the depth map is generated, then answers each tool call.
    Unlike v2's ``ToolSession`` it holds NO grounding dict and NO cross-call
    cache: every call is answered purely from the inline coordinates in its own
    args.
    """

    depth: np.ndarray                  # (Hd, Wd) float32, meters
    conf: np.ndarray                   # (Hd, Wd) float32, confidence
    intrinsics: np.ndarray             # 3x3 K (raw); rescaled to depth grid lazily
    image_wh: tuple[int, int]          # (W_img, H_img); for denorm of 0-1000 coords
    conf_thresh: float = 1.05
    min_pixels: int = 20

    # Lazily-computed cache of the rescaled intrinsics (see K_scaled).
    _K_scaled: Optional[np.ndarray] = None

    # -- 4a. coordinate helpers ---------------------------------------------
    # All model coords arrive as 0-1000 normalised image coords; the depth map
    # may be at a different resolution than the original image. These helpers
    # map norm coords -> the actual depth-grid pixel we sample.

    @property
    def K_scaled(self) -> np.ndarray:
        """Intrinsics rescaled to the depth-map pixel grid (computed once,
        then cached). Needed because K is given at the image/processing
        resolution but we unproject on the depth grid."""
        if self._K_scaled is None:
            self._K_scaled = scale_intrinsics_to_depth(
                np.asarray(self.intrinsics, dtype=np.float64), self.depth.shape
            )
        return self._K_scaled

    def _denorm_point(self, p_norm: tuple[int, int]) -> tuple[int, int]:
        """0-1000 normalised point -> integer pixel on the depth grid.

        Two steps: norm -> image pixel (scale by W-1 / H-1), then, only if the
        depth map resolution differs from the image, image pixel -> depth-grid
        pixel.
        """
        W, H = self.image_wh
        u = int(round(p_norm[0] / 1000.0 * (W - 1)))
        v = int(round(p_norm[1] / 1000.0 * (H - 1)))
        Hd, Wd = self.depth.shape
        if (Hd, Wd) != (H, W):           # depth grid differs from image size
            u = int(round(u * (Wd / max(1, W))))
            v = int(round(v * (Hd / max(1, H))))
        return u, v

    def _bbox_area_px(self, b_norm: tuple[int, int, int, int]) -> int:
        """Area (depth-grid px) of a 0-1000 normalised bbox."""
        Hd, Wd = self.depth.shape
        x0 = b_norm[0] / 1000.0 * (Wd - 1)
        x1 = b_norm[2] / 1000.0 * (Wd - 1)
        y0 = b_norm[1] / 1000.0 * (Hd - 1)
        y1 = b_norm[3] / 1000.0 * (Hd - 1)
        return int(max(1, (x1 - x0) * (y1 - y0)))

    def _denorm_bbox(self, b_norm: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """0-1000 normalised bbox -> depth-grid pixel corners."""
        Hd, Wd = self.depth.shape
        return (
            int(round(b_norm[0] / 1000.0 * (Wd - 1))),
            int(round(b_norm[1] / 1000.0 * (Hd - 1))),
            int(round(b_norm[2] / 1000.0 * (Wd - 1))),
            int(round(b_norm[3] / 1000.0 * (Hd - 1))),
        )

    # -- 4b. the core read: depth straight from one object's inline coords ---

    def _read_object(self, obj: dict) -> Optional[PointStats]:
        """Read depth for ONE object directly from its inline ``point`` (+
        optional ``bbox``). This is the heart of the self-contained design:
        no grounding-text lookup, no name matching, no cross-call cache.

        bbox only controls how big a circular window we median the depth over
        (bigger object -> bigger window, clamped to [5,25]px). Returns a
        PointStats, or None if the point is missing / out of frame / the window
        has too few confident pixels."""
        point = _coerce_point(obj.get("point"))
        if point is None:
            return None

        # bbox is optional and used ONLY to size the depth-sampling window
        # (bigger object -> bigger window). It never shifts the sample point.
        bbox_norm = _coerce_bbox(obj.get("bbox"))
        if bbox_norm is not None:
            radius = _adaptive_radius_px(self._bbox_area_px(bbox_norm))
            bbox_px = self._denorm_bbox(bbox_norm)
        else:
            radius = DEFAULT_RADIUS_PX
            u_n, v_n = point                  # no bbox: degenerate box at point
            bbox_px = (u_n, v_n, u_n, v_n)

        u, v = self._denorm_point(point)
        out = read_depth_at_point(
            self.depth, self.conf, self.K_scaled, (u, v), radius,
            conf_thresh=self.conf_thresh, min_pixels=self.min_pixels,
        )
        if out is None:
            return None
        depth_m, depth_std, n_used, centroid = out
        return PointStats(
            point_xy_pixel=(u, v),
            depth_m=depth_m,
            depth_std_m=depth_std,
            n_pixels_used=n_used,
            radius_px=int(radius),
            centroid_3d_cam=centroid,
            bbox_xyxy_pixel=bbox_px,
            mask_area_pixels=int(self._bbox_area_px(bbox_norm)) if bbox_norm else 0,
            interior_distance_px=float(radius),
        )

    def _read_all(self, objs: list) -> tuple[Optional[dict[str, PointStats]], Optional[str]]:
        """Read every object in an args list. Returns (stats_by_name, None) on
        success, or (None, error_message) on the first failure."""
        stats: dict[str, PointStats] = {}
        for obj in objs:
            if not isinstance(obj, dict):
                return None, "object entry is not a JSON object"
            name = _clean_name(obj.get("name", ""))
            ps = self._read_object(obj)
            if ps is None:
                return None, f"could not read depth for '{name}'"
            stats[name] = ps
        return stats, None

    # -- 4c. public entry point: parse -> read -> dispatch -------------------

    def run_tool_call(self, tool_call_text: str) -> str:
        """Parse a ``<|tool_call_start|>{...}<|tool_call_end|>`` string and
        return the matching ``<|tool_result_start|>{...}<|tool_result_end|>``
        string to append to the model context.

        Flow: extract JSON -> read every object's depth from its inline coords
        -> branch on the tool name. Any failure short-circuits into an error
        result (never raises), so a bad call can't crash the inference loop.
        """
        # Pull the JSON body; if the wrapper tokens are missing, fall back to
        # treating the whole text as the JSON blob.
        m = _TOOL_CALL_RE.search(tool_call_text)
        blob = m.group(1) if m else tool_call_text.strip()
        call = _loads_tolerant(blob)
        if call is None:
            return _tool_result_error("malformed tool call: could not parse JSON")

        name = call.get("name")
        args = call.get("args") or {}
        objs = args.get("objects") or []
        if not objs:
            return _tool_result_error("tool call has no 'objects'")

        # Step 1: read per-object depth from inline coordinates. This step is
        # identical for BOTH tools -- the only difference is what we do after.
        stats, err = self._read_all(objs)
        if stats is None:
            return _tool_result_error(err or "depth read failed")

        # Step 2a: query_point_depth -> just the per-object depth block.
        if name == TOOL_POINT_DEPTH:
            return _tool_result(_depths_from_stats(stats))

        # Step 2b: query_3d_distance -> per-object depth + pairwise distances,
        # all in one self-contained result. Pair names are resolved LOCALLY
        # against the objects in this same call.
        if name == TOOL_3D_DISTANCE:
            keys = list(stats.keys())
            resolved: list[tuple[str, str]] = []
            for pair in args.get("pairs") or []:
                a, b = _pair_endpoints(pair)
                if a is None or b is None:
                    return _tool_result_error(f"malformed pair: {pair!r}")
                ka = _match_local(_clean_name(a), keys)
                kb = _match_local(_clean_name(b), keys)
                if ka is None or kb is None:
                    return _tool_result_error(
                        f"pair references unknown object: {a!r}/{b!r}"
                    )
                resolved.append((ka, kb))
            return _tool_result({
                "objects": _depths_from_stats(stats),
                "distances": _distances_from_stats(resolved, stats),
            })

        return _tool_result_error(f"unknown tool '{name}'")
