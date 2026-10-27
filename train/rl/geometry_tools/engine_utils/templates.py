# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Per-category sample generators + question/assistant renderers.

This is where a tuple of :class:`ObjectRecord`s (already carrying 3D stats from
``builder``) becomes a :class:`SampleDraft`: a concrete question, MCQ options,
the ground-truth answer, the reasoning text, and an :class:`IllusionInfo` tag.

What is SHARED vs PROTOCOL-SPECIFIC
-----------------------------------
* The ``gen_*`` generators and the ``SampleDraft`` they return are
  **protocol-agnostic** -- both v2 and v3 consume the same drafts. They decide
  WHAT the question is and WHY the answer is what it is. Crucially, the draft
  carries NO tool-call objects: just the question content, the participating
  objects, and the reasoning text.
* HOW the assistant turn is rendered (which/how many tool calls, what shape) is
  derived later from ``participants`` + ``category``, separately per protocol:
    - v2  -> ``v2_tool_calls`` + ``render_assistant`` at the bottom of THIS file
      (two calls: query_point_depth then query_3d_distance).
    - v3  -> ``render_v3.selfcontained_tool_calls`` +
      ``render_v3.render_assistant_selfcontained`` (a single self-contained call).
  The builder picks exactly one of these based on ``--v3`` and never
  renders both. ``pairs_from_records`` + ``DEPTH_GEN_LINE`` are the shared
  primitives both renderers reuse so the two protocols stay in lock-step.

Anatomy of a generator (every ``gen_*`` follows this skeleton)
--------------------------------------------------------------
  1. Compute the ground-truth metric (camera depth, or object-to-object 3D
     distance) from the records' precise stats.
  2. Margin filter: return ``None`` if the two quantities are too close to call
     (an absolute floor + a relative fraction), so the GT is unambiguous.
  3. Pick the answer target, shuffle MCQ options, pick a repr mode, randomly
     flip closer<->farther phrasing.
  4. Build the question and the pre-depth grounding text. (NOT the tool calls --
     those are derived at render time from participants+category.)
  5. Write the post-depth rationale (quotes the tool numbers; the rationale
     itself does NOT end with the answer -- the answer line is appended by the
     renderer via ``_answer_phrase``: "A. laptop" for MCQ, "1.4 m" for open).
  6. Compute an :class:`IllusionInfo`: what a 2D-only heuristic would answer and
     whether that differs from the GT (=> this sample REQUIRES metric depth).

Reading order (sections in this file)
-------------------------------------
  SECTION 1  data types        -- ObjectRecord / ToolCall / IllusionInfo / SampleDraft
  SECTION 2  2D-cue helpers    -- the "what would a 2D baseline think" primitives
  SECTION 3  formatting helpers -- bbox/point strings, repr mode, tool-call build
  SECTION 4  question templates -- the verbatim CV-Bench phrasings + paraphrases
  SECTION 5  category groups   -- DEPTH_CATEGORIES / DISTANCE_CATEGORIES + metric MCQ
  SECTION 6  generators        -- the six gen_* functions (the core of this file)
  SECTION 7  renderers         -- render_user / render_assistant* (v2 text assembly)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import PointStats
from .system_prompts import MODE_NO_THINK, NO_THINK_QUESTION_SUFFIX
from .tools import query_3d_distance, query_point_depth


# ===========================================================================
# SECTION 1 -- data types
# ---------------------------------------------------------------------------
# Review note: four dataclasses. ObjectRecord is the INPUT unit (one object +
# its 3D stats). SampleDraft is the OUTPUT unit (one fully-resolved Q/A). The
# norm1000() methods convert pixel coords -> the 0~1000 integers used in the
# prompt/trace. ToolCall is only used by the v2 renderer in SECTION 7.
# ===========================================================================


@dataclass
class ObjectRecord:
    """One instance, with name + bbox + interior point + 3D stats already
    computed. This is the unit the generators consume.
    """

    name: str
    cat_id: int
    stats: PointStats
    bbox_xyxy_pixel: tuple[int, int, int, int]
    point_xy_pixel: tuple[int, int]
    image_wh: tuple[int, int]              # (W, H) of the original image
    # True when this image had >1 instance of the object's category, so its
    # ``name`` was disambiguated ("chair" -> "chair 1"/"chair 2"). Such samples
    # MUST reference objects by bbox/point in the question (never name-only),
    # otherwise the model can't tell which "chair N" is meant. Set by the
    # builder's extraction step; defaults False for the common single-instance
    # case (whose name is the bare category, unchanged).
    ambiguous: bool = False

    def bbox_norm1000(self) -> tuple[int, int, int, int]:
        W, H = self.image_wh
        x0, y0, x1, y1 = self.bbox_xyxy_pixel
        return (
            int(round(x0 / max(1, W) * 1000)),
            int(round(y0 / max(1, H) * 1000)),
            int(round(x1 / max(1, W) * 1000)),
            int(round(y1 / max(1, H) * 1000)),
        )

    def point_norm1000(self) -> tuple[int, int]:
        W, H = self.image_wh
        u, v = self.point_xy_pixel
        return (
            int(round(u / max(1, W) * 1000)),
            int(round(v / max(1, H) * 1000)),
        )


@dataclass
class ToolCall:
    name: str
    args: dict
    result: dict


@dataclass
class IllusionInfo:
    """Whether and how strongly a 2D-only baseline would get fooled.

    ``naive_2d_letter`` is the answer that the most natural pixel-only
    heuristic would emit. ``illusion`` is True iff this differs from the
    ground-truth answer, in which case the sample *requires* metric depth to
    answer correctly. ``strength`` in [0, 1] grades how confident the 2D cue
    is, so an "illusion + high strength" is a sample where a pure-2D model
    would be confidently wrong.
    """

    naive_2d_letter: str
    illusion: bool
    strength: float
    cue: str


@dataclass
class SampleDraft:
    """A fully resolved, PROTOCOL-AGNOSTIC Q/A sample.

    It carries only the question content + reasoning text + the participating
    objects; it does NOT carry any tool-call objects. The tool calls (and the
    full assistant trace) are derived from ``participants`` + ``category`` at
    render time, separately for each protocol:

      * v2 -> :func:`v2_tool_calls` / :func:`render_assistant`  (this file)
      * v3 -> ``render_v3.selfcontained_tool_calls`` / ``render_v3.render_assistant_selfcontained``

    Keeping the draft free of protocol-specific tool calls is what stops the
    old "generator builds v2 calls, v3 throws them away" drift.
    """

    category: str
    question: str
    options: dict
    answer_letter: str
    participants: list[ObjectRecord]
    # How the question presented each object's location (``bbox`` / ``point`` /
    # ``name``). The renderer uses this to phrase the grounding step
    # consistently with the question (e.g. "the bbox is given..." vs "the
    # point is given..."). Both v2 and v3 renderers read it.
    repr_mode: str
    # Task narration ONLY (the "To answer this, I need..." sentence). The
    # localize block ("- the table: <bbox>... <point>...") is added by each
    # renderer at render time, because v2 and v3 need different grounding
    # shapes (v3 must pre-emit BOTH <bbox> and <point> so the self-contained
    # tool call args are derivable from the assistant text).
    pre_depth_text: str
    post_depth_text: str
    illusion: Optional[IllusionInfo] = None

    def to_jsonable(self) -> dict:
        return {
            "category": self.category,
            "question": self.question,
            "options": self.options,
            "answer": self.answer_letter,
            "participants": [
                {
                    "name": r.name,
                    "cat_id": r.cat_id,
                    "bbox_xyxy_pixel": list(r.bbox_xyxy_pixel),
                    "bbox_norm1000": list(r.bbox_norm1000()),
                    "point_xy_pixel": list(r.point_xy_pixel),
                    "point_norm1000": list(r.point_norm1000()),
                    "depth_m": r.stats.depth_m,
                    "depth_std_m": r.stats.depth_std_m,
                    "region_px": r.stats.n_pixels_used,
                    "radius_px": r.stats.radius_px,
                    "interior_distance_px": r.stats.interior_distance_px,
                    "mask_area_pixels": r.stats.mask_area_pixels,
                    "centroid_3d_cam": [float(x) for x in r.stats.centroid_3d_cam],
                }
                for r in self.participants
            ],
            # NOTE: ``tool_calls`` is intentionally NOT here -- it is protocol-
            # specific and added by the builder from the chosen renderer
            # (v2_tool_calls / selfcontained_tool_calls).
            "repr_mode": self.repr_mode,
            "pre_depth_text": self.pre_depth_text,
            "post_depth_text": self.post_depth_text,
            "illusion": (
                None if self.illusion is None
                else {
                    "naive_2d_letter": self.illusion.naive_2d_letter,
                    "illusion": self.illusion.illusion,
                    "strength": self.illusion.strength,
                    "cue": self.illusion.cue,
                }
            ),
        }


# ===========================================================================
# SECTION 2 -- 2D-cue helpers (used to detect "visual illusion" samples)
# ---------------------------------------------------------------------------
# Review note: these compute the cheap 2D appearance signals a pure-2D model
# would lean on (bbox area as a proxy for "closer", bbox-center distance as a
# proxy for "near the anchor"). A generator compares the 2D guess to the metric
# GT; when they DISAGREE the sample is an "illusion" and needs real depth.
# _binary_strength grades how confident the 2D cue is (0..1).
# ===========================================================================


def _bbox_area_norm(r: ObjectRecord) -> float:
    """bbox area, normalised by image area to be scale-invariant."""
    x0, y0, x1, y1 = r.bbox_xyxy_pixel
    W, H = r.image_wh
    return float(max(0, x1 - x0) * max(0, y1 - y0)) / float(max(1, W * H))


def _bbox_center_2d_dist(a: ObjectRecord, b: ObjectRecord) -> float:
    """bbox-center Euclidean distance normalised by image size."""
    ax0, ay0, ax1, ay1 = a.bbox_xyxy_pixel
    bx0, by0, bx1, by1 = b.bbox_xyxy_pixel
    W, H = a.image_wh
    ua = (ax0 + ax1) * 0.5 / max(1, W)
    va = (ay0 + ay1) * 0.5 / max(1, H)
    ub = (bx0 + bx1) * 0.5 / max(1, W)
    vb = (by0 + by1) * 0.5 / max(1, H)
    return float(np.hypot(ua - ub, va - vb))


def _binary_strength(x: float, y: float) -> float:
    """Confidence that two scalar 2D cues differ, on a 0..1 scale."""
    if x + y <= 1e-9:
        return 0.0
    return float(abs(x - y) / (x + y))


# ===========================================================================
# SECTION 3 -- formatting helpers (coords, repr mode, options, tool calls)
# ---------------------------------------------------------------------------
# Review note: pure string/format helpers shared by every generator.
#   * _box_str / _point_str        -> the <bbox>/<point> tags (0~1000 coords)
#   * _pick_repr_mode / _q_obj     -> how an object is MENTIONED in the question
#   * _localize_block              -> the grounding lines in the trace
#   * _tool_call_query_*           -> build v2 ToolCall objects (consumed only by
#                                     _v2_tool_calls/render_assistant in SECTION 7)
#   * _shuffle_options             -> map answer items to randomised A/B/C/D
# ===========================================================================


def _box_str(r: ObjectRecord) -> str:
    bx0, by0, bx1, by1 = r.bbox_norm1000()
    return f"<bbox>{bx0} {by0} {bx1} {by1}</bbox>"


# How an object's location is presented in the *question*:
#   * "bbox" -> ``the door (represented by <bbox>57 94 120 268</bbox>)``
#              matches CV-Bench-3D, whose absolute pixel boxes the eval harness
#              rewrites into this exact ``<bbox>`` 0~1000 form (abs_bbox_to_rel).
#   * "point" -> ``the door (represented by <point>89 181</point>)``
#              a single 2D location instead of a full box; useful for benches /
#              prompts that mark objects with a point.
#   * "name"  -> ``the door`` (no coordinates; the model must ground it itself,
#              preserving the original "self-localize" behaviour).
REPR_MODES = ("bbox", "point", "name")
REPR_MODE_WEIGHTS = (0.5, 0.3, 0.2)


def _pick_repr_mode(
    rng: random.Random,
    participants: "list[ObjectRecord] | None" = None,
) -> str:
    """Pick how the question mentions each object: ``bbox`` / ``point`` /
    ``name``.

    If ANY participant has an ambiguous (disambiguated, e.g. "chair 1") name,
    the ``name`` mode is forbidden: a bare "the chair 1" gives the model no way
    to know WHICH chair, so we must show a bbox/point in the question. We then
    pick between the two coordinate modes with their relative weights.
    """
    if participants and any(getattr(p, "ambiguous", False) for p in participants):
        return rng.choices(("bbox", "point"), weights=(0.5, 0.3), k=1)[0]
    return rng.choices(REPR_MODES, weights=REPR_MODE_WEIGHTS, k=1)[0]


def _q_obj(r: ObjectRecord, mode: str = "bbox") -> str:
    """Object mention for the question, with its location embedded per ``mode``.

    Both "bbox" and "point" use the same ``(represented by <...>)`` wrapper so
    that, whatever rewriter the eval harness uses, the train/eval question
    format stays identical.
    """
    if mode == "point":
        return f"the {r.name} (represented by {_point_str(r)})"
    if mode == "name":
        return f"the {r.name}"
    return f"the {r.name} (represented by {_box_str(r)})"


def _answer_phrase(answer_letter: str, options: dict) -> str:
    """The final answer the assistant emits (no ``Answer:`` prefix).

    Two formats, picked by whether the sample is MCQ or open-ended:

      * MCQ (``options`` non-empty)  -> ``"A. laptop"``
        i.e. letter + the option text. Echoing the option text makes eval
        graders robust to letter-only vs full-text answer formats; the
        intra-letter period (``A.``) is the MCQ label separator, not a
        sentence terminator -- no trailing period.
      * Open-ended (``options`` empty, e.g. ``metric_*_open``)
                                     -> ``"1.4 m"``
        Here ``answer_letter`` already carries the full numeric answer string
        (e.g. ``"1.4 m"``); there's no MCQ label to echo.

    The string is emitted as the LAST line of the assistant turn (outside
    ``</think>`` for the think-style modes; the entire assistant content for
    no-think mode).
    """
    if options:
        return f"{answer_letter}. {options[answer_letter]}"
    return f"{answer_letter}"


def _point_str(r: ObjectRecord) -> str:
    px, py = r.point_norm1000()
    return f"<point>{px} {py}</point>"


def localize_block(
    participants: list[ObjectRecord],
    mode: str = "bbox",
    protocol: str = "v2",
) -> str:
    """Render the grounding step (the bullet list of "- the X: <...>" lines)
    so the assistant has pre-emitted every coordinate its tool call will use.

    ``mode`` reflects how the *question* presented each object's location, and
    only affects the LEAD-IN PROSE so the trace reads naturally (e.g. "the bbox
    is given..." vs "the point is given...").

    ``protocol`` decides WHICH coordinates each line lists, and that decision
    is dictated by what the tool call args will need downstream:

      * ``"v2"`` -- v2 tool call args carry only ``point`` per object. So:
          - "point" / "bbox" modes  -> emit only ``<point>`` per line.
          - "name"  mode            -> emit ``<bbox> <point>`` (no other source).
      * ``"v3"`` -- v3 tool call args carry BOTH ``point`` AND ``bbox`` per
        object. So every line MUST emit ``<bbox> <point>`` regardless of mode,
        otherwise the tool call args contain numbers that never appeared in the
        assistant text (breaks the self-containment / train-infer parity
        invariant that v3 was built for).
    """
    if protocol == "v3":
        lines = [
            f"- the {r.name}: {_box_str(r)} {_point_str(r)}"
            for r in participants
        ]
        if mode == "point":
            lead = ("The location of each object is given as a point. I'll "
                    "outline a bounding box around each so the depth read "
                    "has spatial support:")
        elif mode == "bbox":
            lead = ("The bounding box of each object is given. I'll pick a "
                    "representative surface point inside each box:")
        else:  # "name"
            lead = ("First, let me locate each object in the image and "
                    "pick a point on its surface:")
        return lead + "\n" + "\n".join(lines)

    # v2 protocol (unchanged): tool call only needs ``point``, so grounding
    # only emits the bits that aren't already in the question.
    if mode == "point":
        lines = [f"- the {r.name}: {_point_str(r)}" for r in participants]
        return ("The location of each object is given as a point. I'll read "
                "the metric depth at that point:\n" + "\n".join(lines))
    if mode == "bbox":
        lines = [f"- the {r.name}: {_point_str(r)}" for r in participants]
        return ("The bounding box of each object is given. I'll pick a "
                "representative surface point inside each box:\n"
                + "\n".join(lines))
    lines = [
        f"- the {r.name}: {_box_str(r)} {_point_str(r)}"
        for r in participants
    ]
    return ("First, let me locate each object in the image and pick a point "
            "on its surface:\n" + "\n".join(lines))


# Back-compat alias: previously private. Kept so any code still importing the
# underscore name continues to work; new code should use ``localize_block``.
_localize_block = localize_block


def _tool_call_query_point_depth(participants: list[ObjectRecord]) -> ToolCall:
    objects = [
        {"name": r.name, "point": list(r.point_norm1000())}
        for r in participants
    ]
    args = {"objects": objects}
    stats_by_name = {r.name: r.stats for r in participants}
    result = query_point_depth(stats_by_name)
    return ToolCall(name="query_point_depth", args=args, result=result)


def _tool_call_query_3d_distance(
    participants: list[ObjectRecord],
    pairs: list[tuple[str, str]],
) -> ToolCall:
    args = {"pairs": [{"a": a, "b": b} for a, b in pairs]}
    stats_by_name = {r.name: r.stats for r in participants}
    result = query_3d_distance(pairs, stats_by_name)
    return ToolCall(name="query_3d_distance", args=args, result=result)


def _shuffle_options(rng: random.Random, items: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({letter: item}, {item: letter}) with random letter ordering."""
    letters = ["A", "B", "C", "D"][: len(items)]
    perm = items.copy()
    rng.shuffle(perm)
    options = dict(zip(letters, perm))
    inv = {v: k for k, v in options.items()}
    return options, inv


# ===========================================================================
# SECTION 4 -- question templates
# ---------------------------------------------------------------------------
# Review note: ``{a}`` / ``{b}`` / ``{anchor}`` / ``{objs}`` are filled with
# _q_obj(...) so each object's location is embedded inline. The FIRST entry of
# every list is the verbatim CV-Bench-3D phrasing (the eval target); the rest
# are paraphrases for robustness. A generator picks one at random via rng.choice.
# ===========================================================================

_CAM_Q_CLOSER = [
    "Which object is closer to the camera taking this photo, {a} or {b}?",
    "Consider the real-world 3D location of the objects. "
    "Which object is closer to the camera, {a} or {b}?",
]
_CAM_Q_FURTHER = [
    "Which object is further away from the camera taking this photo, {a} or {b}?",
    "Consider the real-world 3D location of the objects. "
    "Which object is further away from the camera, {a} or {b}?",
]
_MULTI_Q_CLOSER = [
    "Estimate the real-world distances between objects in this image. "
    "Which object is closer to {anchor}, {a} or {b}?",
    "Consider the real-world 3D locations of the objects. "
    "Which is closer to {anchor}, {a} or {b}?",
]
_MULTI_Q_FURTHER = [
    "Estimate the real-world distances between objects in this image. "
    "Which object is farther from {anchor}, {a} or {b}?",
    "Consider the real-world 3D locations of the objects. "
    "Which is farther from {anchor}, {a} or {b}?",
]

# N-way (3-4 candidates) phrasing. ``{objs}`` is an inline, comma-separated
# list of "the X (represented by <bbox>...</bbox>)" mentions.
_CAM_NWAY_CLOSEST = [
    "Which of these objects is closest to the camera taking this photo: {objs}?",
    "Consider the real-world 3D location of the objects. "
    "Which one is closest to the camera: {objs}?",
]
_CAM_NWAY_FARTHEST = [
    "Which of these objects is farthest from the camera taking this photo: {objs}?",
    "Consider the real-world 3D location of the objects. "
    "Which one is farthest from the camera: {objs}?",
]
_MULTI_NWAY_CLOSEST = [
    "Estimate the real-world distances between objects in this image. "
    "Which of these is closest to {anchor}: {objs}?",
    "Consider the real-world 3D locations of the objects. "
    "Which one is closest to {anchor}: {objs}?",
]
_MULTI_NWAY_FARTHEST = [
    "Estimate the real-world distances between objects in this image. "
    "Which of these is farthest from {anchor}: {objs}?",
    "Consider the real-world 3D locations of the objects. "
    "Which one is farthest from {anchor}: {objs}?",
]

# Metric-value phrasing. The options are distances in meters, not object names.
_METRIC_DEPTH_Q = [
    "Estimate the real-world distance from the camera to {a}. "
    "Roughly how many meters away is it?",
    "About how far is {a} from the camera taking this photo, in meters?",
]


# ===========================================================================
# SECTION 5 -- category groups + metric MCQ bucketing
# ---------------------------------------------------------------------------
# Review note: a category is either *depth-driven* (answered from per-object
# camera depth, one query_point_depth over all participants) or *distance-driven*
# (answered from object-to-object 3D distance; participants are
# ``[anchor, cand1, cand2, ...]`` and we measure anchor->each candidate, via
# query_3d_distance). BOTH the v2 renderer (this file) and the v3 self-contained
# renderer dispatch on these two frozensets, so registering a new category here
# wires it into both protocols. _metric_options turns a continuous meter value
# into a 4-way numeric MCQ with a guaranteed-unique nearest option.
# ===========================================================================

CAT_LOCATION_CLOSER = "location_closer_to_camera"
CAT_CAMERA_NWAY = "camera_closest_among"
CAT_METRIC_DEPTH = "metric_depth_value"            # MCQ over numeric options
CAT_METRIC_DEPTH_OPEN = "metric_depth_value_open"  # open-ended: answer the number

CAT_MULTI_CLOSER = "multi_object_closer_to"
CAT_MULTI_NWAY = "multi_object_closest_among"

DEPTH_CATEGORIES = frozenset(
    {CAT_LOCATION_CLOSER, CAT_CAMERA_NWAY, CAT_METRIC_DEPTH, CAT_METRIC_DEPTH_OPEN}
)
DISTANCE_CATEGORIES = frozenset(
    {CAT_MULTI_CLOSER, CAT_MULTI_NWAY}
)


def pairs_from_records(records: list[ObjectRecord]) -> list[tuple[str, str]]:
    """The anchor->candidate name pairs for a distance-driven sample.

    SINGLE SOURCE OF TRUTH for "which pairs do we measure", shared by the v2
    renderer (this file) and the v3 renderer (``render_v3``). The layout
    convention is ``participants = [anchor, cand1, cand2, ...]``, so the pairs
    are always ``anchor -> each candidate``. For the two-object metric/closer
    tasks that is simply the single pair ``(participants[0], participants[1])``.
    """
    anchor = records[0]
    return [(anchor.name, r.name) for r in records[1:]]


def _join_q_objs(records: list[ObjectRecord], mode: str) -> str:
    """Comma-separated inline object mentions, e.g.
    ``the table (...), the lamp (...), or the door (...)``."""
    mentions = [_q_obj(r, mode) for r in records]
    if len(mentions) <= 1:
        return "".join(mentions)
    return ", ".join(mentions[:-1]) + (", or " if len(mentions) > 2 else " or ") + mentions[-1]


def _metric_options(
    rng: random.Random,
    value_m: float,
    n: int = 4,
) -> tuple[dict[str, str], str, float]:
    """Bucket a continuous metric value (meters) into an ``n``-way MCQ.

    Returns ``(options, answer_letter, rounded_value)``. The correct option is
    the rounded true value; distractors are scaled copies kept far enough apart
    that "nearest option to the measurement" is unambiguous.
    """
    v = max(0.1, round(value_m, 1))
    gap = max(0.4, 0.20 * v)            # min separation between any two options
    vals = [v]
    factors = [0.35, 0.5, 0.7, 1.4, 1.8, 2.4, 3.0]
    rng.shuffle(factors)
    for f in factors:
        cand = round(v * f, 1)
        if cand > 0 and all(abs(cand - o) >= gap for o in vals):
            vals.append(cand)
        if len(vals) >= n:
            break
    # additive padding if scaling didn't yield enough distinct buckets
    step = gap
    while len(vals) < n:
        cand = round(max(vals) + step, 1)
        if all(abs(cand - o) >= gap for o in vals):
            vals.append(cand)
        step += gap
    rng.shuffle(vals)
    letters = ["A", "B", "C", "D"][: len(vals)]
    options = {L: f"{o:.1f} m" for L, o in zip(letters, vals)}
    ans_letter = next(L for L, o in zip(letters, vals) if abs(o - v) < 1e-9)
    return options, ans_letter, v


# ===========================================================================
# SECTION 6 -- generators (the six gen_* that build a SampleDraft)
# ---------------------------------------------------------------------------
# Review note: each follows the 6-step skeleton from the module docstring
# (compute GT -> margin filter -> pick target/options -> build question+tools ->
# rationale -> illusion tag). Any step-2 failure returns None and the builder
# silently drops the candidate. Read gen_location_closer_to_camera first (the
# simplest), then the others are variations on it.
# ===========================================================================


def gen_location_closer_to_camera(
    rng: random.Random,
    a: ObjectRecord,
    b: ObjectRecord,
) -> Optional[SampleDraft]:
    """2-object MCQ: which is closer to (or further from) the camera?

    GT signal: per-object camera depth ``depth_m`` (smaller = closer).
    """
    # Step 1: ground truth = each object's camera depth.
    za = a.stats.depth_m
    zb = b.stats.depth_m
    if not (np.isfinite(za) and np.isfinite(zb)):
        return None
    # Step 2: margin filter. 放宽最小深度差阈值，让「近距离对」也能进训练集
    # （之前 max(0.5, 0.15*min) 把所有深度差 < 0.5m 的对都丢掉，模型从没在近距离
    # 区间被训练过）。现在：绝对地板 0.1m + 相对 5%，保留更多有挑战性的近距离样本。
    margin = max(0.1, 0.05 * min(za, zb))
    if abs(za - zb) < margin:
        return None
    closer = a if za < zb else b

    # Step 3: randomise A/B order, repr mode, and whether we ask closer/further.
    options, inv = _shuffle_options(rng, [a.name, b.name])
    mode = _pick_repr_mode(rng, [a, b])
    use_further = rng.random() < 0.5      # 50/50 closer vs further phrasing
    if use_further:
        further = b if closer is a else a
        ans_letter = inv[further.name]    # GT letter = the FURTHER object
        tmpl = rng.choice(_CAM_Q_FURTHER)
    else:
        ans_letter = inv[closer.name]     # GT letter = the CLOSER object
        tmpl = rng.choice(_CAM_Q_CLOSER)
    q = tmpl.format(a=_q_obj(a, mode), b=_q_obj(b, mode))

    # Step 4: task narration ONLY. (Tool calls AND the localize bullet list
    # are NOT built here anymore -- each protocol's renderer adds the localize
    # block in its own shape, and derives tool calls from participants.)
    pre = (
        "To answer this, I need each object's metric distance from the camera. "
        "I'll generate a depth map and read it at a representative surface "
        "point of each object."
    )

    # Step 5: rationale quotes both depths and states which is closer/further.
    # (sup = the deeper/further object, inf = the nearer one.)
    sup, inf = (a, b) if za >= zb else (b, a)
    rat = (
        f"From the tool, the {a.name}'s point is at {za:.2f} m from the camera "
        f"and the {b.name}'s point is at {zb:.2f} m. "
        f"Since {min(za, zb):.2f} < {max(za, zb):.2f}, "
        f"the {inf.name} is closer and the {sup.name} is further."
    )

    # Step 6: illusion tag. 2D cue: larger bbox area -> looks closer. naive_obj
    # is what a 2D model would PICK given the closer/further phrasing; if that
    # letter != GT letter, a 2D-only baseline is wrong => illusion=True.
    area_a = _bbox_area_norm(a)
    area_b = _bbox_area_norm(b)
    naive_closer = a if area_a > area_b else b
    naive_obj = (b if naive_closer is a else a) if use_further else naive_closer
    illusion = IllusionInfo(
        naive_2d_letter=inv[naive_obj.name],
        illusion=(inv[naive_obj.name] != ans_letter),
        strength=_binary_strength(area_a, area_b),
        cue="bbox_area",
    )

    return SampleDraft(
        category="location_closer_to_camera",
        question=q,
        options=options,
        answer_letter=ans_letter,
        participants=[a, b],
        repr_mode=mode,
        pre_depth_text=pre,
        post_depth_text=rat,
        illusion=illusion,
    )


def gen_multi_object_closer_to(
    rng: random.Random,
    anchor: ObjectRecord,
    a: ObjectRecord,
    b: ObjectRecord,
) -> Optional[SampleDraft]:
    """3-object MCQ: of A and B, which is closer to the anchor in 3D?

    GT signal: object-to-object 3D Euclidean distance ``anchor->A`` vs
    ``anchor->B`` (NOT camera depth -- that's the shortcut we fight, see the
    rationale below). Participants are stored as ``[anchor, a, b]``, the layout
    the v3 renderer relies on (``pairs = anchor->each candidate``).
    """
    # Step 1: ground truth = 3D distance from the anchor to each candidate.
    dA = float(np.linalg.norm(anchor.stats.centroid_3d_cam - a.stats.centroid_3d_cam))
    dB = float(np.linalg.norm(anchor.stats.centroid_3d_cam - b.stats.centroid_3d_cam))
    if not (np.isfinite(dA) and np.isfinite(dB)):
        return None
    # Step 2: margin filter. Distances are noisier than depths, so the floor is
    # larger here (0.5m absolute + 30% relative) than in the depth task.
    margin = max(0.5, 0.30 * min(dA, dB))
    if abs(dA - dB) < margin:
        return None

    closer = a if dA < dB else b
    farther = b if closer is a else a

    # Step 3: randomise options / repr mode / closer-vs-further phrasing.
    options, inv = _shuffle_options(rng, [a.name, b.name])
    mode = _pick_repr_mode(rng, [anchor, a, b])
    use_further = rng.random() < 0.5
    target = farther if use_further else closer   # GT object given the phrasing
    ans_letter = inv[target.name]
    tmpl = rng.choice(_MULTI_Q_FURTHER if use_further else _MULTI_Q_CLOSER)
    q = tmpl.format(anchor=_q_obj(anchor, mode), a=_q_obj(a, mode), b=_q_obj(b, mode))

    # Step 4: pre-depth grounding text. Tool calls are derived later from
    # participants+category: v2 -> two calls (query_point_depth then
    # query_3d_distance), v3 -> one self-contained query_3d_distance. The
    # [anchor, a, b] participant order is what both derivations rely on
    # (pairs_from_records => anchor->a, anchor->b).
    pre = (
        "To answer this, I need the 3D distance from the anchor to each "
        "candidate. I'll generate a depth map and read it at a point on each "
        "object, then compute distances."
    )

    # Step 5: rationale. Contrastive clause up front: the question is about
    # object-to-object distance, NOT depth from the camera. This is the exact
    # shortcut the model falls into on CV-Bench Distance (sorting by camera
    # depth_m), so we state the distinction explicitly before quoting numbers.
    relation = "farther from" if use_further else "closer to"
    rat = (
        f"This asks which object is {relation} the {anchor.name} — that is the "
        f"distance between objects, not the distance from the camera — so I "
        f"compare the object-to-object 3D distances rather than the raw "
        f"camera depths.\n"
        f"The 3D distance from the {anchor.name} to the {a.name} is {dA:.2f} m, "
        f"and from the {anchor.name} to the {b.name} is {dB:.2f} m. "
        f"Since {min(dA, dB):.2f} < {max(dA, dB):.2f}, the {closer.name} is "
        f"closest to the {anchor.name} and the {farther.name} is farthest."
    )

    # Step 6: illusion tag. 2D cue: the candidate whose bbox CENTER is nearer
    # the anchor's bbox center in the image "looks" closer. naive_pick maps that
    # 2D guess through the closer/further phrasing; != GT letter => illusion.
    d2A = _bbox_center_2d_dist(anchor, a)
    d2B = _bbox_center_2d_dist(anchor, b)
    naive_close = a if d2A < d2B else b
    naive_pick = (farther if naive_close is closer else closer) if use_further else naive_close
    illusion = IllusionInfo(
        naive_2d_letter=inv[naive_pick.name],
        illusion=(inv[naive_pick.name] != ans_letter),
        strength=_binary_strength(d2A, d2B),
        cue="bbox_center_2d_dist_to_anchor",
    )

    return SampleDraft(
        category="multi_object_closer_to",
        question=q,
        options=options,
        answer_letter=ans_letter,
        participants=[anchor, a, b],
        repr_mode=mode,
        pre_depth_text=pre,
        post_depth_text=rat,
        illusion=illusion,
    )


def gen_camera_closest_among(
    rng: random.Random,
    objs: list[ObjectRecord],
) -> Optional[SampleDraft]:
    """N-object (>=3) MCQ: which object is closest / farthest from the camera?

    Tool call: a single ``query_point_depth`` over all participants (same as
    the 2-object location task, just more objects).

    N-way skeleton: same 6 steps, but the margin filter compares the EXTREME
    object (closest or farthest) against the RUNNER-UP only -- that's the pair
    that decides the answer, so the gap between them must be unambiguous.
    """
    if len(objs) < 3:
        return None
    depths = [o.stats.depth_m for o in objs]
    if not all(np.isfinite(d) for d in depths):
        return None

    # Sort by depth; the answer is one extreme of this order. Require the
    # extreme to clear its runner-up by the margin (else the top-2 are a tie).
    order = sorted(range(len(objs)), key=lambda i: depths[i])
    use_further = rng.random() < 0.5
    if use_further:
        ext, runner = order[-1], order[-2]     # farthest vs 2nd-farthest
        if depths[ext] - depths[runner] < max(0.1, 0.05 * depths[runner]):
            return None
    else:
        ext, runner = order[0], order[1]       # closest vs 2nd-closest
        if depths[runner] - depths[ext] < max(0.1, 0.05 * depths[ext]):
            return None
    target = objs[ext]

    mode = _pick_repr_mode(rng, objs)
    options, inv = _shuffle_options(rng, [o.name for o in objs])
    ans_letter = inv[target.name]
    by_name = {o.name: o for o in objs}
    ordered = [by_name[options[L]] for L in sorted(options)]
    tmpl = rng.choice(_CAM_NWAY_FARTHEST if use_further else _CAM_NWAY_CLOSEST)
    q = tmpl.format(objs=_join_q_objs(ordered, mode))

    pre = (
        "To answer this, I need each object's metric distance from the camera. "
        "I'll generate a depth map and read it at a point on each object."
    )

    listing = ", ".join(f"the {o.name} at {depths[i]:.2f} m"
                        for i, o in enumerate(objs))
    rel = "farthest from" if use_further else "closest to"
    rat = (
        f"From the depth map: {listing}. "
        f"The {target.name} is {rel} the camera."
    )

    # 2D cue: bigger bbox area -> looks closer.
    areas = [_bbox_area_norm(o) for o in objs]
    a_order = sorted(range(len(objs)), key=lambda i: areas[i])
    naive_idx = a_order[0] if use_further else a_order[-1]
    a_ext, a_run = (a_order[0], a_order[1]) if use_further else (a_order[-1], a_order[-2])
    illusion = IllusionInfo(
        naive_2d_letter=inv[objs[naive_idx].name],
        illusion=(inv[objs[naive_idx].name] != ans_letter),
        strength=_binary_strength(areas[a_ext], areas[a_run]),
        cue="bbox_area",
    )

    return SampleDraft(
        category=CAT_CAMERA_NWAY,
        question=q,
        options=options,
        answer_letter=ans_letter,
        participants=list(objs),
        repr_mode=mode,
        pre_depth_text=pre,
        post_depth_text=rat,
        illusion=illusion,
    )


def gen_multi_object_closest_among(
    rng: random.Random,
    anchor: ObjectRecord,
    cands: list[ObjectRecord],
) -> Optional[SampleDraft]:
    """(anchor + N>=3 candidates) MCQ: which candidate is closest / farthest
    from the anchor in 3D?

    Tool call: ``query_3d_distance`` measuring anchor->each candidate (the v3
    self-contained renderer derives this from the [anchor, *cands] layout)."""
    if len(cands) < 3:
        return None
    dists = [float(np.linalg.norm(anchor.stats.centroid_3d_cam - c.stats.centroid_3d_cam))
             for c in cands]
    if not all(np.isfinite(d) for d in dists):
        return None

    order = sorted(range(len(cands)), key=lambda i: dists[i])
    use_further = rng.random() < 0.5
    if use_further:
        ext, runner = order[-1], order[-2]
        if dists[ext] - dists[runner] < max(0.5, 0.20 * dists[runner]):
            return None
    else:
        ext, runner = order[0], order[1]
        if dists[runner] - dists[ext] < max(0.5, 0.20 * dists[ext]):
            return None
    target = cands[ext]

    mode = _pick_repr_mode(rng, [anchor, *cands])
    options, inv = _shuffle_options(rng, [c.name for c in cands])
    ans_letter = inv[target.name]
    by_name = {c.name: c for c in cands}
    ordered = [by_name[options[L]] for L in sorted(options)]
    tmpl = rng.choice(_MULTI_NWAY_FARTHEST if use_further else _MULTI_NWAY_CLOSEST)
    q = tmpl.format(anchor=_q_obj(anchor, mode), objs=_join_q_objs(ordered, mode))

    parts = [anchor, *cands]
    pre = (
        "To answer this, I need the 3D distance from the anchor to each "
        "candidate. I'll generate a depth map, read a point on each object, "
        "then compute distances."
    )

    rel = "farthest from" if use_further else "closest to"
    listing = ", ".join(f"to the {c.name} is {dists[i]:.2f} m"
                        for i, c in enumerate(cands))
    rat = (
        f"This asks which object is {rel} the {anchor.name} — that is the "
        f"distance between objects, not the distance from the camera — so I "
        f"compare the object-to-object 3D distances rather than the raw camera "
        f"depths.\nThe 3D distance from the {anchor.name} {listing}. "
        f"The {target.name} is {rel} the {anchor.name}."
    )

    # 2D cue: bbox-center 2D distance to the anchor.
    d2 = [_bbox_center_2d_dist(anchor, c) for c in cands]
    d_order = sorted(range(len(cands)), key=lambda i: d2[i])
    naive_idx = d_order[-1] if use_further else d_order[0]
    d_ext, d_run = (d_order[-1], d_order[-2]) if use_further else (d_order[0], d_order[1])
    illusion = IllusionInfo(
        naive_2d_letter=inv[cands[naive_idx].name],
        illusion=(inv[cands[naive_idx].name] != ans_letter),
        strength=_binary_strength(d2[d_ext], d2[d_run]),
        cue="bbox_center_2d_dist_to_anchor",
    )

    return SampleDraft(
        category=CAT_MULTI_NWAY,
        question=q,
        options=options,
        answer_letter=ans_letter,
        participants=parts,
        repr_mode=mode,
        pre_depth_text=pre,
        post_depth_text=rat,
        illusion=illusion,
    )


def gen_metric_depth_value(
    rng: random.Random,
    obj: ObjectRecord,
    open_ended: bool = False,
) -> Optional[SampleDraft]:
    """Single-object metric question: how far is the object from the camera (m)?

    Tool call: ``query_point_depth`` over the one object; the answer is the
    measured ``depth_m``. Always needs the tool (a metric number can't be read
    off 2D), so ``illusion=True``.

    ``open_ended`` selects the answer format:
      * False (``metric_depth_value``)      -> MCQ over numeric distractors.
      * True  (``metric_depth_value_open``) -> answer the number directly.
    """
    z = obj.stats.depth_m
    if not np.isfinite(z) or z <= 0:
        return None

    mode = _pick_repr_mode(rng, [obj])
    tmpl = rng.choice(_METRIC_DEPTH_Q)
    q = tmpl.format(a=_q_obj(obj, mode))

    pre = (
        "To answer this, I need the object's metric distance from the camera. "
        "I'll generate a depth map and read it at a point on the object."
    )

    if open_ended:
        v = max(0.1, round(z, 1))
        options: dict = {}
        answer = f"{v:.1f} m"
        category = CAT_METRIC_DEPTH_OPEN
        rat = (
            f"From the depth map, the {obj.name}'s surface point is at {z:.2f} m "
            f"from the camera."
        )
    else:
        options, answer, v = _metric_options(rng, z)
        category = CAT_METRIC_DEPTH
        rat = (
            f"From the depth map, the {obj.name}'s surface point is at {z:.2f} m "
            f"from the camera. Among the options, {v:.1f} m is the closest."
        )
    illusion = IllusionInfo(
        naive_2d_letter=answer, illusion=True, strength=1.0,
        cue="metric_value_needs_depth",
    )

    return SampleDraft(
        category=category,
        question=q,
        options=options,
        answer_letter=answer,
        participants=[obj],
        repr_mode=mode,
        pre_depth_text=pre,
        post_depth_text=rat,
        illusion=illusion,
    )


# ---------------------------------------------------------------------------
# Assistant message renderer
# ---------------------------------------------------------------------------


# Modality / branch boundary tokens. These should be added to the model
# tokenizer as single special tokens so the inference loop can detect them
# reliably (see ``engine_utils/runtime.py``).
GENERATE_DEPTH_START = "<GENERATE_DEPTH>"
GENERATE_DEPTH_END = "</GENERATE_DEPTH>"
GENERATE_IMAGE_START = "<GENERATE_IMAGE>"
GENERATE_IMAGE_END = "</GENERATE_IMAGE>"

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_RESULT_START = "<|tool_result_start|>"
TOOL_RESULT_END = "<|tool_result_end|>"

_DEPTH_TOKEN = f"{GENERATE_DEPTH_START}{GENERATE_DEPTH_END}"

# Single source of truth for the "I'm about to emit a depth map" narration.
# Shared by the v2 renderer (below) and the v3 self-contained renderer so the
# wording can't silently drift between the two protocols.
DEPTH_GEN_LINE = "Now I will generate a metric depth map of the scene."


def _format_tool_call(tc: ToolCall) -> str:
    payload = {"name": tc.name, "args": tc.args}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_CALL_START}{body}{TOOL_CALL_END}"


def _format_tool_result(tc: ToolCall) -> str:
    body = json.dumps(tc.result, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_RESULT_START}{body}{TOOL_RESULT_END}"


def render_user(draft: SampleDraft, mode: str | None = None) -> str:
    """Render the USER message of the chat record.

    Body: ``<question>\\n<options or "answer with a number" hint>``. The body
    is IDENTICAL across all three modes -- same image, same question -- which
    is what keeps the mode comparison fair.

    The ONLY per-mode difference here is no_think: it has no system prompt, so
    its conditioning instruction rides on the question as a short-answer
    suffix (:data:`NO_THINK_QUESTION_SUFFIX`). think / adaptive_think leave the
    body untouched and carry their instruction in the system prompt instead.
    """
    # Open-ended questions (e.g. metric *_open categories) carry no options;
    # ask for the number directly instead of listing A/B/C/D choices.
    if not draft.options:
        text = draft.question + "\nAnswer with a number in meters."
    else:
        letters_lines = [f"{k}. {v}" for k, v in draft.options.items()]
        text = draft.question + "\n" + "\n".join(letters_lines)
    if mode == MODE_NO_THINK:
        text = text + "\n" + NO_THINK_QUESTION_SUFFIX
    return text


def render_assistant_no_depth(
    draft: SampleDraft,
    protocol: str = "v2",
) -> str:
    """Direct-answer trace for samples a 2D appearance cue already solves
    (``illusion=False``). No ``<GENERATE_DEPTH>``, no tool calls.

    This is the *negative half* of the depth-gating supervision: paired with
    the full depth trace on illusion samples (and ``--balance-illusion`` to
    keep the two halves 50/50), it teaches the model to answer directly when
    the 3D relation is visually unambiguous and to reserve depth generation
    for the ambiguous / illusion cases. The "decide whether I need depth"
    judgment lives in the assistant content itself (no system prompt), so it
    generalizes across task / prompt phrasings.

    The bullet list shape MIRRORS what the depth-branch grounding for the
    SAME sample would emit, so the model learns ONE grounding format and only
    learns to gate the depth generation. ``protocol`` therefore matches the
    caller's renderer:

      * ``"v3"`` -> ``<bbox> <point>`` per object (same as the v3 depth
        renderer's grounding -- the self-containment style is uniform).
      * ``"v2"`` -> ``<bbox>`` only per object (legacy v2 behaviour).
    """
    if protocol == "v3":
        lines = [
            f"- the {r.name}: {_box_str(r)} {_point_str(r)}"
            for r in draft.participants
        ]
    else:
        lines = [f"- the {r.name}: {_box_str(r)}" for r in draft.participants]
    ground = ("First, let me locate the objects in the image:\n"
              + "\n".join(lines))
    decide = ("The relative 3D arrangement is visually clear from the image, "
              "so I can answer directly without generating a depth map.")
    return "\n\n".join([
        ground, decide,
        _answer_phrase(draft.answer_letter, draft.options),
    ])


def _v2_tool_calls(draft: SampleDraft) -> list[ToolCall]:
    """Derive the v2 (two-call) tool trace from participants + category.

    This is the v2 counterpart of ``render_v3.selfcontained_tool_calls``:
    both read the SAME ``participants`` layout, so the two protocols can never
    disagree about *which* objects/pairs are involved -- they only differ in
    HOW the calls are shaped.

      * distance-driven  -> [query_point_depth(all), query_3d_distance(pairs)]
                            where pairs = anchor->each candidate.
      * depth-driven     -> [query_point_depth(all)]

    (Previously each generator built these objects and stored them on the
    draft; v3 then recomputed + threw them away. Deriving here keeps the draft
    protocol-agnostic and removes that duplication.)
    """
    parts = draft.participants
    if draft.category in DISTANCE_CATEGORIES:
        return [
            _tool_call_query_point_depth(parts),
            _tool_call_query_3d_distance(parts, pairs_from_records(parts)),
        ]
    return [_tool_call_query_point_depth(parts)]


def v2_tool_calls(draft: SampleDraft) -> list[dict]:
    """The v2 ``tool_calls`` list (``{name, args, result}`` dicts) to store in
    the JSONL record. Mirrors ``selfcontained_tool_calls`` for the v3 path; the
    builder picks one based on ``--v3``."""
    return [
        {"name": tc.name, "args": tc.args, "result": tc.result}
        for tc in _v2_tool_calls(draft)
    ]


def render_assistant(draft: SampleDraft) -> str:
    """Render the full v2 assistant turn (pre-depth grounding, depth gen, the
    two-call tool trace, rationale) as a single string. The result is the exact
    text the model is trained to reproduce in SFT, with ``<tool_result>`` blocks
    loss-masked at training time.

    Depth-gating: when the sample is *not* an illusion (a 2D appearance cue
    already yields the correct answer), we render the short no-depth direct
    answer instead, so the model learns *when* to skip depth. Illusion samples
    (and any sample without illusion info) keep the full depth+tool trace.
    """
    if draft.illusion is not None and not draft.illusion.illusion:
        return render_assistant_no_depth(draft)

    # Build pre-depth section = task narration + v2 localize bullet list. The
    # bullet list lives at render time (not on the draft) so v2 and v3 each
    # emit the coordinate set that THEIR tool call args will reference.
    grounding = localize_block(draft.participants, draft.repr_mode, protocol="v2")
    parts: list[str] = [draft.pre_depth_text.rstrip() + "\n\n" + grounding]
    parts.append(DEPTH_GEN_LINE)
    parts.append(_DEPTH_TOKEN)
    for tc in _v2_tool_calls(draft):
        parts.append(_format_tool_call(tc))
        parts.append(_format_tool_result(tc))
    parts.append(draft.post_depth_text.rstrip())
    # Final answer line ("A. laptop" or "1.4 m") comes AFTER the rationale.
    parts.append(_answer_phrase(draft.answer_letter, draft.options))
    return "\n\n".join(parts)
