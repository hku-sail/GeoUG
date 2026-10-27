"""Camera-motion *understanding* built from the multi-view *generation* parquets.

The generation arm trains on (image, pose-text) -> image: the pose is handed to the model
in the prompt and the loss lands on the rendered view, so nothing ever asks the model to
estimate a pose. The evaluation asks the inverse, (image, image) -> pose. This module
reverses the same parquets -- both views through the ViT, loss on the pose text -- so the
inverse direction gets a gradient. No new data is collected; only the side carrying the
loss changes.

Why the labels are recomputed instead of read off `instruction`
---------------------------------------------------------------
`instruction` cannot be reversed into a target. Measured against the raw pose pair in
`meta_info.w2c` (12 files x 10 rows per source):

    source        components named   rows omitting a visible   named components the
                  per row            (>=40 px) component      pose contradicts
    scannetpp     2.16               90.0%                    27%, median 50 px, max 501
    re10k_1_5     2.32                0.0%                     4%, median  1 px, max  11
    re10k_26_35   2.04               29.7%                     1%, median  6 px, max  11
    dl3dv_1_5     2.60               12.7%                    29%, median  6 px, max 142
    dl3dv_26_35   3.03               74.0%                     7%, median 23 px, max 455

re10k and dl3dv disagreements sit near zero, where the sign of a component is a coin flip
nobody could read off the images -- harmless. ScanNet++ is a different matter: 59% of its
contradicted components carry >=40 px of image motion, i.e. about a third of its rows name
at least one direction that the camera visibly did not go. The flip rate also falls
monotonically with pair interval (dl3dv 70.3 -> 63.3 -> 44.3 -> 21.0%, re10k 9.0 -> 6.7 ->
5.7 -> 2.3%), which is the signature of a fixed smoothing lag divided by growing true
motion rather than a coordinate convention error. prepare_vqa_step_five.pair_values
documents the cause: step 1 measures on an EMA-smoothed trajectory (alpha=0.05), whose lag
is ~0.3 m against ~0.03 m of true motion over these short segments, and "on this pool that
inverted the direction of roughly 40% of labels".

Both poses and intrinsics are in `meta_info`, so the truth is recoverable and none of this
has to be trusted. `pose_values` below reimplements the raw-pair geometry.

Which components the target names
---------------------------------
The two largest by *image* motion, or one if only one is visible, following the gates the
benchmark's compound family already validated:

  a named component moves >= `min_component_px` pixels
      Otherwise the sentence asserts something the images do not show. Pixels rather than
      metres and degrees because those two are not comparable and the pixels are: a
      translation of t at depth d moves the image f*t/d, a rotation of theta moves it
      f*theta.
  the second named component leads the third by >= `dominance`
      Which of two near-tied components gets named would otherwise be a coin flip, making
      the target a property of the tie-break rather than of the motion. Naming "every
      significant component" is not an escape: on this pool moving a significance
      threshold from 5 to 8 degrees changes the answer on 40.6% of pairs.
  rotation under `max_rotation_deg`, largest component under half the frame width
      Past those, directions relative to the first image stop being readable at all, and
      at 180 degrees the two roll directions land on the same picture.

There is deliberately no sharpness gate, unlike the benchmark's compound family, which
needed one after a blurred ETH3D SLAM pair turned up in the samples render. Measured here,
this pool does not have that problem: median Laplacian variance runs 456 (re10k_16_25) to
1039 (dl3dv_6_15) against SLAM's 71, and the benchmark's thresholds would reject 0-3% of
kept pairs from three of the four buckets checked. re10k_16_25 is the exception at 23%, so
a run that wants it can pass thresholds through in the same shape as the other gates.

Magnitudes are deliberately absent from the target. Metric translation is not recoverable
from two monocular views -- t and depth enter the projection only as t/d -- so asking for
"0.3 m" trains the model to guess a number the images do not contain. Rotation angle *is*
recoverable, but is left out too so the answer space stays exactly the 12 direction
primitives the benchmark scores.

Contamination
-------------
The ScanNet++ generation pool contains the benchmark's own test scenes. The cause is in
data_engine_mv_scannetpp/run.sh, whose step 0 omits `--scene-list`; prepare_data_step0.py
then falls back to "every subdirectory of data-root" and swept all 862 scenes carrying
poses, 815 train plus all 47 val, instead of the 856 in nvs_sem_train.txt. Measured
against the shipped benchmark: every one of its 47 ScanNet++ scenes is in the pool,
25917 of 625611 pool rows come from those scenes, and 179 of 2290 benchmark candidate
pairs are the identical frame pair (7% of the shipped `rotation` and `lateral` questions).

`HELD_OUT_SCENES` drops those rows as they are read, which is why ScanNet++ can be listed
in a mixture at all. Deleting them from the parquets was measured and rejected: the val
rows sit in 1781 of 2086 files at a median 3% each with no file wholly val, so removing
4% of rows means rewriting 85% of 1.4 TB, and those parquets are what the generation arm
actually read -- editing them in place would make its result unreproducible.

Sources still register separately (`multi_view_und_<source>`) and the mixture yaml still
chooses between them, but that choice is no longer what keeps the test set out.

Note on --visual_gen True
-------------------------
These samples carry no VAE image. train.sft.train all-reduces the skip
flag with MAX, so a packed sequence holding no VAE image drops the step on every rank. In
a generation-enabled run this dataset therefore needs a mandatory generation group beside
it, exactly as cotrain_b_vst500k_mindcube_mvgen_v3.yaml already arranges. A pure
understanding run (--visual_gen False) is unaffected.
"""
from __future__ import annotations

import collections
import io
import json
import random
import zlib

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from ..vlm_dataset import IMAGE_FLAG, VLMParquetIterableDataset

# axis -> (name when the signed value is positive, name when negative)
T_AXES = {"x": ("Truck Right", "Truck Left"),
          "y": ("Pedestal Down", "Pedestal Up"),
          "z": ("Dolly In", "Dolly Out")}
R_AXES = {"yaw": ("Pan Left", "Pan Right"),
          "pitch": ("Tilt Down", "Tilt Up"),
          "roll": ("Roll CCW", "Roll CW")}
FLIP = {a: b for pair in list(T_AXES.values()) + list(R_AXES.values())
        for a, b in (pair, pair[::-1])}
PRIMITIVES = [d for pair in list(T_AXES.values()) + list(R_AXES.values()) for d in pair]

# ScanNet++ nvs_sem_val.txt verbatim, the split the benchmark's ScanNet++ questions are
# built from. Inlined rather than read from the split file because training otherwise
# touches no path outside the parquets, and a guard that disappears when a mount is absent
# is the same failure that let this leak in. re10k and dl3dv scene names are 16 and 64 hex
# characters against these 10, so no other source can collide with the set.
HELD_OUT_SCENES = frozenset((
    "09c1414f1b", "0d2ee665be", "13c3e046d7", "1ada7a0617", "21d970d8de", "25f3b7a318",
    "27dd4da69e", "286b55a2bf", "31a2c91c43", "3864514494", "38d58a7a31", "3db0a1c8f3",
    "3e8bba0176", "3f15a9266d", "40aec5fffa", "45b0dac5e3", "5748ce6f01", "578511c8a9",
    "5942004064", "5eb31827b7", "5ee7c22ba0", "5f99900f09", "6115eddb86", "7831862f02",
    "7b6477cb95", "7bc286c1b6", "825d228aec", "9071e139d9", "99fa5c25e1", "a24f64f7fb",
    "a8bf42d646", "a980334473", "ac48a9b736", "acd95847c5", "b0a08200c9", "bcd2436daf",
    "bde1e479ad", "c49a8c6cff", "c4c04e6d6c", "c50d2d1d42", "c5439f4607", "cc5237fd77",
    "d755b3d9d8", "e398684d27", "e7af285f7d", "f2dc06b1d2", "f3685d06a9", "f3d64c30f8",
    "f9f95681fd", "fb5a96b1a2",
))

# Same wording the benchmark's options use, so a correct answer reads as one of them.
PHRASE = {
    "Truck Left": "moved left", "Truck Right": "moved right",
    "Pedestal Up": "moved up", "Pedestal Down": "moved down",
    "Dolly In": "moved forward", "Dolly Out": "moved backward",
    "Pan Left": "turned left", "Pan Right": "turned right",
    "Tilt Up": "tilted up", "Tilt Down": "tilted down",
    "Roll CCW": "rolled counter-clockwise", "Roll CW": "rolled clockwise",
}

# A stem is one draw from each of four slots rather than one draw from a list of finished
# stems. The 34 clauses below compose into 400 stems per family, and each clause is read
# once; the flat alternative is 400 strings per family to read, where a single variant that
# quietly drops the frame anchor mislabels every pair it is drawn for. That clause is not
# decoration: the labels are computed in the first camera's own axes, and in these handheld
# captures that frame sits a median 38 degrees off the room's vertical, so "up" without the
# anchor is not a question with one answer. Every OPENING fixes the capture order, every
# AGENT says the camera moved rather than the scene, every FRAME pins the axes to the first
# image, and every ASK carries whatever its family has to state to be well posed.
OPENING = [
    "These two images show the same scene from two camera poses.",
    "The two images are the same scene photographed from two camera poses, in capture "
    "order.",
    "Here are two views of one scene, taken by a single moving camera in capture order.",
    "The two pictures come from one moving camera and are shown in the order they were "
    "captured.",
    "Two frames of the same scene are shown, the first followed by the second.",
]
AGENT = [
    "Answer about how the camera itself moved, not about how the scene appears to shift.",
    "Report the motion of the camera rather than the apparent motion of the scene content.",
    "Describe what the camera did, not what the contents of the frame seem to do.",
    "The scene is static, so everything that differs between the frames comes from the "
    "camera; describe the camera's own motion.",
]
FRAME = [
    "Directions are anchored to the first image: \"up\" is toward its top edge and "
    "\"right\" is toward its right edge; this frame rides with the camera and often does "
    "not line up with up and down in the room.",
    "Read every direction relative to the first image's frame, where \"up\" means toward "
    "its top edge and \"right\" toward its right edge. That frame tilts with the camera "
    "and need not agree with the room's.",
    "All directions are in the first image's own frame: its top edge is \"up\" and its "
    "right edge is \"right\". Because the camera is often held at an angle, this is not "
    "the same as up and down in the room.",
    "Take \"up\" and \"right\" to mean toward the top and right edges of the first image "
    "rather than up and right in the room; the two can differ by a large angle.",
]
ASK = {
    "open": [
        "How did the camera move between the first and the second image?",
        "Describe the camera's motion from the first view to the second.",
        "What did the camera do between the two views?",
        "State how the camera moved and turned to get from the first view to the second.",
        "In what way did the camera's pose change from the first image to the second?",
    ],
    "lateral": [
        "The camera's translation was mainly sideways or vertical rather than forward or "
        "backward. Ignoring any rotation, which way did it move?",
        "Its displacement was mostly sideways or vertical rather than toward or away from "
        "the scene. Setting aside how it turned, in which direction did it move?",
        "The camera travelled mainly across the frame rather than into or out of it. "
        "Which direction did it translate, ignoring any turning?",
        "Most of the camera's displacement was sideways or vertical, not forward or "
        "backward. Disregarding rotation, which way did it go?",
        "The dominant part of the camera's translation lies in the image plane rather "
        "than along its viewing direction. Which way did it move, leaving rotation aside?",
    ],
    "rotation": [
        "Yaw swings the camera left or right, pitch tilts it up or down, and roll spins "
        "it about the direction it is facing, clockwise as seen in the picture. Ignoring "
        "any translation, how did the camera rotate from the first view to the second "
        "view?",
        "Yaw turns the camera left or right, pitch tips it up or down, and roll rotates "
        "the picture about the axis the camera looks along. Setting any translation "
        "aside, how did it turn?",
        "A yaw turns the camera sideways, a pitch tilts it vertically, and a roll twists "
        "the image about the viewing direction, clockwise as it appears in the frame. "
        "Disregarding any change of position, which rotation did the camera undergo?",
        "Yaw is a left or right swing, pitch an up or down tilt, and roll a twist about "
        "the direction of view, read clockwise in the picture. Leaving translation aside, "
        "how did the camera rotate?",
        "The camera can yaw left or right, pitch up or down, or roll clockwise or "
        "counter-clockwise about its viewing axis. Ignoring where it moved to, how did "
        "it turn?",
    ],
    "compound": [
        "Between the first and the second image the camera both moved and turned. Which "
        "of the following describes the camera's motion? Each option names the two "
        "largest parts of that motion.",
        "The camera both changed position and changed orientation between the two views. "
        "Which option describes what it did? Every option names the two largest "
        "components of the motion.",
        "This pair involves both a translation and a rotation. Pick the option that "
        "describes the camera's motion; each one names its two biggest parts.",
        "The camera moved and turned at the same time. Which of the options below "
        "matches its motion? Each lists the two largest components.",
        "Both a shift in position and a change in heading occurred between the two "
        "frames. Which option describes them? Each option names the two dominant "
        "components.",
    ],
}

# Everything below is copied verbatim from the benchmark builders -- prepare_vqa_step_five
# for the single-axis families, build_compound for the compound one -- rather than derived
# from PHRASE above. The two sides word options differently on purpose: the single-axis
# families use short noun phrases and the compound family full sentences. A training
# question that paraphrases either one is no longer the format the evaluation asks, which
# is the whole reason for asking it in this shape, so these have to be the same strings.
LETTERS = "ABCDEF"
LATERAL_OPTIONS = ["Truck Left", "Truck Right", "Pedestal Up", "Pedestal Down"]
ROTATION_OPTIONS = ["Pan Left", "Pan Right", "Tilt Up", "Tilt Down", "Roll CW", "Roll CCW"]
OPTION_TEXT = {
    "Dolly In": "Forward", "Dolly Out": "Backward",
    "Truck Left": "Left (toward the left edge of the first image)",
    "Truck Right": "Right (toward the right edge of the first image)",
    "Pedestal Up": "Up (toward the top edge of the first image)",
    "Pedestal Down": "Down (toward the bottom edge of the first image)",
    "Pan Left": "Turn left (yaw left)", "Pan Right": "Turn right (yaw right)",
    "Tilt Up": "Tilt up (pitch up)", "Tilt Down": "Tilt down (pitch down)",
    "Roll CW": "Roll clockwise", "Roll CCW": "Roll counter-clockwise",
}
# The compound family anchors every frame-relative direction to the first image. Dolly and
# roll are absent because neither needs one: forward and clockwise mean the same thing in
# any frame that shares the optical axis.
MCQ_ANCHOR = {
    "Truck Left": " (toward the left edge of the first image)",
    "Truck Right": " (toward the right edge of the first image)",
    "Pedestal Up": " (toward the top edge of the first image)",
    "Pedestal Down": " (toward the bottom edge of the first image)",
    "Pan Left": " (toward the left edge of the first image)",
    "Pan Right": " (toward the right edge of the first image)",
    "Tilt Up": " (toward the top edge of the first image)",
    "Tilt Down": " (toward the bottom edge of the first image)",
}
MCQ_STEM = {
    "lateral": (
        "These two images come from one moving camera, in capture order. Answer about how "
        "the camera itself moved, not about how the scene appears to shift. Directions are "
        "anchored to the first image: \"up\" is toward its top edge and \"right\" is toward "
        "its right edge; this frame rides with the camera and often does not line up with "
        "up and down in the room. The camera's translation was mainly sideways or vertical "
        "rather than forward or backward. Ignoring any rotation, which way did it move?"
    ),
    "rotation": (
        "These two images come from one moving camera, in capture order. Answer about how "
        "the camera itself turned, not about how the scene appears to shift. Directions are "
        "anchored to the first image: \"up\" is toward its top edge and \"right\" is toward "
        "its right edge; this frame rides with the camera and often does not line up with "
        "up and down in the room. Yaw swings the camera left or right, pitch tilts it up or "
        "down, and roll spins it about the direction it is facing, clockwise as seen in the "
        "picture. Ignoring any translation, how did the camera rotate from the first view "
        "to the second view?"
    ),
    "compound": (
        "These two images show the same scene from two camera poses. Between the first "
        "and the second image the camera both moved and turned. Which of the following "
        "describes the camera's motion? Each option names the two largest parts of that "
        "motion. Describe how the camera itself moved, not the apparent motion of the "
        "scene content, and read all directions relative to the first image's frame."
    ),
}
ROTATION_NOTE = "Between the two views the camera also rotated by about {deg:.0f} degrees."
DISCLOSE_ABOVE_DEG = 5.0

# Share of multiple-choice samples asked with the benchmark's stem word for word. The rest
# are composed from the slots, so the model does not learn one string, but the evaluation's
# exact wording still has to appear in training or nothing guarantees it generalises *to*
# that one -- it is the only phrasing that will ever be scored.
VERBATIM_SHARE = 0.10

# The evaluation's POST_PROMPT, verbatim from benchmark/VLMEvalKit/.../cammotion.py.
POST_LETTER = ("Answer with the option's letter from the given choices directly. "
               "Enclose the option's letter within ``.")
POST_LETTER_TEXT = ("Answer with the option's letter followed by its text, exactly as it "
                    "appears in the list.")


def pose_values(w2c_a, w2c_b) -> dict[str, float]:
    """Signed magnitude of all 12 motion primitives between the two views.

    Measured on the raw pose pair. Kept local so training does not depend on
    benchmark-construction code.
    """
    A, B = np.asarray(w2c_a, dtype=float), np.asarray(w2c_b, dtype=float)
    rot_s, rot_e = Rot.from_matrix(A[:3, :3]), Rot.from_matrix(B[:3, :3])
    yaw, pitch, roll = (rot_e * rot_s.inv()).as_euler("yxz", degrees=True)
    centre_s = -rot_s.inv().apply(A[:3, 3])
    centre_e = -rot_e.inv().apply(B[:3, 3])
    dx, dy, dz = rot_s.apply(centre_e - centre_s)
    return {
        "Dolly In": dz, "Dolly Out": -dz,
        "Truck Right": dx, "Truck Left": -dx,
        "Pedestal Down": dy, "Pedestal Up": -dy,
        "Pan Left": yaw, "Pan Right": -yaw,
        "Tilt Down": pitch, "Tilt Up": -pitch,
        "Roll CCW": roll, "Roll CW": -roll,
    }


def total_rotation_deg(w2c_a, w2c_b) -> float:
    """Geodesic angle of the relative rotation.

    The geodesic angle rather than the largest Euler component, because three moderate
    components can add up to a turn no single one of them reveals.
    """
    A, B = np.asarray(w2c_a, dtype=float), np.asarray(w2c_b, dtype=float)
    rel = Rot.from_matrix(B[:3, :3]) * Rot.from_matrix(A[:3, :3]).inv()
    return float(np.degrees(rel.magnitude()))


def meta_node(meta: dict) -> dict:
    """The level holding `w2c` / `fxfycxcy` / `scene_name`.

    ScanNet++ keeps them at the top of meta_info; re10k and dl3dv nest them one level down.
    """
    return meta.get("meta_info", meta)


def read_pose(meta: dict):
    """(w2c pair, focal, frame width).

    `fxfycxcy` holds a 3x3 intrinsic matrix per view despite the name. Frame width comes
    from 2*cx rather than the stored image, so it is consistent with the focal used for
    the pixel conversion even where the parquet holds a resized copy.
    """
    node = meta_node(meta)
    w2c, K = node.get("w2c"), node.get("fxfycxcy")
    if not w2c or len(w2c) < 2 or not K:
        return None
    k0 = np.asarray(K[0], dtype=float)
    return (np.asarray(w2c[0], float), np.asarray(w2c[1], float)), \
        float(k0[0, 0]), 2.0 * float(k0[0, 2])


def scene_depth(depth_cell) -> float | None:
    """Median depth of the first view, in metres. The maps are float64 metres already."""
    if depth_cell is None or len(depth_cell) == 0:
        return None
    arr = np.array(Image.open(io.BytesIO(depth_cell[0]["bytes"])), dtype=np.float64)
    good = arr[np.isfinite(arr) & (arr > 0)]
    return float(np.median(good)) if good.size else None


def components(w2c_pair, focal, depth):
    """Every axis of the pose difference with the image motion it accounts for, largest first."""
    v = pose_values(*w2c_pair)
    out = []
    for pos, neg in T_AXES.values():
        mag = abs(float(v[pos]))
        out.append({"name": pos if v[pos] > 0 else neg, "px": focal * mag / depth})
    for pos, neg in R_AXES.values():
        deg = abs(float(v[pos]))
        out.append({"name": pos if v[pos] > 0 else neg, "px": np.deg2rad(deg) * focal})
    return sorted(out, key=lambda c: -c["px"])


def describe(comps, min_component_px, dominance):
    """The names the target should mention, or None if no reading is well posed.

    Both branches demand the same thing: whatever is named has to lead whatever is left
    unnamed by `dominance`. Dropping that requirement from the single-axis branch -- as the
    first version did, gating only on the second component falling under the floor -- puts
    a sliding threshold back into the label. It showed up in the samples render: a pair
    with Truck Right at 46 px and Pan Left at 33 px was labelled "moved right", while the
    same pair with Pan Left at 41 px would have been labelled "moved right and turned
    left". Nearly identical images, contradictory targets, and the boundary is the floor's
    exact value. 28% of single-axis labels sat under 2x, and requiring dominance costs 14%
    of kept pairs, which is the cheaper side of the trade.
    """
    if comps[0]["px"] < min_component_px:
        return None                                  # nothing visible happened
    if comps[1]["px"] < min_component_px:
        if comps[0]["px"] < dominance * comps[1]["px"]:
            return None                              # a near tie the label would hide
        return [comps[0]["name"]]                    # genuinely single-axis
    if comps[1]["px"] < dominance * comps[2]["px"]:
        return None                                  # which one is second is a coin flip
    return [comps[0]["name"], comps[1]["name"]]


def stem_for(family, rng, rot_deg=0.0):
    """One stem for this family: the benchmark's own wording, or a composed paraphrase.

    The disclosure sentence is spliced into the middle of the verbatim stem, because that
    is where the benchmark puts it, and appended as its own sentence in the composed case,
    where there is no fixed phrase to splice against. Both leave the asker the same
    information: how far the camera turned, but not about which axis. Naming the axis
    leaks -- handheld motion couples pitch with vertical translation and yaw with sideways
    translation, and over the candidate pool a prompt-only axis rule reached 76%.
    """
    disclose = family == "lateral" and rot_deg > DISCLOSE_ABOVE_DEG
    # The open-ended form has no benchmark counterpart to quote, so it is always composed.
    if family != "open" and rng.random() < VERBATIM_SHARE:
        stem = MCQ_STEM[family]
        if disclose:
            head, tail = stem.split(" Ignoring any rotation,")
            return f"{head} {ROTATION_NOTE.format(deg=rot_deg)} Ignoring that rotation,{tail}"
        return stem
    parts = [rng.choice(OPENING), rng.choice(AGENT), rng.choice(FRAME)]
    if disclose:
        parts.append(ROTATION_NOTE.format(deg=rot_deg))
    parts.append(rng.choice(ASK[family]))
    return " ".join(parts)


def mcq(names, rot_deg, rng):
    """The benchmark's own question for this label, or None if no family asks it.

    Returns (question, answer letter, answer option text). Which family a label belongs to
    follows from its shape: two named components is what `compound` asks, one named
    rotation is `rotation`, one named lateral translation is `lateral`. A lone Dolly In or
    Out has no family -- `lateral` drops the depth axis on purpose, since forward motion
    shows up only as a radial scale change and moves the image about half as far as an
    equal sideways move -- so those rows stay open-ended rather than being asked a
    question the evaluation never asks.

    The gates upstream are the same ones the compound family validated, so a two-part
    label already satisfies what that family requires of a question: both named
    components over 40 px, the second leading the third by 2x, rotation under the cap.
    The single-axis families ask less than this module already enforces, since they judge
    dominance within one family of axes while `describe` judges it across all six.
    """
    if len(names) == 2:
        p, s = names
        combos = [(p, s), (FLIP[p], s), (p, FLIP[s]), (FLIP[p], FLIP[s])]
        texts = [f"The camera {PHRASE[a]}{MCQ_ANCHOR.get(a, '')} and "
                 f"{PHRASE[b]}{MCQ_ANCHOR.get(b, '')}." for a, b in combos]
        answer, family = 0, "compound"
    elif names[0] in ROTATION_OPTIONS:
        texts = [OPTION_TEXT[m] for m in ROTATION_OPTIONS]
        answer, family = ROTATION_OPTIONS.index(names[0]), "rotation"
    elif names[0] in LATERAL_OPTIONS:
        texts = [OPTION_TEXT[m] for m in LATERAL_OPTIONS]
        answer, family = LATERAL_OPTIONS.index(names[0]), "lateral"
    else:
        return None
    stem = stem_for(family, rng, rot_deg)

    order = list(range(len(texts)))
    rng.shuffle(order)                     # uniform answer position, so no letter prior
    body = "\n".join(f"{LETTERS[i]}. {texts[j]}" for i, j in enumerate(order))
    return f"{stem}\nOptions:\n{body}", LETTERS[order.index(answer)], texts[answer]


def conversation(names, rot_deg, rng, mcq_share):
    """The two turns, images and question in, answer out. Loss lands on the second.

    A function rather than inline so check_multi_view_und.py can render exactly what a
    run sees. `rng` is seeded from the row id, so a given pair always gets the same
    question; when balancing is on it has consumed one draw by this point, which is a
    constant number of draws either way and so does not make the choice order-dependent.

    Half the pool is asked as multiple choice because that is the only shape the
    evaluation scores, and half stays open-ended because the closed form supervises very
    little: its answer is one letter, and a model can reach it by elimination over four
    option strings without ever naming a direction. The open half carries the full
    sentence and cannot be answered that way.

    No stem is reused across the pool beyond what the slots repeat, so the model cannot
    reach the answer by keying on the prompt. See `stem_for`.

    The two answer shapes get *different* instruction lines rather than sharing one. The
    same prompt cannot be trained towards both `C` and "C. The camera ...", which would
    make the target shape unpredictable from the prompt and the loss inconsistent; the
    instruction line is what makes it predictable. The letter-only line is the
    evaluation's own POST_PROMPT verbatim, so that half is the eval's exact format.
    """
    built = mcq(names, rot_deg, rng) if rng.random() < mcq_share else None
    if built is None:
        return [
            {"from": "human", "value": IMAGE_FLAG * 2 + stem_for("open", rng)},
            {"from": "gpt", "value": "The camera "
             + " and ".join(PHRASE[n] for n in names) + "."},
        ]
    question, letter, text = built
    if rng.random() < 0.5:
        post, answer = POST_LETTER, f"`{letter}`"
    else:
        post, answer = POST_LETTER_TEXT, f"{letter}. {text}"
    return [
        {"from": "human", "value": IMAGE_FLAG * 2 + f"{question}\n{post}"},
        {"from": "gpt", "value": answer},
    ]


class MultiViewUnderstandingIterableDataset(VLMParquetIterableDataset):
    """The generation parquets read backwards: two views in, pose text out."""

    def __init__(self, *args, min_component_px=40.0, dominance=2.0,
                 max_rotation_deg=60.0, max_frac_width=0.5, balance=True,
                 mcq_share=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_component_px = min_component_px
        self.dominance = dominance
        self.max_rotation_deg = max_rotation_deg
        self.max_frac_width = max_frac_width
        self.balance = balance
        self.mcq_share = mcq_share
        self._arrived = collections.Counter()

    def keep_for_balance(self, names, rng) -> bool:
        """Level all twelve directions by thinning each one to the rarest one's rate.

        The raw pool is nowhere near level. Over 420 samples of the arm C mixture the named
        directions ran Pan Left 27.5% down to Pedestal Up 0.3%, an 80x spread, and within
        the z axis Dolly In outnumbered Dolly Out 2.8 to 1. Both kinds of skew are worth
        removing rather than explaining later: the models being measured already answer
        from a directional prior, and a training set carrying its own would make a gain
        from prior-matching look like a gain from perception.

        The rule keeps a candidate with probability (rarest arrival count + 1) / (this
        direction's arrival count + 1), judged on *arrivals* -- every candidate that clears
        the gates -- rather than on what has been kept. Arrivals accumulate in proportion
        to each direction's true rate, so the ratio converges to a constant and the kept
        rate converges to the rarest direction's rate for every direction alike. That is
        exact levelling rather than compression, and it needs no hardcoded target shares.
        Arrivals are counted for every direction the target *names*, not just the primary,
        so a second-named component cannot smuggle the pool's shape back in. The
        probability is taken from the *rarest* named direction, which matters because roll
        mostly arrives as the junior partner of a yaw or a dolly: of 525 pairs clearing the
        caps, roll was second in 7.6% but largest in only 1.3%. Governing by the commonest
        named direction instead thins those pairs at yaw's rate and roll can then never
        rise, which is what the first version did and why it left roll at 2.1% and 3.0%
        while the other ten directions reached 5.6-13.6%. Coupled labels cannot have their
        marginals set independently by rejection alone, so the choice is which side to
        favour, and the scarce side is the one worth favouring.

        TWO REJECTED ALTERNATIVES, both measured:

          Thinning on *kept* counts with probability (floor+1)/(kept+1). Solving the
          steady state gives kept rate ~ sqrt(arrival rate), so an 80x spread compresses
          only to about 9x. It compresses; it does not level.

          A hard quota, reject once a direction leads the minimum by `balance_slack`. It
          binds immediately -- it cut the spread from 80x to 18.4x at slack 16 -- but it
          deadlocks. re10k produces essentially no roll at all (300 sampled rows of
          re10k_26_35 contained zero roll primaries), and a quota keyed to the minimum over
          all twelve directions blocks every direction once each sits `slack` above a
          minimum that will never rise. Such a source emits 12 x slack samples and then
          stalls for the rest of the run. The probabilistic form cannot deadlock, since its
          acceptance probability is bounded below by 1/(arrivals+1).

        WHAT IT ACHIEVES, AND WHAT IT COSTS. Measured over 420 samples of the arm C
        mixture, all twelve directions land between 4.9% and 13.8% against a uniform 8.33%,
        a 2.8x spread where the unthinned stream reads 36.5x:

            Pan Left     13.8%    Pedestal Up   9.1%    Tilt Up     5.6%
            Pan Right    13.6%    Truck Right   8.3%    Roll CW     5.6%
            Dolly In     11.9%    Dolly Out     8.0%    Tilt Down   5.5%
            Truck Left    7.4%    Pedestal Down 6.2%    Roll CCW    4.9%

        Yaw stays on top because it is abundant enough to accumulate even at a low keep
        probability, and because roll-paired samples inject yaw along with the roll.

        The cost is throughput: 0.61 samples/s per worker against 9.84 unthinned, a 16x
        slowdown, since the loader reads on the order of thirty rows per sample it keeps.
        That is inherent to insisting on equality rather than an artefact of this rule --
        retention cannot exceed twelve times the rarest direction's share. Roll is that
        constraint, and the pool does hold roll: of 525 pairs clearing the caps, roll was
        visible at 40 px or more in 17.3%, second largest in 7.6%, largest in only 1.3%.
        The two-largest rule is what discards it.

        If 16x is too slow for a real run, the fix is architectural rather than statistical.
        Reading only `id`, `instruction` and `meta_info` costs 0.25 s per 300-row file
        against 2.5-6.6 s once `depth` is included, because the depth column is 554-1239 MB
        per file against 1 MB for the rest. So an offline pass can index the pool and
        materialise an exactly balanced row list, after which the loader reads only rows it
        will keep. The remaining way to lift roll specifically is to manufacture it:
        rotating the second view in plane by theta is exactly a roll of theta about the
        optical axis, so the label stays exact and the volume is unlimited.

        Counters are per instance, and each source in each dataloader worker holds its own,
        which is fine here: the rule is a rate ratio rather than a running difference, so
        every instance converges to the same shape independently.
        """
        for n in names:
            self._arrived[n] += 1
        floor = min(self._arrived[p] for p in PRIMITIVES)
        keep = max((floor + 1) / (self._arrived[n] + 1) for n in names)
        return rng.random() < keep

    def build_names(self, row):
        """(primitives to label this pair with, total rotation), or None if gated out.

        The rotation comes back with the names because the lateral question has to state
        it once it passes DISCLOSE_ABOVE_DEG.
        """
        meta = json.loads(row["meta_info"])
        if meta_node(meta).get("scene_name") in HELD_OUT_SCENES:
            return None
        if "names" in row:
            # A materialised parquet: this row already cleared the gates offline, and the
            # depth map it would take to re-derive the label is not in the file. The scene
            # guard above still runs, so the held-out check does not depend on the build
            # having been done right. An empty list cannot occur -- such rows are not
            # written -- but reads as "dropped" rather than crashing if one ever appears.
            names = json.loads(row["names"])
            return (names, float(row["rot_deg"])) if names else None
        pose = read_pose(meta)
        if pose is None:
            return None
        w2c_pair, focal, width = pose
        rot = total_rotation_deg(*w2c_pair)
        if rot > self.max_rotation_deg:
            return None
        # ScanNet++ names its depth column differently from re10k and dl3dv.
        cell = row["depth"] if "depth" in row else row.get("depth_maps")
        depth = scene_depth(cell)
        if not depth:
            return None

        comps = components(w2c_pair, focal, depth)
        if comps[0]["px"] > self.max_frac_width * width:
            return None
        names = describe(comps, self.min_component_px, self.dominance)
        return None if names is None else (names, rot)

    def parse_row(self, row):
        built = self.build_names(row)
        if built is None:
            return dict()
        names, rot = built
        rng = random.Random(zlib.crc32(str(row["id"]).encode()))
        if self.balance and not self.keep_for_balance(names, rng):
            return dict()
        row = row.copy()
        row["conversations"] = conversation(names, rot, rng, self.mcq_share)
        return super().parse_row(row)
