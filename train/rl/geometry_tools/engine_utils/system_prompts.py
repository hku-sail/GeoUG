# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""System prompts and mode tokens for the three thinking modes.

This module is the **single source of truth** for the strings that bind the
training trace to the user-visible system prompt:

  * ``Think``           -> always generate a metric depth map + tool call,
                          then answer (forces the depth branch even when
                          a 2D cue would suffice).
  * ``Adaptive think``  -> gate on ``illusion``: visually unambiguous samples
                          answer directly, ambiguous ones go through the
                          depth+tool branch (current default behaviour).
  * ``No think``        -> answer directly with no <think>, no depth, no tool;
                          the trace is just the answer phrase. Has NO system
                          prompt; instead its question gets a short-answer
                          suffix (see ``NO_THINK_QUESTION_SUFFIX``).

Train / eval consistency
------------------------
The data engine bakes the per-sample system prompt into the JSONL record.
The eval harness MUST import these same constants (or copy them verbatim) so
the prompt the model is trained on is byte-identical to the prompt it sees at
inference time -- otherwise the mode-conditioning silently drifts.

Fair ablation
-------------
The image + question BODY are identical across all three modes. The only
input difference is the conditioning instruction: think / adaptive_think
carry it in the system prompt, while no_think carries it as a short-answer
suffix on the question (and has no system prompt). This keeps the comparison
fair -- same image, same question body -- while still letting each mode be
triggered unambiguously at inference time.

The two ``Think``-style modes wrap their reasoning in ``<think>...</think>``
and emit the final answer phrase OUTSIDE the closing tag (no "Answer:"
prefix -- just the answer), e.g.::

    <think>
    ... reasoning, depth gen, tool call/result ...
    </think>

    B. dog

``No think`` mode emits ONLY that answer phrase (no ``<think>``) and has no
system prompt; its question is suffixed with ``NO_THINK_QUESTION_SUFFIX``.
"""

from __future__ import annotations


# ===========================================================================
# Mode keys (use these as identifiers everywhere -- JSONL ``mode`` field, CLI
# ``--modes``, dispatcher dict in render_v3, etc.).
# ===========================================================================
MODE_THINK = "think"
MODE_ADAPTIVE_THINK = "adaptive_think"
MODE_NO_THINK = "no_think"

ALL_MODES: tuple[str, ...] = (MODE_THINK, MODE_ADAPTIVE_THINK, MODE_NO_THINK)


# ===========================================================================
# <think> tokens. Same string the assistant emits at training time, and the
# same string the inference harness should split on at eval time.
# ===========================================================================
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


# ===========================================================================
# System prompts. ``Think`` and ``Adaptive think`` get a real system prompt;
# ``No think`` has NONE -- it relies on the question-suffix below to control
# behaviour. The two prompts share the same <think>...</think> framing; the
# ONLY difference is the final sentence: ``Think`` says "You must use your
# generation ability" (force the depth-gen + tool branch every time), while
# ``Adaptive think`` says "You can adaptively use your generation ability"
# (gate it on whether the 3D arrangement is visually clear).
# ===========================================================================

SYSTEM_PROMPT_THINK = (
    "You are a helpful assistant. You should first think about the reasoning "
    "process in your mind and then provide the user with the answer. The "
    "reasoning process is enclosed within <think> </think> tags, i.e. "
    "<think> reasoning process here </think> answer here. You must use your "
    "generation ability to assist your thinking before answering."
)

SYSTEM_PROMPT_ADAPTIVE_THINK = (
    "You are a helpful assistant. You should first think about the reasoning "
    "process in your mind and then provide the user with the answer. The "
    "reasoning process is enclosed within <think> </think> tags, i.e. "
    "<think> reasoning process here </think> answer here. You can adaptively "
    "use your generation ability to assist your thinking before answering."
)

SYSTEM_PROMPT_NO_THINK = ""  # no system prompt for no_think (see suffix below)


# Suffix appended to the user question when ``mode == "no_think"``. This is
# no_think's ONLY instruction (it has no system prompt): a standard LLaVA-style
# short-answer cue. The question BODY (the part before this line) stays
# byte-identical to the think / adaptive_think question, so the modes remain a
# fair comparison -- they share the same image + question body and differ only
# in (system prompt) vs (this short-answer suffix).
NO_THINK_QUESTION_SUFFIX = "Answer the question with a single word or phrase."


# Lookup tables. Keep these the ONLY way callers map mode -> strings.
SYSTEM_PROMPTS: dict[str, str] = {
    MODE_THINK:          SYSTEM_PROMPT_THINK,
    MODE_ADAPTIVE_THINK: SYSTEM_PROMPT_ADAPTIVE_THINK,
    MODE_NO_THINK:       SYSTEM_PROMPT_NO_THINK,
}


def system_prompt_for(mode: str) -> str:
    """Return the system prompt for ``mode``, or raise if mode is unknown."""
    if mode not in SYSTEM_PROMPTS:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {ALL_MODES}"
        )
    return SYSTEM_PROMPTS[mode]
