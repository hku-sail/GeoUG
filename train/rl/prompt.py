"""Prompt shared by v6 SFT, RL rollout, and evaluation."""

_STEM = (
    "You are a helpful assistant. You should first think about the reasoning "
    "process in your mind and then provide the user with the answer. The "
    "reasoning process is enclosed within <think> </think> tags, i.e. "
    "<think> reasoning process here </think> answer here. "
)

UNIFIED_PROMPT = _STEM + (
    "Within the reasoning process, emit <DECIDE> once you are ready to commit, "
    "immediately followed by either <GENERATE_DEPTH> to generate a depth map and "
    "keep reasoning, or </think> to answer directly."
)

