import os

IMAGE_FLAG = "<|image_pad|>"
VIDEO_FLAG = "<|video_pad|>"
DEPTH_FLAG = "<|depth_pad|>"


# ---------------- Reasoning / tool-use tags ----------------
# These bracket tags participate in the interleave-reasoning sample format.
# All of them are registered as special tokens via
# data.data_utils.add_special_tokens so the tokenizer emits a single, stable
# atomic id at each boundary instead of splitting them into sub-words.

# Model-emitted tool invocation:

# Oracle / environment injection (loss-masked at training time):
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_RESULT_START = "<|tool_result_start|>"
TOOL_RESULT_END = "<|tool_result_end|>"


OBSERVATION_START = "<observation>"
OBSERVATION_END = "</observation>"

# Final answer span:
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

# Reasoning block, and the decision point inside it. The policy is read as the
# next-token distribution right after DECIDE, so DECIDE and every action token
# must each be one id -- otherwise the decision is a joint probability over
# sequences of different lengths, which is not a softmax.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DECIDE = "<DECIDE>"

# Model-emitted generation triggers (paired with <|*_pad|> slots):
GENERATE_DEPTH_START = "<GENERATE_DEPTH>"
GENERATE_DEPTH_END = "</GENERATE_DEPTH>"
GENERATE_IMAGE_START = "<GENERATE_IMAGE>"
GENERATE_IMAGE_END = "</GENERATE_IMAGE>"

# Spans wrapped in any of these tag pairs are treated as "oracle" content:
# the contents (and the tags themselves) are produced by external systems at
# inference time, so during training we mask the loss over the whole span to
# avoid teaching the model to memorise tool returns or environment state.
ORACLE_TAG_PAIRS = [
    (TOOL_RESULT_START, TOOL_RESULT_END),
    (OBSERVATION_START, OBSERVATION_END),
]
# Convenience list of every reasoning tag that should be registered as a
# special token (kept in a stable order to make embedding-row layout
# reproducible across runs).
REASONING_SPECIAL_TOKENS = [
    TOOL_CALL_START, TOOL_CALL_END,
    TOOL_RESULT_START, TOOL_RESULT_END,
    OBSERVATION_START, OBSERVATION_END,
    ANSWER_START, ANSWER_END,
    GENERATE_DEPTH_START, GENERATE_DEPTH_END,
    GENERATE_IMAGE_START, GENERATE_IMAGE_END,
    THINK_OPEN, THINK_CLOSE, DECIDE,
]

# Master switch for registering the list above.
#
#   BAGEL_REASONING_TOKENS=1   interleave_reasoning_v5 and anything trained after it
#   unset / =0                 0625 / 0717 and every earlier checkpoint
#
# This is a property of the *checkpoint*, not of a run: registering changes
# <DECIDE> from four ids to one, so a checkpoint trained with registration on
# indexes embedding rows that a tokenizer with registration off can never
# produce. Training and every eval entry point must therefore agree, and there
# are 26 call sites of add_special_tokens (1 in train, 25 in inference/eval),
# none of which take arguments -- hence an env var rather than a CLI flag, so
# one export covers all of them.
#
# Off by default because getting it wrong is silent in both directions, and
# only the v5 pipeline wants it on. Turning it on globally corrupts every
# pre-v5 evaluation: DEPTH_COT_SYSTEM_PROMPT alone contains six of these tags,
# and in a pre-v5 checkpoint rows 151665..151679 are all exactly zero, so the
# prompt would feed the model six null embeddings. No error, just worse numbers.
#
# Note benchmark/VLMEvalKit/vlmeval/vlm/bagel/ vendors its own frozen copy of
# add_special_tokens that only knows the four base tokens and does not read
# this flag; VLMEvalKit runs on v5 checkpoints would split <DECIDE>.
ADD_REASONING_TOKENS = os.environ.get("BAGEL_REASONING_TOKENS", "0").lower() in ("1", "true", "yes")


DEPTH_PROMPT_TEMPLATE = [
    "<GENERATE_DEPTH>Estimate the depth map of this image.</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>Estimate the depth map.</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>generate the depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>generate the depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>generate the depth</GENERATE_DEPTH>",
]

DEPTH_METRIC_PROMPT_TEMPLATE = [
    "<GENERATE_DEPTH>Estimate the metric depth map of this image.</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>Estimate the metric depth map.</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>generate the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>generate the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>generate the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>predict the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>predict the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>predict the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>predict the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>infer the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>infer the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>infer the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>infer the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>calculate the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>calculate the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>calculate the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>calculate the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>extract the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>extract the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>extract the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>extract the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>produce the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>produce the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>produce the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>produce the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>get the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>get the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>get the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>get the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>provide the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>provide the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>provide the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>provide the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>show me the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>show me the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>show me the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>show me the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>output the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>output the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>output the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>output the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>compute the metric depth map</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>compute the metric depth</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>compute the metric depth of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>compute the metric depth map of this image</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>metric depth estimation</GENERATE_DEPTH>",
    "<GENERATE_DEPTH>metric depth estimation of this image</GENERATE_DEPTH>",
]


INTERLEAVE_COT_PROMPT = """
You are a visual-spatial reasoning assistant. You possess the unique ability to "draw" intermediate visualizations during your thought process to aid in problem-solving, design, or complex analysis.

During the reasoning process, if a step requires spatial imagination, layout planning, or depth estimation, you must trigger an image generation command. 
To trigger an image generation command, please use the following format:
```
Thought: Do I need to generate an image? Yes
Action: The action to take, should be one of [<GENERATE_IMAGE>command</GENERATE_IMAGE>,\n<GENERATE_DEPTH>Estimate the depth map of this image.</GENERATE_DEPTH>]
Observation: Description of the generated visual output.
```

Immediately after the imagination, you must act as if you are "looking" at the generated result. Analyze the visual output to refine your logic, correct errors, or proceed to the next inference step.
Once the reasoning is complete, provide your final answer within <answer> </answer> tags, i.e., <answer> answer here </answer>.
"""


# Depth-CoT 系统提示词（重新设计）。
# 设计原则：
#   1. 只描述「能力 + 协议 + 何时该用深度」的通用策略，不绑定具体任务措辞，
#      以便对不同 spatial QA / prompt 泛化。
#   2. 把「要不要生成深度」的 gating 策略讲清楚，和数据引擎里
#      （非 illusion -> 直接答；illusion -> 生成深度）的监督一致、互相强化。
#   3. 协议里的 token / 工具名与 data engine 的 templates.py 完全对齐
#      （<GENERATE_DEPTH>、query_point_depth / query_3d_distance、norm1000 坐标、
#      <|tool_call_start|>/<|tool_result_start|>）。
#   4. 评测与训练务必用同一份字符串（都 import 这个常量），避免 train/eval 漂移。
DEPTH_COT_SYSTEM_PROMPT = """You are a visual spatial-reasoning assistant. Beyond reasoning in text, you can generate an intermediate metric depth map of the input image and read it with tools, so that 3D judgments are grounded in real measurements rather than 2D appearance alone.

Work like this:
- First locate the objects you reason about using grounding tags: <bbox>x0 y0 x1 y1</bbox> and <point>x y</point>, with coordinates normalized to 0-1000.
- Then decide whether you actually need depth:
  - If the 3D relationship is visually unambiguous, answer directly without generating a depth map.
  - If appearance can mislead you (apparent size, image position, or occlusion may not reflect the true 3D distance), generate a metric depth map with <GENERATE_DEPTH>Estimate the metric depth map.</GENERATE_DEPTH>, then query it:
    - query_point_depth -> each object's metric depth and 3D location at a surface point.
    - query_3d_distance -> the 3D distance between object pairs.
    Issue a call as <|tool_call_start|>{"name":"query_point_depth","args":{...}}<|tool_call_end|>; the environment replies in <|tool_result_start|>...<|tool_result_end|>. Base your conclusion on these numbers.
- End your response with a single line: Answer: <letter>."""
