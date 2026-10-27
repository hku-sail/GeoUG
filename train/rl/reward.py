# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""RL 的 verifier 与 reward。

verifier
--------
题库是四选一（`build_rl_debug_set.py` 只取四类比较题），答案是 'A'/'B'/'C'/'D'。
抽取顺序照搬 `inference_depth_cot_v3.py::extract_answer_letter_v3` 的语义：
`</think>` 之后才是答案区，先找显式字母，再退到选项文本匹配。这里重写而不是 import，
是为了不把 RL 侧绑死在评测模块的 transcript 结构上 —— rollout 产出的是纯文本。

reward
------
baseline 用最简形式::

    r = correct - lambda_cost * used_depth

`lambda_cost = 0` 时就是纯提准，门控只受「深度到底帮不帮得上」驱动；
调大则开始惩罚生成。两端都要跑，因为 RESULT.md V20 说明门控当前几乎无判别力
（gate_value = +0.57），先确认 RL 能把它推离随机，再谈省算力。

**这不是 PLAN.md §9 最终想要的形式。** 那里论证的是带预算约束的拉格朗日式
「在触发率 <= 预算的条件下最大化准确率」，lambda 由对偶上升自动调。
固定 lambda 的问题是它和「一次深度值多少准确率」的兑换比隐式绑定，
而那个比值在训练中会漂。baseline 阶段先不引入这层，跑通再说。

四选一而不是 D3 的二选一，好处是蒙对概率 25% 而非 50%，
同样的组内方差下 reward 的信噪比高一倍。
"""
from __future__ import annotations

import json
import re

_LETTERS = ("A", "B", "C", "D")
_THINK_END = "</think>"


def extract_letter(text: str, options: dict | None = None) -> str:
    """从 rollout 的完整助手文本里抽出选项字母；抽不出来返回 ""。"""
    if not text:
        return ""
    # 只看答案区：最后一个 </think> 之后。没有 </think> 就看全文。
    tail = text.rsplit(_THINK_END, 1)[-1]
    # 去掉特殊 token 和 GENERATE_* 标记，免得干扰字母匹配。
    cleaned = re.sub(r"<\|.*?\|>", " ", tail)
    cleaned = re.sub(r"</?GENERATE_[A-Z]+>", " ", cleaned)
    cleaned = re.sub(r"<\/?think>|<DECIDE>", " ", cleaned)

    m = re.search(r"\bAnswer\s*[:：]\s*([A-D])\b", cleaned, re.I)
    if m:
        return m.group(1).upper()
    # "B. parked car" / "(B)" / 单独的 "B"
    m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", cleaned)
    if m:
        return m.group(1).upper()

    # 退到选项文本匹配：只有恰好命中一个选项时才采信。
    if options:
        low = cleaned.lower()
        hits = [k for k, v in options.items() if v and str(v).lower() in low]
        if len(hits) == 1:
            return hits[0].upper()
    return ""


def score(text: str, answer: str, used_depth: bool, options=None,
          lambda_cost: float = 0.0) -> dict:
    """算一条 rollout 的 reward。

    返回的 dict 除了 `reward` 还带着分解项，训练时直接当 metric 打到 wandb ——
    只看合成后的 reward 是看不出「涨的是准确率还是省了生成」的。
    """
    if isinstance(options, str):          # parquet 里存的是 json 字符串
        options = json.loads(options) if options else None
    pred = extract_letter(text, options)
    correct = float(bool(pred) and pred == str(answer).strip().upper())
    cost = float(bool(used_depth))
    return {
        "reward": correct - lambda_cost * cost,
        "correct": correct,
        "used_depth": cost,
        "parsed": float(bool(pred)),      # 解析失败率要单独盯：它会伪装成「答错」
        "prediction": pred,
    }
