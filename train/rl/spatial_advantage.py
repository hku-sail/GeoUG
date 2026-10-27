# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""组内 advantage：把分母里的 λ 去掉。

要修的 bug（TRAINING_CODE_REVIEW.md 的 P0）
-------------------------------------------
上游 `stat_tracking.py:36` 在 `global_std=True` 时算的是

    std = np.std(rewards)          # rewards 只是**当前这一道题**的那一组

因为 `common.py::compute_advantages` 每次只 gather 一道题的 G 条 rollout、
算完立刻 `stat_tracker.clear()`。所谓 "global" 其实是**单题组内**。

后果：对**组内全对**（或全错）的那批题，`correct` 项完全打平，只剩成本项 ——

    r_i  = 1 − λ·d_i
    A_i  = (r_i − mean r) / std r
         = −λ(d_i − mean d) / (λ·std d + 1e-4)
         ≈ −(d_i − mean d) / std d          <-- **λ 被约掉**

实测（`PerPromptStatTracker(global_std=True)`，8 直答 / 8 深度、全对）：

    λ=0.05 -> |A| = 0.996      λ=0.1 -> 0.998      λ=0.5 -> 0.9996

也就是**任何 λ>0 都给出满幅 ±1 的反深度 advantage**，而真正该起作用的、
正确率有方差的组只有 `0.1λ` 量级 —— 差 10 倍。

这直接解释了「λ=0.05/0.07/0.1 曲线重合」和「λ=0 到 0.05 的突变」，
所以那组实验**不能**用来论证任务本身双稳。见 CODE.md §证据四 的更正。

修法：分母换成不含 λ 的固定尺度
--------------------------------
    A_i = (r_i − mean_prompt(r)) / adv_scale

* **分子保持 `correct − λ·d` 不动**，所以 λ 的物理含义（"一次深度值多少准确率"）
  和 Δ 仍然同量纲、可直接比较。报告建议的
  `normalize(correct) − λ·centered(d)` 会把两项用不同尺度缩放，
  破坏这个对应关系，所以没采用（这是报告第 4 条留的那个口子）。
* **`adv_scale` 是常数**，不随组变、不含 λ。于是：
  - 全对组：`A_i = −λ(d_i − mean d)/adv_scale`，**随 λ 线性变化**，不再被约掉；
  - 可学组：`A_depth − A_direct = (Δ̂ − λ)/adv_scale`，正是想要的量。
* 顺带一个**期望中的**副作用：奖励方差小的组（全对/全错）现在贡献的梯度**更小**，
  而不是和可学组一样满幅。这是对的 —— 那些组本来就没什么信息。

`adv_scale` 取 0.5 是因为准确率约 0.5 时伯努利的 std 就是 0.5，
这样修完之后可学组的 advantage 幅值和修之前同量级，学习率不用跟着改。
"""
from __future__ import annotations

import numpy as np

from train.rl.engine.alpha_grpo.stat_tracking import PerPromptStatTracker


class FixedScaleStatTracker(PerPromptStatTracker):
    """`(r − 组内均值) / adv_scale`，`adv_scale` 是常数。

    只覆盖 `type='grpo'` 这条路 —— 其余（rwr / sft / dpo / remax / grpo_anchor）
    我们不用，落到父类去，免得复制一份会漂的实现。
    """

    def __init__(self, global_std=False, adv_scale=0.5):
        super().__init__(global_std=global_std)
        self.adv_scale = float(adv_scale)

    def update(self, prompts, rewards, type="grpo", group_mean=None):  # noqa: A002
        if type != "grpo" or group_mean is not None:
            return super().update(prompts, rewards, type=type, group_mean=group_mean)

        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        advantages = np.zeros_like(rewards)
        for prompt in np.unique(prompts):
            mask = prompts == prompt
            r = rewards[mask]
            # 仍然把这组塞进 self.stats，好让 `get_stats()` 报的 group_size 不变。
            self.stats.setdefault(prompt, []).extend(r)
            self.history_prompts.add(hash(prompt))
            advantages[mask] = (r - r.mean()) / self.adv_scale
        return advantages
