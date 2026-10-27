# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""`spatial_grpo`：GRPO 加一条「占位 step 不产生梯度」的规则。

为什么需要它
------------
`train.py` 的训练循环是**逐 step** 调 `accelerator.backward()` 的，而我们已经把
`sync_each_batch` 打开（见 spatial_train.py），于是**每次 backward 都是一次
DeepSpeed 梯度规约**——一次集合通信。

我们的轨迹长度跟分支走：

    直答分支   [text]                                              -> 1 次 backward
    深度分支   [text, input_image, input_image, text, input_text, text] -> 3 次

组是跨进程成的（每张卡同一道题、各采一条轨迹），分支又是采样决定的，于是
4 张卡很容易一张走深度、一张直答 —— backward 次数 1 vs 3，集合通信对不上，
第一次 step 就死锁在 NCCL 里。（实测：rank 3 卡在 `allreduce_gradients`，
rank 1/2 已经跑到下一轮 rollout。）

治法
----
把每条轨迹的**计 loss step 数补齐到常数 K**（`config.train.loss_steps_per_sample`），
补出来的占位 step 照常前向、照常 backward、照常规约，但 loss 恒为 0 ——
通信次数对齐了，梯度不受污染。

为什么不换个补法
----------------
- **拆分文本段**凑数不行：`prepare_prompts` 每调一次都会给这段文本包一对
  bos/eos（`bagel.py:515`），把一段拆成三段就等于往序列里塞了两对本不存在的
  token，重放出来的就不是采样时那条轨迹了。
- **只对第一段计 loss** 也不行：直答分支的第一段就是整条轨迹（含答案），
  深度分支的第一段只到 `<GENERATE_DEPTH>`，两条分支被训练的部分不对等。

占位 step 的 loss 写成 `logp.sum() * 0.0` 而不是 `torch.zeros(())`：
要留着通往模型的那条图，backward 才会真的走一遍、触发规约。

再一条：分段必须按 token 数重新加权
------------------------------------
补齐 step 数解决了死锁，但**留下一个静默的分支偏置**，PLAN.md §10.5 要核的就是它。

`train.py:936` 是**逐 step** `accelerator.backward()`，梯度在一个 sample 内累加，
也就是一条轨迹的总 loss = **各 loss step 之和**。而序列级（GSPO）重要性比值下，
每个 step 的 loss 是个标量 ≈ `-A·ratio`，`ratio` 在 ppo_epoch=1 时恒为 1。于是：

    深度分支  3 个计 loss step  -> 总 loss ≈ -3A
    直答分支  1 个计 loss step  -> 总 loss ≈ -1A   （另外 2 个是零梯度的占位）

**深度分支白拿 3 倍梯度**，而这个倍数和「深度好不好」毫无关系，纯粹因为注入把它
切成了三段。门控会被这个伪信号推向开火 —— 和 §2.2 担心的 2.48:1 长度偏置是同一
个病，只是从 token 级搬到了 step 级，而且更狠（3:1，且是结构性的、不随数据变）。

治法是给每个 step 的 loss 乘上 `n_token(step) / n_token(整条轨迹)`：

* 两条分支的**权重和都恰好是 1**，分支偏置消失；
* 这不是随手一归一化，而是**分段 GSPO 的正确离散化**。序列级比值是
  `s = exp(mean(log_ratio))`，按 token 数加权平均各段的 `mean(log_ratio)`
  正好还原整条轨迹的 `mean(log_ratio)`：

      Σ_i (n_i/N)·(1/n_i)Σ_{t∈i} log r_t  =  (1/N)Σ_t log r_t

  （唯一的偏差是 min/clip 逐段生效而非全局生效，那是一阶小量。）

用等权 `1/段数` 也能消掉分支偏置，但那等于把一个 3 token 的段和一个 300 token 的段
看得一样重，丢掉了 GSPO 的长度归一化语义。所以按 token 加权。

权重和要归到 K，不是 1（我自己引入的回归）
-------------------------------------------
DeepSpeed 的 `backward()` 会把每次的 loss 除以
`gradient_accumulation_steps`，而 `train.py:298` 把它设成
`gas × backward_calls_per_sample` = gas × K：

    engine.py:3468   loss = loss / self.gradient_accumulation_steps()

原来每段权重都是 1、一个 sample 三段加起来是 **K**，除以 gas×K 之后正好是
「按 sample 取平均」—— 上游这个尺度是对的。我按 token 加权让权重和变成 **1**，
于是有效梯度**被缩小了 K=3 倍**（`TRAINING_CODE_REVIEW.md` 的 P1）。
它不会重新引入分支偏置，但会让 config 里的 `learning_rate` 和实际步长差 3 倍，
也让这套实验的 lr 没法和不分段的实验比。

所以权重乘回 K：`w_i = K · n_i / N`，`Σ w_i = K`，和上游一致。
"""
from __future__ import annotations

import torch  # noqa: E402

from train.rl.engine.alpha_grpo.algorithms import register
from train.rl.engine.alpha_grpo.algorithms.grpo import GRPOAlgorithm

# 占位 step 用的文本。它会被前向一遍并进入上下文，所以要短；
# 又因为占位 step 永远排在轨迹末尾，进不进上下文都无所谓。
PAD_TEXT = "."

_LOSS_STEPS = ("think_text", "text")


def is_pad(gen_step) -> bool:
    return bool(getattr(gen_step, "is_pad", False))


def _n_tokens(gen_step) -> int:
    """这一段 rollout 时记下的 token 数。

    用 rollout 存的那份而不是训练时重算的那份，是为了让分子分母同源 ——
    两者本该相等（同一段文本、同一个 tokenizer，见 spatial_inferencer 的说明），
    混用的话万一哪天不等，权重会悄悄不归一。
    """
    lp = gen_step.text_per_token_log_probs
    return 0 if lp is None else int(lp.reshape(-1).shape[0])


def step_weight(sample, gen_step, k=1) -> float:
    """`k · n_token(本段) / n_token(整条轨迹)`，占位段不计入分母。

    `k` 取 `backward_calls_per_sample`，让 `Σ w_i = k` 而不是 1 —— 理由见模块
    docstring（DeepSpeed 按 gas×k 缩放，权重和必须是 k 才和上游同尺度）。

    整条轨迹一个 token 都没有（理论上进不来，`_pad_loss_steps` 会先丢掉）时退回 k，
    不除零。
    """
    total = sum(_n_tokens(s) for s in sample["generation_steps"]
                if s.step_type in _LOSS_STEPS and not is_pad(s))
    n = _n_tokens(gen_step)
    if total <= 0 or n <= 0:
        return float(k)
    return k * n / total


@register("spatial_grpo")
class SpatialGRPOAlgorithm(GRPOAlgorithm):

    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info,
                          gen_step=None, text_per_token_log_probs=None, **kwargs):
        if is_pad(gen_step):
            # 什么都不往 info 里塞：train.py 收尾时是 `mean(stack(v))`，
            # 各卡的列表长度本来就不用一致，少塞几项比塞几个 0 更干净
            # （塞 0 会把 text_policy_loss 的均值往 0 拉）。
            return info, text_per_token_log_probs.sum() * 0.0

        info, loss = super().compute_text_loss(
            config, sample, accelerator, info,
            gen_step=gen_step, text_per_token_log_probs=text_per_token_log_probs,
            **kwargs)

        # 分支偏置的修正 + 权重和归到 K，见模块 docstring。
        k = getattr(config.train, "backward_calls_per_sample", None) or 1
        w = step_weight(sample, gen_step, k=k)
        # 核对用（PLAN.md §10.5 要的那组读数）：每段的 token 数和权重，按分支分开看。
        # `seg_weight` 的均值应当是 K/段数，各分支的**和**都应当是 K。
        info["seg_tokens"].append(torch.tensor(float(_n_tokens(gen_step))))
        info["seg_weight"].append(torch.tensor(w))
        return info, loss * w
