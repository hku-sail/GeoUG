# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""`spatial_gate_draft` task + `spatial_grpo_draft` 算法：用 v6 草答做门控的反事实基线。

`spatial_gate` / `spatial_grpo` 原样保留，作为 ablation（整条轨迹一个 advantage）。

要修的问题
----------
v6 的轨迹是 `…My initial answer without depth: X\\n\\n<DECIDE>` 再二选一：
直答时终答 = 草答；开深度时按工具读数重新作答。原方案整条轨迹一个 advantage
`A = (r − 组均值 r) / adv_scale`，`r = correct − λ·used_depth`，有三个毛病：

1. 组内 16 条全开深度时每条都扣同一个 λ，均值里抵消，门控没有梯度 ——
   进了「全开」就出不来（v6 λ=0.05 在 step 9–10 有 28/37 组是这样）。
2. 草答已经对、深度白开的那条，只要组均值 < 1−λ 就拿正 advantage。
3. 草答错、深度救回来时，错的草答字母跟着拿正 advantage。

分段 advantage
--------------
草答写在门控动作之前，所以「不开深度的结果」`c_draft` 对每条轨迹都是已知的，
而且只依赖动作之前的状态 —— 拿它当门控的基线不引入偏差。按 token 切四段：

| 段 | 范围 | advantage（均除以 adv_scale） |
|---|---|---|
| grounding | 草答句之前 | `r − 组均值 r`（= 原方案） |
| 草答 | 草答句 … `<DECIDE>`（含） | `c_draft − 组均值 c_draft` |
| 门控 | `<DECIDE>` 后的第一个 token | `r − c_draft` |
| 执行 | 门控之后直到轨迹结束 | 深度分支 `c − 组内开深度那些的均值 c`；直答分支 `c − c_draft` |

门控这一项**不减组均值**：减了 λ 就又在全开组里抵消了。它本身已经是
`Q(s, 动作) − Q(s, 直答)`，直答恒为 0，深度 = 修正 +(1−λ) / 白开 −λ /
没修好 −λ / 改错 −(1+λ)。

草答抽不出字母时 `c_draft` 无定义：门控和执行退回原方案的 `r − 组均值 r`，
草答段照常按 `c_draft = 0` 算（写不出草答本身就该罚）。轨迹没到 `<DECIDE>`
或切分失败时整条退回原方案。

序列级（GSPO）比值按段各算一个，损失按段 token 数加权 —— 每个 token 的梯度权重仍是
`A_段 / N`，只有一段时与 `spatial_grpo` 逐位相同（`test_draft_advantage.py` 对拍）。

各段可切换（ablation 接口，`V6_RL_2_REVIEW.md` §8–10）
------------------------------------------------------
上表是默认值（= 第二版）。`config.train` 里四个开关：

| 开关 | 取值 |
|---|---|
| `draft_ground_adv` | `r`：`r − r̄`（默认）；`c`：`c − c̄`，不含 λ；`depth`：开深度样本 `c − c̄_depth`，直答 0 |
| `draft_draft_adv` | `grpo`：`c_draft − c̄_draft`（默认）；`zero`：0；`format`：`parsed − 组均值 parsed` |
| `draft_gate_adv` | `sampled`：`r − c_draft`（默认）；`two_branch`：见下 |
| `force_depth_per_group` | 每组随机强制 m 条开深度（默认 0） |

`draft_gate_coef` 乘在门控 advantage 上（默认 1，即与第二版同尺度）。

`two_branch` 是以 `V = π_d·Q_depth + (1−π_d)·Q_direct` 为基线的采样动作策略梯度，
`Q_direct = c_draft` 精确已知：

- 自然开深度：`(1−π_d)·(c − λ − c_draft)`（用自己的 c，无偏）；
- 自然直答：`−π_d·(Q̂_depth − λ − c_draft)`，`Q̂_depth` 取组内草答对错相同的
  **强制**开深度样本的 c 均值（没有强制样本时用自然开深度的）；该格为空时退回合并两格，
  都没有则为 0；
- 强制样本：0（门控 token 不是采出来的）。

`π_d` 由门控 token 的 rollout log-prob 得到：开深度 `exp(lp)`，直答 `1 − exp(lp)`。
SFT 500 步在决策点上两 token 质量 ≥ 0.9976，误差可忽略。
门控 token 不是这两个之一时退回 `A_std`。
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import torch

from accelerate.utils import gather_object  # noqa: E402

from train.rl.engine.alpha_grpo.algorithms import register as register_algorithm
from data.constant import DECIDE, GENERATE_DEPTH_START, THINK_CLOSE  # noqa: E402
from train.rl.reward import extract_letter  # noqa: E402
from train.rl.spatial_algorithm import SpatialGRPOAlgorithm, is_pad, step_weight, _LOSS_STEPS  # noqa: E402
from train.rl.spatial_task import SpatialGateTask  # noqa: E402
from train.rl.engine.alpha_grpo.tasks import register as register_task

# Must match the draft cue used to render the released SFT data.
DRAFT_CUE = "My initial answer without depth:"


def draft_letter(text, options=None):
    """草答句里的字母；没有草答句、没有 `<DECIDE>` 或抽不出返回 ""。"""
    i = text.find(DRAFT_CUE)
    j = text.find(DECIDE, i + 1) if i >= 0 else -1
    if j < 0:
        return ""
    if isinstance(options, str):              # parquet 里存的是 json 字符串
        options = json.loads(options) if options else None
    return extract_letter(text[i + len(DRAFT_CUE):j], options)


def region_sizes(tokenizer, text, n_logprobs):
    """决策段的 `(n_grounding, n_draft, n_gate, n_exec)`，按 per-token log-prob 的位置切。

    log-prob 覆盖 `encode(text) + [eos]`：`prepare_prompts` 包成 `[bos]+ids+[eos]`，
    `forward_cache_update_text` 再错一位（bagel.py）。切不出来返回 None。
    训练用的是慢速 `Qwen2Tokenizer`，没有 offset mapping，所以草答句的边界靠
    「前缀的编码恰好是整段编码的前缀」来定，不成立就不单列草答段。
    """
    ids = tokenizer.encode(text)
    if len(ids) + 1 != n_logprobs:
        return None
    decide_id = tokenizer.convert_tokens_to_ids(DECIDE)
    if decide_id not in ids:
        return None
    p = ids.index(decide_id)
    if p + 1 >= len(ids):                       # <DECIDE> 之后没有动作 token
        return None
    n_pre = p + 1
    n_ground = n_pre
    c = text.find(DRAFT_CUE)
    if 0 <= c < text.find(DECIDE):
        head = tokenizer.encode(text[:c])
        if ids[:len(head)] == head:
            n_ground = len(head)
    return n_ground, n_pre - n_ground, 1, n_logprobs - n_pre - 1


def pi_depth(tokenizer, text, gate_lp, used_depth):
    """门控 token 的 log-prob -> `π(开深度)`；门控 token 不是两种动作之一返回 nan。"""
    ids = tokenizer.encode(text)
    decide_id = tokenizer.convert_tokens_to_ids(DECIDE)
    if decide_id not in ids or ids.index(decide_id) + 1 >= len(ids):
        return float("nan")
    a = ids[ids.index(decide_id) + 1]
    want = tokenizer.convert_tokens_to_ids(GENERATE_DEPTH_START if used_depth else THINK_CLOSE)
    if a != want:
        return float("nan")
    p = float(np.exp(gate_lp))
    return p if used_depth else 1.0 - p


def draft_advantages(prompts, r, c, c_draft, draft_ok, depth, adv_scale, *,
                     forced=None, pi_d=None, lam=0.0,
                     ground="r", draft="grpo", gate="sampled", gate_coef=1.0):
    """各条轨迹各段的 advantage（均已除以 adv_scale），形状都是 (G,)。

    返回 `(adv, q_hat, q_src)`：`adv` 的键是 std / ground / draft / gate / exec；
    `q_hat` 是直答样本用的 `Q̂_depth`（其余 nan），`q_src` 是它的来源
    （1 同格，2 合并两格，0 没有）。各模式见模块 docstring。
    """
    prompts = np.asarray(prompts)
    n = len(r)
    forced = np.zeros(n, bool) if forced is None else np.asarray(forced, bool)
    adv = {k: np.zeros(n) for k in ("std", "ground", "draft", "gate", "exec")}
    exec_depth = np.zeros(n)
    q_hat, q_src = np.full(n, np.nan), np.zeros(n, int)
    for p in np.unique(prompts):
        m = prompts == p
        adv["std"][m] = r[m] - r[m].mean()
        md = m & depth
        if md.any():
            exec_depth[md] = c[md] - c[md].mean()
        if draft == "grpo":
            adv["draft"][m] = c_draft[m] - c_draft[m].mean()
        elif draft == "format":
            ok = draft_ok.astype(float)
            adv["draft"][m] = ok[m] - ok[m].mean()
        elif draft != "zero":
            raise ValueError(f"draft_draft_adv={draft!r}")
        if ground == "c":
            adv["ground"][m] = c[m] - c[m].mean()
        if gate == "two_branch":
            pool = md & draft_ok & (forced if (md & forced).any() else ~forced)
            for i in np.flatnonzero(m & ~depth & draft_ok & ~forced):
                cell = pool & (c_draft == c_draft[i])
                if cell.any():
                    q_hat[i], q_src[i] = c[cell].mean(), 1
                elif pool.any():
                    q_hat[i], q_src[i] = c[pool].mean(), 2

    if ground == "r":
        adv["ground"] = adv["std"].copy()
    elif ground == "depth":
        adv["ground"] = np.where(depth, exec_depth, 0.0)
    elif ground != "c":
        raise ValueError(f"draft_ground_adv={ground!r}")

    if gate == "sampled":
        a_gate = np.where(draft_ok, r - c_draft, adv["std"])
    elif gate == "two_branch":
        pi = np.full(n, np.nan) if pi_d is None else np.asarray(pi_d, float)
        ok = draft_ok & ~np.isnan(pi)
        pi = np.nan_to_num(pi)
        on_depth = (1 - pi) * (r - c_draft)
        on_direct = np.where(np.isnan(q_hat), 0.0, -pi * (np.nan_to_num(q_hat) - lam - c_draft))
        a_gate = np.where(ok, np.where(depth, on_depth, on_direct), adv["std"])
    else:
        raise ValueError(f"draft_gate_adv={gate!r}")
    adv["gate"] = np.where(forced, 0.0, a_gate) * gate_coef

    adv["exec"] = np.where(depth, exec_depth, np.where(draft_ok, c - c_draft, adv["std"]))
    return {k: v / adv_scale for k, v in adv.items()}, q_hat, q_src


@register_task("spatial_gate_draft")
class SpatialGateDraftTask(SpatialGateTask):
    """rollout 与 `spatial_gate` 相同（外加可选的强制开深度），只换 advantage 的算法。"""

    def rollout_force(self, config, prompt, global_step):
        """每组随机挑 `force_depth_per_group` 张卡强制开深度。

        一组 = 同一道题在全部卡上各一条（G = world_size），所以各卡用同一个种子
        就能算出同一组选择，不需要通信。
        """
        m = int(getattr(config.train, "force_depth_per_group", 0) or 0)
        if m <= 0:
            return None
        world = self.accelerator.num_processes
        seed = zlib.crc32(f"{getattr(config, 'seed', 0)}:{global_step}:{prompt}".encode())
        chosen = np.random.default_rng(seed).choice(world, size=min(m, world), replace=False)
        return "depth" if self.accelerator.process_index in chosen else None

    def reward_postprocess(self, config, sample, stat_tracker, ctx=None):
        if sample["is_postprocessed"]:
            return sample, False
        lam = self.cost_lambda(config)            # 本条奖励用的 λ；_log_group 可能更新它
        self._log_group(config, sample, ctx)

        acc = ctx.accelerator
        meta = sample["prompt_metadata"][0]
        text = sample["think_texts"][0]
        letter = draft_letter(text, meta.get("options"))
        r = sample["rewards"]
        dev = acc.device
        r["draft_parsed"] = torch.tensor([float(bool(letter))], device=dev)
        r["draft_correct"] = torch.tensor(
            [float(bool(letter) and letter == str(meta["answer"]).strip().upper())], device=dev)

        loss_steps = [s for s in sample["generation_steps"]
                      if s.step_type in _LOSS_STEPS and not is_pad(s)]
        k_decide, sizes = None, None
        for k, s in enumerate(loss_steps):
            if DECIDE in s.texts[0]:
                n_lp = int(s.text_per_token_log_probs.reshape(-1).shape[0])
                k_decide, sizes = k, region_sizes(self.inferencer.tokenizer, s.texts[0], n_lp)
                break
        split_ok = sizes is not None
        r["split_ok"] = torch.tensor([float(split_ok)], device=dev)
        pi_d = float("nan")
        if split_ok:
            s = loss_steps[k_decide]
            gate_lp = float(s.text_per_token_log_probs.reshape(-1)[sizes[0] + sizes[1]])
            pi_d = pi_depth(self.inferencer.tokenizer, s.texts[0], gate_lp,
                            bool(r["used_depth"].item()))
        r["pi_depth"] = torch.tensor([pi_d], device=dev)

        g = {key: acc.gather(r[key].detach().clone()).double().cpu().numpy()
             for key in ("avg", "correct", "used_depth", "draft_parsed", "draft_correct",
                         "split_ok", "forced", "pi_depth")}
        prompts = gather_object(sample["prompts"])
        letters = gather_object([letter])
        tr = config.train
        adv, _, q_src = draft_advantages(
            prompts, g["avg"], g["correct"], g["draft_correct"],
            g["draft_parsed"].astype(bool), g["used_depth"].astype(bool), tr.adv_scale,
            forced=g["forced"].astype(bool), pi_d=g["pi_depth"], lam=lam,
            ground=getattr(tr, "draft_ground_adv", "r"),
            draft=getattr(tr, "draft_draft_adv", "grpo"),
            gate=getattr(tr, "draft_gate_adv", "sampled"),
            gate_coef=float(getattr(tr, "draft_gate_coef", 1.0)))
        mine = {k: float(v[acc.process_index]) for k, v in adv.items()}
        # pi_depth 在切分失败时是 nan；wandb 的 reward_* 按均值汇总，不能带 nan。
        r["pi_depth"] = torch.nan_to_num(r["pi_depth"], nan=0.0)

        if split_ok:
            n_ground, n_draft, n_gate, n_exec = sizes
            for k, s in enumerate(loss_steps):
                n = int(s.text_per_token_log_probs.reshape(-1).shape[0])
                if k < k_decide:
                    s.adv_regions = [(n, mine["ground"])]
                elif k == k_decide:
                    s.adv_regions = [(n_ground, mine["ground"]), (n_draft, mine["draft"]),
                                     (n_gate, mine["gate"]), (n_exec, mine["exec"])]
                else:
                    s.adv_regions = [(n, mine["exec"])]
        else:
            for s in loss_steps:
                s.adv_regions = [(int(s.text_per_token_log_probs.reshape(-1).shape[0]),
                                  mine["std"])]

        # 只看真正会用到的量：切分失败的轨迹只用 A_std。
        ok = g["split_ok"].astype(bool)
        live = np.abs(adv["std"][~ok]).sum() + sum(
            np.abs(adv[k][ok]).sum() for k in ("ground", "draft", "gate", "exec"))
        if acc.is_main_process:
            self._log_draft(g, adv, letters, q_src,
                            show_q=getattr(tr, "draft_gate_adv", "sampled") == "two_branch")
            print("think_text", sample["think_texts"], flush=True)
        if live == 0:
            if acc.is_local_main_process:
                print(f"Skipping rollout for step {ctx.global_step} - all advantages are zero")
            sample["is_postprocessed"] = True
            return sample, True

        t = torch.tensor([[mine["std"]]], device=dev, dtype=torch.float64)
        num_train_timesteps = max(int(config.sample.num_steps * config.train.timestep_fraction), 1)
        sample["advantages"] = t.repeat(1, num_train_timesteps)
        sample["text_advantages"] = sample["advantages"]
        sample["is_postprocessed"] = True
        return sample, False

    @staticmethod
    def _log_draft(g, adv, letters, q_src, show_q):
        """草答 → 终答的四种情况计数，外加各段 advantage。和 `[gate]` 那行配着看。

        - 四种情况（fix / redundant / futile / harm）只数**自然**开深度的，强制的另起一段；
        - `conf`：众数草答字母占已解析草答的比例（1.0 = 组内完全笃定），置信度磨尖的代理；
        - `π_d`：自然样本的均值；`Q̂`：直答样本用的 Q̂_depth 来源（同格 / 合并 / 无）。
        """
        d = g["used_depth"].astype(bool)
        ok = g["draft_parsed"].astype(bool)
        f = g["forced"].astype(bool)
        dc, c = g["draft_correct"].astype(bool), g["correct"].astype(bool)

        def cases(mask):
            return (f"fix {int((mask & ~dc & c).sum())} redundant {int((mask & dc & c).sum())} "
                    f"futile {int((mask & ~dc & ~c).sum())} harm {int((mask & dc & ~c).sum())}")

        parsed = [x for x, o in zip(letters, ok) if o]
        conf = max(parsed.count(x) for x in set(parsed)) / len(parsed) if parsed else float("nan")
        pi = g["pi_depth"][~f]
        pi = pi[~np.isnan(pi)]
        line = (f"[draft] parsed {int(ok.sum())}/{len(ok)} | draft acc "
                f"{dc[ok].mean() if ok.any() else float('nan'):.2f} conf {conf:.2f} | depth: "
                f"{cases(d & ok & ~f)}")
        if f.any():
            line += f" | forced {int(f.sum())}: {cases(d & ok & f)}"
        line += (f" | π_d {pi.mean() if len(pi) else float('nan'):.3f}"
                 f" | split_ok {int(g['split_ok'].sum())}/{len(ok)}")
        need_q = ~d & ok & ~f
        if show_q and need_q.any():
            line += (f" | Q̂ 同格/合并/无 {int((q_src[need_q] == 1).sum())}/"
                     f"{int((q_src[need_q] == 2).sum())}/{int((q_src[need_q] == 0).sum())}")
        print(line, flush=True)
        print("rewards", g["avg"])
        for k in ("std", "ground", "draft", "gate", "exec"):
            print(f"A_{k}", np.round(adv[k], 3))


@register_algorithm("spatial_grpo_draft")
class SpatialGRPODraftAlgorithm(SpatialGRPOAlgorithm):
    """`gen_step.adv_regions` 存在时按段算序列级比值和损失；否则同 `spatial_grpo`。"""

    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info,
                          gen_step=None, text_per_token_log_probs=None, **kwargs):
        regions = getattr(gen_step, "adv_regions", None)
        if is_pad(gen_step) or regions is None:
            return super().compute_text_loss(
                config, sample, accelerator, info,
                gen_step=gen_step, text_per_token_log_probs=text_per_token_log_probs, **kwargs)

        tr = config.train
        assert tr.text_importance_sampling_level == "sequence" and tr.text_beta == 0 \
            and tr.top_entropy_quantile >= 1.0 and not tr.rollout_correction, \
            "spatial_grpo_draft 只实现了 GSPO、无 KL、无熵掩码、无 rollout correction 这一种配置"

        lp = text_per_token_log_probs.reshape(-1)
        old = gen_step.text_per_token_log_probs.reshape(-1).to(lp.device)
        log_ratio = lp - old if tr.true_ratio else lp - lp.detach()
        sizes = [n for n, _ in regions]
        assert sum(sizes) == lp.numel(), (sizes, lp.numel())

        n_total = lp.numel()
        loss = lp.sum() * 0.0
        for (n, a), chunk in zip(regions, log_ratio.split(sizes)):
            if n == 0:
                continue
            ratio = torch.exp(chunk.mean())
            clipped = torch.clamp(ratio, 1 - tr.text_clip_range, 1 + tr.text_clip_range_high)
            loss = loss - (n / n_total) * torch.min(ratio * a, clipped * a)

        ratio_all = torch.exp(log_ratio.mean()).detach()
        info["text_policy_loss"].append(loss.detach())
        info["text_clipfrac"].append(((1.0 - ratio_all) > tr.text_clip_range).float())
        info["text_clipfrac_high"].append(((ratio_all - 1.0) > tr.text_clip_range_high).float())
        if kwargs.get("text_entropies") is not None:
            info["text_entropies"].append(kwargs["text_entropies"].mean())
        info["text_ratio"].append(ratio_all)

        k = getattr(tr, "backward_calls_per_sample", None) or 1
        w = step_weight(sample, gen_step, k=k)
        info["seg_tokens"].append(torch.tensor(float(n_total)))
        info["seg_weight"].append(torch.tensor(w))
        return info, loss * w
