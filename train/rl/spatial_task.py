# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""`spatial_gate` task：把深度门控轨迹接进 GeoUG RL engine 的训练循环。

它在整条链路里的位置
--------------------
    spatial_train.py  装配（打补丁、注册、起 GeoUG RL engine 的 main）
    spatial_dataset.py  出题（parquet -> prompt / image / metadata）
 →  spatial_task.py    每步跑一条轨迹、算奖励、装成 sample
    spatial_inferencer.py  轨迹本身（门控、深度注入、log-prob）

一个 sample 只装一条 rollout
----------------------------
别的 task（t2i / reflect）一个 sample 里装 `num_image_per_prompt` 条轨迹，
共享一套 `contexts`。我们不行：深度分支比直答分支多两个 `input_image` step，
同一套 step 序列装不下两种分支。

所以 `batch_size = 1`，组靠**跨进程**成：`DistributedKRepeatBatchSampler(k=world_size)`
让每张卡拿到同一道题，`compute_advantages` 按 prompt 字符串 gather 成一组，
**G = world_size**。8 卡 G=8。

奖励为什么同步算
----------------
别的 task 走 `self.compute_rewards(...)` 扔进线程池，因为它们的 reward 是
另一个大模型（qwenvl / viescore），一次几百毫秒。我们的 verifier 是
`reward.py` 里的正则抽字母，微秒级 —— 扔进线程池只是多一层 future。
"""
from __future__ import annotations

from functools import partial

import torch

from train.rl.reward import score

from train.rl.engine.alpha_grpo.common import (
    AdvantageContext, GenerationStep, compute_advantages)
from train.rl.engine.alpha_grpo.tasks import BagelBaseTask, register

from train.rl.spatial_algorithm import PAD_TEXT  # noqa: E402
from train.rl.spatial_eval import (  # noqa: E402
    D3EvalSet as SpatialEvalSet, format_metrics, run_online_eval,
)

_LOSS_STEPS = ("think_text", "text")


def _pad_loss_steps(steps, k):
    """把计 loss 的 step 数补齐到 k，返回补了几个；超了返回 None。

    各卡的 backward 次数必须一样，否则梯度规约对不上、直接死锁 ——
    原因和取舍写在 spatial_algorithm.py 的 docstring 里。
    """
    n = sum(1 for s in steps if s.step_type in _LOSS_STEPS)
    if n == 0 or n > k:
        return None
    for _ in range(k - n):
        pad = GenerationStep(step_type="text", texts=[PAD_TEXT])
        pad.is_pad = True          # spatial_grpo 认这个标记，给它 0 loss
        steps.append(pad)
    return k - n


@register("spatial_gate")
class SpatialGateTask(BagelBaseTask):
    """图 + 题 -> 门控轨迹 -> 四选一答案。"""

    # 写成类属性而不是在 __init__ 里设，是为了不覆盖基类的构造签名。
    _eval_set = None        # 在线评测集，`eval()` 第一次被调用时才建
    _lambda = None          # 成本项的拉格朗日乘子，见 cost_lambda()
    _trig_acc: list = []    # 一个 optimizer batch 内的触发率，见 _update_lambda()

    def rollout(self, config, tokenizer, batch_data, global_step, stat_tracker, **kwargs):
        prompts, metadatas, images = batch_data
        assert len(prompts) == 1, (
            f"spatial_gate 的 batch_size 必须是 1（见模块 docstring），收到 {len(prompts)}。"
            f"组靠 DistributedKRepeatBatchSampler 跨进程成，不靠批内。")
        meta = metadatas[0]
        accelerator = self.accelerator

        from trl.models import unwrap_model_for_generation  # noqa: PLC0415

        inferencer = self.inferencer
        model = inferencer.model
        with unwrap_model_for_generation(model, accelerator) as unwrapped:
            inferencer.model = unwrapped
            with accelerator.autocast():
                out = inferencer.rollout_one(
                    image_pil=images[0],
                    question=prompts[0],
                    image_key=meta["image_key"],
                    system_prompt=config.sample.system_prompt,
                    intrinsics=meta["intrinsics"],
                    # 用原图尺寸，不是 parquet 里那张缩到 1024 的。题面里的
                    # <point> 是 norm1000，ToolSession 拿 image_wh 反归一化，
                    # 它必须和 intrinsics 处在同一个坐标系（cx = W/2 = 960）。
                    image_wh=meta["image_wh"],
                    max_rounds=config.sample.max_rounds,
                    max_text_tokens=config.sample.max_text_tokens,
                    temperature=config.sample.think_text_temperature,
                    top_p=config.sample.think_text_top_p,
                    top_k=config.sample.think_text_top_k,
                    force=self.rollout_force(config, prompts[0], global_step),
                )
            inferencer.model = model

        # 轨迹形状对齐。丢弃的条件有两个：一条 loss step 都没有（第一轮就吐空），
        # 或者段数超过 K（模型连着发了两次工具调用）。
        #
        # 丢不丢必须**全组一致**：一张卡 `continue` 掉而别的卡往下走，
        # 后面 compute_advantages 里的 gather 就少一个人，一样死锁。
        n_pad = _pad_loss_steps(out["steps"], config.train.loss_steps_per_sample)
        keep = torch.tensor([0.0 if n_pad is None else 1.0], device=accelerator.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(keep, op=torch.distributed.ReduceOp.MIN)
        if keep.item() == 0.0:
            return None, True

        breakdown = score(
            out["text"], meta["answer"], out["used_depth"],
            options=meta.get("options"), lambda_cost=self.cost_lambda(config),
        )
        dev = accelerator.device
        rewards = {
            "avg": torch.tensor([breakdown["reward"]], device=dev, dtype=torch.float32),
            "correct": torch.tensor([breakdown["correct"]], device=dev, dtype=torch.float32),
            "used_depth": torch.tensor([breakdown["used_depth"]], device=dev, dtype=torch.float32),
            "parsed": torch.tensor([breakdown["parsed"]], device=dev, dtype=torch.float32),
            "reached_decide": torch.tensor([float(out["reached_decide"])], device=dev,
                                           dtype=torch.float32),
            # 补了几个空 step。均值接近 K-1 说明基本全在直答分支，
            # 接近 0 说明基本全在深度分支 —— 顺手当分支占比看。
            "n_pad_steps": torch.tensor([float(n_pad)], device=dev, dtype=torch.float32),
            "forced": torch.tensor([float(out["forced"])], device=dev, dtype=torch.float32),
            "depth_natural": torch.tensor([float(out["used_depth"] and not out["forced"])],
                                          device=dev, dtype=torch.float32),
        }

        prompt_ids = tokenizer(
            prompts, padding="max_length", max_length=config.model.max_sequence_length,
            truncation=True, return_tensors="pt",
        ).input_ids.to(dev).long()

        sample = {
            "prompts": prompts,
            "prompt_ids": prompt_ids,
            "prompt_metadata": metadatas,
            "contexts": out["contexts"],
            "generation_steps": out["steps"],
            # compute_advantages 用它统计助手文本长度；一条轨迹给一个字符串，
            # 否则统计的是"每段"而不是"每条"的长度。
            "think_texts": [out["text"]],
            "rewards": rewards,
            "is_postprocessed": False,
            "task": "spatial_gate",
        }

        ctx = AdvantageContext(accelerator=accelerator, tokenizer=tokenizer,
                               global_step=global_step, prompt_ids=prompt_ids)
        sample["reward_postprocess_fn"] = partial(self.reward_postprocess, ctx=ctx)

        if config.train.async_reward_fn:
            return sample, False
        return sample["reward_postprocess_fn"](config, sample, stat_tracker=stat_tracker)

    def rollout_force(self, config, prompt, global_step):
        """这条 rollout 要不要强制门控动作（`rollout_one(force=...)`）。本 task 不强制。"""
        return None

    # ------------------------------------------------------------------
    # 成本项的拉格朗日乘子（PLAN.md §9.5）
    # ------------------------------------------------------------------
    @staticmethod
    def _budget(config):
        """目标触发率 B；`<=0`（默认 -1.0）表示关闭预算约束、用固定 λ。

        用负数哨兵而不是 None，是因为 ml_collections 不许从命令行覆盖默认值为
        None 的字段 —— 写 None 的话 `--config.train.cost_budget=0.46` 会直接
        `Flags parsing error` 起不来。
        """
        b = getattr(config.train, "cost_budget", -1.0)
        return None if b is None or b <= 0 else float(b)

    def cost_lambda(self, config):
        """当前的 λ。预算约束关掉时就是配置里的固定值。"""
        if self._budget(config) is None:
            return config.train.lambda_cost
        if self._lambda is None:
            self._lambda = float(config.train.lambda_cost)
        return self._lambda

    def _update_lambda(self, config, trigger_rate):
        """对偶上升：`λ <- clip(λ + η·(p̂ − B), 0, λ_max)`，**按 optimizer step 更新**。

        为什么不是固定 λ（PLAN.md §9.5 的论证）：固定 λ 把「一次深度值多少准确率」
        这个兑换比写死了，而那个比值在训练中会漂 —— 实测两个 benchmark 上它根本
        不是一个数（D3 上深度值 +7.5 分，CV-3D 上值 −0.2 分）。把准确率当目标、
        成本当**约束**，λ 由触发率自己顶出来，就不用手工扫。

        **必须所有卡都调用**，而且输入必须是 all-reduce 过的同一个数 ——
        各卡 λ 不一致的话，同一组 rollout 的奖励就不在同一个尺度上，
        组内标准化直接失去意义，而且不会报任何错。

        **更新节奏**（`TRAINING_CODE_REVIEW.md` 的 P1）：这个函数每道题都被调用，
        但参数是攒满 `gradient_accumulation_steps` 道题才更新一次。如果 λ 也逐题
        更新，同一个 optimizer batch 里的样本就用了 16 个不同的目标函数，
        而 `lambda_lr` 的实际含义还会随 gas 和数据过滤率漂。
        所以这里只累积触发率，攒满 gas 道题才真正走一步对偶上升 ——
        `lambda_lr` 的定义因此是**每次 optimizer update 的步长**，与 gas 无关。

        一步滞后是有意的：本批的奖励用的是上一批末尾更新出来的 λ。
        对偶上升本来就允许这个滞后，而且避免了「用本批奖励反推本批 λ」的循环。
        """
        b = self._budget(config)
        if b is None:
            return
        # 攒到 optimizer step 边界再动 λ。gas 用 config 里的值，不看
        # accelerator.sync_gradients —— 后者在逐 step backward 的循环里
        # 每个 micro step 都是 True（我们开了 sync_each_batch），不是 optimizer 边界。
        self._trig_acc.append(float(trigger_rate))
        gas = int(getattr(config.train, "gradient_accumulation_steps", 1) or 1)
        if len(self._trig_acc) < gas:
            return
        p_hat = sum(self._trig_acc) / len(self._trig_acc)
        self._trig_acc = []
        lr = getattr(config.train, "lambda_lr", 0.05)
        hi = getattr(config.train, "lambda_max", 1.0)
        self._lambda = float(min(max(self.cost_lambda(config) + lr * (p_hat - b), 0.0), hi))

    # 实例方法而不是 classmethod：λ 是 task 的状态，得能拿到 `self`。
    # 上游 `reflect.py:293` / `self_reward_t2i.py:144` 绑的也是 `self.reward_postprocess`，
    # 所以这是照它们的约定走，不是特例。
    def reward_postprocess(self, config, sample, stat_tracker, ctx=None):
        """奖励已在 rollout 里算好，这里剩组归一化 + 摘要 + 更新 λ。"""
        if sample["is_postprocessed"]:
            return sample, False
        self._log_group(config, sample, ctx)
        return compute_advantages(config, sample, stat_tracker, ctx)

    def _log_group(self, config, sample, ctx):
        """把一组 G 条 rollout 的分支构成打成一行，并按实测触发率更新 λ。

        为什么非要打：`collect_metrics` 那些 `reward_*` 只进 wandb。
        `rewards [...]` 只有对错（λ=0 时 reward 就是 correct），
        看不出这一组里有几条走了深度。而「走深度的那几条对得更多吗」正是这件事
        要优化的量，不打出来就只能盯着 loss 猜。

        `acc@depth` / `acc@direct` 是**这一步的单题 Δ 估计**，
        和 `rl_debug_v2/measured_gap.json` 里离线量的 Δ 同口径，可以直接对。
        某一侧一条 rollout 都没有时打 `-`，不打 0.0 —— 两者意思完全不同。

        gather 和 λ 更新在**所有卡**上做（gather 本身是集合操作，
        提前 return 会死锁；λ 不一致会让组内标准化失去意义）。只有打印限主进程。
        """
        acc = ctx.accelerator
        r = sample["rewards"]
        g = {k: acc.gather(r[k].detach().clone()).float().cpu()
             for k in ("correct", "used_depth", "parsed", "reached_decide", "forced")}

        n = g["correct"].numel()
        depth = g["used_depth"].bool()
        # 触发率只数自然采样的：强制的那几条会把它系统性抬高。
        natural = ~g["forced"].bool()
        trigger = float(depth[natural].float().mean()) if natural.any() else 0.0
        lam = self.cost_lambda(config)
        self._update_lambda(config, trigger)          # 所有卡，输入同源

        if not acc.is_main_process:
            return
        def _rate(mask):
            return f"{g['correct'][mask].mean():.2f}" if mask.any() else "-"
        budget = self._budget(config)
        tail = "" if budget is None else f" | lambda {lam:.3f}->{self._lambda:.3f} (B={budget})"
        n_forced = int((~natural).sum())
        forced_txt = f" (强制 {n_forced}，自然 {int(depth[natural].sum())}/{int(natural.sum())})" \
            if n_forced else ""
        print(f"[gate] step {ctx.global_step} | G={n} "
              f"| acc {int(g['correct'].sum())}/{n} "
              f"| depth {int(depth.sum())}/{n}{forced_txt} "
              f"| parsed {int(g['parsed'].sum())}/{n} "
              f"| decide {int(g['reached_decide'].sum())}/{n} "
              f"| acc@depth {_rate(depth)} acc@direct {_rate(~depth)}{tail}", flush=True)

    def collect_metrics(self, sample, info):
        """把 reward 拆开报。

        只看合成后的 reward 看不出涨的是什么：`lambda_cost > 0` 时
        少生成一次深度和多答对一题在 reward 上是同一个数。
        `parsed` 要单独盯 —— 解析失败会伪装成「答错」。
        """
        for key, value in sample["rewards"].items():
            info[f"reward_{key}"].append(torch.mean(value))

        # 再按 `gate/` 前缀打一份：wandb 按 `/` 分组，这样训练触发率、
        # 评测触发率（`spatial_eval.py` 打 `gate/trigger_eval`）和 λ 会落在
        # **同一个面板**里。散在 `reward_used_depth` / `eval_depth_frac` 两个
        # 不相关的名字下时，图上画不到一起，正是最该并排看的那三条。
        # 含强制样本。自然触发率 = reward_depth_natural / (1 − reward_forced)；
        # 不按样本条件 append，是因为 train.py 按 key 做跨卡 reduce，各卡 key 必须一致。
        info["gate/trigger_train"].append(torch.mean(sample["rewards"]["used_depth"]))
        if self._lambda is not None:
            info["gate/lambda"].append(torch.tensor(float(self._lambda)))

    def eval(self, test_dataloader, config, global_step, autocast, **kwargs):
        """每 `config.eval_freq` 步在 D3 固定子集上跑一次 adaptive。

        `test_dataloader` 不用 —— 它是训练集同分布的留出集（`validate.parquet`），
        而我们真正想盯的是 **D3 上的跨域行为**，尤其是触发率有没有塌。
        评测集另外从 `config.eval_questions` 读，深度走缓存、不跑扩散，
        近似的边界写在 `spatial_eval.py` 的 docstring 里（**触发率是精确的，
        `acc@depth` 是近似的**）。

        `config.eval_freq` 设得比总步数大就等于关掉。
        """
        if not getattr(config, "eval_questions", None):
            return {}

        # 第一次用时才建，避免不做评测的 run 白读一遍 parquet。
        if self._eval_set is None:
            self._eval_set = SpatialEvalSet(
                config.eval_questions, config.eval_depth_cache,
                num_samples=config.eval_num_samples, seed=config.eval_seed,
            )

        metrics = run_online_eval(self, config, global_step, self._eval_set)
        if self.accelerator.is_main_process and metrics:
            print(format_metrics(metrics, global_step), flush=True)
            # 跟着 logger_type 走，别写死 wandb —— `logger_type != "wandb"` 时
            # train.py 走的是 `accelerator.log`（tensorboard），写死 wandb 的话
            # 一切 TB 这几个指标就静默丢了。
            if config.logger_type == "wandb":
                import wandb  # noqa: PLC0415
                wandb.log(metrics, step=global_step)
            else:
                self.accelerator.log(metrics, step=global_step)
        return metrics
