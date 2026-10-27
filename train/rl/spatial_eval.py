# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""训练途中的在线评测：每 `eval_freq` 步在 D3 的一个固定子集上跑一次 adaptive。

要回答的问题
------------
训练日志里的 `[gate]` 是**训练集上**的滑动读数，而且每步换一道题 —— 它能告诉你
门控在动，但不能告诉你动的方向在留出集上算不算数。离线评测（`gate/eval_d3.sh`）
是权威口径，但要先存 ckpt、合 LoRA、再起 16 卡跑一小时，反馈太慢。

这个在线评测填中间那段：**2 分钟出一个 D3 子集上的 acc + 触发率**，
用来在训练过程中看趋势、及早发现门控塌缩。

为什么能这么便宜：深度走缓存，不跑扩散
--------------------------------------
发布数据包中的 D3 depth cache 是全部 800 题的深度缓存，
来自 SFT 4k 那轮 `force_depth` 的产物。一次扩散 ~90 秒，读缓存是毫秒级 ——
128 题 / 16 卡 = 每卡 8 题，整轮约 2 分钟。

**这个近似错在哪，必须记住。** LoRA 打在 `language_model` 上，而深度 latent 也是
这个 LLM 生成的，所以策略一变，它真正会生成的深度图就和缓存里那张不同了。
于是：

* **触发率是精确的** —— 门控决策发生在生成深度**之前**，不受缓存影响；
* **`acc@depth` 是近似的** —— 用的是 SFT 那版的深度图。

所以在线评测的定位是**趋势指示器**，尤其是盯触发率有没有塌。
要报进论文的数字仍然只能来自 `gate/eval_d3.sh`（真跑扩散、全量 800 题）。

其它对齐离线口径的地方
----------------------
* **采样解码**（`do_sample=True`，温度对齐 rollout）。
  **这里原本用的是贪心，实测证明那是错的，别改回去。**

  离线评测默认贪心，我照抄了，结果 step 0 的锚点上触发率是 0/32，
  而同样 32 题、同样权重的离线跑是 9/32。查下来两边都没错 ——
  **底层的量本身是简并的**：SFT 4k 在全部 800 题上 `p_depth` 只取两个值
  （`0.468791` 581 题 / `0.500000` 219 题），`frac_gt_0.5 = 0`。
  离线那 27.4% 的触发率恰好等于 219/800，也就是**全部来自
  `p_depth` 严格等于 0.5 时 `argmax` 的打平规则**（按 token id 取先者）。
  换一条解码路径，平局就可能判向另一边 —— 于是 0% 和 27.4% 都是"对"的。

  贪心把一个 0.47-vs-0.50 的近似平局放大成由数值噪声决定的 0%/100% 阶跃，
  当趋势指示器用完全失效。采样反映的是真实策略（p≈0.47 -> 约 47% 触发），
  和训练 rollout 同口径，噪声是可量化的二项噪声（n=128 时 se≈0.044）。
* **`intrinsics=None`**，和离线 D3 评测一致（退回 `f=max(W,H)` 近似）。
* 子集用固定种子抽一次，**每次评测都是同一批题**，否则趋势没法读。
* **每题的采样种子只由 `eval_seed` 和题目 id 决定**（公共随机数）：同一题在不同 step
  用同一串随机数，step 间的差异里就不再混入「这次恰好采到另一条分支」的噪声。
  训练 RNG 不受影响（`_RngGuard` 在外面兜着）。

前沿（`config.eval_force_depth=True` 时）
----------------------------------------
每题再跑一次 `force="depth"`。同一题的种子相同，所以 `<DECIDE>` 之前的草答和自然那次
**逐 token 相同**，强制那次同时给出「直答 = 草答」和「开深度」两个结果（v6 里直答终答 = 草答）：

* `eval_fa_acc`：草答准确率（= force_answer）；RL 不训草答时它应基本不变；
* `eval_fd_acc`：force_depth 准确率；
* `eval_oracle_acc`：每题取两者较好的那个；
* `eval_random_acc`：同自然触发率下的随机门控 `(1−t)·fa + t·fd`；
  `eval_gain_vs_random = eval_acc − eval_random_acc` 是门控本身的贡献；
* `eval_fix_rate` / `eval_harm_rate`：草答错时深度修正率、草答对时深度改错率；
* `eval_auc_pd`：自然那次的 `π_d` 预测草答错的 AUC（置信度信号有没有被磨掉）。

代价是评测时间翻倍（192 题约 8 分钟 -> 16 分钟）。
"""
from __future__ import annotations

import io
import json
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train.rl.reward import extract_letter


class D3EvalSet:
    """D3 题面 + 深度缓存。固定子集，只在第一次用时读盘。"""

    def __init__(self, questions_path, depth_cache_dir, num_samples=128, seed=0):
        import pyarrow.parquet as pq  # noqa: PLC0415

        table = pq.read_table(questions_path)
        rows = table.to_pylist()
        rows.sort(key=lambda r: r["id"])          # 先定序，parquet 的行序不保证稳定
        if 0 < num_samples < len(rows):
            # 固定种子抽一次：每次评测同一批题，趋势才读得出来。
            idx = np.random.RandomState(seed).permutation(len(rows))[:num_samples]
            rows = [rows[i] for i in sorted(idx)]
        self.rows = rows
        self.depth_cache_dir = Path(depth_cache_dir)

        missing = [r["id"] for r in rows
                   if not (self.depth_cache_dir / f"{r['id']}.npy").exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(rows)} 题缺深度缓存（例：{missing[:3]}），"
                f"目录 {self.depth_cache_dir}")

    def __len__(self):
        return len(self.rows)


def _load_image(cell):
    """题面里的 image 可能是路径，也可能是内嵌 JPEG 字节。"""
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    return Image.open(cell).convert("RGB")


class _RngGuard:
    """进出时保存 / 恢复全部 RNG 状态。

    为什么必须有（`TRAINING_CODE_REVIEW.md` 的 P2）：在线评测是**采样**的，
    而且跑在训练进程、训练 RNG 流里。不隔离的话，改动
    `eval_freq` / `eval_num_samples` / `eval_before_train`，甚至评测轨迹的长度，
    都会改变**后续训练 rollout** 的随机序列 —— 于是「同一个 sampler seed
    就是严格对齐的 A/B」这个假设不成立，而我们之前的 λ 对照正是这么假设的。

    恢复放在 `__exit__`，所以单题抛异常也会执行。
    """

    def __enter__(self):
        import random  # noqa: PLC0415
        self._py = random.getstate()
        self._np = np.random.get_state()
        self._pt = torch.get_rng_state()
        self._cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return self

    def __exit__(self, *exc):
        import random  # noqa: PLC0415
        random.setstate(self._py)
        np.random.set_state(self._np)
        torch.set_rng_state(self._pt)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False


def _auc(score, label):
    """`score` 预测 `label=1` 的 AUC（Mann–Whitney，平局记 0.5）；某类为空返回 nan。"""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def _frontier(recs, trigger):
    """强制开深度那一轮的逐题记录 -> 前沿指标（见模块 docstring）。"""
    ok = [x for x in recs if x["fd_draft"] is not None]
    if not ok:
        return {}
    fa = np.array([x["fd_draft"] for x in ok], float)
    fd = np.array([x["fd_hit"] for x in ok], float)
    m = {"eval_fa_acc": fa.mean(), "eval_fd_acc": fd.mean(),
         "eval_oracle_acc": np.maximum(fa, fd).mean(),
         "eval_random_acc": (1 - trigger) * fa.mean() + trigger * fd.mean(),
         "eval_fix_rate": fd[fa == 0].mean() if (fa == 0).any() else float("nan"),
         "eval_harm_rate": 1 - fd[fa == 1].mean() if (fa == 1).any() else float("nan")}
    nat = [x for x in recs if x["pi_d"] is not None and x["draft"] is not None]
    m["eval_auc_pd"] = _auc([x["pi_d"] for x in nat], [x["draft"] == 0 for x in nat])
    return {k: float(v) for k, v in m.items()}


@torch.no_grad()
def run_online_eval(task, config, global_step, eval_set):
    """在 `eval_set` 上跑一遍 adaptive，返回指标 dict（各卡一致）。

    分片按 `rank::world_size`，各卡只汇总三个标量再 all-reduce ——
    不用 `gather` 是因为各卡分到的题数可能不等，gather 要求形状一致。
    前沿要逐题配对，那部分走 `gather_object`（允许各卡长度不等）。

    整个过程包在 `_RngGuard` 里，评测不会扰动训练的随机流（见那个类的说明）。
    """
    from accelerate.utils import gather_object  # noqa: PLC0415
    from trl.models import unwrap_model_for_generation  # noqa: PLC0415

    from spatial_draft import draft_letter, pi_depth, region_sizes  # noqa: PLC0415  (循环 import)

    acc = task.accelerator
    inferencer = task.inferencer
    model = inferencer.model
    rank, world = acc.process_index, acc.num_processes
    rows = eval_set.rows[rank::world]
    force_depth = bool(getattr(config, "eval_force_depth", False))

    was_training = model.training
    model.eval()

    def run(row, force=None):
        torch.manual_seed(zlib.crc32(f"{config.eval_seed}:{row['id']}".encode()))
        try:
            return inferencer.rollout_one(
                image_pil=_load_image(row["image"]),
                question=row["question"],
                image_key=row["id"],
                system_prompt=config.sample.system_prompt,
                intrinsics=None,        # 和离线 D3 评测一致
                image_wh=None,
                max_rounds=config.sample.max_rounds,
                max_text_tokens=config.sample.max_text_tokens,
                # 采样，不是贪心 —— 理由见模块 docstring，
                # 贪心在这个近似平局的决策点上会退化成噪声驱动的阶跃。
                do_sample=bool(getattr(config, "eval_do_sample", True)),
                temperature=config.sample.think_text_temperature,
                top_p=config.sample.think_text_top_p,
                top_k=config.sample.think_text_top_k,
                depth_dir=eval_set.depth_cache_dir,
                force=force,
            )
        except Exception as e:  # noqa: BLE001
            # 单题失败不能把整轮评测（以及后面的 all-reduce）带走
            print(f"[eval] rank{rank} id={row['id']} force={force} 失败: {e}", flush=True)
            return None

    def grade(row, out):
        options = row.get("options")
        if isinstance(options, str) and options:
            options = json.loads(options)
        gt = str(row["answer"]).strip().upper()
        pred = extract_letter(out["text"], options)
        d = draft_letter(out["text"], options)
        return pred, int(bool(pred) and pred == gt), (int(d == gt) if d else None)

    def gate_prob(out):
        s = out["steps"][0]
        if s.text_per_token_log_probs is None:
            return None
        n_lp = int(s.text_per_token_log_probs.reshape(-1).shape[0])
        sizes = region_sizes(inferencer.tokenizer, s.texts[0], n_lp)
        if sizes is None:
            return None
        lp = float(s.text_per_token_log_probs.reshape(-1)[sizes[0] + sizes[1]])
        p = pi_depth(inferencer.tokenizer, s.texts[0], lp, out["used_depth"])
        return None if p != p else p

    n = n_correct = n_depth = n_depth_correct = n_parsed = n_decide = 0
    recs = []
    try:
        with _RngGuard(), unwrap_model_for_generation(model, acc) as unwrapped:
            inferencer.model = unwrapped
            with acc.autocast():
                for row in rows:
                    out = run(row)
                    n += 1
                    if out is None:
                        continue
                    pred, hit, draft = grade(row, out)
                    n_correct += hit
                    n_parsed += int(bool(pred))
                    n_decide += int(out["reached_decide"])
                    if out["used_depth"]:
                        n_depth += 1
                        n_depth_correct += hit
                    if not force_depth:
                        continue
                    rec = {"draft": draft, "pi_d": gate_prob(out),
                           "fd_hit": None, "fd_draft": None}
                    fo = run(row, force="depth")
                    if fo is not None and fo["forced"] and fo["used_depth"]:
                        _, rec["fd_hit"], rec["fd_draft"] = grade(row, fo)
                    recs.append(rec)
    finally:
        inferencer.model = model
        if was_training:
            model.train()

    t = torch.tensor([n, n_correct, n_depth, n_depth_correct, n_parsed, n_decide],
                     device=acc.device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    n, n_correct, n_depth, n_depth_correct, n_parsed, n_decide = t.tolist()
    frontier = {}
    if force_depth:                                   # 集合操作，各卡都要走到
        recs = gather_object(recs)
        if n:
            frontier = _frontier(recs, n_depth / n)
            if frontier:
                frontier["eval_gain_vs_random"] = n_correct / n - frontier["eval_random_acc"]
    if n == 0:
        return {}

    n_skip = n - n_depth
    return {
        **frontier,
        "eval_n": n,
        "eval_acc": n_correct / n,
        "eval_depth_frac": n_depth / n,
        "eval_acc_when_depth": (n_depth_correct / n_depth) if n_depth else float("nan"),
        "eval_acc_when_skip": ((n_correct - n_depth_correct) / n_skip) if n_skip else float("nan"),
        "eval_parsed_frac": n_parsed / n,
        "eval_decide_frac": n_decide / n,
        # `gate/` 前缀的那份是给 wandb 分组用的：和 spatial_task 打的
        # `gate/trigger_train` / `gate/lambda` 落在同一个面板，
        # 三条曲线并排才看得出「门控往哪走、λ 推了多少」。
        "gate/trigger_eval": n_depth / n,
        "gate/acc_eval": n_correct / n,
    }


def format_metrics(m, global_step):
    if not m:
        return f"[eval] step {global_step} 没有任何结果"
    fmt = lambda x: "-" if x != x else f"{x:.3f}"  # noqa: E731  (nan != nan)
    line = (f"[eval] step {global_step} | D3 n={int(m['eval_n'])} "
            f"| acc {m['eval_acc']:.3f} "
            f"| 触发 {m['eval_depth_frac']:.3f} "
            f"| acc@depth {fmt(m['eval_acc_when_depth'])} "
            f"acc@skip {fmt(m['eval_acc_when_skip'])} "
            f"| parsed {m['eval_parsed_frac']:.3f} decide {m['eval_decide_frac']:.3f}")
    if "eval_fd_acc" in m:
        line += (f"\n[eval] step {global_step} 前沿 | 草答 {fmt(m['eval_fa_acc'])} "
                 f"force_depth {fmt(m['eval_fd_acc'])} oracle {fmt(m['eval_oracle_acc'])} "
                 f"随机门控 {fmt(m['eval_random_acc'])} 增益 {fmt(m['eval_gain_vs_random'])} "
                 f"| 修正率 {fmt(m['eval_fix_rate'])} 改错率 {fmt(m['eval_harm_rate'])} "
                 f"| AUC(π_d→草答错) {fmt(m['eval_auc_pd'])}")
    return line
