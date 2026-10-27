# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""GeoUG spatial-gate RL training configuration.

用法::

    --config train/rl/config_spatial.py:spatial_gate_draft_v3

冒号后面那截**不是函数名**，是传给 `get_config()` 的参数
（`ml_collections/config_flags.py:684`：`config_module.get_config(*split_path[1:])`）。
所以下面要有一个 `get_config(name)` 做分发 —— 内置 engine 的 `config/bagel.py`
也是这个形状。

只写和 engine 默认值**不同**的项，其余继承 `config/bagel.py::_base_config`。
每一条偏离都注明理由 —— 不注明的偏离半个月后没人敢动。
"""
import importlib.util
import os
from pathlib import Path

from train.rl.prompt import UNIFIED_PROMPT

_ROOT = Path(__file__).resolve().parents[2]
_ENGINE_CONFIG = Path(__file__).resolve().parent / "engine" / "config" / "bagel.py"
_SPEC = importlib.util.spec_from_file_location("_geoug_bagel_rl_config", _ENGINE_CONFIG)
_bagel_cfg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_bagel_cfg)


def _required_path(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} must be set to an existing public/local path")
    return os.path.abspath(os.path.expanduser(value))


TRANSFORM_CONFIG = os.environ.get(
    "RL_TRANSFORM_CONFIG", str(_ROOT / "configs" / "rl" / "depth_cot_multi_dsp.yaml"))


def spatial_gate():
    config = _bagel_cfg._base_config()
    model_path = _required_path("MODEL_PATH")

    # ===== model =====
    config.model.bagel_model_path = model_path
    config.model.vae_path = os.path.join(model_path, "ae.safetensors")
    # 评测用的同一份 transform yaml。深度归一化口径必须和评测一致，
    # 否则 RL 注入的深度图和评测时模型自己生成的不是同一种东西。
    config.model.transform_config = TRANSFORM_CONFIG
    # **这个键是死的，设什么都没用。** `config/bagel.py:24` 定义了它，但全仓库
    # 没有一处读它：`train.py::load_bagel_weight` 写死 `ema.safetensors`，唯一的
    # 逃生口是环境变量 `BAGEL_MODEL_WEIGHT`。
    #
    # 而写死的这个恰好是**对的**，所以不要去"修"它：评测侧
    # （`bagel_model_interface.py:183`）和 `measure_branch_gap.py:120` 都是
    # `is_ema=True`，RESULT.md 的 V18/V19/V20 和 rl_debug_v2 的实测 Δ
    # （never 0.650 / always 0.745 / oracle 0.968）全部量在 ema 权重上。
    # 改成 model.safetensors 的话 RL 的起点和所有基线数字就不是同一个模型了。
    config.model.load_from_ema = True

    # ===== 任务与数据 =====
    config.train.task = "spatial_gate"
    config.prompt_fn = "spatial_gate"          # -> SpatialGateDataset
    config.dataset = _required_path("RL_DATASET")
    config.depth_cache_dir = _required_path("DEPTH_CACHE_DIR")

    # ===== rollout =====
    config.sample.system_prompt = UNIFIED_PROMPT
    # 深度分支的自然形状是三段生成：`...<GENERATE_DEPTH>` / 工具调用 / `...</think>答案`，
    # 中间插两次注入（深度图、工具结果）。直答分支一段走到底。
    # 训练循环逐段 backward，各卡段数必须一致，所以这里把段数**钉死**成 3：
    # 少了由 spatial_task 补空 step，多了（模型连发两次工具调用）整组丢掉。
    # 两个值要一起改。
    config.sample.max_rounds = 3
    config.train.loss_steps_per_sample = 3
    config.train.backward_calls_per_sample = 3   # = K，accelerate 的累积计数要对上
    config.sample.max_text_tokens = 512
    config.sample.think = True                 # 让 compute_advantages 统计文本长度

    # 一个 sample 只装一条轨迹，组靠跨进程成 —— 见 spatial_task.py 的 docstring。
    # k = num_image_per_prompt 必须等于 num_gpus，DistributedKRepeatBatchSampler
    # 里 `total_samples % k == 0` 才成立，且 m=1（每个全局 batch 一道题）。
    config.sample.train_batch_size = 1
    config.sample.test_batch_size = 1
    config.sample.num_image_per_prompt = 8     # = num_gpus，启动脚本会覆盖
    config.num_gpus = 8

    # 没有图像生成步，但 compute_advantages 要拿它算 advantages 的重复次数
    # （`num_steps * timestep_fraction`）。设 1 = 每条轨迹一份 advantage。
    config.sample.num_steps = 1

    # 采样必须打开且不要截断得太狠：RESULT.md V20 实测决策点的 p_depth 只有
    # 0.4688 / 0.5 两个值、logit 差 0.125 nat —— 采样天然让两条分支各占一半，
    # 这是 GRPO 组内有方差的前提。top_p 压太低会把弱势分支掐死，门控就没得学了。
    config.sample.think_text_temperature = 1.0
    config.sample.think_text_top_p = 1.0
    config.sample.think_text_top_k = None

    # ===== 奖励 =====
    # verifier 是 reward.py 里的正则抽字母，微秒级，同步算完就归一化，
    # 不走 GeoUG RL engine 的线程池（那是给 qwenvl / viescore 这类大模型 reward 用的）。
    config.train.async_reward_fn = False
    config.reward_fn = {}                      # 不用 multi_score 里的任何一个
    config.text_reward_fn = None
    config.train.use_think_text_format_reward = False
    # r = correct - lambda_cost * used_depth。
    #
    # **0 只能用来冒烟，不能久跑**（RESULT.md V29 实测）。两个原因：
    #
    # 1. λ=0 时 reward 的最优策略就是 always-depth（always 0.847 > never 0.709），
    #    RL 会老老实实收敛到「永远开深度」—— 那正是我们要对比的 baseline。
    # 2. 47% 的题 8 次 rollout 全对，组内 reward 全是 1、std=0，
    #    `alpha_grpo/common.py:149` 直接跳过整组。训练集有效利用率只有 49%，
    #    另一半的 rollout 算力纯空转。λ>0 之后跳到 95%
    #    （门控是硬币，每题的 n_depth 都落在 1~7，从没 0/8 或 8/8，
    #     所以全对的题也有「用了深度 vs 没用」的方差）。
    #
    # 论文发布策略使用 0.1；0 只用于显式请求的 always-depth smoke baseline。
    # 最优门控 = 「Δ > λ 才生成」，所以扫 λ 得到的就是论文那张
    # 准确率-成本前沿图。
    #
    # λ 是平滑旋钮**只因为** `config.sample.global_std` 默认 True
    # （`config/base.py:52`，我们没覆盖）。若改成组内 std，对全对的题
    # 分子分母同时含 λ、会被完全约掉，无论 λ 多小都是满幅的「别用深度」
    # 梯度，深度分支会被压死。改之前先看 RESULT.md V29 第 3 节。
    config.train.lambda_cost = 0.1

    # ---- 可选：把固定 λ 换成预算约束（PLAN.md §9.5），默认**关闭** ----
    # `max E[correct] s.t. E[used_depth] <= B`，λ 由对偶上升按实测触发率自动调：
    #     λ <- clip(λ + η·(p̂ − B), 0, λ_max)
    # 实现在 `spatial_task.cost_lambda` / `_update_lambda`。
    #
    # **默认关闭（= 用上面的固定 λ），这是 V32 之后有意的选择。**
    # §9.5 当初的动机是「防止准确率被成本项悄悄换掉」，而 V32 测出来的前沿在
    # λ∈[0,0.10] 是平的、oracle 自己就用 46% 的深度。也就是说预算不是瓶颈，
    # 硬卡一个 B 反而会把策略推到前沿更差的一端（λ=0.2 -> 31.9% / 0.861）。
    # 真要开的话 **B 取 0.45~0.50**（对齐 oracle 的 46%），不要取 0.3。
    #
    # 用 **-1.0 而不是 None** 表示关闭：ml_collections 不允许从命令行覆盖
    # 默认值为 None 的字段（`Type <class 'NoneType'> ... is not supported for
    # overriding`），写 None 的话 `--config.train.cost_budget=0.46` 直接起不来。
    config.train.cost_budget = -1.0    # 目标触发率上限 B；<=0 表示关闭

    # advantage 的固定分母：`A = (r − 组内均值) / adv_scale`。
    # **这是 P0 的修复**（`spatial_advantage.py`）。上游 `global_std=True` 算的是
    # 单题组内的 `std(r)`，而 r 含 λ —— 对组内全对的题，λ 在分子分母里被完全约掉，
    # 于是任何 λ>0 都给出满幅 ±1 的反深度 advantage（实测 λ=0.05 -> 0.996，
    # λ=0.5 -> 0.9996）。换成常数分母之后 λ 才真正线性生效。
    # 0.5 是准确率 ≈0.5 时伯努利的 std，取它是为了让可学组的 advantage 幅值
    # 和修复前同量级，学习率不必跟着改。
    config.train.adv_scale = 0.5
    config.train.lambda_lr = 0.05      # 对偶上升步长 η
    config.train.lambda_max = 1.0      # 截断，别让成本项把准确率信号吃光

    # ===== 优化 =====
    # LoRA（GeoUG RL engine 的默认，打在 LLM 的全部 attention + MLP 上）。
    # 全参在 4 x A100-80G 上按 ZeRO-2 估算要 ~47 GB/卡（fp32 优化器状态占 21 GB），
    # 再叠上我们每个 sample 要存的三份 KV 上下文，余量太薄。
    # 要切全参就 `--config.use_lora=False --config.train.learning_rate=1e-6`，
    # 并把 deepspeed_config 换成 zero3。
    config.use_lora = True
    config.lora_rank = 32
    config.lora_alpha = 64
    config.train.learning_rate = 5e-5
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 1
    config.train.deepspeed_config = str(
        _ROOT / "train" / "rl" / "engine" / "accelerate_configs" /
        "deepspeed_zero2.json")
    config.train.algorithm = "spatial_grpo"    # = grpo + 占位 step 零梯度
    config.train.text_importance_sampling_level = "sequence"   # GSPO
    config.train.recompute_log_prob = False    # ppo_epoch=1 且组内不重复，不需要
    config.num_epochs = 1

    # ===== 在线评测（每 eval_freq 步在 D3 子集上跑一次 adaptive）=====
    # 深度走缓存不跑扩散。近似的边界见
    # `spatial_eval.py` 的 docstring：**触发率精确，acc@depth 是近似的**。
    # 这是趋势指示器，报数仍然用 `gate/eval_d3.sh` 的全量离线评测。
    config.eval_freq = 10
    # 分层子集：split × domain × 捷径(quad) × 难度(bin) 32 格各 6 题、格内 A/B 各半，
    # 与 800 题的构成逐项一致（`data_pipeline/make_d3_eval_subset.py`）。
    # 用全集改成 d3_800_rl.parquet；`eval_num_samples > 0` 时在文件里再随机抽。
    config.eval_questions = _required_path("EVAL_QUESTIONS")
    config.eval_depth_cache = _required_path("EVAL_DEPTH_CACHE")
    config.eval_num_samples = -1
    config.eval_seed = 0                 # 固定子集 + 每题固定采样种子，跨 step 可比
    # **采样，不是贪心。** 决策点是个近似平局（SFT 4k 的 p_depth 只有
    # 0.4688 / 0.5000 两个值），贪心会把它变成由浮点打平决定的 0%/100% 阶跃 ——
    # 实测同样 32 题，贪心在线跑 0/32、离线跑 9/32，两边都"对"。详见 spatial_eval.py。
    config.eval_do_sample = True
    # 每题再跑一次 force_depth，给出草答 / force_depth / oracle / 随机门控的前沿和 AUC。
    # 评测时间翻倍，所以默认关；指标定义见 spatial_eval.py。
    config.eval_force_depth = False
    # step 0 先量一次：没有这个锚点，后面的曲线只能看出"在动"，
    # 看不出"相对 RL 之前动了多少"。代价就 2 分钟。
    config.train.eval_before_train = True

    config.wandb.project = "spatial-gate-rl"
    return config


def spatial_gate_draft():
    """v6：用决策点前的草答做门控的反事实基线，advantage 按 token 分四段。

    奖励 `r = correct − λ·used_depth` 不变，只改 advantage 怎么分给 token，
    见 `spatial_draft.py`。`spatial_gate` 保留为整条轨迹一个 advantage 的 ablation。
    """
    config = spatial_gate()
    config.train.task = "spatial_gate_draft"
    config.train.algorithm = "spatial_grpo_draft"
    # 各段 advantage 的开关，取值见 spatial_draft.py 的 docstring。默认值 = 第二版。
    config.train.draft_ground_adv = "r"
    config.train.draft_draft_adv = "grpo"
    config.train.draft_gate_adv = "sampled"
    config.train.draft_gate_coef = 1.0
    config.train.force_depth_per_group = 0
    return config


def spatial_gate_draft_v3():
    """第三版（`V6_RL_2_REVIEW.md` §8.6 / §10.4）：只训门控和深度执行。

    - 草答段 0：不去优化门控要读的置信度（v6 SFT 里草答也不计 loss）；
    - grounding 只用开深度样本的 `c − c̄_depth`，不含 λ，直答 0；
    - 门控用 two_branch，每组强制 4 条开深度给 `Q̂_depth`；
    - 在线评测带 force_depth 前沿。
    """
    config = spatial_gate_draft()
    config.train.draft_ground_adv = "depth"
    config.train.draft_draft_adv = "zero"
    config.train.draft_gate_adv = "two_branch"
    config.train.force_depth_per_group = 4
    config.eval_force_depth = True
    return config


def spatial_gate_debug():
    """100 题跑通用。数据集换成 debug 目录，日志密一点。

    用的是 `rl_debug_v2` —— 从 proxy_check 里按**实测的** Δ 挑出来的（见
    `data_pipeline/build_debug_subset.py`）。上一版 debug 集是随机挑的，冒烟时
    3 道题 x G=8 共 24 条 rollout 全对，组内 reward 方差为 0、advantage 恒 0，
    看不出训练环到底通没通。这一版保证组内有方差。
    """
    config = spatial_gate()
    config.dataset = os.environ.get("RL_DEBUG_DATASET", config.dataset)
    config.depth_cache_dir = os.environ.get(
        "RL_DEBUG_DEPTH_CACHE", config.depth_cache_dir)
    config.num_gpus = 4
    config.sample.num_image_per_prompt = 4
    config.save_freq = 10000                   # 跑通阶段不存 ckpt
    config.eval_freq = 10                      # 每 10 步在 D3 子集上看一眼
    config.logger_type = "wandb"
    return config


_CONFIGS = {
    "spatial_gate": spatial_gate,
    "spatial_gate_draft": spatial_gate_draft,
    "spatial_gate_draft_v3": spatial_gate_draft_v3,
    "spatial_gate_debug": spatial_gate_debug,
}


def get_config(name="spatial_gate_draft_v3"):
    if name not in _CONFIGS:
        raise ValueError(f"未知配置 {name!r}，可选：{sorted(_CONFIGS)}")
    return _CONFIGS[name]()
