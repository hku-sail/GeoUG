# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""GeoUG RL 训练入口。

训练循环、GRPO 基类、带 RL hooks 的 BAGEL 实现已经内置在 ``engine/``，
不再依赖外部 GeoUG RL engine checkout 或运行时符号链接。深度 transform 和决策 token
也已经物理合入内置的 ``engine/Bagel/data``。

空间任务仍通过替换三个模块级名字接入通用训练循环。之所以可以这么做，是因为
``engine/alpha_grpo/train.py`` 在模块级
   `from X import Y` 把它们绑成了自己的全局名，改全局名就等于改它用的实现：

   | 名字 | 原值 | 换成 | 为什么 |
   |---|---|---|---|
   | `train.InterleaveInferencer` | GeoUG RL engine 的 | `SpatialDepthInferencer` | 加深度模态和门控轨迹 |
   | `train.PerPromptStatTracker` | GeoUG RL engine 的 | `FixedScaleStatTracker` | 把 advantage 分母里的 λ 去掉，见 spatial_advantage.py |
   | `dataset.DATASET_MAP` | 4 个条目 | +`spatial_gate` | 读我们的 parquet |
   | `tasks.REGISTRY` | 4 个条目 | +`spatial_gate` | `@register` 自己注册，import 一下就行 |

``add_special_tokens`` 来自内置 BAGEL data 包；其 reasoning-token 顺序与 SFT
实现保持一致。

起训::

    bash scripts/train_rl.sh
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_ENGINE = _HERE / "engine"

# `<DECIDE>` 不注册就会被切成 4 个 token，4k ckpt 是带注册训的。
# 放在所有 import 之前：`data.constant` 在模块加载时就读这个开关。
os.environ.setdefault("BAGEL_REASONING_TOKENS", "1")

# Support both the module entry point and direct execution of this file.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from train.rl.bootstrap import prepare_engine_imports

prepare_engine_imports()

import importlib  # noqa: E402

from train.rl.engine.alpha_grpo import dataset as ag_dataset  # noqa: E402
from absl import app, flags  # noqa: E402
from ml_collections import config_flags  # noqa: E402

from train.rl.spatial_advantage import FixedScaleStatTracker  # noqa: E402
from train.rl.spatial_dataset import SpatialGateDataset  # noqa: E402
from train.rl.spatial_inferencer import make_inferencer_cls  # noqa: E402
import train.rl.spatial_algorithm  # noqa: E402,F401  —— @register("spatial_grpo")
import train.rl.spatial_task  # noqa: E402,F401  —— import 即通过 @register 注册进 tasks.REGISTRY
import train.rl.spatial_draft  # noqa: E402,F401  —— spatial_gate_draft / spatial_grpo_draft

CONFIG_FILE = str(_HERE / "config_spatial.py")


def _load_alphagrpo_train():
    """加载 `alpha_grpo/train.py`。两个坑，都必须绕。

    1. **不能 `import train`**：`Bagel/` 下也有一个 `train` 包，谁在 sys.path 前面
       谁赢。GeoUG RL engine 自己没这个问题 —— 它是把 train.py 当脚本跑的，那时
       train.py 就是 `__main__`，根本不走 import。我们是当模块用，只能按文件路径加载。

    2. **它的 `--config` 默认值会在 DEFINE 时就被解析**：
       `config_flags.DEFINE_config_file("config", "config/bagel.py", ...)`，
       而 `config/bagel.py::get_config(name)` 要一个参数。默认值没带 `:函数名`，
       于是 import 阶段直接抛 `IllegalFlagValueError`。

       绕法是在 exec 期间把 `DEFINE_config_file` 换成空操作，之后由我们自己用
       带函数名的默认值重新定义。命令行的 `--config` 照常覆盖。
    """
    real_define = config_flags.DEFINE_config_file
    config_flags.DEFINE_config_file = lambda *a, **kw: None
    try:
        mod = importlib.import_module("train.rl.engine.alpha_grpo.train")
    finally:
        config_flags.DEFINE_config_file = real_define

    config_flags.DEFINE_config_file(
        "config", f"{CONFIG_FILE}:spatial_gate_draft_v3", "Training configuration.")
    return mod


def _force_sync_each_batch():
    """让 `accelerator.accumulate()` 永远不进 `no_sync`。

    accelerate 0.34 的 `accumulate()` 在非同步步上会套 `model.no_sync()`，而
    DeepSpeed 的 engine 对此有两条硬 assert：ZeRO>=2 不许 no_sync（梯度是切分的），
    且 no_sync 里不许 `step()` —— 偏偏 `train.py:954` 每个 micro step 都无条件
    `optimizer.step()`。两条必踩一条。

    DeepSpeed 本来就自己按 ds_config 里的 `gradient_accumulation_steps` 决定哪一步
    真正更新参数，accelerate 这层 no_sync 是多余的。`sync_each_batch=True` 正是
    官方给的关掉它的开关，语义上对 DeepSpeed 就是「每个 micro batch 都规约梯度」。
    """
    import accelerate.accelerator as _acc

    real_plugin = _acc.GradientAccumulationPlugin

    def _plugin(*args, **kwargs):
        kwargs.setdefault("sync_each_batch", True)
        return real_plugin(*args, **kwargs)

    _acc.GradientAccumulationPlugin = _plugin


_force_sync_each_batch()

ag_train = _load_alphagrpo_train()
FLAGS = flags.FLAGS

ag_dataset.DATASET_MAP["spatial_gate"] = (SpatialGateDataset, {})


def main(argv):
    config = FLAGS.config

    # 把 inferencer 的构造和 spatial 的初始化绑在一起。`train.py:419` 调的是
    # `InterleaveInferencer(model, vae_model, tokenizer, vae_t, vit_t, new_token_ids)`
    # —— 六个位置参数，塞不进 transform 配置和深度缓存路径，所以在 __init__ 里补。
    base_cls = make_inferencer_cls()
    transform_config = config.model.transform_config
    depth_cache_dir = config.depth_cache_dir

    class BoundSpatialInferencer(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setup_spatial(transform_config, depth_cache_dir)

    ag_train.InterleaveInferencer = BoundSpatialInferencer

    # advantage 的分母去 λ 化（spatial_advantage.py 的 docstring 讲了为什么）。
    # `train.py:565` 是 `PerPromptStatTracker(config.sample.global_std)` —— 单个
    # 位置参数，所以 adv_scale 只能这样绑进去。
    adv_scale = config.train.adv_scale
    ag_train.PerPromptStatTracker = (
        lambda global_std: FixedScaleStatTracker(global_std, adv_scale=adv_scale))

    ag_train.main(argv)


if __name__ == "__main__":
    app.run(main)
