# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""深度门控的 inferencer：GeoUG RL engine 的 `InterleaveInferencer` + 深度模态 + 门控轨迹。

架构：继承它的 inferencer，而不是转接我们的
--------------------------------------------
最初的方案是把 GeoUG RL engine 加载好的模型包成 `BagelModel` 的形状，喂给
`DepthCoTInferencerV5`，再把结果适配回 GeoUG RL engine 的批量 API。三层转接。

放弃它的理由是量出了一个事实：**深度模态在注入路径上只是换了一个 transform。**
`inference_depth_cot.py:261` 那行 ——

    transforms = self.depth_transform if is_depth else self.vae_transform

—— 就是 `is_depth` 的全部。`which_vit="gen"` 同理，只是 `vit_gen_transform` 换
`vit_transform`。后来又逐行比对了两个分支的 `prepare_vae_images` /
`prepare_vit_images` / `prepare_prompts`：**完全一致**（只差尾空格和一行
`.to('cuda')`）。BAGEL 模型本体在 spatial_ug 里一行都没为深度改过
（`grep -c depth modeling/bagel/bagel.py` = 0）。

既然如此，直接继承 GeoUG RL engine 的 `InterleaveInferencer`、加两个 transform 属性和
一个 `rollout_one`，转接层全部消失。深度模块作为实体文件包含在仓库中，
由 bootstrap 固定使用 GeoUG 的完整 `data` 包，不依赖外部目录或符号链接。

log-prob 怎么来：不要自己在解码循环里收
---------------------------------------
GeoUG RL engine 的做法是两步（`inferencer.py:919-931`）：

    texts = self.gen_text(ctx, ...)                                  # 自由采样
    ctx, lp = self.update_context_text(texts, ctx, return_logprobs=True)  # 重过一遍

第二步是 teacher-forced 的单次前向，顺带推进上下文。

这看起来绕，但它避开了一个**不会报错**的坑：采样吐出的是 token id 序列，
而把它 decode 成文本、训练循环再 encode 回去，得到的 id 序列**不保证相同**
（BPE 的合并边界会变）。如果 rollout 的 log-prob 记在采样 id 上、而
`compute_text_logprob` 算在重编码 id 上，GRPO 的重要性比就逐 token 错位，
梯度打到别的 token 上，全程静默。

走 GeoUG RL engine 这条路，rollout 和训练用的是同一个函数、同一份重编码结果，
一致性是结构保证的，不是靠约定。代价是每段文本多一次前向。

stop marker
-----------
`gen_text` 不支持 stop marker，只能生成到 eos。我们需要在 `</GENERATE_DEPTH>` /
`</tool_call>` 处停下来注入东西。做法是**自由生成完再截断**：截断后的那段喂给
`update_context_text`，上下文和 log-prob 都只覆盖保留的部分，丢掉的尾巴纯属浪费
算力（那段是模型在没拿到深度图的情况下自己编的，本来也不能要）。

注意 stop marker 里**没有 `<DECIDE>`**，和评测的 `STOP_MARKERS` 一致
（`inference_depth_cot.py:271`）。评测的 v5 在第一轮额外用 `stop_markers=(DECIDE,)`
切一刀，那是为了读决策点的整词表 logits 算 `p_depth`；RL 不需要那个读数
（真正要的是**采出来的动作**，它就在保留的文本里），多切一刀反而会把一段 CoT
拆成两个 step、和评测的上下文推进顺序错开。

轨迹长什么样
------------
`contexts` 是**条件部分**（system -> und-RGB -> 题面），训练循环从这里开始重放；
`generation_steps` 是动作和观测交替：

    直答分支   [think_text(CoT+<DECIDE></think>+答案)]
    深度分支   [think_text(CoT+<DECIDE><GENERATE_DEPTH>...),
                input_image(生成条件 RGB), input_image(深度图),
                think_text(<|tool_call_start|>...), input_text(工具结果),
                ..., text(答案)]

`input_image` / `input_text` 在 GeoUG RL engine 里是**不计 loss** 的条件步
（`common.py` 写明 "input conditioning — no loss"）。注入的深度图和工具结果正好
落在这两个类型上，所以 PLAN.md §2.3 担心的「要给注入 token 打掩码」是结构自带的，
不用手写 mask。

条件部分为什么必须烤进 `contexts` 而不能当 step
-----------------------------------------------
题面那张 RGB 走的是 `vae=False, vit=True, which_vit="und"`。而训练循环重放
`input_image` 时调的是 `update_context_image(gen_step.images, gen_context)`
—— **不带 kwargs**（`train.py:713/771/828/924`，recompute / KL / 训练三处都一样）。
我们改不了它的调用形式，所以带特殊 kwargs 的图只能烤进 `contexts`。

深度分支那两张图躲不掉（它们在生成中途才出现），只能让被调方自己认出喂进来的是
什么 —— 见 `update_context_image` 的类型分派。

GRPO 的组怎么成
---------------
`compute_advantages` 按 **prompt 字符串**分组，分组池是 `accelerator.gather`
跨进程收上来的那一批（`common.py:94-104`）。而一个 `sample` 只有一套 `contexts`
（BAGEL 的 packed KV），同一 sample 内的多条 rollout 必须共享 step 序列 ——
深度分支比直答分支多两个 step，做不到。

所以每个 sample 只装 1 条 rollout，**G = world_size**。配
`DistributedKRepeatBatchSampler(batch_size=1, k=world_size)` 即可，
4 卡 G=4，8 卡 G=8。
"""
from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_ENGINE = _HERE / "engine"

from train.rl.bootstrap import prepare_engine_imports

prepare_engine_imports()

from data.constant import (  # noqa: E402
    DECIDE, GENERATE_DEPTH_END, GENERATE_DEPTH_START, GENERATE_IMAGE_END, THINK_CLOSE,
    TOOL_CALL_END, TOOL_CALL_START, TOOL_RESULT_END, TOOL_RESULT_START,
)

# `rollout_one(force=...)` 在 `<DECIDE>` 之后强行写入的门控 token。
FORCE_TOKENS = {"depth": GENERATE_DEPTH_START, "answer": THINK_CLOSE}
from data.data_utils import pil_img2rgb  # noqa: E402

# 评测的 `DepthCoTInferencer.STOP_MARKERS`（inference_depth_cot.py:271）逐项一致。
_STOP_MARKERS = (GENERATE_DEPTH_END, GENERATE_IMAGE_END, TOOL_CALL_END)

# 数据引擎的 ToolSession —— 和训练数据里的工具结果是同一份实现，不能换。
#
# 必须是 **runtime_v3** 的 `SelfContainedToolSession`，不是 `runtime.ToolSession`：
# v2 那个要先 `ingest_grounding(助手文本)` 把 CoT 里的 `<bbox>/<point>` 解析进
# `_grounding`，`query_3d_distance` 才查得到物体（`runtime.py:518`，那条路**没有**
# 内联坐标兜底）；v3 协议把 point/bbox 内联进 tool call 的 args，运行时直接读。
# 一旦用错，`query_3d_distance` 必然回 `{"error":"could not read depth for ..."}`，
# 而模型看到 error 会照着先验编距离 —— 不报错，只是 reward 全是噪声。
# 评测侧走的就是 v3 这份（`inference_depth_cot_v3.py:377`）。
_DATA_ENGINE_ROOT = os.environ.get(
    "GEOMETRY_TOOL_ROOT", str(_HERE / "geometry_tools"))
_TOOL_SESSION_CLS = None


def _tool_session_cls():
    global _TOOL_SESSION_CLS
    if _TOOL_SESSION_CLS is None:
        if not Path(_DATA_ENGINE_ROOT).is_dir():
            raise FileNotFoundError(
                f"geometry tool runtime not found: {_DATA_ENGINE_ROOT}; "
                "set GEOMETRY_TOOL_ROOT or install train/rl/geometry_tools")
        if _DATA_ENGINE_ROOT not in sys.path:
            sys.path.insert(0, _DATA_ENGINE_ROOT)
        from engine_utils.runtime_v3 import SelfContainedToolSession  # noqa: PLC0415
        _TOOL_SESSION_CLS = SelfContainedToolSession
    return _TOOL_SESSION_CLS


def _truncate_at_marker(text, markers=_STOP_MARKERS):
    """截到第一个 stop marker（含）为止；一个都没有就原样返回。

    返回 `(截断后的文本, 命中的 marker 或 None)`。
    """
    hits = [(text.find(m), m) for m in markers if m in text]
    if not hits:
        return text, None
    pos, marker = min(hits)
    return text[: pos + len(marker)], marker


def build_transforms(transform_config_path):
    """按评测用的同一份 yaml 造出四个 transform。

    照搬 `benchmark/bagel_model_interface.py:104-160` 的口径 —— **必须逐字节一致**，
    否则 RL 时注入的深度和评测时的不是同一张图，学到的东西评测上复现不出来。
    那边是一条 45 行的 if 链，这里压成 dict 分派，语义不变。

    不支持的 normalization 类型直接抛：静默退回到 `vae_transform` 会让深度图按 RGB
    口径归一化，模型照样能跑、数值全错。
    """
    import yaml  # noqa: PLC0415

    from data.transforms import ImageTransform  # noqa: PLC0415
    from data.transforms_depth import DepthTransformBase  # noqa: PLC0415
    from data.depth_encoding.disparity_transform import DepthDisparityTransform  # noqa: PLC0415
    from data.depth_encoding.multiscale_disparity_transform import (  # noqa: PLC0415
        DepthMultiScaleDisparityTransform,
    )
    from data.depth_encoding.power_transform import DepthPowerTransform  # noqa: PLC0415
    from data.depth_encoding.scale_shift_transform import ScaleShiftDepthNormalizer  # noqa: PLC0415

    cfg = yaml.safe_load(open(transform_config_path))
    vae_t = ImageTransform(**cfg["image_transform_args"])
    vit_t = ImageTransform(**cfg["vit_image_transform_args"])
    vit_gen_t = (ImageTransform(**cfg["vit_image_gen_transform_args"])
                 if "vit_image_gen_transform_args" in cfg else vit_t)

    depth_t = None
    if "depth_transform_args" in cfg:
        args = dict(cfg["depth_transform_args"])
        norm = args.get("depth_normalization", {"type": "scale_shift"})
        kind = norm["type"]
        # stack 的取值跟着原实现：只有多尺度视差是 False（它自己产出三通道）。
        makers = {
            "scale_shift": (lambda: ScaleShiftDepthNormalizer(), True),
            "power_transform": (lambda: DepthPowerTransform(
                lam=norm.get("lam", -3.0), c=norm.get("c", 10.0 / 3.0)), True),
            "disparity_transform": (lambda: DepthDisparityTransform(
                c=norm.get("c", 5.0)), True),
            "multiscale_disparity_transform": (lambda: DepthMultiScaleDisparityTransform(
                c_values=norm.get("c", [2.0, 8.0, 32.0])), False),
        }
        if kind not in makers:
            raise ValueError(f"depth_normalization 类型 {kind} 不支持")
        make_norm, stack = makers[kind]
        depth_t = DepthTransformBase(**args, normalize=make_norm(), stack=stack)

    return vae_t, vit_t, vit_gen_t, depth_t


def make_inferencer_cls():
    """造出 `SpatialDepthInferencer` 类。

    在函数里定义而不是模块级，是因为基类要从 GeoUG RL engine 的 `Bagel.inferencer` 导入 ——
    模块级 import 会让 `reward.py` / 数据管线这些不依赖 GeoUG RL engine 的脚本也被拖着走。
    """
    if str(_ENGINE / "Bagel") not in sys.path:
        sys.path.insert(0, str(_ENGINE / "Bagel"))
    from Bagel.inferencer import InterleaveInferencer, move_generation_input_to_device  # noqa: PLC0415
    from train.rl.engine.alpha_grpo.common import GenerationStep  # noqa: PLC0415

    class SpatialDepthInferencer(InterleaveInferencer):
        """GeoUG RL engine 的 inferencer + 深度模态 + 门控 rollout。

        相对基类只多四样东西：
          1. `depth_transform` / `vit_gen_transform` 两个 transform；
          2. `update_context_image` 认得 `is_depth` / `which_vit`；
          3. `load_cached_depth()` —— 深度读缓存，不跑扩散；
          4. `rollout_one()` —— 多轮门控轨迹。
        """

        # 基类默认走 flex_attention，但 Qwen2.5-7B 是 28 query head / 4 KV head，
        # 每个 KV head 分到 7 个 query head，不是 2 的幂，inductor 直接拒绝下降。
        # spatial_ug 自己的评测路径本来也没用 flex，走的是逐 sample 的 sdpa，
        # 关掉它顺便让 rollout 的 logprob 和评测对得上。
        use_flex_attention = False

        def setup_spatial(self, transform_config_path, depth_cache_dir):
            """四个 transform 全部从评测那份 yaml 重建，覆盖基类的两个。

            **不能只补深度和 vit-gen 两个**：`train.py:152` 是用
            `config.model.vit_max_image_size` 那几个标量造 vae / vit transform 的，
            而 yaml 里 `vit_image_transform_args` 还带一个 `max_pixels: 2_007_040`，
            标量那条路造不出来。少这一项，同一张图在 RL 里和评测里会被切成
            不同数量的 patch —— 不报错，只是学到的东西评测上复现不出来。
            """
            (self.vae_transform, self.vit_transform,
             self.vit_gen_transform, self.depth_transform) = build_transforms(
                transform_config_path)
            self.depth_cache_dir = Path(depth_cache_dir)
            return self

        # --------------------------------------------------------------
        def update_context_image(self, images, gen_context, vae=True, vit=True,
                                 which_vit=None, is_depth=None):
            """基类同名方法 + 两处 transform 选择。

            前向调用形式（`self.model(..., forward_mode=...)`）必须和基类一致 ——
            模型被 DDP / DeepSpeed 包过，绕开 `__call__` 直接调方法会跳过
            梯度同步的 hook。

            `which_vit` / `is_depth` 传 None 时走**类型分派**，这是给训练循环用的：
            它重放 `input_image` 步时不带 kwargs（见模块 docstring）。我们改不了它的
            调用形式，只能让被调方自己认出喂进来的是什么：

                torch.Tensor (1,H,W) -> 深度图        -> depth_transform
                PIL.Image            -> 生成条件 RGB  -> vit_gen_transform

            认不出来就抛。默认按 RGB 处理的话，深度会用 RGB 的归一化口径过 VAE，
            **不报任何错**，只是重放出来的上下文和 rollout 时的不是同一个。
            """
            if not isinstance(images, list):
                images = [images]

            if which_vit is None and is_depth is None:
                probe = images[0]
                if isinstance(probe, np.ndarray):
                    images = [torch.from_numpy(im) if isinstance(im, np.ndarray) else im
                              for im in images]
                    probe = images[0]
                if isinstance(probe, torch.Tensor):
                    images = [im.unsqueeze(0) if im.ndim == 2 else im for im in images]
                    assert all(im.ndim == 3 and im.shape[0] == 1 for im in images), (
                        f"重放 input_image 时遇到形状 {tuple(images[0].shape)} 的张量，"
                        f"认不出是什么。深度图应为 (1,H,W) 或 (H,W)。")
                    vae, vit, which_vit, is_depth = True, False, "und", True
                else:
                    assert hasattr(probe, "convert"), (
                        f"重放 input_image 时遇到 {type(probe).__name__}，"
                        f"既不是深度张量也不是 PIL 图。")
                    vae, vit, which_vit, is_depth = True, True, "gen", False
            which_vit = which_vit or "und"
            is_depth = bool(is_depth)

            # 张量（深度图）统一挪到本卡：rollout 时它来自缓存（CPU），
            # 重放时它来自 step 里存的副本（也是 CPU）。两条路走同一句。
            images = [im.to(self.device).contiguous() if isinstance(im, torch.Tensor) else im
                      for im in images]

            assert vae or vit
            past_key_values = gen_context["past_key_values"]
            kv_lens, ropes = gen_context["kv_lens"], gen_context["ropes"]
            has_mod = hasattr(self.model, "module")
            prepare_vae = (self.model.module if has_mod else self.model).prepare_vae_images
            prepare_vit = (self.model.module if has_mod else self.model).prepare_vit_images

            if vae:
                # ← 深度模态在注入路径上的**全部**差异就是这一行的 transform 选择。
                transforms = self.depth_transform if is_depth else self.vae_transform
                generation_input, kv_lens, ropes = prepare_vae(
                    curr_kvlens=kv_lens, curr_rope=ropes, images=images,
                    transforms=transforms, new_token_ids=self.new_token_ids,
                )
                generation_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                    for k, v in generation_input.items()}
                past_key_values = self.model(
                    self.vae_model, past_key_values,
                    use_flex_attention=self.use_flex_attention, **generation_input,
                    forward_mode="forward_cache_update_vae")

            if vit:
                transforms = self.vit_gen_transform if which_vit == "gen" else self.vit_transform
                generation_input, kv_lens, ropes = prepare_vit(
                    curr_kvlens=kv_lens, curr_rope=ropes, images=images,
                    transforms=transforms, new_token_ids=self.new_token_ids,
                )
                generation_input = {k: v.to(self.device) for k, v in generation_input.items()}
                past_key_values = self.model(
                    past_key_values, use_flex_attention=self.use_flex_attention,
                    **generation_input, forward_mode="forward_cache_update_vit")

            gen_context["kv_lens"] = kv_lens
            gen_context["ropes"] = ropes
            gen_context["past_key_values"] = past_key_values
            return gen_context

        # --------------------------------------------------------------
        def load_cached_depth(self, image_key, depth_dir=None):
            """读预生成的米制深度 (H, W)，留在 CPU 上。

            缺了就抛，不静默退回去跑扩散 —— 一次扩散 110 秒，一个 epoch 会从几分钟
            变成十几小时，而且看起来"只是有点慢"。

            `depth_dir` 给在线评测用：评测集（D3）和训练集的缓存不是一个目录。
            """
            p = Path(depth_dir or self.depth_cache_dir) / f"{image_key}.npy"
            if not p.exists():
                raise FileNotFoundError(
                    f"深度缓存缺 {image_key}：{p}。"
                    "请下载发布的 depth-cache artifact，或按 RL_TRAINING.md 准备缓存。")
            return torch.from_numpy(np.load(p))

        def _make_tool_session(self, depth_metric, intrinsics, image_wh):
            """基于深度图建一个 ToolSession。

            逐行对应 `benchmark/interleave_reasoning/inference_depth_cot.py`
            的 `_make_tool_session`（那边改了这里也要改）。没有直接 import 是因为
            那个模块会把 spatial_ug 的 `modeling` / `BagelModel` 一起拖进来，
            而本进程的 `modeling` 必须是 GeoUG RL engine 那份。

            conf 给全 1 且 `conf_thresh=0`：模型不预测置信度，让所有像素都通过。
            """
            depth_np = depth_metric.detach().float().cpu().numpy()
            if intrinsics is not None:
                K = np.asarray(intrinsics, dtype=np.float64)
            else:
                W, H = int(image_wh[0]), int(image_wh[1])
                f = float(max(W, H))
                K = np.array([[f, 0.0, W / 2.0],
                              [0.0, f, H / 2.0],
                              [0.0, 0.0, 1.0]], dtype=np.float64)
            return _tool_session_cls()(
                depth=depth_np, conf=np.ones_like(depth_np), intrinsics=K,
                image_wh=(int(image_wh[0]), int(image_wh[1])), conf_thresh=0.0,
            )

        @staticmethod
        def _compact_tool_result(injected):
            """把工具结果里的 JSON 体重新用紧凑分隔符序列化，和训练渲染
            （`templates._format_tool_result`，`separators=(",", ":")`）逐字节一致。
            """
            if not (injected.startswith(TOOL_RESULT_START)
                    and injected.endswith(TOOL_RESULT_END)):
                return injected
            body = injected[len(TOOL_RESULT_START):-len(TOOL_RESULT_END)]
            try:
                payload = json.loads(body)
            except Exception:  # noqa: BLE001
                return injected
            compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            return f"{TOOL_RESULT_START}{compact}{TOOL_RESULT_END}"

        # --------------------------------------------------------------
        @torch.no_grad()
        def gen_text_after_prefix(self, gen_context, prefix_ids, max_length=500,
                                  do_sample=True, temperature=1.0, top_p=None, top_k=None):
            """同一段里先写入 `prefix_ids`、再接着采样；返回 prefix + 续写的文本。

            不用 `gen_text(prefix_think_token_ids=...)`：`bagel.generate_text` 在多 token
            那一步之后把位置加的是 `query_lens[0]`（= 前缀长 + 1）而不是 1，续写的 token
            会整体错开 len(prefix) 个位置。这里拆两次调用，都不经过那一行的错位：

            1. `bos + prefix[:-1]` 一次前向写进 KV（`max_length=1`，采出的那个 token 丢掉）；
            2. 以 `prefix[-1]` 为起始 token、位置和 KV 下标自己算，走单 token 路径续写。
            """
            ctx = deepcopy(gen_context)
            kv, kv_lens, ropes = ctx["past_key_values"], ctx["kv_lens"], ctx["ropes"]
            assert len(kv_lens) == 1 and len(prefix_ids) >= 2
            head, last = list(prefix_ids[:-1]), int(prefix_ids[-1])
            common = dict(do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
                          end_token_id=self.new_token_ids["eos_token_id"],
                          use_flex_attention=self.use_flex_attention)

            inp = move_generation_input_to_device(self.model.prepare_start_tokens(
                kv_lens, ropes, self.new_token_ids, prefix_think_token_ids=head), self.device)
            _, kv = self.model.generate_text(past_key_values=kv, max_length=1,
                                             return_kv_cache=True, **common, **inp)

            n_kv = int(kv_lens[0]) + len(head) + 1                 # + bos
            inp = move_generation_input_to_device({
                "packed_start_tokens": torch.tensor([last], dtype=torch.long),
                "packed_query_position_ids": torch.tensor([int(ropes[0]) + len(head) + 1],
                                                          dtype=torch.long),
                "key_values_lens": torch.tensor([n_kv], dtype=torch.int),
                "packed_key_value_indexes": torch.arange(n_kv, dtype=torch.long),
            }, self.device)
            seq = self.model.generate_text(past_key_values=kv, max_length=max_length,
                                           **common, **inp)
            text = self.tokenizer.decode(head + seq[:, 0].tolist())
            return text.split("<|im_end|>")[0]

        # --------------------------------------------------------------
        @torch.no_grad()
        def rollout_one(self, image_pil, question, image_key, system_prompt,
                        intrinsics=None, image_wh=None,
                        max_rounds=6, max_text_tokens=512,
                        temperature=1.0, top_p=1.0, top_k=0,
                        do_sample=True, depth_dir=None, force=None):
            """跑一条完整的门控轨迹。

            上下文的搭法（system -> und-RGB -> 题面）和 `DepthCoTInferencerV5.run`
            完全一致 —— 训练数据就是按这个顺序展开的。

            `force="depth"/"answer"`：`<DECIDE>` 之前照常采样，之后强行写入对应的门控
            token（`FORCE_TOKENS`），再在**同一段**里接着采样。自然采到的动作恰好相同时
            不重采。没到 `<DECIDE>` 就不强制，`forced` 为 False。

            返回 dict::

                contexts    (gen, cfg_text, cfg_img) 三元组，训练循环从这里重放
                steps       GenerationStep 列表
                think_texts 计 loss 的文本段（给 compute_advantages 统计长度）
                text        拼起来的完整助手文本（给 verifier）
                used_depth  这条轨迹走没走深度分支
                forced      门控 token 是不是强制写入的
                n_rounds    实际轮数
            """
            rgb = pil_img2rgb(image_pil)
            if image_wh is None:
                image_wh = (rgb.width, rgb.height)

            gen_context = self.init_gen_context(1)
            cfg_img_context = self.init_gen_context(1)  # text-only 分支，永远不放图
            if system_prompt:
                gen_context = self.update_context_text([system_prompt], gen_context)
                cfg_img_context = self.update_context_text([system_prompt], cfg_img_context)
            gen_context = self.update_context_image(
                [rgb], gen_context, vae=False, vit=True, which_vit="und")
            gen_context = self.update_context_text([question], gen_context)
            cfg_img_context = self.update_context_text([question], cfg_img_context)

            # 训练循环从这三个上下文开始重放 generation_steps，所以必须在生成前快照。
            # cfg_text / cfg_img 只被 image step 消费，我们没有 image step ——
            # 它们对 loss 没有影响，给出合法值即可（照 batch_generate_texts 的口径）。
            contexts = (deepcopy(gen_context), deepcopy(gen_context), cfg_img_context)

            steps, texts, think_texts = [], [], []
            used_depth, tool_session, rnd, forced = False, None, 0, False

            for rnd in range(max_rounds):
                raw = self.gen_text(
                    gen_context, max_length=max_text_tokens, do_sample=do_sample,
                    temperature=temperature, top_p=top_p, top_k=top_k)[0]
                if force and not forced and DECIDE in raw:
                    head = raw[:raw.index(DECIDE) + len(DECIDE)]
                    tok = FORCE_TOKENS[force]
                    if not raw[len(head):].lstrip().startswith(tok):
                        # 前缀喂在 bos 之后、同一段里，不是新起一段（SFT 里决策段就是一段）。
                        raw = self.gen_text_after_prefix(
                            gen_context, self.tokenizer.encode(head + tok),
                            max_length=max_text_tokens, do_sample=do_sample,
                            temperature=temperature, top_p=top_p, top_k=top_k)
                    forced = True
                text, marker = _truncate_at_marker(raw)
                text = text.strip()
                if not text:
                    break

                triggered_depth = (marker == GENERATE_DEPTH_END
                                   or GENERATE_DEPTH_START in text)
                triggered_tool = (marker == TOOL_CALL_END
                                  or (TOOL_CALL_START in text and TOOL_CALL_END in text))

                gen_context, lp = self.update_context_text(
                    [text], gen_context, return_logprobs=True, temperature=temperature)
                # think_text 和 text 在 GeoUG RL engine 里都计 loss，差别只在 cfg 上下文的
                # 推进规则 —— 我们没有 image step，那条规则不生效。分开纯为日志可读：
                # 中间的触发段是 think，最后一段是答案。
                steps.append(GenerationStep(
                    step_type="think_text" if (triggered_depth or triggered_tool) else "text",
                    texts=[text], text_per_token_log_probs=lp))
                texts.append(text)
                think_texts.append(text)

                if triggered_depth:
                    used_depth = True
                    # 生成条件 RGB：评测里这一步是喂给扩散当条件的。走缓存不需要扩散，
                    # 但**这张图仍要进上下文** —— 后续文本 attend 得到它，
                    # 少喂一次就和评测不一致了。
                    gen_context = self.update_context_image(
                        [rgb], gen_context, vae=True, vit=True, which_vit="gen")
                    steps.append(GenerationStep(step_type="input_image", images=[rgb]))

                    depth = self.load_cached_depth(image_key, depth_dir)   # (H, W)，CPU
                    tool_session = self._make_tool_session(depth, intrinsics, image_wh)
                    depth_input = depth.unsqueeze(0).contiguous()      # (1, H, W)
                    gen_context = self.update_context_image(
                        [depth_input], gen_context, vae=True, vit=False, is_depth=True)
                    # 深度图是"观测"不是"动作"，梯度不该流过它 -> input_image（不计 loss）。
                    steps.append(GenerationStep(step_type="input_image",
                                                images=[depth_input]))
                    continue

                if triggered_tool:
                    injected = (
                        f'{TOOL_RESULT_START}{{"error":"no depth generated"}}{TOOL_RESULT_END}'
                        if tool_session is None
                        else self._compact_tool_result(tool_session.run_tool_call(text)))
                    gen_context = self.update_context_text([injected], gen_context)
                    cfg_img_context = self.update_context_text([injected], cfg_img_context)
                    texts.append(injected)
                    # 工具结果是环境返回的，同样不是模型的动作 -> 不计 loss。
                    steps.append(GenerationStep(step_type="input_text", texts=[injected]))
                    continue

                break

            return {
                "contexts": contexts,
                "steps": steps,
                "think_texts": think_texts,
                "text": "\n".join(texts),
                "used_depth": used_depth,
                "forced": forced,
                "reached_decide": DECIDE in "\n".join(texts),
                "n_rounds": rnd + 1,
            }

    return SpatialDepthInferencer
