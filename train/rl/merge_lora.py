# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""把 RL 训出来的 LoRA 合进 base 权重，产出一个评测脚本能直接吃的 ckpt 目录。

为什么要合
----------
GeoUG RL engine 存的是 PEFT adapter（`checkpoint-N/hf_model/adapter_model.safetensors`，
几百 MB），而 `eval_d3_v5_mp.py` / `bagel_model_interface.py` 要的是一个带
`ema.safetensors` 的 ckpt 目录（`--model_path`）。中间需要这一步。

为什么不改评测侧去支持 `--lora_path`
------------------------------------
那样更省磁盘，但要改动评测模型的加载路径。合权重这条路评测侧不需要改，
代价只是每个 checkpoint 多写一份约 29 GB 的权重。

输出目录的小文件（config / tokenizer / ae.safetensors）全部做**符号链接**指回 base，
只有 `ema.safetensors` 是新写的。这样既省空间，也保证 tokenizer 和 base 逐字节一致 ——
词表顺序决定 embedding 行布局，复制一份出来漂了就是静默错配。

口径
----
* `W' = W + (B @ A) * (alpha / r)`，和 peft 的 `merge_and_unload` 一致；
* 加法在 **fp32** 上做再转回 base 的 dtype。bf16 只有 8 位尾数，
  直接在 bf16 上累加会把小的 LoRA 增量吃掉一部分；
* 只写 `ema.safetensors`。评测侧默认 `is_ema=true`（`bagel_model_interface.py:183`），
  RESULT.md 的所有基线也都量在 ema 上，所以合出来的这份要占住这个名字才对得上。

**为什么每一步都断言**：LoRA 合错的典型表现不是报错，而是**合了个寂寞** ——
key 对不上就一个都没合，模型还是 base，评测跑出来和 SFT 一模一样，
而你会以为是 RL 没学到东西。所以：一个 LoRA pair 都没匹配上直接抛，
匹配数对不上 adapter 里的 pair 数也直接抛。

用法::

    python -m train.rl.merge_lora \
        --lora /tmp/spatial_gate_debug/checkpoint-49/hf_model \
        --out /path/to/merged-checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# base 里除权重外还要带上的文件。全部做软链，不复制。
_SIDECAR = (
    "ae.safetensors", "config.json", "llm_config.json", "vit_config.json",
    "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "vocab.json", "merges.txt",
)

# peft 存出来的 key 长这样：
#   base_model.model.<base 里的 key 前缀>.lora_A.weight
# 去掉头尾就是 base 的模块名。`.default` 是有些 peft 版本会插进来的 adapter 名。
_LORA_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<mod>.+?)\.lora_(?P<ab>[AB])(?:\.default)?\.weight$")


def _collect_pairs(adapter):
    """adapter state dict -> {base 模块名: {'A': tensor, 'B': tensor}}。"""
    pairs = {}
    for k, v in adapter.items():
        m = _LORA_RE.match(k)
        if not m:
            # embedding / modules_to_save 之类的我们没开，出现了就是预期外，别静默跳过
            raise ValueError(f"adapter 里有认不出来的 key：{k}")
        pairs.setdefault(m.group("mod"), {})[m.group("ab")] = v
    bad = [m for m, d in pairs.items() if set(d) != {"A", "B"}]
    if bad:
        raise ValueError(f"{len(bad)} 个模块只有半边 LoRA（例：{bad[:3]}）")
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lora", required=True, help="checkpoint-N/hf_model 目录")
    ap.add_argument("--base", required=True, help="base checkpoint 目录")
    ap.add_argument("--out", required=True, help="输出 ckpt 目录")
    ap.add_argument("--scaling", type=float, default=None,
                    help="覆盖 alpha/r；不给就从 adapter_config.json 读")
    args = ap.parse_args()

    base, lora, out = Path(args.base), Path(args.lora), Path(args.out)
    base_weights = base / "ema.safetensors"
    if not base_weights.exists():
        raise SystemExit(f"base 权重不存在：{base_weights}")

    # ---- adapter ----
    cand = [lora / "adapter_model.safetensors", lora / "adapter_model.bin"]
    apath = next((p for p in cand if p.exists()), None)
    if apath is None:
        raise SystemExit(f"{lora} 里没有 adapter_model.safetensors / .bin")
    adapter = (load_file(str(apath)) if apath.suffix == ".safetensors"
               else torch.load(apath, map_location="cpu"))

    acfg_path = lora / "adapter_config.json"
    acfg = json.loads(acfg_path.read_text()) if acfg_path.exists() else {}
    r, alpha = acfg.get("r"), acfg.get("lora_alpha")
    scaling = args.scaling if args.scaling is not None else (alpha / r)
    print(f"[merge] adapter={apath.name}  r={r} alpha={alpha} scaling={scaling}")

    pairs = _collect_pairs(adapter)
    print(f"[merge] LoRA pair 数 = {len(pairs)}")

    # ---- 合 ----
    out.mkdir(parents=True, exist_ok=True)
    merged, n_hit = {}, 0
    with safe_open(str(base_weights), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"[merge] base 张量数 = {len(keys)}")
        for k in keys:
            w = f.get_tensor(k)
            mod = k[:-len(".weight")] if k.endswith(".weight") else None
            if mod is not None and mod in pairs:
                A = pairs[mod]["A"].to(torch.float32)   # (r, in)
                B = pairs[mod]["B"].to(torch.float32)   # (out, r)
                delta = (B @ A) * scaling
                if delta.shape != w.shape:
                    raise ValueError(
                        f"{mod}: LoRA 增量 {tuple(delta.shape)} 和权重 "
                        f"{tuple(w.shape)} 形状不符")
                w = (w.to(torch.float32) + delta).to(w.dtype)
                n_hit += 1
            merged[k] = w

    # 合了个寂寞是这件事最常见的失败模式，所以这里是硬断言
    if n_hit == 0:
        raise SystemExit(
            "一个 LoRA 模块都没匹配上 base 的 key —— 合出来就是 base 本身。"
            f"\nadapter 侧例子：{sorted(pairs)[:3]}"
            f"\nbase 侧例子：{keys[:3]}")
    if n_hit != len(pairs):
        raise SystemExit(f"只合上 {n_hit}/{len(pairs)} 个 LoRA 模块，剩下的 key 对不上")
    print(f"[merge] 合上 {n_hit}/{len(pairs)} 个模块")

    save_file(merged, str(out / "ema.safetensors"), metadata={"format": "pt"})

    # ---- 小文件做软链，不复制 ----
    for name in _SIDECAR:
        src, dst = base / name, out / name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src.resolve(), dst)

    print(f"[merge] 写好 {out}")
    print(f"[merge] 推理：python -m inference.spatial --model-path {out} --image IMAGE --question QUESTION")


if __name__ == "__main__":
    main()
