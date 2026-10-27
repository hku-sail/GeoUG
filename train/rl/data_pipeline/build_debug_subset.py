# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""从一个 RL parquet 里切出「深度已缓存」的小子集，用来跑通训练。

为什么要按缓存过滤：`load_cached_depth` 缺文件就抛。这是故意的 —— 静默退回去跑
扩散会让一个 epoch 从几分钟变成十几小时。跑通阶段只想要能跑的那些题，
所以这里先和缓存目录求交集。

用法::

    python -m train.rl.data_pipeline.build_debug_subset \\
        --src /path/to/validate.parquet \\
        --depth-cache /path/to/depth-cache \\
        --n 100 --out /path/to/rl-debug
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

_CAMERA_TABLE = "camera_table.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--depth-cache", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    cached = {p.stem for p in Path(args.depth_cache).glob("*.npy")}
    df = pd.read_parquet(src)
    keep = df[df.image_key.isin(cached)]
    print(f"  {len(df)} 题 / {df.image_key.nunique()} 图 -> "
          f"缓存命中 {len(keep)} 题 / {keep.image_key.nunique()} 图")
    if len(keep) < args.n:
        print(f"  ! 只有 {len(keep)} 题可用，少于要的 {args.n}")
    keep = keep.sample(n=min(args.n, len(keep)), random_state=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(out / "train.parquet", index=False)
    # SpatialGateDataset 的 test split 落到 validate.parquet 上；跑通阶段不评测，
    # 给个最小的占位，让 build_dataloader 的两个 split 都能建起来。
    keep.head(4).to_parquet(out / "validate.parquet", index=False)
    shutil.copy(src.parent / _CAMERA_TABLE, out / _CAMERA_TABLE)

    print(f"  -> {out}  train {len(keep)} 题 / {keep.image_key.nunique()} 图")
    print("     类别", dict(keep.category.value_counts()))
    print("     来源", dict(keep.source.value_counts()))


if __name__ == "__main__":
    main()
