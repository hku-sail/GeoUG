# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""从 `rl_train_v2` 切一份**各维度配平**的小训练集（默认 800 题）。

`rl_train_v2` 按源均匀，但物-物题几乎全来自 nuScenes（ScanNet++ 只有 345 道），
随机取前 800 题时 ScanNet++ 物-物只有 26 道。这里按格子硬配额：

    来源(2) x 题型(4) = 8 格，每格 n/8 题
    格内：问近 / 问远 各半
          多选题型（*_among）3 选 1 : 4 选 1 = 60 : 40
          同一选项数下答案字母均匀
    全局：每张图只出一题

某个小格凑不满时，先在同一 (方向, 选项数) 里换字母补，再在整格里补，补了会打印出来。
图、相机表、深度缓存都是 `rl_train_v2` 的子集，训练侧不用重算任何东西。

用法::

    python -m train.rl.data_pipeline.build_balanced_subset \
        --src-dir /path/to/rl-train \
        --out-dir /path/to/rl-train-balanced
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

_FAR = r"farthest|farther|further"
_NOPT_SHARE = {2: {2: 1.0}, 3: {3: 0.6, 4: 0.4}}   # 按「是不是 among 题」取


def _quotas(total, shares):
    """最大余数法把 total 按 shares 分成整数。"""
    keys = list(shares)
    raw = np.array([total * shares[k] for k in keys])
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: total - base.sum()]:
        base[i] += 1
    return dict(zip(keys, base.tolist()))


def pick_cell(cell, n, rng):
    among = cell["nopt"].iloc[0] > 2
    nopt_share = _NOPT_SHARE[3 if among else 2]
    shares = {}
    for d in ("near", "far"):
        for k, s in nopt_share.items():
            for letter in "ABCD"[:k]:
                shares[(d, k, letter)] = 0.5 * s / k
    quota = _quotas(n, shares)

    cell = cell.iloc[rng.permutation(len(cell))]
    taken, short = [], 0
    for key, q in quota.items():
        pool = cell[(cell["dir"] == key[0]) & (cell["nopt"] == key[1]) & (cell["answer"] == key[2])]
        got = pool.head(q)
        taken.append(got)
        short += q - len(got)
    sel = pd.concat(taken)
    if short:
        # 先在同 (方向, 选项数) 里补，再整格补
        for level in (["dir", "nopt"], []):
            if short == 0:
                break
            rest = cell.drop(sel.index)
            need = sel.groupby(level).size() if level else None
            if level:
                target = {k[:2]: 0 for k in quota}
                for k, q in quota.items():
                    target[k[:2]] += q
                fill = []
                for (d, k), t in target.items():
                    have = int(need.get((d, k), 0))
                    if t > have:
                        fill.append(rest[(rest["dir"] == d) & (rest["nopt"] == k)].head(t - have))
                fill = pd.concat(fill) if fill else rest.head(0)
            else:
                fill = rest.head(short)
            fill = fill.head(short)
            sel = pd.concat([sel, fill])
            short -= len(fill)
    if short:
        raise ValueError(f"格子只有 {len(cell)} 题，凑不出 {n}")
    return sel


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("-n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Path(args.src_dir)
    df = pd.read_parquet(src / "train.parquet")
    cols = list(df.columns)
    df["nopt"] = df["options"].map(lambda s: len(json.loads(s)))
    df["dir"] = np.where(df["question"].str.contains(_FAR, case=False), "far", "near")

    # 一图一题：先打乱再按图去重，保留哪道题是随机的
    df = df.iloc[rng.permutation(len(df))].drop_duplicates("image_key")

    cells = df.groupby(["source", "category"])
    per_cell = args.n // len(cells)
    assert per_cell * len(cells) == args.n, f"n={args.n} 要能被 {len(cells)} 格整除"
    sel = pd.concat(pick_cell(c, per_cell, rng) for _, c in cells)
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
    assert sel["id"].is_unique and sel["image_key"].is_unique

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel[cols].to_parquet(out / "train.parquet", index=False)
    for f in ("validate.parquet", "camera_table.json"):
        shutil.copy(src / f, out / f)

    print(f"{len(sel)} 题 / {sel['image_key'].nunique()} 图 -> {out}\n")
    print(pd.crosstab(sel["category"], sel["source"], margins=True), "\n")
    print(pd.crosstab([sel["source"], sel["category"]], [sel["dir"]]), "\n")
    print(pd.crosstab([sel["source"], sel["category"]], [sel["nopt"], sel["answer"]]), "\n")
    print("illusion", sel["illusion"].value_counts().to_dict())


if __name__ == "__main__":
    main()
