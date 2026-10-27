# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""从 D3 800 题里分层抽在线评测子集，各分层维度的构成与 800 题一致。

D3 800 是完整析因设计：split(depth/distance) × domain(indoor/outdoor) ×
quad(11 = 捷径与 GT 一致 / 00 = 捷径与 GT 相反) × bin(4 档相对读数差，即难度)，
32 格每格 25 题，每格内 A/B 各半（实测 12/13 或 13/12，全体 400/400）。

所以子集按格等量抽：每格 `per_cell` 题，格内 A/B 各一半。`per_cell=6` 得 192 题 ——
200 除不开 32 格，192 是所有交互项都严格平衡的最近值。

    python -m train.rl.data_pipeline.make_d3_eval_subset \
        --rl-parquet /path/to/d3_800_rl.parquet \
        --questions /path/to/3dcuebench.parquet \
        --depth-cache /path/to/depth-cache \
        --out /path/to/d3_192_rl.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STRATA = ["split", "domain", "quad", "bin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-parquet", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--depth-cache", required=True)
    ap.add_argument("--per-cell", type=int, default=6, help="偶数，格内 A/B 各一半")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.per_cell % 2 == 0, "per_cell 必须是偶数，格内 A/B 才能各一半"

    rl = pd.read_parquet(args.rl_parquet)
    q = pd.read_parquet(args.questions, columns=["question_id"] + STRATA + ["answer"])
    q = q.rename(columns={"question_id": "id", "answer": "answer_q"})
    df = rl.merge(q.drop(columns=["bin"]), on="id", how="inner", validate="1:1")
    assert len(df) == len(rl) == 800, (len(df), len(rl))
    assert (df["answer"] == df["answer_q"]).all(), "两份 parquet 的答案对不上"
    assert (df["category"] == df["split"]).all()

    rng = np.random.RandomState(args.seed)
    picked = []
    for key, cell in df.sort_values("id").groupby(STRATA, sort=True):
        for ans in ("A", "B"):
            pool = cell[cell["answer"] == ans]
            k = args.per_cell // 2
            assert len(pool) >= k, (key, ans, len(pool))
            picked.append(pool.iloc[rng.permutation(len(pool))[:k]])
    sub = pd.concat(picked).sort_values("id").drop(columns=["answer_q", "split"])

    missing = [i for i in sub["id"] if not (Path(args.depth_cache) / f"{i}.npy").exists()]
    assert not missing, f"缺深度缓存：{missing[:3]}"

    out = args.out
    sub.to_parquet(out, index=False)
    print(f"写好 {out}：{len(sub)} 题")
    for col in ["category", "domain", "quad", "bin", "answer", "source"]:
        full = df[col].value_counts(normalize=True).sort_index().round(3).to_dict()
        part = sub[col].value_counts(normalize=True).sort_index().round(3).to_dict()
        print(f"  {col:8s} 子集 {part}\n  {'':8s} 全集 {full}")


if __name__ == "__main__":
    main()
