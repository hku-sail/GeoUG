# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""挑 100 道**已知会产生梯度**的题，用来跑通 GRPO 训练环。

上一版 debug 集是随机挑的，结果冒烟时 3 道题 x G=8 共 24 条 rollout 全对：
组内 reward 方差为 0，GRPO 的组归一化优势恒为 0，一点梯度都没有。
那种情况下训练环通没通根本看不出来 —— loss 是 0，指标不动，和死循环长得一样。

现在 proxy_check 那 600 题的分支落差已经实测过了（每分支 8 次采样），
可以直接按实测值挑，两类各占一半：

    可学题   |Δ| > 0 且 |Δ| >= 2·se   —— 门控选对选错会改变 reward，
             保证组内有方差、advantage 非零、梯度真的流起来
    双对题   两条分支都稳对          —— 只有成本项在起作用，
             用来验证 λ_cost 那一路（模型该学会「这种题别生成深度」）

只有可学题的话，门控会学成「永远生成深度」也能拿满分，成本项验不到；
只有双对题就退回原来 advantage 恒 0 的老问题。所以必须混。

这 238 张图的深度缓存是现成的（`depth_cache_validate`），不用再生成。

用法::

    python -m train.rl.data_pipeline.build_debug_subset_v2 \
        --gap /path/to/branch-gap.json \
        --questions /path/to/proxy-check.parquet \
        --camera-table /path/to/camera-table.json \
        --out-dir /path/to/rl-debug
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--gap", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--camera-table", required=True)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--learnable-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    gap = pd.DataFrame(json.loads(Path(args.gap).read_text()))
    gap["p_depth"] = gap["force_depth"].map(lambda d: d["p"])
    gap["p_answer"] = gap["force_answer"].map(lambda d: d["p"])
    gap = gap[["id", "p_depth", "p_answer", "delta", "se"]]

    q = pd.read_parquet(args.questions)
    df = q.merge(gap, on="id", how="inner")
    print(f"{len(df)} 题有实测落差")

    se = df["se"].fillna(0.0)
    learnable = (df["delta"].abs() > 1e-9) & (df["delta"].abs() >= 2 * se)
    both_right = (df["p_depth"] >= 0.99) & (df["p_answer"] >= 0.99)
    print(f"  可学题 {int(learnable.sum())}   双对题 {int(both_right.sum())}")

    n_learn = int(round(args.n * args.learnable_frac))
    # 可学题里挑 |Δ| 最大的：信号最强，训练环一旦通了立刻看得出 reward 分化。
    take_l = df[learnable].reindex(
        df[learnable]["delta"].abs().sort_values(ascending=False).index).head(n_learn)
    pool_b = df[both_right & ~learnable]
    take_b = pool_b.iloc[rng.permutation(len(pool_b))].head(args.n - len(take_l))
    sel = pd.concat([take_l, take_b])
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in q.columns]           # 落盘的列和题库完全一致，下游不用改
    # `_build_single_dataloader` 固定要 train + test 两个 split，test 给同一批题
    # （debug 阶段不看泛化，只看训练环通不通）。
    sel[cols].to_parquet(out / "train.parquet", index=False)
    sel[cols].head(16).to_parquet(out / "validate.parquet", index=False)
    (out / "camera_table.json").write_text(Path(args.camera_table).read_text())
    # 把实测落差也存一份，debug 时可以拿来核对 reward 对不对得上
    sel[["id", "p_depth", "p_answer", "delta", "se"]].to_json(
        out / "measured_gap.json", orient="records", indent=2)

    print(f"\n{len(sel)} 题 -> {out}")
    print(f"  可学 {len(take_l)}（|Δ| 中位 {take_l['delta'].abs().median():.2f}）"
          f"  双对 {len(take_b)}")
    print(f"  类别 {sel['category'].value_counts().to_dict()}")
    print(f"  预期 never-depth {sel['p_answer'].mean():.3f}"
          f"  always-depth {sel['p_depth'].mean():.3f}"
          f"  oracle {np.maximum(sel['p_depth'], sel['p_answer']).mean():.3f}")


if __name__ == "__main__":
    main()
