# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""造一个能**证伪** illusion 代理的对照集。

为什么原来的 validate.parquet 不够用
------------------------------------
`build_rl_trainset.py` 默认 `--illusion-only`，所以它切出来的 validate.parquet
是 300 题 **全是 illusion=True**。拿它跑 measure_branch_gap 只能得到一个数
（这批题上深度帮了多少），没有对照组，illusion 到底是不是有效代理根本无从判断 ——
如果模型在 illusion=False 的题上深度也帮同样多，那 illusion 就是个没用的筛子，
我们等于白扔了池子里 75% 的题。

对照怎么设计
------------
天真的做法是从池子里随便抽 300 道 illusion=False 当对照。不行：illusion 的取值
和场景难度是相关的（2D 线索会指错的场景，往往物体更挤、遮挡更多），两组差异会被
场景难度污染，分不清是 illusion 起作用还是场景难度起作用。

这里用**同图配对**：负样本只从 validate 已有的那 238 张图里取。同一个场景、同一批
物体、同一个相机位姿，唯一变的是这道题的 2D 线索指不指错。场景难度被配对消掉了。

顺带一个实际好处：图集不变，深度缓存还是那 238 张，这一步不额外花生成成本。

产出
----
`proxy_check.parquet`，600 题 = 300 正（illusion=True，原样搬 validate.parquet）
+ 300 负（illusion=False，同图，category 分布对齐正样本）。

下一步::

    torchrun --nproc_per_node=8 /path/to/measure_branch_gap.py \\
        -n 16 --questions /path/to/proxy_check.parquet \\
        --depth-cache /path/to/depth_cache_validate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--positives", required=True)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--train", required=True,
                    help="用来确认负样本没有偷偷落进训练图集")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pos = pd.read_parquet(args.positives)
    assert bool(pos["illusion"].all()), "正样本臂应该全是 illusion=True"
    keys = set(pos["image_key"])
    train_keys = set(pd.read_parquet(args.train, columns=["image_key"])["image_key"])
    assert not (keys & train_keys), "验证图集和训练图集重叠了，build_rl_trainset 的切分有问题"

    pool = pd.concat([pd.read_parquet(p) for p in args.pool], ignore_index=True)
    neg_pool = pool[(~pool["illusion"]) & (pool["image_key"].isin(keys))]
    print(f"正样本 {len(pos)} 题 / {len(keys)} 图；同图负样本池 {len(neg_pool)} 题")

    # category 分布对齐正样本：两组的题型配比一样，免得「深度有没有用」的差异
    # 其实是题型差异（比如 closest_among 天生比 closer_to 难）。
    want = pos["category"].value_counts().to_dict()
    picked = []
    for cat, k in want.items():
        sub = neg_pool[neg_pool["category"] == cat]
        # 同一张图最多出一道负样本，和正样本一样保持场景多样性
        sub = sub.iloc[rng.permutation(len(sub))].groupby("image_key", sort=False).head(1)
        got = sub.head(k)
        if len(got) < k:
            print(f"  ! {cat}: 要 {k}，只有 {len(got)}")
        picked.append(got)

    neg = pd.concat(picked).reset_index(drop=True)
    # 缺口用其余类别补满，保证两臂题量相等（配对检验不要求等量，但等量最省事）
    short = len(pos) - len(neg)
    if short > 0:
        rest = neg_pool[~neg_pool["id"].isin(set(neg["id"]))]
        rest = rest.iloc[rng.permutation(len(rest))].head(short)
        print(f"  补 {len(rest)} 题（不限 category）填平 {short} 的缺口")
        neg = pd.concat([neg, rest]).reset_index(drop=True)

    out = pd.concat([pos, neg[pos.columns]], ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    assert out["id"].is_unique
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    print(f"\n{len(out)} 题 / {out['image_key'].nunique()} 图 -> {args.out}")
    print(pd.crosstab(out["category"], out["illusion"]))
    print(pd.crosstab(out["source"], out["illusion"]))


if __name__ == "__main__":
    main()
