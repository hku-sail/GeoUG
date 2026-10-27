# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""把各源的题池混成最终的 RL 训练集。

筛选用 `illusion`，不用多次采样测 Δ —— 为什么
----------------------------------------------
原计划是每题两条分支各采样 16 次，按 `|Δ| >= 2·se` 收。算一下成本就知道不行：
一次纯文本 rollout 约 18 秒（D3 的 `force_answer` 800 题 4 卡跑了约 1 小时反推）。

    Δ 筛选 85,944 道比较题：85944 x 32 x 18s = **13,750 GPU 小时**
    一个 RL epoch（G=8）：  85944 x  8 x 18s = **3,438 GPU 小时**

**筛选比它想省的那个 epoch 贵 4 倍。** 而且 GRPO 组内 G 条 rollout 奖励全相同时，
组归一化的优势恒为 0、不产生梯度 —— 无信息的题它自己就丢了，筛选唯一的收益是省下
那几条 rollout，但花的比省的多。

engine 免费给了个等价物：`illusion` 是确定性算出来的「只看 2D 线索会不会答错」
（`templates.py` 各 `gen_*` 的 Step 6，拿 bbox 面积 / 2D 中心距当代理），正是
`depth_only` 象限的免费代理。

验过了：**这个代理只在物-物题上成立**
--------------------------------------
illusion 说的是「2D 捷径会错」，我们要的是「模型不看深度会错」。两者不是一回事：
模型的直答分支不是纯 2D 捷径，它还有语义先验（知道桌子一般比杯子大）。
所以在 600 题的同图对照集上验了一次（`probes/analyze_proxy.py`，见 DATA.md §4）：

    整体   AUC 0.498，配对 Wilcoxon p=0.39   —— 作为全局筛子，零价值
    物-物  Δ_True−Δ_False +0.231  AUC 0.684
    物-相机 Δ_True−Δ_False −0.124  AUC 0.418   —— 反号
    交互项 t=5.72, p<1e-7

所以默认改成 `--illusion-scope objobj`：只在 `multi_object_*` 上筛。
而且要认清收益的量级 —— 可学题率 不筛 25.3% / 全筛 28.7% / 只筛物-物 27.5%，
**筛选最多买到 2 个点**。这不是一道值得为它扔掉 73% 池子的工序。
细节见 `_apply_illusion` 的 docstring。

配比
----
- **室内 / 室外各半**：D3 评测就是 50/50（`build_benchmark.py --indoor 0.5`）。
  只训室内的话门控学到的策略未必转得到室外。
- **单图设上限**：同一张图 4.8 道题，不设限的话少数几个场景会主导 batch，
  GRPO 的组间方差被场景相关性污染。
- **类别均衡到最小的那一类**：避免 `location_closer_to_camera`（题量最大）
  把门控往单一题型上带。

用法::

    python -m train.rl.data_pipeline.build_rl_trainset \\
        --inputs /path/to/indoor-train.parquet \\
                 /path/to/outdoor-train.parquet \\
        --indoor-sources scannetpp_train \\
        --illusion-scope objobj --n 10000 --validate-n 300 \\
        --out-dir /path/to/rl-train
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 不能当物体问的名字。命中任一选项就整题扔掉。
#
# 两类：
#   1. **标注工具的产物**。ScanNet++ 的词表里混着 `remove`(1493 实例) / `split`(1028)，
#      合计 3%。「the remove 和 the split 哪个更近」是无意义的题，但它一路不报错，
#      会安静地进训练集当噪声。
#   2. **结构面**。墙、地、天花板铺满画面、深度跨度极大，"哪个更近" 没有良定义的答案。
#      adapter 里已经按整串精确匹配扔过一轮，这里补词边界子串匹配 —— 前者漏掉了
#      `shower wall` / `door frame` 这种带修饰词的。
_BAD_NAME = re.compile(
    r"(?<![a-z])("
    r"remove|split|objects?|unknown|other|misc|none|null|item|thing|stuff"
    r"|wall|floor|ceiling|door|doorframe|window|windowsill|panel|pipe|cables?"
    r")(?![a-z])"
)


def option_names(question):
    """题面里 `A. xxx` 那几行的物体名。engine 的 `render_user` 保证是这个格式。"""
    out = []
    for line in question.split("\n")[1:]:
        if len(line) > 3 and line[1] == "." and line[0].isalpha():
            # 去掉同类重名的编号后缀（"chair 2" -> "chair"）
            out.append(re.sub(r"\s+\d+$", "", line[2:].strip().lower()))
    return out


def take_balanced(df, n, by, rng):
    """从 `df` 里按 `by` 这些列均衡地取 n 条：各组等额，取不满的组由其他组补。"""
    if n <= 0 or df.empty:
        return df.iloc[:0]
    groups = [g for _, g in df.groupby(by, sort=False)]
    rng.shuffle(groups)
    out, quota = [], n
    # 多轮分配：每轮给每组一个等额名额，组取空了就退出，剩余名额自然流向还有货的组。
    while quota > 0 and groups:
        per = max(1, quota // len(groups))
        nxt = []
        for g in groups:
            k = min(per, len(g), quota)
            if k <= 0:
                continue
            out.append(g.iloc[:k])
            quota -= k
            if len(g) > k:
                nxt.append(g.iloc[k:])
        if not nxt:
            break
        groups = nxt
    return pd.concat(out) if out else df.iloc[:0]


def _apply_illusion(df, scope):
    """按 `scope` 施加 illusion 筛选。

    默认 `objobj` 是**实测出来的**，不是拍脑袋（600 题同图对照集，每分支采样 8 次，
    见 `probes/analyze_proxy.py` 和 DATA.md §4）：

        整体          illusion 预测「深度会帮忙」的 AUC = 0.498，配对 Wilcoxon p=0.39
                      —— 干净的零结果，illusion 作为全局筛子毫无价值

        物-物  multi_object_*        Δ_True−Δ_False = +0.231   AUC 0.684
        物-相机 camera/location       Δ_True−Δ_False = −0.124   AUC 0.418
        交互项 t = 5.72, p < 1e-7    —— 两个方向相反的效应，整体的零是它们抵消出来的

    为什么会反号：`camera_closest_among` / `location_closer_to_camera` 问的是
    「哪个离相机近」，而 illusion 的判据是 bbox 面积。模型压根不靠 bbox 面积答这类题 ——
    它有语义尺寸先验（知道桌子比杯子大）加透视线索（在画面里的高度、遮挡、触地点）。
    所以 illusion=True 在这类题上只意味着「物体尺寸反常」，那对**深度分支**同样是干扰。
    物-物题问的是物体之间的远近，没有相机中心的捷径可用，2D 中心距确实就是模型手上
    有的那点信息，illusion 这时才真的对上。

    要注意收益本来就不大：可学题率 不筛 25.3% / 全筛 28.7% / 只筛物-物 27.5%。
    **筛选最多买到 2 个点**，所以不值得为它扔掉 73% 的池子 —— 物-物之外一律不筛。
    物-物内部值得筛是因为那里差距大（illusion=False 只有 7.8-20.5% 可学）。
    """
    if scope == "none":
        return df
    keep = df["illusion"] if scope == "all" else (
        df["illusion"] | ~df["category"].str.startswith("multi_object"))
    out = df[keep].reset_index(drop=True)
    print(f"  illusion 筛（scope={scope}）{len(df)} -> {len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--indoor-sources", nargs="+", default=["scannetpp_train"],
                    help="这些 source 算室内，其余算室外")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=10000, help="训练集题量")
    ap.add_argument("--validate-n", type=int, default=300,
                    help="切出来的留出子集（按图切，和训练集不重叠）")
    # proxy_check 那 238 张图已经量过分支落差、深度也缓存好了。把它们挡在训练集外面，
    # 训练中途想看门控学没学到东西，随时可以在同一批题上重跑 measure_branch_gap 对比。
    ap.add_argument("--exclude-images", nargs="*",
                    default=[],
                    help="这些 parquet 里出现过的 image_key 一律不进训练/留出集")
    ap.add_argument("--indoor-frac", type=float, default=0.5,
                    help="室内占比。D3 评测是 50/50")
    ap.add_argument("--per-image-cap", type=int, default=2)
    ap.add_argument("--illusion-scope", choices=("objobj", "all", "none"), default="objobj",
                    help="illusion 筛在哪些题型上生效。见下面 _apply_illusion 的实测依据")
    ap.add_argument("--keep-bad-names", action="store_true",
                    help="保留含标注产物 / 结构面名字的题（默认扔掉，见 _BAD_NAME）")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.concat([pd.read_parquet(p) for p in args.inputs], ignore_index=True)
    print(f"读入 {len(df)} 题，来源 {sorted(df['source'].unique())}")

    if args.exclude_images:
        drop = set()
        for p in args.exclude_images:
            drop |= set(pd.read_parquet(p, columns=["image_key"])["image_key"])
        n0 = len(df)
        df = df[~df["image_key"].isin(drop)].reset_index(drop=True)
        print(f"  排除 {len(drop)} 张已测图 {n0} -> {len(df)}")

    df = _apply_illusion(df, args.illusion_scope)

    if not args.keep_bad_names:
        ok = ~df["question"].map(
            lambda q: any(_BAD_NAME.search(n) for n in option_names(q)))
        print(f"  名字过滤扔掉 {int((~ok).sum())}（标注产物 + 结构面），剩 {int(ok.sum())}")
        df = df[ok].reset_index(drop=True)

    # 单图限额。先按 illusion_strength 降序，这样留下的是 2D 线索指错得最狠的那几道 ——
    # 同一张图里最能区分「看不看深度」的题。
    if "illusion_strength" in df.columns:
        # 按**类别内分位数**排，不按原始值。illusion_strength 是各 gen_* 各算各的
        # （物-物用 2D 中心距之比，物-相机用 bbox 面积之比），跨类别不可比。
        # 直接全局排序的后果实测过：物-相机题的原始值系统性更大，于是每张图的两个
        # 名额几乎都被它们赢走，物-物题从 8,018 掉到 1,636（损失 80%），
        # 下游类别均衡再怎么配也配不出来。
        df = df.assign(
            _rank=df.groupby("category")["illusion_strength"].rank(pct=True)
        ).sort_values("_rank", ascending=False).drop(columns="_rank")
    df = df.groupby("image_key", sort=False).head(args.per_image_cap).reset_index(drop=True)
    print(f"  单图限 {args.per_image_cap} 题后 {len(df)}（{df['image_key'].nunique()} 张图）")

    df["indoor"] = df["source"].isin(args.indoor_sources)
    n_in = int(round((args.n + args.validate_n) * args.indoor_frac))
    n_out = (args.n + args.validate_n) - n_in

    picked = []
    for flag, want in ((True, n_in), (False, n_out)):
        sub = df[df["indoor"] == flag]
        got = take_balanced(sub.sample(frac=1.0, random_state=args.seed), want,
                            ["category"], rng)
        picked.append(got)
        tag = "室内" if flag else "室外"
        print(f"  {tag}: 要 {want}，池子 {len(sub)}，取到 {len(got)}")
    sel = pd.concat(picked).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 验证子集按 image_key 切，不按行 —— 同一张图的题落在两边会让「验证」偷看训练集。
    keys = sel["image_key"].drop_duplicates().tolist()
    rng.shuffle(keys)
    val_keys, n_val = set(), 0
    for k in keys:
        if n_val >= args.validate_n:
            break
        c = int((sel["image_key"] == k).sum())
        val_keys.add(k)
        n_val += c
    val = sel[sel["image_key"].isin(val_keys)].reset_index(drop=True)
    train = sel[~sel["image_key"].isin(val_keys)].head(args.n).reset_index(drop=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.drop(columns=["indoor"]).to_parquet(out / "train.parquet", index=False)
    val.drop(columns=["indoor"]).to_parquet(out / "validate.parquet", index=False)

    for name, d in (("train", train), ("validate", val)):
        print(f"\n{name}: {len(d)} 题，{d['image_key'].nunique()} 张图")
        print(f"  室内 {100 * d['source'].isin(args.indoor_sources).mean():.0f}%")
        for k, v in d["category"].value_counts().items():
            print(f"    {k}: {v}")
    print(f"\n-> {out}/train.parquet, {out}/validate.parquet")
    print("\n下一步：\n"
          f"  1) 建深度缓存（按 image_key 去重，实测序翻转 ~1%，安全）\n"
          f"  2) 在 validate.parquet 上跑 measure_branch_gap.py -n 16，"
          f"验 illusion 和真实分支落差的相关性")


if __name__ == "__main__":
    main()
