# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""从 engine 的 qa.jsonl 抽一张 `image_key -> (intrinsics, image_wh)` 的侧表。

为什么需要
----------
RL rollout 走深度分支时会建一个 ToolSession，模型靠它读点深度、算 3D 距离。
ToolSession 要相机内参。拿不到就退回到「主点居中、f = max(W,H)」的近似
（`inference_depth_cot.py::_make_tool_session`）—— ScanNet++ 上真值 f 是 1426.9，
近似给 1920，**差 35%**。而题目的标准答案是按真内参从 GT 几何算出来的。

结果是模型读到的距离和它该答的数系统性不一致。这种错不报异常，
只会让深度分支学不出东西，然后被误读成「深度没用」。

为什么不直接重造 parquet
------------------------
`engine_jsonl_to_rl_parquet.py` 当初没带这两个字段。补进去要重跑转换 + 重跑
`build_rl_trainset.py`，而后者的抽样一动，选中的题就可能变，正在跑的深度缓存
（按 image_key 存）就白跑了。侧表按 image_key 关联，两边都不用动。

一张图一条记录，18,598 张图的表只有几 MB。

用法::

    python -m train.rl.data_pipeline.build_camera_table \\
        --jsonl /path/to/scannetpp/qa.jsonl \\
                /path/to/nuscenes/qa.jsonl \\
        --out /path/to/camera_table.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def image_key_of(image_path):
    """和 `engine_jsonl_to_rl_parquet.py` 里的算法必须一致 —— 它是两边的连接键。"""
    return hashlib.sha1(str(image_path).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    table, conflicts = {}, 0
    for path in args.jsonl:
        n = 0
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            key = image_key_of(d["image_path"])
            entry = {"intrinsics": d["intrinsics_scaled"], "image_wh": d["image_wh"]}
            if key in table and table[key] != entry:
                # 同一张图两条不同的内参 = 上游出了问题，不能悄悄取其中一个。
                conflicts += 1
            table[key] = entry
            n += 1
        print(f"  {n:>7} 行  {path}")

    if conflicts:
        raise SystemExit(f"{conflicts} 个 image_key 对应多套内参，先查上游")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table))
    print(f"  {len(table)} 张图 -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
