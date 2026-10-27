# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Prepare DA-2K textual questions from upstream images and annotations.

Annotations use [row, col] pixels. Output points use [x, y], independently
normalized to 0-1000 on each axis. Images remain unmarked. A fixed random seed
randomizes the base option order; this does not guarantee exact A/B balance.
The output parquet references local images; pack_data_parquet.py can embed them.
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image




# 与 D3 / v3 训练一致：题干里是 "the <名字> (represented by <point>x y</point>)"，
# 选项行是裸名字。两个点没有语义名字，用中性的序数名，且避开 "point A/B" —— 那会和
# 选项字母撞车，轮换后出现 "A. point B" 这种自相矛盾的行。
_NAME_1, _NAME_2 = "first point", "second point"


def _norm1000(pt_rowcol, width, height):
    """``[row, col]`` 绝对像素 -> 每轴归一到 0~1000 的 ``(x, y)``。"""
    row, col = float(pt_rowcol[0]), float(pt_rowcol[1])
    x = int(round(min(max(col / max(width, 1), 0.0), 1.0) * 1000))
    y = int(round(min(max(row / max(height, 1), 0.0), 1.0) * 1000))
    return [x, y]


def _mention(name, pt):
    return f"the {name} (represented by <point>{pt[0]} {pt[1]}</point>)"


def build_rows(root: Path, seed: int):
    ann = json.loads((root / "annotations.json").read_text())
    rng = random.Random(seed)
    rows, oob = [], 0

    for rel in sorted(ann):
        img_path = root / rel
        with Image.open(img_path) as im:          # 只读文件头拿尺寸，不解码像素
            W, H = im.size
        real = os.path.realpath(img_path)         # 存 realpath：torchrun worker 的挂载
                                                  # 命名空间可能看不见 /opt/tiger 软链
        for k, pair in enumerate(ann[rel]):
            p_close, p_far = pair["point1"], pair["point2"]
            assert pair["closer_point"] == "point1", f"{rel}#{k} 的 closer_point 不是 point1"
            for p in (p_close, p_far):
                if not (0 <= p[0] < H and 0 <= p[1] < W):
                    oob += 1

            # 固定种子打乱：否则答案恒为 A
            close_first = rng.random() < 0.5
            pt_a_src, pt_b_src = ((p_close, p_far) if close_first else (p_far, p_close))
            answer = "A" if close_first else "B"

            pt_a = _norm1000(pt_a_src, W, H)
            pt_b = _norm1000(pt_b_src, W, H)
            a, b = _mention(_NAME_1, pt_a), _mention(_NAME_2, pt_b)

            rows.append(dict(
                question_id=f"{rel}#{k}",
                # native 口径：沿用 DA-2K 自己的"which point"问法，与 v3 的"which object"
                # 只差一个词，两者之差就是格式漂移本身（与 D3 的 --prompt_style 同一用法）。
                question=(f"Which point is closer to the camera taking this photo, {a} or {b}?"
                          f"\nA. {_NAME_1}\nB. {_NAME_2}"),
                answer=answer,
                option_a=_NAME_1, option_b=_NAME_2,
                pt_a=pt_a, pt_b=pt_b,
                anchor=None, pt_anchor=None,      # 只有 D3 的 distance split 用得上
                split="depth",                    # 走 _CAM_Q_CLOSER 模板
                bin=rel.split("/")[1],            # DA-2K 的 8 个类别 -> 报表的 category 列
                source="DA-2K",
                domain=rel.split("/")[1],
                image=real,
                image_w=W, image_h=H,
                grid_w=1000, grid_h=1000,         # 点已归一，下游换算是恒等的
                px_close=p_close, px_far=p_far,   # 留原始像素坐标便于复查 / 可视化
            ))
    return rows, oob


def main():
    p = argparse.ArgumentParser("把 DA-2K 做成 textual 版（坐标写进题干，图片不画标记）")
    p.add_argument("--root", required=True, help="Directory containing annotations.json and images/")
    p.add_argument("--out_dir", default="outputs/da2k_textual")
    p.add_argument("--seed", type=int, default=0, help="打乱 A/B 基础顺序的种子")
    args = p.parse_args()

    rows, oob = build_rows(Path(args.root), args.seed)
    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "questions.parquet", index=False)

    print(f"写出 {len(df)} 条 -> {out / 'questions.parquet'}")
    print(f"  图片数      : {df['image'].nunique()}")
    print(f"  答案分布    : {dict(Counter(df['answer']))}")
    print(f"  类别分布    : {dict(Counter(df['bin']))}")
    print(f"  越界点      : {oob}")
    missing = [q for q in df['image'].unique() if not os.path.exists(q)]
    print(f"  读不到的图  : {len(missing)}")


if __name__ == "__main__":
    main()
