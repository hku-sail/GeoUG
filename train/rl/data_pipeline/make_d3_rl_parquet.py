# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""把 D3 的 800 题转成 RL 脚本认的格式，**prompt 取自评测记录而不是题库**。

为什么不直接读 questions.parquet
--------------------------------
题库的 `question` 列和评测实际喂进模型的 prompt **不是一个东西**：

    parquet : "Consider the relationship in the 3D world. Which is closer to the
               camera?\\n(A) the bus (represented by the point <point>571 578</point>)..."
    评测喂的: "Which object is closer to the camera taking this photo, the bus
               (represented by <point>571 578</point>) or the parked car ...?\\nA. bus\\nB. parked car"

评测脚本按 `--prompt_style` 重写过。拿 parquet 那版去跑，模型见到的是没在 SFT 里
见过的格式 —— 这正是 V21 踩过的坑：`build_rl_debug_set.py` 当时用了不含选项列表的
`question`，深度分支解析率掉到 50%，而且在报表上伪装成「答错」。

所以这里的 prompt 一律从 `predictions_full.jsonl` 的 `prompt` 字段取，
那是逐字节喂进去的原文，没有任何重写余地。

深度缓存
--------
不用重新生成：`force_depth` 那一跑开了 `--save_depth`，
`transcripts/<qid>/depth_0.npy` 就是 float32 米制深度，和
`build_depth_cache.py` 的产物同格式。软链过去即可（省 4 卡 6 小时）::

    SRC=<eval_dir>/force_depth/transcripts
    DST=/path/to/depth_cache_d3_800
    mkdir -p $DST
    for d in $SRC/*/; do ln -sf "$(realpath $d/depth_0.npy)" "$DST/$(basename $d).npy"; done

用法::

    python -m train.rl.data_pipeline.make_d3_rl_parquet \\
        --eval-dir /path/to/evaluation-output \\
        --questions /path/to/3dcuebench.parquet \\
        --out /path/to/d3_800_rl.parquet
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_OPT_LINE = re.compile(r"^([A-D])\.\s*(.+)$", re.M)


def main():
    ap = argparse.ArgumentParser("D3 800 题 -> RL 格式 parquet")
    ap.add_argument("--eval-dir", required=True,
                    help="含 force_depth/predictions_full.jsonl 的评测目录")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = {r["question_id"]: r for r in (
        json.loads(l) for l in
        open(Path(args.eval_dir) / "force_depth" / "predictions_full.jsonl"))}
    q = pd.read_parquet(args.questions).set_index("question_id")

    rows, n_noopt = [], 0
    for qid, rec in preds.items():
        if qid not in q.index:
            continue
        src = q.loc[qid]
        prompt = rec["prompt"]
        # 选项从 prompt 末尾的 "A. xxx" 行里解析 —— 和模型看到的是同一份文本，
        # 不从 option_a / option_b 推导（那两列带 "the " 前缀，对不上）。
        opts = {m.group(1): m.group(2).strip() for m in _OPT_LINE.finditer(prompt)}
        if not opts:
            n_noopt += 1
        img = src["image"]
        rows.append({
            "id": qid,
            "image": img["bytes"] if isinstance(img, dict) else img,
            "question": prompt,
            "answer": str(src["answer"]),
            "options": json.dumps(opts, ensure_ascii=False),
            "category": str(src["split"]),       # depth / distance
            "source": str(src["source"]),
            "image_id": str(src["image_id"]),
            "bin": str(src["bin"]),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(f"写出 {len(rows)} 题 -> {out}")
    if n_noopt:
        print(f"  !! {n_noopt} 题没解析出选项行，verifier 会退回文本匹配")
    print("  category:", dict(pd.Series([r["category"] for r in rows]).value_counts()))
    print("  answer  :", dict(pd.Series([r["answer"] for r in rows]).value_counts()))


if __name__ == "__main__":
    main()
