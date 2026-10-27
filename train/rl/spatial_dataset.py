# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""RL 训练集的 Dataset：读 `rl_train_v1/{split}.parquet`，产 (题面, 元数据, 图)。

对接的是 GeoUG RL engine 的 `ImageTextDataset` 那套约定 ——
`collate_fn` 返回 `(prompts, metadatas, images)`，`_build_single_dataset` 用
`(dataset_path, split=...)` 构造。

为什么保留 arrow table 而不转 pandas
------------------------------------
`train.parquet` 有 1.7 GB，绝大部分是内嵌的 JPEG 字节。`pd.read_parquet` 会把
image 列变成一列 Python `bytes` 对象；dataloader fork 出 worker 之后，
每次访问都会动这些对象的引用计数，把写时复制的页一页页复制下来 ——
4 个 worker x 8 张卡，内存会涨到不可接受。

arrow 的 buffer 是不可变的 C 内存，fork 之后真正共享，取某一行才
materialize 那一行。所以整条路上不碰 pandas。

内参从哪来
----------
parquet 里没有内参（转换脚本当初漏了）。ToolSession 拿不到内参会退回
「f = max(W,H)」的近似，ScanNet++ 上和真值差 35%，而标签是按真内参算的。
所以旁边放一张 `camera_table.json`（`data_pipeline/build_camera_table.py` 生成），
按 image_key 关联。缺一张图就抛 —— 静默用近似正是要避免的那种错。
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

_CAMERA_TABLE = "camera_table.json"

# `_build_single_dataloader` 固定要 train 和 test 两个 split，而我们的目录里
# 切出来的那份叫 validate（`build_rl_trainset.py` 的命名）。这里做个映射，
# 而不是在磁盘上再复制一份。
_SPLIT_ALIAS = {"test": "validate"}


class SpatialGateDataset(Dataset):
    def __init__(self, dataset, split="train"):
        root = Path(dataset)
        path = root / f"{_SPLIT_ALIAS.get(split, split)}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在（split={split!r}）")

        self.table = pq.read_table(path)
        self.cameras = json.loads((root / _CAMERA_TABLE).read_text())

        cols = set(self.table.column_names)
        missing = {"id", "image", "question", "answer", "options", "image_key"} - cols
        if missing:
            raise ValueError(f"{path} 缺列 {sorted(missing)}")

        keys = set(self.table["image_key"].to_pylist())
        absent = keys - set(self.cameras)
        if absent:
            raise ValueError(
                f"{len(absent)} 张图不在 {root / _CAMERA_TABLE} 里（例：{sorted(absent)[:3]}）。"
                f"重跑 data_pipeline/build_camera_table.py，源要覆盖这个 split 的所有源。")

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        row = {k: self.table[k][idx].as_py()
               for k in ("id", "image", "question", "answer", "options",
                         "category", "source", "image_key")}
        cam = self.cameras[row["image_key"]]
        return {
            "prompt": row["question"],
            "image": Image.open(io.BytesIO(row["image"])),
            "metadata": {
                "id": row["id"],
                "answer": row["answer"],
                "options": row["options"],
                "category": row["category"],
                "source": row["source"],
                "image_key": row["image_key"],
                "intrinsics": cam["intrinsics"],
                "image_wh": cam["image_wh"],
            },
        }

    @staticmethod
    def collate_fn(examples):
        return ([e["prompt"] for e in examples],
                [e["metadata"] for e in examples],
                [e["image"] for e in examples])
