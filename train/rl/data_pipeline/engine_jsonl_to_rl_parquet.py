# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""data engine 的 JSONL -> RL 训练用的轻量 parquet。

为什么不走 build_v5_parquet.py
------------------------------
那条路会产出 **913 GB**（v5 的 483,570 行，coco 304 GB + ade20k 610 GB），因为每行内嵌
一张深度图 PNG —— SFT 要训生成器，深度图是监督目标，必须带。

RL 不训生成器。深度走我们自己预生成的缓存（`build_depth_cache.py`），parquet 里只需要
出题必需的五列。10K 题约 5 GB，差两个量级。

同时这里**不做** v5 的文本手术（`<DECIDE>` 改写、双分支切分）。那些是 SFT 的监督信号，
RL 时轨迹由模型自己采样出来，我们只提供题面和答案。

schema 对齐 `build_rl_debug_set.py:138-143`，这样 `measure_branch_gap.py` /
`smoke_rollout.py` / `build_depth_cache.py` 不用改：

    id / image(bytes) / question / answer / options(JSON str) / category

另外多带几列做筛选和归因用：`illusion`（2D 捷径会不会答错）、`source`、`image_key`
（同图的题共享 —— 深度缓存按它去重，7.5 题/图意味着能省 85% 的扩散调用）。

用法::

    python -m train.rl.data_pipeline.engine_jsonl_to_rl_parquet \\
        --jsonl /path/to/data-engine/qa.jsonl \\
        --source scannetpp_train \\
        --out /path/to/rl_scannetpp.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# 只保留比较类题目。
#
# metric_depth_value / _open 是「估个米数」，答案是连续量、判分靠数值容差，
# 和门控要学的「深度有没有用」是两件事：这类题**必然**需要深度（engine 自己把它们
# 的 illusion 标成 100%），放进 RL 只会把触发率推向 1，学不出取舍。
# 需要时用 --categories all 放开。
COMPARISON = {
    "location_closer_to_camera", "camera_closest_among",
    "multi_object_closer_to", "multi_object_closest_among",
}


def main():
    ap = argparse.ArgumentParser("engine JSONL -> RL parquet")
    ap.add_argument("--jsonl", required=True, nargs="+")
    ap.add_argument("--source", required=True, help="写进 source 列，便于混源后归因")
    ap.add_argument("--out", required=True)
    ap.add_argument("--categories", default="comparison",
                    choices=["comparison", "all"])
    ap.add_argument("--max-side", type=int, default=0,
                    help=">0 时把长边缩到这个值再存，省体积；0 保留原图字节")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    keep_cat = None if args.categories == "all" else COMPARISON
    rows, img_cache, n_skip = [], {}, 0

    for jp in args.jsonl:
        for line in tqdm(open(jp), desc=Path(jp).name):
            r = json.loads(line)
            if keep_cat and r.get("category") not in keep_cat:
                continue
            opts = {k: v for k, v in (r.get("options") or {}).items() if v is not None}
            if not opts:
                n_skip += 1
                continue

            ip = r["image_path"]
            if ip not in img_cache:
                try:
                    with open(ip, "rb") as f:
                        raw = f.read()
                    if args.max_side:
                        im = Image.open(io.BytesIO(raw)).convert("RGB")
                        if max(im.size) > args.max_side:
                            s = args.max_side / max(im.size)
                            im = im.resize((round(im.width * s), round(im.height * s)),
                                           Image.LANCZOS)
                            buf = io.BytesIO()
                            im.save(buf, format="JPEG", quality=95)
                            raw = buf.getvalue()
                except Exception:                           # noqa: BLE001
                    img_cache[ip] = None
                    raw = None
                if raw is not None:
                    img_cache[ip] = raw
            img = img_cache[ip]
            if img is None:
                n_skip += 1
                continue

            # image_key 让同一张图的题共享深度缓存。用路径的哈希而不是 image_id ——
            # 后者是 adapter 里 hash() 取模出来的，跨源可能撞。
            ikey = hashlib.sha1(ip.encode()).hexdigest()[:16]
            qid = f"{args.source}_{ikey}_{len(rows):06d}"
            ill = r.get("illusion") or {}
            rows.append({
                "id": qid,
                "image": img,
                "question": r["user_text"],
                "answer": str(r["answer"]),
                "options": json.dumps(opts, ensure_ascii=False),
                "category": r.get("category", ""),
                "illusion": bool(ill.get("illusion", False)),
                # 2D 线索指错方向的强度。免费的难度代理：强度越高，纯 2D 越会自信地答错，
                # 深度带来的收益也就越大。用来在 illusion 内部再排序。
                "illusion_strength": float(ill.get("strength", 0.0) or 0.0),
                "source": args.source,
                "image_key": ikey,
            })
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"写出 {len(df)} 题 -> {out}  ({out.stat().st_size / 2**30:.2f} GB)")
    print(f"  跳过 {n_skip} 条（无选项 / 图读不出）")
    print(f"  唯一图 {df['image_key'].nunique()} 张，{len(df) / max(df['image_key'].nunique(), 1):.1f} 题/图")
    print(f"  illusion 占比 {100 * df['illusion'].mean():.1f}%")
    print("  类别分布:")
    for k, v in df["category"].value_counts().items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
