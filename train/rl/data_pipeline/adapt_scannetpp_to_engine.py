# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""把 ScanNet++ train 转成 data engine 认的三件套：COCO JSON + 图像目录 + 深度 npz。

三个源里 ScanNet++ 的资产最全，是唯一不需要补任何东西的：

    图像   data/{scene}/iphone/rgb/{frame}.jpg              1920x1440
    mask   标注 JSON 里的 `mask` 字段，**真 COCO-RLE，全分辨率**
    名字   标注 JSON 里的 `label`，真实语义词表（book / cabinet / trash can ...）
    深度   data/{scene}/iphone/depth/{frame}.png            uint16 毫米，192x256，100% 稠密
    内参   data/{scene}/iphone/pose_intrinsic_imu.json      逐帧

对比另外两个源：CA-1M 的框是**类别无关**的（1493 个实例全叫 "object"，得先补一轮 VLM
起名才能出题）；nuScenes 的 `segmentation` 是空列表、也没有稠密深度。所以先跑这个源。

关于深度的取舍
--------------
ScanNet++ 有两套深度：激光扫描 mesh 渲染的（`rasterout_dir`）和 iPhone LiDAR 的
（`data/{scene}/iphone/depth`）。前者质量高，是 benchmark 侧 `build_scannetpp_tables.py`
用的 —— 但它**只有 val 下载了**，train 的 856 个场景目录全是空的。

所以这里用 iPhone LiDAR。它 192x256、比 RGB 低 7.5 倍，精度也不如 mesh 渲染，但
**无空洞**，室内 1-5 米范围够用。误差能不能翻标签，有两道保险：题目本来就按
relgap >= 0.10 过滤；真翻了的题会在下游多次采样的 Δ 筛选里落进 `both_wrong` 被剔掉。

标签来自 GT 深度而不是 DA3
--------------------------
v5 的 SFT 数据是从 **DA3 预测深度**读出答案的，这里换成真实传感器深度。

理由：D3 评测的标签来自 GT。标签用 DA3 的话，我们是在优化「和 DA3 一致」却拿 GT 评测。
模型生成的深度本来就是 DA3 风格的，它和 GT 的差距正是 `force_depth` 在 D3 上只有 90%
而非 100% 的原因 —— 这个差距该让门控学着绕开，而不是被标签抹平。副作用是新数据上深度
分支的准确率会低于 v5 同域数据，那是真实的。

读**表面**深度而不是 OBB 中心
-----------------------------
标注里每个物体都带 `obb`（激光扫描出来的，亚厘米精度），拿它的 centroid 算距离看着更准。
但 engine 的 `centroid_3d_cam` 是**表面点反投影**出来的（mask 内点 + 深度窗口读数），
v5 的答案也是这个口径。换成 OBB 中心会让「距离」的语义和 SFT 时不一致 —— 对薄而长的
物体（桌子、沙发）两者能差半米。所以宁可用精度低一些但口径一致的表面深度。

npz 的三个字段
--------------
- `depth`：毫米转米，最近邻重采样到 RGB 网格。**不能用双线性** —— 深度边缘插值会造出
  物理上不存在的中间值，那些值恰好骑在物体和背景之间，会污染窗口读数。
- `conf`：engine 的过滤条件是 `conf > conf_thresh`，默认 **1.05**（不是 0-1 口径，
  见 `geometry.py:205`）。有效处写 2.0、无效处写 0.0。
- `intrinsics`：**主点必须写成图像正中**。`geometry.scale_intrinsics_to_depth` 从主点
  反推源分辨率（`Wp = 2*cx`），它的输出 cx 恒等于 W/2 —— engine 的几何从根上假设中心
  主点（DA3 确实如此）。把主点写成中心、焦距写真值，这个函数就退化成恒等变换。代价是
  每个物体横向偏移 `δ·z/fx`（δ 是真实主点偏移，这里约 3 像素），3 米处约 0.6 厘米。

用法::

    python -m train.rl.data_pipeline.adapt_scannetpp_to_engine \\
        --root /path/to/scannetpp --out /path/to/engine_src/scannetpp_train \\
        --workers 32 --max-frames-per-scene 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# engine 侧口径（builder.py:145，geometry.py:205）：过滤条件是 conf > 1.05。
CONF_VALID = 2.0
CONF_HOLE = 0.0

# 不能当物体的标签。这里先扔，省掉 RLE 解码和落盘。
#
# 除了结构面（铺满画面、深度跨度极大，"哪个更近" 没有良定义的答案），
# ScanNet++ 的词表里还混着**标注工具的产物**：`remove` 1493 个实例、`split` 1028 个，
# 合计占 3%。「the remove 和 the split 哪个更近」是无意义的题，而且一路不报错。
#
# 注意这是**整串精确匹配**，抓不到 `shower wall` / `door frame` 这种带修饰词的。
# 那一层由 `build_rl_trainset.py::_BAD_NAME` 的词边界子串匹配兜底。
STRUCTURAL = {
    "wall", "floor", "ceiling", "door", "doorframe", "door frame", "window",
    "window frame", "windowsill", "panel", "pipe", "cables",
    "object", "objects", "remove", "split", "unknown", "other",
}
MIN_MASK_PX = 400          # RGB 网格上；小于这个的物体在 192x256 的深度上不足 7 个像素


def load_frame_intrinsics(scene, data_dir):
    """读整个场景的逐帧内参，返回 {frame_stem: 3x3}。一个场景读一次。"""
    p = data_dir / scene / "iphone" / "pose_intrinsic_imu.json"
    if not p.exists():
        return {}
    try:
        meta = json.loads(p.read_text())
    except Exception:                                       # noqa: BLE001
        return {}
    return {k: np.asarray(v["intrinsic"], dtype=np.float64)
            for k, v in meta.items() if isinstance(v, dict) and "intrinsic" in v}


def process_scene(scene, ann_dir, data_dir, out_dir, max_frames):
    from pycocotools import mask as mask_utils

    ann_files = sorted((ann_dir / scene).glob("*.jpg.json"))[:max_frames]
    if not ann_files:
        return [], []
    Ks = load_frame_intrinsics(scene, data_dir)
    rgb_dir = data_dir / scene / "iphone" / "rgb"
    dep_dir_src = data_dir / scene / "iphone" / "depth"
    img_out, dep_out = out_dir / "images", out_dir / "depth"

    images, anns = [], []
    for af in ann_files:
        frame = af.name[: -len(".jpg.json")]              # frame_000000
        rgb_p, dep_p = rgb_dir / f"{frame}.jpg", dep_dir_src / f"{frame}.png"
        if not rgb_p.exists() or not dep_p.exists() or frame not in Ks:
            continue
        try:
            payload = json.loads(af.read_text())
            with Image.open(rgb_p) as im:
                W, H = im.size
            depth_raw = np.asarray(Image.open(dep_p)).astype(np.float32) / 1000.0
        except Exception:                                   # noqa: BLE001
            continue

        objs = []
        for a in payload.get("annotations", []):
            label = str(a.get("label", "")).strip().lower()
            if not label or label in STRUCTURAL:
                continue
            m = a.get("mask")
            if not isinstance(m, dict) or "counts" not in m:
                continue
            # `size` 在 JSON 里是字符串 "[1440, 1920]"，不是 list。
            size = m["size"]
            if isinstance(size, str):
                size = json.loads(size)
            if list(size) != [H, W]:
                continue
            rle = {"size": [int(size[0]), int(size[1])],
                   "counts": m["counts"].encode("ascii") if isinstance(m["counts"], str)
                             else m["counts"]}
            area = int(mask_utils.area(rle))
            if area < MIN_MASK_PX:
                continue
            x, y, w, h = [float(v) for v in mask_utils.toBbox(rle)]
            objs.append((label, rle, area, [x, y, w, h]))

        if len(objs) < 2:            # 少于两个物体出不了比较题
            continue

        # 深度重采样到 RGB 网格。最近邻 —— 双线性会在深度边缘造出假的中间值。
        if depth_raw.shape != (H, W):
            import cv2
            depth = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            depth = depth_raw

        stem = f"{scene}_{frame}"
        link = img_out / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(rgb_p.resolve())

        K = Ks[frame]
        K_out = np.array([[K[0, 0], 0.0, W / 2.0],
                          [0.0, K[1, 1], H / 2.0],
                          [0.0, 0.0, 1.0]], dtype=np.float64)   # 主点居中，见 docstring
        conf = np.where(depth > 0, CONF_VALID, CONF_HOLE).astype(np.float32)
        np.savez_compressed(dep_out / f"{stem}.npz",
                            depth=depth.astype(np.float32), conf=conf, intrinsics=K_out)

        img_id = abs(hash(stem)) % (2 ** 31)
        images.append({"id": img_id, "file_name": f"{stem}.jpg", "width": W, "height": H})
        for label, rle, area, bbox in objs:
            anns.append({
                "id": 0,                                    # 汇总时统一编号
                "image_id": img_id,
                "category_name": label,
                "bbox": bbox,
                "segmentation": {"size": rle["size"], "counts": rle["counts"].decode("ascii")},
                "area": area,
                "iscrowd": 0,
            })
    return images, anns


def main():
    ap = argparse.ArgumentParser("ScanNet++ train -> data engine 三件套")
    ap.add_argument("--root", required=True, help="ScanNet++ dataset root")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-frames-per-scene", type=int, default=8,
                    help="标注本身已经抽过帧了，这里再截断以免单场景占比过高")
    ap.add_argument("--limit-scenes", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    ann_dir = root / "iphone_semantic_2d_subsample_1" / "annotations"
    data_dir = root / "data"
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    scenes = sorted(p.name for p in ann_dir.iterdir() if p.is_dir())
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    print(f"{len(scenes)} 个场景 -> {out}")

    all_images, all_anns = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(
                    process_scene, s, ann_dir, data_dir, out, args.max_frames_per_scene
                ): s
                for s in scenes}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                imgs, anns = f.result()
            except Exception as e:                          # noqa: BLE001
                print(f"  {futs[f]} 失败: {e}")
                continue
            all_images += imgs
            all_anns += anns

    cat_id = {n: i + 1 for i, n in enumerate(sorted({a["category_name"] for a in all_anns}))}
    for i, a in enumerate(all_anns):
        a["id"] = i + 1
        a["category_id"] = cat_id[a.pop("category_name")]

    coco = {
        "info": {"description": "ScanNet++ train adapted for the 3DSR data engine"},
        "licenses": [], "images": all_images, "annotations": all_anns,
        "categories": [{"id": i, "name": n}
                       for n, i in sorted(cat_id.items(), key=lambda kv: kv[1])],
    }
    (out / "instances.json").write_text(json.dumps(coco))
    print(f"写出 {len(all_images)} 图 / {len(all_anns)} 实例 / {len(cat_id)} 类 -> {out}")


if __name__ == "__main__":
    main()
