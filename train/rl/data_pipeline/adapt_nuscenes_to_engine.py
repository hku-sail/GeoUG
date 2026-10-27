# Copyright 2026 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""把 nuScenes train 转成 data engine 认的三件套：COCO JSON + 图像目录 + 深度 npz。

三个源里 nuScenes 缺得最多，得自己补两样
----------------------------------------
    图像   samples/CAM_*/...jpg                            1600x900，现成
    名字   category_name，10 个真实类别                     现成
    3D 框  bbox_cam3d = [x,y,z,l,h,w,yaw]，相机系           现成
    mask   `segmentation` 字段是**空列表**                  要补
    深度   只有稀疏 LiDAR，没有稠密深度图                    要补

补法：**把 3D 框当几何体光栅化**。对每个像素做一次射线-长方体求交，取近面交点的深度，
多个物体之间用 z-buffer 取最近的。这一步同时产出两样东西 ——

    depth[u,v] = 该像素看到的最近物体表面的深度
    mask_i     = z-buffer 里归属于物体 i 的像素（**天然带遮挡**，被挡住的部分不算它的）

比「拿 2D 框当 mask + 填个常数深度」强在两处：遮挡关系是对的，所以 engine 的
`pick_interior_point` 不会把前车的像素当成后车的；深度是**近面**而不是中心，
和 v5 的口径一致。

为什么近面 vs 中心这件事要较真
------------------------------
engine 读的是表面深度（mask 内点 + 深度窗口读数），v5 的答案也是这个口径。nuScenes 里
物体尺寸差异极大 —— 公交车 12 米、行人 0.5 米。用中心深度的话：

    20 米处的卡车（长 8 米）中心 20 米、近面 16 米
    18 米处的轿车（长 4 米）中心 18 米、近面 16 米

中心说轿车近，近面说一样近。题目按 relgap >= 0.10 筛，这种翻转会实打实进训练集。

坐标约定是实测出来的，不是查文档
--------------------------------
- `(x,y,z)` 是**几何重心**：投影它得到的点和标注的 `center2d`、2D 框中心三者重合。
- 维度顺序 `(l,h,w)` 映射到局部 `(x,y,z)` 轴、绕 **y 轴**（相机系 y 朝下）转 yaw。
  拿 800 个标注比对「投影包围盒 vs 标注 bbox」：`lhw` 中位误差 0.80 像素，
  `whl` 是 13.93 像素。

npz 的三个字段，口径同另外两个 adapter
--------------------------------------
- `conf`：engine 的过滤条件是 `conf > 1.05`（不是 0-1 口径）。物体像素写 2.0，背景 0.0。
  背景没有深度不要紧 —— engine 只在物体内点周围开窗读数，不碰背景。
- `intrinsics`：主点强制写图像正中。`geometry.scale_intrinsics_to_depth` 从主点反推源
  分辨率，它的输出 cx 恒等于 W/2，engine 的几何从根上假设中心主点。nuScenes 的真实主点
  约 (816, 492)，图像中心 (800, 450) —— 差 16/42 像素，20 米处横向偏 0.25/0.66 米。
  比另外两个源大，但仍远小于 10% 的 relgap 门槛。

用法::

    python -m train.rl.data_pipeline.adapt_nuscenes_to_engine \\
        --root /path/to/nuScenes --out /path/to/engine_src/nuscenes_train \\
        --stride 20 --workers 32
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

CONF_VALID = 2.0
CONF_HOLE = 0.0

MIN_MASK_PX = 300       # 太小的物体在 1600x900 上读不出稳定的深度窗口
MIN_DEPTH_M = 3.0       # 太近的物体多半只拍到一角，框也不准
MAX_DEPTH_M = 60.0      # 远处 3D 框标注噪声大


def box_corners(bbox_cam3d):
    """3D 框的八个角点，相机系。维度顺序 (l,h,w) -> 局部 (x,y,z)，绕 y 轴转 yaw。"""
    x, y, z, l, h, w, yaw = bbox_cam3d
    s = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
    c = s * np.array([l, h, w])
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    return (R @ c.T).T + np.array([x, y, z]), R, np.array([x, y, z]), np.array([l, h, w]) / 2.0


def ray_box_near_depth(K, R, center, half, u0, u1, v0, v1):
    """在像素窗口 [u0:u1, v0:v1] 内做射线-长方体求交，返回近面深度（未命中为 inf）。

    射线取 D = (X/Z, Y/Z, 1)，这样交点参数 t **就是相机系深度**，省一次换算。
    在长方体的局部坐标系里用 slab 法求交。
    """
    us = np.arange(u0, u1)
    vs = np.arange(v0, v1)
    U, V = np.meshgrid(us, vs)
    # 反投影出方向（z 分量为 1）
    dx = (U - K[0, 2]) / K[0, 0]
    dy = (V - K[1, 2]) / K[1, 1]
    D = np.stack([dx, dy, np.ones_like(dx)], axis=-1)      # (h, w, 3)

    Rt = R.T
    Dl = D @ Rt.T                                          # 方向转到局部系
    Ol = -(Rt @ center)                                    # 相机原点在局部系的位置

    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / Dl
        t1 = (-half - Ol) * inv
        t2 = (half - Ol) * inv
    tmin = np.maximum.reduce([np.minimum(t1, t2)[..., i] for i in range(3)])
    tmax = np.minimum.reduce([np.maximum(t1, t2)[..., i] for i in range(3)])

    hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin) & np.isfinite(tmax)
    depth = np.where(hit, np.maximum(tmin, 0.0), np.inf)
    return depth


def process_image(info, anns, root, out_dir):
    """一张图 -> (COCO image 条目, COCO annotation 列表)，并落盘 npz + 图像软链。"""
    from pycocotools import mask as mask_utils

    src = root / info["file_name"]
    if not src.exists():
        return None, []
    W, H = 1600, 900
    K = np.asarray(info["cam_intrinsic"], dtype=np.float64)

    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    owner = np.full((H, W), -1, dtype=np.int32)
    kept = []

    for a in anns:
        b = a["bbox_cam3d"]
        if not (MIN_DEPTH_M <= b[2] <= MAX_DEPTH_M):
            continue
        C, R, center, half = box_corners(b)
        if (C[:, 2] <= 0.1).any():                  # 有角点在相机后面，投影会翻
            continue
        u = K[0, 0] * C[:, 0] / C[:, 2] + K[0, 2]
        v = K[1, 1] * C[:, 1] / C[:, 2] + K[1, 2]
        u0, u1 = int(np.clip(np.floor(u.min()), 0, W)), int(np.clip(np.ceil(u.max()) + 1, 0, W))
        v0, v1 = int(np.clip(np.floor(v.min()), 0, H)), int(np.clip(np.ceil(v.max()) + 1, 0, H))
        if u1 - u0 < 4 or v1 - v0 < 4:
            continue

        d = ray_box_near_depth(K, R, center, half, u0, u1, v0, v1)
        sub = zbuf[v0:v1, u0:u1]
        upd = d < sub                               # z-buffer：更近的才覆盖
        if not upd.any():
            continue
        sub[upd] = d[upd].astype(np.float32)
        owner[v0:v1, u0:u1][upd] = len(kept)
        kept.append(a)

    if len(kept) < 2:
        return None, []

    stem = info["file_name"].replace("/", "_").replace(".jpg", "")
    objs = []
    for i, a in enumerate(kept):
        m = owner == i
        area = int(m.sum())
        if area < MIN_MASK_PX:
            continue
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        x, y, w, h = [float(t) for t in mask_utils.toBbox(rle)]
        objs.append({"id": 0, "image_id": 0,
                     "category_name": a["category_name"].split(".")[-1].replace("_", " "),
                     "bbox": [x, y, w, h],
                     "segmentation": {"size": list(rle["size"]),
                                      "counts": rle["counts"].decode("ascii")},
                     "area": area, "iscrowd": 0})
    if len(objs) < 2:
        return None, []

    link = out_dir / "images" / f"{stem}.jpg"
    if not link.exists():
        link.symlink_to(src.resolve())
    depth = np.where(np.isfinite(zbuf), zbuf, 0.0).astype(np.float32)
    conf = np.where(depth > 0, CONF_VALID, CONF_HOLE).astype(np.float32)
    K_out = np.array([[K[0, 0], 0.0, W / 2.0],
                      [0.0, K[1, 1], H / 2.0],
                      [0.0, 0.0, 1.0]], dtype=np.float64)   # 主点居中，见 docstring
    np.savez_compressed(out_dir / "depth" / f"{stem}.npz",
                        depth=depth, conf=conf, intrinsics=K_out)

    img_id = abs(hash(stem)) % (2 ** 31)
    for o in objs:
        o["image_id"] = img_id
    return {"id": img_id, "file_name": f"{stem}.jpg", "width": W, "height": H}, objs


def main():
    ap = argparse.ArgumentParser("nuScenes train -> data engine 三件套")
    ap.add_argument("--root", required=True, help="nuScenes dataset root")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--stride", type=int, default=20,
                    help="nuScenes 关键帧 2Hz，相邻帧几乎相同；隔 N 张取一张")
    ap.add_argument("--cameras", default="CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,"
                                         "CAM_BACK,CAM_BACK_LEFT,CAM_BACK_RIGHT")
    ap.add_argument("--limit-images", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    coco_train = root / "nuscenes_infos_temporal_train_mono3d.coco.json"
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    print("读 COCO JSON（1.15M 标注，约 1 分钟）...")
    d = json.loads(coco_train.read_text())
    cams = tuple(f"samples/{c.strip()}/" for c in args.cameras.split(","))
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)
    infos = [i for i in d["images"] if i["file_name"].startswith(cams)]
    infos = infos[:: args.stride]
    if args.limit_images:
        infos = infos[: args.limit_images]
    print(f"{len(infos)} 张图（stride={args.stride}）-> {out}")

    all_images, all_anns = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process_image, i, by_img[i["id"]], root, out) for i in infos]
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                im, objs = f.result()
            except Exception as e:                          # noqa: BLE001
                print(f"  失败: {e}")
                continue
            if im is None:
                continue
            all_images.append(im)
            all_anns += objs

    cat_id = {n: i + 1 for i, n in enumerate(sorted({a["category_name"] for a in all_anns}))}
    for i, a in enumerate(all_anns):
        a["id"] = i + 1
        a["category_id"] = cat_id[a.pop("category_name")]

    coco = {"info": {"description": "nuScenes train adapted for the 3DSR data engine"},
            "licenses": [], "images": all_images, "annotations": all_anns,
            "categories": [{"id": i, "name": n}
                           for n, i in sorted(cat_id.items(), key=lambda kv: kv[1])]}
    (out / "instances.json").write_text(json.dumps(coco))
    print(f"写出 {len(all_images)} 图 / {len(all_anns)} 实例 / {len(cat_id)} 类 -> {out}")


if __name__ == "__main__":
    main()
