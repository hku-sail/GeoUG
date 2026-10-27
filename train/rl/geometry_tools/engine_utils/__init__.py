# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared depth-distance data-engine modules.

These modules are dataset-independent. They operate on COCO-format instance
annotations + DA3-style depth ``.npz`` files. To plug a new dataset (COCO,
ADE20K, ...) into the pipeline, write a thin wrapper that defines defaults
for the few dataset-specific paths and filters; see ``coco/build_3dsr_data.py``
and ``ade20k/build_3dsr_data.py`` for examples.

Modules:
    geometry  -- per-instance 3D point extraction (RANSAC-free)
    tools     -- query_point_depth + query_3d_distance (the model-facing tools)
    templates -- per-category sample generators + sequence renderer
    builder   -- COCO-to-JSONL pipeline (configurable defaults)
    visualize -- self-contained HTML visualisation of a JSONL
"""
