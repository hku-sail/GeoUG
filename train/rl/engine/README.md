# GeoUG RL engine

This directory contains the project-owned training engine required by GeoUG
RL. It is versioned as part of GeoUG and is not loaded from an external
checkout. Components unrelated to the released method are intentionally
excluded.

Included components:

- `alpha_grpo/train.py`: distributed rollout and optimization loop;
- `alpha_grpo/common.py`: generation-step and advantage types;
- `alpha_grpo/algorithms/grpo.py`: GRPO base loss;
- task and dataset registries;
- EMA, statistics, and training utilities;
- `Bagel/`: BAGEL inferencer and model implementation with RL hooks;
- GeoUG reasoning-token and depth transforms integrated into `Bagel/data`;
- base configuration and DeepSpeed JSON files.

The engine is released under Apache-2.0. Its license is retained at
`alpha_grpo/LICENSE`, and source files carry SPDX headers.

