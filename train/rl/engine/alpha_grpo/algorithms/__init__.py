# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Algorithm hub for GeoUG RL engine training.

Usage:
    from algorithms import get_algorithm
    text_algo = get_algorithm(config.train.algorithm)
    image_algo = get_algorithm(config.train.image_algorithm)
    info, loss = text_algo.compute_text_loss(config, sample, accelerator, info, ...)
    info, loss = image_algo.compute_image_loss(config, sample, accelerator, info, ...)

To add a new algorithm (e.g., NFT):
    1. Create algorithms/nft.py
    2. Implement @register('nft') class with compute_text_loss / compute_image_loss
    3. Add `from . import nft` below
"""

REGISTRY = {}


def register(name):
    """Support registering one algorithm under multiple names: @register(('grpo', 'ppo'))"""
    def decorator(cls):
        names = name if isinstance(name, (list, tuple)) else [name]
        for n in names:
            REGISTRY[n] = cls
        return cls
    return decorator


def get_algorithm(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]


class BaseAlgorithm:
    @classmethod
    def compute_text_loss(cls, config, sample, accelerator, info, gen_step=None, **kwargs):
        """Compute text-level RL loss.

        Args:
            gen_step: GenerationStep with rollout/ref data for the current text step.
                - gen_step.text_per_token_log_probs: rollout (or recomputed) log probs
                - gen_step.ref_text_per_token_log_probs: ref model log probs (when text_beta > 0)
            text_per_token_log_probs: Per-token log probs from current forward pass
            text_entropies: Per-token entropy values
            text_token_lens: Token lengths per sequence
            get_high_entropy_mask: Callable to compute high-entropy token mask
        """
        raise NotImplementedError

    @classmethod
    def compute_image_loss(cls, config, sample, accelerator, info, gen_step=None, j=0, **kwargs):
        """Compute image-level RL loss.

        Args:
            gen_step: GenerationStep with rollout/ref data for the current image step.
                - gen_step.image_log_probs[:, j]: rollout (or recomputed) log probs at timestep j
                - gen_step.ref_prev_latents_means[:, j]: ref model predicted mean (when beta > 0)
                - gen_step.ref_model_output[:, j]: ref model raw output (when beta > 0)
            j: Timestep index within the sde_window
            image_log_probs: Log probs from current forward pass
            prev_latents_mean: Predicted mean of previous latents
            std_dev_t: Standard deviation at timestep t
            model_output: Raw model output at current timestep
        """
        raise NotImplementedError


# GeoUG only ships the GRPO base used by spatial_grpo. Other GeoUG RL engine
# algorithms were intentionally not vendored into the release.
from . import grpo
