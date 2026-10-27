from __future__ import annotations

import numpy as np

from train.rl.engine.alpha_grpo.stat_tracking import PerPromptStatTracker
from train.rl.spatial_advantage import FixedScaleStatTracker


def test_paper_config_defaults_to_lambda_point_one(monkeypatch, tmp_path):
    for name in (
        "MODEL_PATH",
        "RL_DATASET",
        "DEPTH_CACHE_DIR",
        "EVAL_QUESTIONS",
        "EVAL_DEPTH_CACHE",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    from train.rl.config_spatial import get_config

    config = get_config("spatial_gate_draft_v3")
    assert config.train.lambda_cost == 0.1


def _advantages(tracker, correct, depth, cost):
    reward = np.asarray(correct, dtype=np.float64) - cost * np.asarray(depth)
    return tracker.update(["question"] * len(reward), reward)


def test_cost_advantage_scales_with_lambda():
    depth = np.array([0] * 8 + [1] * 8)
    correct = np.ones(16)

    low = _advantages(FixedScaleStatTracker(adv_scale=0.5), correct, depth, 0.05)
    high = _advantages(FixedScaleStatTracker(adv_scale=0.5), correct, depth, 0.1)

    np.testing.assert_allclose(np.abs(high).mean(), 2 * np.abs(low).mean())
    assert high[depth == 1].mean() < 0 < high[depth == 0].mean()


def test_release_tracker_does_not_cancel_lambda_like_group_std():
    depth = np.array([0] * 8 + [1] * 8)
    correct = np.ones(16)

    upstream_low = _advantages(
        PerPromptStatTracker(global_std=True), correct, depth, 0.05
    )
    upstream_high = _advantages(
        PerPromptStatTracker(global_std=True), correct, depth, 0.5
    )
    fixed_low = _advantages(FixedScaleStatTracker(adv_scale=0.5), correct, depth, 0.05)
    fixed_high = _advantages(FixedScaleStatTracker(adv_scale=0.5), correct, depth, 0.5)

    assert np.isclose(
        np.abs(upstream_high).mean() / np.abs(upstream_low).mean(), 1.0, rtol=0.02
    )
    assert np.isclose(
        np.abs(fixed_high).mean() / np.abs(fixed_low).mean(), 10.0, rtol=1e-6
    )
