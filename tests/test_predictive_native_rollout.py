from __future__ import annotations

import numpy as np

from seahorse.evaluation.predictive.rollout import rollout_window_native_sampler


def _history():
    return {"times": np.asarray([1.0], dtype=np.float32),
            "locations": np.asarray([[0.0, 0.0]], dtype=np.float32)}


def test_native_nonpositive_draw_becomes_failed_short_rollout_not_zero_wait_event():
    def invalid(_runner, _history, _device):
        return 1.0, np.asarray([.2, .3], dtype=np.float32)

    times, locs, next_history, meta = rollout_window_native_sampler(
        None, invalid, _history(), window_start=1.0, window_end=2.0,
        max_events=2, bridge_retries=1, device=None,
    )
    assert times.size == 0 and locs.shape == (0, 2)
    assert next_history["times"].tolist() == [1.0]
    assert meta["native_failed_rollout"] and meta["native_short_rollout"]
    assert meta["native_invalid_reasons"] == {"nonpositive_wait": 1}


def test_native_float32_collapsed_draw_is_recorded_without_jittering():
    def collapsed(_runner, _history, _device):
        return 1.0 + 1e-10, np.asarray([.2, .3], dtype=np.float32)

    times, _, next_history, meta = rollout_window_native_sampler(
        None, collapsed, _history(), window_start=1.0, window_end=2.0,
        max_events=1, bridge_retries=1, device=None,
    )
    assert times.size == 0
    assert next_history["times"].tolist() == [1.0]
    assert meta["native_invalid_reasons"] == {"float32_time_collapse": 1}
