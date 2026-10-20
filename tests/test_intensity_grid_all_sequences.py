"""Intensity recovery conditions on every test sequence, not only the first."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from seahorse.evaluation import intensity as intensity_mod


class _CountingModel:
    """Intensity equal to the number of history events (any grid point)."""

    def eval(self):
        return self


def _fake_eval_intensity(*, model, t_query, s_grid, history_times, history_locs, **kwargs):
    del model, t_query, kwargs
    return np.full(s_grid.shape[0], float(history_times.shape[0]), dtype=np.float32)


def test_direct_grid_averages_over_test_histories(monkeypatch):
    monkeypatch.setattr(intensity_mod, "eval_intensity", _fake_eval_intensity)
    runner = SimpleNamespace(model=_CountingModel(), norm_stats={"normalize": False})
    seqs = [
        {"times": np.array([0.0, 1.0, 2.0]), "locations": np.zeros((3, 2), dtype=np.float32)},
        {"times": np.array([0.0, 0.5, 1.0, 1.5]), "locations": np.zeros((4, 2), dtype=np.float32)},
    ]
    xs = ys = np.linspace(0.0, 1.0, 2, dtype=np.float32)
    ts = np.array([0.25, 1.25, 2.0], dtype=np.float32)
    grid = intensity_mod._direct_intensity_grid(runner, seqs, xs, ys, ts, torch.device("cpu"))
    # t=0.25: histories of size 1 and 1; t=1.25: 2 and 3; t=2.0: 3 and 4.
    np.testing.assert_allclose(grid[:, 0, 0], [1.0, 2.5, 3.5])
    single = intensity_mod._direct_intensity_grid(runner, seqs[:1], xs, ys, ts, torch.device("cpu"))
    np.testing.assert_allclose(single[:, 0, 0], [1.0, 2.0, 3.0])


def test_rollout_coherence_declares_lower_is_better():
    from seahorse.evaluation.metrics.generative import RolloutCoherence

    assert RolloutCoherence.direction == "lower"
    assert RolloutCoherence.deprecated is True
