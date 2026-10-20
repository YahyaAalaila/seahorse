"""Shared next-event protocol tests independent of fitting implementations."""

from types import SimpleNamespace

import numpy as np
import pytest

from seahorse.evaluation import evaluate
from seahorse.evaluation.artifacts import build_predictive_samples_key
from seahorse.evaluation.registry import metric_by_name
from seahorse.evaluation.statistical import (
    compute_statistical_next_event_nll,
    compute_statistical_predictive_samples,
)


def sequence(times, **metadata):
    return {"times": times, "locations": [[0.5, 0.5]] * len(times), **metadata}


class PrefixProbe:
    support = (0.0, 1.0, 0.0, 1.0)
    nll_method = "test_wait_score"

    def __init__(self):
        self.histories = []
        self.state_version = 1

    def next_event_nll(self, history, next_time, next_location):
        self.histories.append(history)
        return next_time - history["times"][-1]

    def sample_next_event(self, history, *, n_samples, rng):
        self.histories.append(history)
        waits = np.asarray([1.0, 1.0, 10.0]) if n_samples == 3 else rng.exponential(size=n_samples)
        return history["times"][-1] + waits, np.full((n_samples, 2), 0.5)

    def to_state(self):
        return {"version": self.state_version}


def runner():
    return SimpleNamespace(
        runner_name="statstpp", model=PrefixProbe(),
        capabilities=SimpleNamespace(nll_kind="exact", has_density=False, has_intensity=False, has_native_sampler=True),
    )


def test_nll_uses_actual_prefixes_and_pools_targets_without_future_metadata():
    r = runner()
    seqs = [sequence([0, 1, 4], t0=0, t1=99, marks=[2, 3, 4]), sequence([0, 10])]
    score = compute_statistical_next_event_nll(r, seqs)
    assert score["mean_nll"] == pytest.approx(14 / 3)
    assert [h["times"].tolist() for h in r.model.histories] == [[0], [0, 1], [0]]
    assert [h["t1"] for h in r.model.histories] == [0, 1, 0]
    assert r.model.histories[0]["marks"].tolist() == [2]
    assert score["sequence_index"].tolist() == [0, 0, 1]
    assert score["target_event_index"].tolist() == [1, 2, 1]


def test_shared_public_metrics_use_median_for_mae_mean_for_rmse():
    r = runner()
    report = evaluate(r, [sequence([0, 1])], k_pred=3, metric_profile="predictive",
                      allowed_artifact_families={"predictive_samples"},
                      metrics=[metric_by_name("temporal_mae"), metric_by_name("temporal_rmse")])
    assert report["temporal_mae"].value == 0.0
    assert report["temporal_rmse"].value == 3.0
    # The same public NLL metric agrees with the runner's per-target score.
    report = evaluate(r, [sequence([0, 1, 4]), sequence([0, 10])], metrics=[metric_by_name("nll")])
    assert report["nll"].value == pytest.approx(14 / 3)


def test_statistical_samples_are_reproducible_and_cache_tracks_fitted_state(tmp_path):
    r = runner()
    seqs = [sequence([0, 1, 4])]
    a = compute_statistical_predictive_samples(r, seqs, k=4, seed=19)
    b = compute_statistical_predictive_samples(r, seqs, k=4, seed=19)
    np.testing.assert_array_equal(a.next_times, b.next_times)
    assert a.history_length.tolist() == [1, 2]
    first = build_predictive_samples_key(r, seqs, k=4, seed=19, device="cpu")
    r.model.state_version = 2
    second = build_predictive_samples_key(r, seqs, k=4, seed=19, device="cpu")
    assert first.digest != second.digest
    metric = metric_by_name("temporal_mae")
    args = dict(metrics=[metric], metric_profile="predictive", allowed_artifact_families={"predictive_samples"}, k_pred=4, seed=19, artifact_dir=tmp_path)
    computed = evaluate(r, seqs, **args)
    cached = evaluate(r, seqs, artifact_mode="load_only", **args)
    assert computed["temporal_mae"].value == cached["temporal_mae"].value


def test_ties_rejected_and_zero_density_not_discarded():
    r = runner()
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_statistical_next_event_nll(r, [sequence([0, 0])])
    r.model.next_event_nll = lambda *args: float("inf")
    score = compute_statistical_next_event_nll(r, [sequence([0, 1])])
    assert np.isinf(score["mean_nll"])
    assert score["n_nonfinite_scored_contexts"] == 1
    assert score["n_missing_contexts"] == 0
