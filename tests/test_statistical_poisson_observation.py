"""Observation-window and calendar invariants of the native Poisson controls."""

import numpy as np
import pytest

from seahorse.models.statistical.poisson import PoissonSTPP
from seahorse.models.statistical.core import calendar_offset


def seq(times, start):
    return {"times": times, "locations": [[0.5, 0.5]] * len(times),
            "t0": 0.0, "t1": 1.0, "window_start": start, "time_axis": "utc"}


def test_seasonal_phase_survives_per_window_rebasing():
    model = PoissonSTPP((0, 1, 0, 1), period=2, n_phases=2)
    model.fit([seq([0.5], "1970-01-01T00:00:00Z"),
               seq([0.1, 0.3, 0.7], "1970-01-02T00:00:00Z")], seed=0)
    np.testing.assert_allclose(model.rates, [1, 3])
    h0 = seq([0.1], "1970-01-01T00:00:00Z")
    h1 = seq([0.1], "1970-01-02T00:00:00Z")
    h0["t1"] = h1["t1"] = 0.1
    assert model.next_event_nll(h0, 0.5, np.asarray([0.5, 0.5])) == pytest.approx(0.4)
    assert model.next_event_nll(h1, 0.5, np.asarray([0.5, 0.5])) == pytest.approx(-np.log(3) + 1.2)


def test_empty_windows_contribute_exposure_and_zero_rate_forecasts_are_empty():
    model = PoissonSTPP((0, 1, 0, 1), alpha=0)
    diagnostics = model.fit([seq([], "1970-01-01"), seq([], "1970-01-02")], seed=0)
    assert diagnostics["exposure_by_phase"] == [2]
    assert not diagnostics["spatial_distribution_identified"]
    assert model.full_window_nll(seq([], "1970-01-01")) == 0
    assert model.sample_future({"times": [], "locations": [], "t1": 1}, horizon=1,
                               rng=np.random.default_rng(0)) == {"times": [], "locations": []}


def test_forecast_begins_at_observation_end_after_quiet_interval():
    model = PoissonSTPP((0, 1, 0, 1))
    model.fit([seq([0.5], "1970-01-01")], seed=0)
    draws, _ = model.sample_next_event({"times": [0.5], "locations": [[0.5, 0.5]], "t1": 8},
                                       n_samples=10, rng=np.random.default_rng(0))
    assert np.all(draws > 8)
    assert model.next_event_nll({"times": [0.5], "locations": [[0.5, 0.5]], "t1": 8},
                                8.2, np.array([0.5, 0.5])) == pytest.approx(0.2)


def test_phase_exposure_is_exact_over_many_periods_and_negative_origins():
    model = PoissonSTPP((0, 1, 0, 1), period=2, n_phases=2)
    np.testing.assert_allclose(model._phase_exposure(-2.5, 2000.5), [1001.5, 1001.5])


def test_synthetic_descriptive_window_origin_uses_only_declared_offset():
    synthetic = {"window_start": "synthetic:0", "time_unit": "synthetic_horizon",
                 "calendar_offset_from_window_start": .125}
    assert calendar_offset(synthetic) == pytest.approx(.125)
    # Do not broaden this to arbitrary non-day calendar records: a real unit
    # mismatch must still fail rather than silently inventing a phase origin.
    with pytest.raises(ValueError, match="time units in days"):
        calendar_offset({"window_start": "2020-01-01", "time_unit": "hours"})


def test_synthetic_seasonal_reference_uses_one_horizon_period():
    model = PoissonSTPP((0, 1, 0, 1), period=1, n_phases=1)
    diagnostics = model.fit([
        {"t0": 0., "t1": 1., "times": [.2, .7], "locations": [[.5, .5], [.5, .5]],
         "window_start": "synthetic:0", "time_unit": "synthetic_horizon"},
    ], seed=0)
    assert diagnostics["exposure_by_phase"] == [1.]
    assert model.period == 1 and model.n_phases == 1


def test_restored_model_rejects_invalid_probabilities():
    model = PoissonSTPP((0, 1, 0, 1))
    model.fit([seq([0.5], "1970-01-01")], seed=0)
    state = model.to_state()
    state["probabilities"] = [0.5]
    with pytest.raises(ValueError, match="normalized distribution"):
        PoissonSTPP.from_state(state)
