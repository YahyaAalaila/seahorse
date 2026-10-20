"""Contract tests for explicit forecast sampler routes."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from seahorse.evaluation.forecast import forecast_horizon, forecast_next, score_horizon_counts
from seahorse.models.statistical.poisson import PoissonSTPP


def _history(**extra):
    value = {"times": np.asarray([0.2, 0.7]), "locations": np.asarray([[0.25, 0.25], [0.5, 0.5]]),
             "t0": 0.0, "t1": 0.7, "calendar_window_start": "1970-01-01T00:00:00Z",
             "calendar_offset_from_window_start": 0.0, "time_unit": "days"}
    value.update(extra)
    return value


def _stat_runner(model, preset="homogeneous_poisson"):
    return SimpleNamespace(runner_name="statstpp", config=SimpleNamespace(model=SimpleNamespace(preset=preset)), model=model)


def _fitted_poisson(*, seasonal=False, zero=False):
    model = PoissonSTPP((0, 1, 0, 1), period=2.0 if seasonal else None, n_phases=2 if seasonal else 1, alpha=0.0)
    if zero:
        train = [{"t0": 0., "t1": 1., "times": [], "locations": []}]
    elif seasonal:
        train = [{"t0": 0., "t1": 2., "times": np.linspace(1.01, 1.99, 40),
                  "locations": np.tile([[.5, .5]], (40, 1))}]
    else:
        train = [{"t0": 0., "t1": 2., "times": [.25, .75, 1.25, 1.75], "locations": [[.5, .5]] * 4}]
    model.fit(train, seed=3)
    return model


def test_actual_poisson_horizon_has_expected_count_and_quiet_origin():
    history = _history(t1=1.2)
    out = forecast_horizon(_stat_runner(_fitted_poisson()), history, t_end=3.2, n_samples=900, seed=7)
    assert set(out["sample_status"]) <= {"complete", "no_event"}
    assert all(np.all(row >= 1.2) for row in out["sample_times"])
    assert abs(np.mean([len(row) for row in out["sample_times"]]) - 4.0) < .35


def test_actual_seasonal_poisson_preserves_calendar_offset_and_quiet_origin():
    history = _history(t1=.7, calendar_offset_from_window_start=1.0)
    out = forecast_next(_stat_runner(_fitted_poisson(seasonal=True), "seasonal_poisson"), history, n_samples=20, seed=4)
    assert out["status"] == "ok" and np.all(out["valid"])
    assert np.all(out["times"] > .7)
    assert np.median(out["times"]) < .9


def test_actual_zero_rate_poisson_is_valid_empty_fixed_horizon():
    out = forecast_horizon(_stat_runner(_fitted_poisson(zero=True)), _history(), t_end=1.2, n_samples=3, seed=1)
    assert out["sample_status"].tolist() == ["no_event"] * 3
    assert all(row.shape == (0,) for row in out["sample_times"])
    assert score_horizon_counts(out, [], t_start=.7, t_end=1.2, bin_width=.25)["count_rmse"] == 0.0


def test_actual_factorized_route_is_seeded_and_does_not_mutate_context():
    from seahorse.models.configs.factorized import PoissonGMMConfig
    model = PoissonGMMConfig(hidden_dim=8, spatial_dim=2).build_model(); model.eval()
    runner = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(preset="poisson_gmm")), model=model,
        norm_stats={"normalize": False, "time_mean": 0., "time_std": 1., "loc_mean": [0., 0.], "loc_std": [1., 1.]})
    history = _history(); before = history["times"].copy(), history["locations"].copy()
    first, second = forecast_next(runner, history, n_samples=4, seed=17), forecast_next(runner, history, n_samples=4, seed=17)
    assert first["backend"] == "factorized_exact_temporal_full_plane_gmm_v1"
    np.testing.assert_array_equal(first["times"], second["times"])
    np.testing.assert_array_equal(history["times"], before[0]); np.testing.assert_array_equal(history["locations"], before[1])


def test_factorized_horizon_rebases_each_retained_history_and_restores_time_axis():
    """The actual sampler sees the same t=0 context convention as training."""
    import torch
    from seahorse.models.configs.factorized import PoissonGMMConfig

    model = PoissonGMMConfig(hidden_dim=8, spatial_dim=2).build_model(); model.eval()
    with torch.no_grad():
        model.event_model.temporal_model.log_lambda.fill_(3.0)
    seen = []
    encode = model.state_model.encode_sampling_history
    def recording_encode(*, times, locations, lengths):
        seen.append(times.detach().cpu().numpy().copy())
        return encode(times=times, locations=locations, lengths=lengths)
    model.state_model.encode_sampling_history = recording_encode
    runner = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(preset="poisson_gmm")), model=model,
        norm_stats={"normalize": False, "time_mean": 0., "time_std": 1., "loc_mean": [0., 0.], "loc_std": [1., 1.]})
    history = _history(times=np.asarray([0., .2]), t1=.2)
    out = forecast_horizon(runner, history, t_end=1.2, n_samples=1, seed=11, history_limit=2)
    assert out["sample_status"].tolist() == ["complete"]
    assert len(out["sample_times"][0]) >= 2
    assert all(float(frame[0, 0]) == 0.0 for frame in seen)
    generated = out["sample_times"][0]
    assert np.all(np.diff(generated) > 0) and np.all((generated >= .2) & (generated < 1.2))
    assert out["neural_history_frame"].startswith("rebase retained")


def _tiny_diffusion_runner(device):
    import torch
    from seahorse.registry import build_model
    torch.manual_seed(11)
    config = {"token_delta_t_min": .1, "token_delta_t_range": .4, "token_loc_min": (0., 0.), "token_loc_range": (1., 1.),
              "decoder": {"type": "diffusion_stpp", "hidden_units": 8, "timesteps": 2, "sampling_timesteps": 1}}
    model = build_model(config=config, preset="diffusion_stpp", spatial_dim=2, hidden_dim=8, n_marks=0).to(device)
    times = torch.tensor([[.1, .3, .6]], dtype=torch.float32, device=device)
    locs = torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4); opt.zero_grad()
    model(times=times, locations=locs, lengths=torch.tensor([3], device=device))["loss"].backward(); opt.step(); model.eval()
    return SimpleNamespace(
        config=SimpleNamespace(model=SimpleNamespace(preset="diffusion_stpp")), model=model,
        norm_stats={"normalize": False, "time_mean": 0., "time_std": 1.,
                    "loc_mean": [0., 0.], "loc_std": [1., 1.]},
    )


@pytest.mark.parametrize("device_name", [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="CUDA is unavailable")),
])
def test_actual_diffusion_native_forecast_is_seeded_after_save_reload(tmp_path, device_name):
    import torch
    device = torch.device(device_name)
    runner = _tiny_diffusion_runner(device); history = _history(); before = history["times"].copy(), history["locations"].copy()
    first = forecast_next(runner, history, n_samples=2, seed=29, device=device)
    path = tmp_path / "tiny_diffusion.pt"; torch.save(runner.model.state_dict(), path)
    reloaded = _tiny_diffusion_runner(device); reloaded.model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    second = forecast_next(reloaded, _history(), n_samples=2, seed=29, device=device)
    assert first["backend"] == "diffusion_stpp_native_next_event"
    np.testing.assert_array_equal(first["times"], second["times"]); np.testing.assert_array_equal(first["locations"], second["locations"])
    np.testing.assert_array_equal(history["times"], before[0]); np.testing.assert_array_equal(history["locations"], before[1])


@pytest.mark.parametrize("device_name", [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="CUDA is unavailable")),
])
def test_actual_smash_native_forecast_is_seeded_after_one_update(device_name):
    import torch
    from seahorse.registry import build_model

    torch.manual_seed(41)
    config = {
        "encoder": {"d_model": 8, "d_rnn": 8, "d_inner": 8, "n_layers": 1,
                    "n_head": 1, "d_k": 8, "d_v": 8, "dropout": 0.0},
        "decoder": {"type": "smash", "num_noise": 1, "samplingsteps": 1,
                    "n_samples": 1, "langevin_step": .001},
        "token_time_min_raw": .1, "token_time_range_raw": .4,
        "token_time_min_log": -2.3, "token_time_range_log": 1.4,
        "token_loc_min": (0., 0.), "token_loc_range": (1., 1.),
    }
    device = torch.device(device_name)
    model = build_model(config=config, preset="smash", spatial_dim=2, hidden_dim=8, n_marks=0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4); opt.zero_grad()
    times = torch.tensor([[.1, .3, .6]], dtype=torch.float32, device=device)
    locs = torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]], dtype=torch.float32, device=device)
    model(times=times, locations=locs, lengths=torch.tensor([3], device=device))["loss"].backward(); opt.step(); model.eval()
    runner = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(preset="smash")), model=model,
                             norm_stats={"normalize": False})
    history = _history(); before = history["times"].copy(), history["locations"].copy()
    first = forecast_next(runner, history, n_samples=1, seed=23, device=device)
    second = forecast_next(runner, history, n_samples=1, seed=23, device=device)
    assert first["backend"] == "smash_native_next_event" and bool(first["valid"][0])
    np.testing.assert_array_equal(first["times"], second["times"])
    np.testing.assert_array_equal(first["locations"], second["locations"])
    np.testing.assert_array_equal(history["times"], before[0]); np.testing.assert_array_equal(history["locations"], before[1])


def test_shape_invalid_draw_and_event_cap_are_not_scored():
    class Bad:
        def sample_next_event(self, history, *, n_samples, rng): return np.zeros((n_samples, 1)), np.zeros((n_samples, 2))
    with pytest.raises(RuntimeError, match="invalid next-event shape"):
        forecast_next(_stat_runner(Bad()), _history())
    capped = {"status": "ok", "sample_status": ["event_cap_exceeded"], "sample_times": [np.asarray([])], "sample_locations": [np.empty((0, 2))]}
    assert np.isnan(score_horizon_counts(capped, [], t_start=.7, t_end=1.2, bin_width=.25)["count_crps"])


def test_spatiotemporal_overflow_and_out_of_horizon_rejection():
    forecast = {"status": "ok", "sample_status": ["complete"], "sample_times": [np.asarray([.75])], "sample_locations": [np.asarray([[2., 2.]])]}
    result = score_horizon_counts(forecast, [.75], t_start=.7, t_end=1.2, bin_width=.25,
        truth_locations=[[2., 2.]], spatial_bounds=(0, 1, 0, 1), spatial_grid=(3, 2))
    assert result["count_rmse"] == 0.0
    with pytest.raises(ValueError, match="within"):
        score_horizon_counts(forecast, [1.2], t_start=.7, t_end=1.2, bin_width=.25)


def test_spatial_upper_boundary_is_in_last_cell_and_overflow_keeps_time_bin():
    forecast = {"status": "ok", "sample_status": ["complete"], "sample_times": [np.asarray([.2])],
                "sample_locations": [np.asarray([[.99, .99]])]}
    edge = score_horizon_counts(forecast, [.2], t_start=0., t_end=1., bin_width=.5,
        truth_locations=[[1., 1.]], spatial_bounds=(0., 1., 0., 1.), spatial_grid=(2, 2))
    assert edge["count_crps"] == edge["count_rmse"] == 0.
    forecast["sample_times"] = [np.asarray([.8])]
    forecast["sample_locations"] = [np.asarray([[2., 2.]])]
    shifted = score_horizon_counts(forecast, [.2], t_start=0., t_end=1., bin_width=.5,
        truth_locations=[[2., 2.]], spatial_bounds=(0., 1., 0., 1.), spatial_grid=(1, 1))
    # Two time bins × (one interior cell plus one overflow cell).
    assert shifted["count_crps"] == shifted["count_mae"] == .5
    assert shifted["count_rmse"] == pytest.approx(np.sqrt(.5))


def test_unsupported_never_thins_rectangle():
    assert forecast_next(SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(preset="auto_stpp"))), _history())["status"] == "unsupported"
