from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest

from seahorse.experiments import campaign_forecast as forecast


def _fixture(monkeypatch, tmp_path, calls, behavior):
    dataset = [object(), object()]
    runner = SimpleNamespace(runner_name="statstpp")
    coord = SimpleNamespace(input_time_unit="days")
    manifest = {"sha256": "manifest-a", "source": {"sha256": "source-a"},
                "datasets": {"d": {"kind": "synthetic"}},
                "spec": {"protocol": {"history_events": 20, "predictive_targets": 2,
                    "predictive_target_seed": 7, "predictive_samples": 2}}}
    job = {"id": "d__model__seed1", "dataset": "d"}
    checkpoint = tmp_path / "model" / "checkpoint.ckpt"; checkpoint.parent.mkdir(); checkpoint.write_bytes(b"first")
    (tmp_path / "status.json").write_text(json.dumps({"checkpoint": str(checkpoint), "manifest_sha256": "manifest-a"}))
    monkeypatch.setattr("seahorse.experiments.campaign_run.select_indices", lambda n, limit, seed: [0, 1])
    monkeypatch.setattr(forecast, "raw_context", lambda dataset, index, coord: {
        "history": {"times": np.asarray([.1, .2]), "locations": np.asarray([[0., 0.], [1., 1.]])},
        "next_time": .3, "next_location": np.asarray([.5, .5]), "observation_metadata": {}})
    def next_event(runner, history, *, n_samples, seed, device):
        calls.append(seed)
        result = behavior(seed)
        if isinstance(result, BaseException): raise result
        return result
    monkeypatch.setattr(forecast, "forecast_next", next_event)
    return runner, dataset, coord, manifest, job


def _draw(*, valid=True):
    return {"status": "ok", "backend": "fake", "valid": np.asarray([valid, valid]),
            "times": np.asarray([.35, .4]), "locations": np.asarray([[.4, .4], [.6, .6]])}


def test_predictive_timeout_resumes_completed_targets_without_recomputing(monkeypatch, tmp_path):
    calls = []
    runner, dataset, coord, manifest, job = _fixture(monkeypatch, tmp_path, calls,
        lambda seed: _draw() if seed == 7 else TimeoutError("scheduler limit"))
    with pytest.raises(TimeoutError):
        forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)
    progress = json.loads((tmp_path / "predictive_status.json").read_text())
    assert list(progress["rows"]) == ["0"]
    monkeypatch.setattr(forecast, "forecast_next", lambda *args, **kwargs: calls.append(kwargs["seed"]) or _draw())
    result = forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)
    assert result["predictive"]["successful_targets"] == 2
    assert calls == [7, 8, 8]


def test_predictive_progress_binding_mismatch_is_refused(monkeypatch, tmp_path):
    calls = []
    runner, dataset, coord, manifest, job = _fixture(monkeypatch, tmp_path, calls, lambda seed: _draw())
    forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)
    manifest["sha256"] = "manifest-b"
    with pytest.raises(ValueError, match="binding differs"):
        forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)


def test_predictive_progress_refuses_changed_checkpoint_contents(monkeypatch, tmp_path):
    calls = []
    runner, dataset, coord, manifest, job = _fixture(monkeypatch, tmp_path, calls, lambda seed: _draw())
    forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)
    (tmp_path / "model" / "checkpoint.ckpt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding differs"):
        forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)


def test_invalid_samples_remain_recorded_failure_on_resume(monkeypatch, tmp_path):
    calls = []
    runner, dataset, coord, manifest, job = _fixture(monkeypatch, tmp_path, calls,
        lambda seed: _draw(valid=seed != 7))
    result = forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)
    assert result["predictive"]["successful_targets"] == 1
    assert result["predictive"]["failures"][0]["target"] == 0
    before = list(calls)
    forecast.evaluate_forecasts(runner, dataset, coord, manifest, job, tmp_path)
    assert calls == before
