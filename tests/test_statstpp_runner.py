import numpy as np
import pytest

from seahorse.runner import STPPRunner, StatisticalSTPPRunner


def _sequences():
    return [{"t0": 0., "t1": 1., "times": [.2, .7], "locations": [[.2, .2], [.7, .7]]}]


def test_statistical_fit_save_to_new_directory_and_reload(tmp_path):
    runner = STPPRunner.from_config_source("homogeneous_poisson", None, cli_values={
        "model": {"support": [0., 1., 0., 1.]}, "data": {"normalize": False},
        "logging": {"out_dir": str(tmp_path / "runs")}, "statistical": {"method": "analytic"},
    })
    assert isinstance(runner, StatisticalSTPPRunner)
    result = runner.fit(_sequences(), _sequences(), _sequences())
    target = tmp_path / "portable"
    runner.save(target)
    restored = StatisticalSTPPRunner.load(target)
    assert restored.model.to_state() == runner.model.to_state()
    assert result.test_nll == restored.evaluate(_sequences())["mean_nll"]


def test_statistical_model_config_builds_declared_support():
    from seahorse.config import STPPConfig

    config = STPPConfig.from_source(
        preset="spatial_poisson",
        cli_values={"model": {"support": [0.0, 2.0, 0.0, 3.0]}},
    )
    model = config.model.build_model()
    assert tuple(model.support) == (0.0, 2.0, 0.0, 3.0)
    assert model.preset == "spatial_poisson"


def test_statistical_estimator_evaluates_and_samples_without_torch_modules(tmp_path):
    from seahorse.api import STPPEstimator

    estimator = STPPEstimator(
        "homogeneous_poisson",
        config_overrides={
            "model": {"support": [0.0, 1.0, 0.0, 1.0]},
            "data": {"normalize": False},
            "logging": {"out_dir": str(tmp_path / "runs")},
        },
        device="cpu",
    ).fit(_sequences(), _sequences(), _sequences())
    metrics = estimator.evaluate(_sequences())
    samples = estimator.predict_next(_sequences(), n_samples=3)
    assert np.isfinite(metrics["test_nll"])
    assert samples["next_times"].shape == (1, 3)
    assert samples["sampling_succeeded"].all()


def test_cli_predictive_metrics_load_statistical_checkpoint(tmp_path):
    """The public CLI must not call ``.to()`` or ``.eval()`` on a stat model."""
    import json
    import subprocess
    import sys

    runner = STPPRunner.from_config_source(
        "homogeneous_poisson", None,
        cli_values={"model": {"support": [0., 1., 0., 1.]},
                    "data": {"normalize": False},
                    "logging": {"out_dir": str(tmp_path / "runs")}},
    )
    runner.fit(_sequences(), _sequences(), _sequences())
    checkpoint = runner.save(tmp_path / "checkpoint")
    data_path = tmp_path / "test.jsonl"
    data_path.write_text(json.dumps(_sequences()[0]) + "\n")
    output = tmp_path / "metrics"
    completed = subprocess.run(
        [sys.executable, "-m", "seahorse", "evaluate", "metrics", "--run", str(checkpoint),
         "--data", str(data_path), "--metric-profile", "predictive", "--out", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "metrics.json").exists()


@pytest.mark.filterwarnings("ignore:the spatial mass varies.*:UserWarning")
def test_hawkes_source_mle_runner_round_trip_has_analytic_gaussian_nll(tmp_path):
    """Smoke the copied source MLE, runner checkpoint and prefix evaluator."""
    first = {"t0": 0., "t1": 10., "times": [.5, 1.1, 1.8, 2.8, 4., 5.1, 6.5, 8., 9.2],
             "locations": [[.2, .2], [.3, .4], [.5, .3], [.7, .8], [.8, .7], [1., 1.1], [1.3, 1.2], [1.5, 1.6], [1.7, 1.5]]}
    second = {"t0": 0., "t1": 10., "times": [.4, 1.6, 2.6, 3.7, 4.8, 6., 7.3, 8.6, 9.5],
              "locations": [[1.7, .2], [1.6, .4], [1.4, .5], [1.2, .7], [1.1, .8], [.9, 1.], [.7, 1.2], [.5, 1.4], [.3, 1.6]]}
    sequences = [first, second]
    empty = {"t0": 0., "t1": 2., "times": [], "locations": []}
    runner = STPPRunner.from_config_source(
        "hawkes_exp_gaussian", None,
        cli_values={
            "model": {"support": [0., 2., 0., 2.], "edge_correction": "renormalise", "quadrature_order": 32},
            "data": {"normalize": False},
            "statistical": {"max_iterations": 600, "optimizer": "Nelder-Mead"},
            "logging": {"out_dir": str(tmp_path / "runs")},
        },
    )
    result = runner.fit([*sequences, empty], sequences, sequences)
    restored = STPPRunner.load(runner.save(tmp_path / "checkpoint"))
    assert result.nll_kind == "exact"
    assert result.nll_footnote == ""
    assert result.test_nll_method == "analytic_continuous_next_event_density"
    assert result.extra_metrics["diagnostics"]["observation_exposure"] == 88.0
    assert np.isclose(result.test_nll, restored.evaluate(sequences)["mean_nll"])


def test_cli_fit_yaml_vectors_and_json_vector_overrides_round_trip(tmp_path):
    """Public CLI preserves support/grid instead of treating vectors as HPO choices."""
    import json
    import subprocess
    import sys

    config = tmp_path / "stat.yaml"
    config.write_text(
        """data:\n  normalize: false\nmodel:\n  preset: spatial_poisson\n  support: [0.0, 1.0, 0.0, 1.0]\n  grid: [2, 2]\ntraining: {}\nlogging: {}\n"""
    )
    sequence = _sequences()[0]
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(sequence) + "\n")
    checkpoint = tmp_path / "checkpoint"
    fitted = subprocess.run(
        [sys.executable, "-m", "seahorse", "fit", "--config", str(config),
         "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
         "--test", str(tmp_path / "test.jsonl"), "--save", str(checkpoint), "--override",
         "model.support=[0,1,0,1]", "model.grid=[2,2]"],
        check=False, capture_output=True, text=True,
    )
    assert fitted.returncode == 0, fitted.stderr
    restored = STPPRunner.load(checkpoint)
    assert tuple(restored.model.support) == (0.0, 1.0, 0.0, 1.0)
    assert tuple(restored.model.grid) == (2, 2)
    output = tmp_path / "metrics"
    evaluated = subprocess.run(
        [sys.executable, "-m", "seahorse", "evaluate", "metrics", "--run", str(checkpoint),
         "--data", str(tmp_path / "test.jsonl"), "--metric-profile", "predictive", "--out", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert (output / "metrics.json").exists()
