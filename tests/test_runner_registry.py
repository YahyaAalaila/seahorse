from seahorse.models.configs import ConfigRegistry
from seahorse.runner import STPPRunner, StatisticalSTPPRunner

def test_registered_statistical_presets_route_without_lightning():
    assert ConfigRegistry.runner_kind("homogeneous_poisson") == "statstpp"
    runner = STPPRunner.from_config_source("homogeneous_poisson", None, cli_values={"model": {"support": [0, 1, 0, 1]}})
    assert isinstance(runner, StatisticalSTPPRunner)


def test_mixed_benchmark_routes_neural_and_statistical_cells(tmp_path):
    """A mixed grid uses per-preset configs; support never leaks into neural GMM."""
    from seahorse.benchmark import Benchmark
    from seahorse.config import BenchmarkConfig, STPPConfig

    seqs = [{"t0": 0.0, "t1": 1.0, "times": [.1, .25, .45, .7, .9],
             "locations": [[.1, .2], [.2, .4], [.4, .3], [.7, .7], [.8, .6]]}]
    neural = STPPConfig.from_source(
        preset="poisson_gmm",
        cli_values={"data": {"normalize": False}, "training": {"n_epochs": 1},
                    "logging": {"out_dir": str(tmp_path / "neural")}},
    )
    statistical = STPPConfig.from_source(
        preset="homogeneous_poisson",
        cli_values={"model": {"support": [0., 1., 0., 1.]}, "data": {"normalize": False},
                    "logging": {"out_dir": str(tmp_path / "statistical")}},
    )
    benchmark = Benchmark(
        presets=["poisson_gmm", "homogeneous_poisson"],
        splits={"toy": (seqs, seqs, seqs)},
        config=BenchmarkConfig(seeds=[3], n_workers=1, backend="joblib"),
        hpo_configs={"poisson_gmm": neural, "homogeneous_poisson": statistical},
    )
    table = benchmark.run()
    assert {run.preset for run in table.runs} == {"poisson_gmm", "homogeneous_poisson"}
    assert {run.test_nll_contexts for run in table.runs} == {4}
    assert {run.test_nll_scored_contexts for run in table.runs} == {4}
    assert {run.nll_report_space for run in table.runs} == {"raw"}


def test_statistical_hpo_trial_does_not_construct_torch_data_module(tmp_path, monkeypatch):
    """Exercise the Ray trial closure with a synchronous fake Tune backend."""
    import sys
    import types

    from seahorse.benchmark.hpo import run_hpo
    from seahorse.config import TuningConfig

    monkeypatch.setattr("seahorse.benchmark.hpo._current_rss_mb", lambda: 1.0)
    monkeypatch.setattr("seahorse.benchmark.hpo._peak_rss_mb", lambda: 1.0)
    reports = []
    fake_tune = types.ModuleType("ray.tune")
    fake_tune.choice = lambda values: values[0]
    fake_tune.uniform = lambda lo, hi: lo
    fake_tune.loguniform = lambda lo, hi: lo
    fake_tune.randint = lambda lo, hi: lo
    fake_tune.report = reports.append

    def run(trainable, *, config, **_kwargs):
        trainable({key: value for key, value in config.items()})
        return types.SimpleNamespace(best_config=config)

    fake_tune.run = run
    fake_ray = types.ModuleType("ray")
    fake_ray.tune = fake_tune
    fake_ray.is_initialized = lambda: True
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.tune", fake_tune)

    seqs = [{"t0": 0.0, "t1": 1.0, "times": [.2, .7], "locations": [[.2, .2], [.7, .7]]}]
    raw = {
        "model": {"preset": "homogeneous_poisson", "support": [0., 1., 0., 1.], "alpha": [0.1, 0.2]},
        "data": {"normalize": False, "seed": 5},
        "training": {"n_epochs": 1},
        "logging": {"out_dir": str(tmp_path / "runs")},
    }
    result = run_hpo(raw, TuningConfig(n_trials=1, scheduler="none", search_alg="random"),
                     train_seqs=seqs, val_seqs=seqs)
    assert result.model.preset == "homogeneous_poisson"
    assert reports and reports[0]["val_metric_key"] == "full_window_nll"
