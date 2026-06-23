"""Researcher tutorial: build, register, fit, and benchmark a new STPP model."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from tutorial_utils import (
    load_tutorial_splits,
    plot_model_comparison,
    write_results_table_html,
    write_tutorial_dataset,
)


_CACHE_ROOT = Path(tempfile.gettempdir()) / "seahorse_tutorial_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE_ROOT / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(_CACHE_ROOT / "xdg")

from unified_stpp.runner import STPPRunner


def _runner_overrides(output_dir: Path, *, seed: int, epochs: int, batch_size: int) -> dict:
    return {
        "logging": {"out_dir": str(output_dir / "runs")},
        "training": {
            "device": "cpu",
            "seed": seed,
            "n_epochs": epochs,
            "batch_size": batch_size,
            "checkpoint_select": "best",
            "test_nll_space": "native",
        },
        "data": {
            "protocol": "raw",
            "normalize": False,
            "batch_size": batch_size,
            "num_workers": 0,
            "seed": seed,
            "adapter_kwargs": {"training_view": "full_sequence"},
        },
        "model": {"hidden_dim": 16},
    }


def _fit_demo_model(
    *,
    output_dir: Path,
    train: list[dict],
    val: list[dict],
    test: list[dict],
    seed: int,
    epochs: int,
    batch_size: int,
) -> dict:
    runner = STPPRunner.from_config_source(
        "demo_gru_gaussian",
        None,
        cli_values=_runner_overrides(
            output_dir,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        ),
    )
    result = runner.fit(train, val, test, dataset_id="tutorial_events")
    return {
        "model": "Demo GRU-Gaussian",
        "preset": "demo_gru_gaussian",
        "test_nll": float(result.test_nll),
        "mean_seq_nll": float(result.test_nll),
        "run_dir": str(result.run_dir),
        "note": "new tutorial model",
    }


def _run_tiny_benchmark(
    *,
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Path:
    from unified_stpp.benchmark import Benchmark
    from unified_stpp.config import BenchmarkConfig

    bench_dir = output_dir / "benchmark"
    splits = load_tutorial_splits(dataset_dir)
    benchmark_config = BenchmarkConfig(
        seeds=[int(seed)],
        n_workers=1,
        backend="sequential",
        protocol="raw",
        normalize=False,
        checkpoint_select="best",
        test_nll_space="native",
    )
    bench = Benchmark(
        presets=["poisson_gmm", "hawkes_gmm", "demo_gru_gaussian"],
        splits={"tutorial_events": (splits["train"], splits["val"], splits["test"])},
        config=benchmark_config,
        base_overrides={
            "model": {"hidden_dim": 16},
            "training": {
                "n_epochs": int(epochs),
                "batch_size": int(batch_size),
                "device": "cpu",
            },
            "data": {
                "num_workers": 0,
                "adapter_kwargs": {"training_view": "full_sequence"},
            },
        },
        out_dir=bench_dir,
        argv=[
            "bench",
            "--presets",
            "poisson_gmm",
            "hawkes_gmm",
            "demo_gru_gaussian",
            "--dataset",
            str(dataset_dir),
        ],
        splits_dir_str=str(dataset_dir),
        raw_overrides=[
            f"training.n_epochs={epochs}",
            f"training.batch_size={batch_size}",
            "training.device=cpu",
            "data.protocol=raw",
            "data.normalize=false",
            "model.hidden_dim=16",
        ],
    )
    table = bench.run()
    table.report(str(bench_dir), metric=benchmark_config.primary_metric)
    return bench_dir


def run_tutorial(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output).resolve()
    dataset_dir = output_dir / "data" / "tutorial_events"
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tutorial_dataset(
        dataset_dir,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
    )
    splits = load_tutorial_splits(dataset_dir)

    demo_row = _fit_demo_model(
        output_dir=output_dir,
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    rows = [demo_row]

    bench_dir = None
    if not args.skip_benchmark:
        bench_dir = _run_tiny_benchmark(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "demo_model_score.json").write_text(json.dumps(rows, indent=2))
    comparison_png = plot_model_comparison(
        rows,
        figure_dir / "demo_model_score.svg",
        title="New model smoke result",
    )
    comparison_html = write_results_table_html(
        rows,
        table_dir / "demo_model_score.html",
        title="Research model result",
    )
    summary = {
        "dataset_dir": str(dataset_dir),
        "demo_model": demo_row,
        "benchmark_dir": str(bench_dir) if bench_dir else None,
        "comparison_svg": str(comparison_png),
        "comparison_html": str(comparison_html),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/tutorials/building_a_research_model")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-val", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser


if __name__ == "__main__":
    print(json.dumps(run_tutorial(build_parser().parse_args()), indent=2))
