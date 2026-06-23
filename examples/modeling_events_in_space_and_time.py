"""End-to-end average-user tutorial: Modeling Events in Space and Time."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from tutorial_utils import (
    load_tutorial_splits,
    plot_eda_panel,
    plot_model_comparison,
    write_event_movie_html,
    write_results_table_html,
    write_tutorial_dataset,
)


_CACHE_ROOT = Path(tempfile.gettempdir()) / "seahorse_tutorial_plot_cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_CACHE_ROOT / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(_CACHE_ROOT / "xdg")

from unified_stpp import STPPEstimator


def _base_overrides(output_dir: Path, *, seed: int, batch_size: int) -> dict:
    return {
        "logging": {"out_dir": str(output_dir / "runs")},
        "training": {
            "device": "cpu",
            "seed": seed,
            "batch_size": batch_size,
            "checkpoint_select": "best",
        },
        "data": {
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": 0,
        },
    }


def _poisson_overrides(output_dir: Path, *, seed: int, batch_size: int) -> dict:
    cfg = _base_overrides(output_dir, seed=seed, batch_size=batch_size)
    cfg["data"].update({"protocol": "standard", "normalize": True})
    return cfg


def _auto_overrides(output_dir: Path, *, seed: int, batch_size: int, lookback: int) -> dict:
    cfg = _base_overrides(output_dir, seed=seed, batch_size=batch_size)
    cfg["data"].update(
        {
            "protocol": "raw",
            "normalize": False,
            "adapter_kwargs": {
                "training_view": "sliding_window",
                "lookback": lookback,
                "lookahead": 1,
            },
        }
    )
    cfg["model"] = {
        "hidden_dim": 16,
        "decoder": {
            "lookback": lookback,
            "lookahead": 1,
            "max_history": lookback,
            "n_prodnet": 1,
            "hidden_size": 16,
            "num_layers": 1,
            "temporal_mc_samples": 2,
        },
    }
    return cfg


def _fit_and_score(
    *,
    label: str,
    preset: str,
    overrides: dict,
    train: list[dict],
    val: list[dict],
    test: list[dict],
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    model = STPPEstimator(preset, config_overrides=overrides, device="cpu", seed=seed)
    model.fit(
        train,
        val,
        test,
        epochs=epochs,
        batch_size=batch_size,
        dataset_id="tutorial_events",
    )
    scores = model.evaluate(test, metric_profile="core")
    return {
        "model": label,
        "preset": preset,
        "test_nll": float(scores["test_nll"]),
        "mean_seq_nll": float(scores["mean_seq_nll"]),
        "run_dir": str(model.runner._run_dir),
        "note": "baseline" if preset == "poisson_gmm" else "history-aware neural STPP",
    }


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

    eda_path = plot_eda_panel(splits, figure_dir / "eda_panel.svg")
    movie_path = write_event_movie_html(splits["test"][0], figure_dir / "event_movie.html")

    rows = [
        _fit_and_score(
            label="Poisson-GMM",
            preset="poisson_gmm",
            overrides=_poisson_overrides(output_dir, seed=args.seed, batch_size=args.batch_size),
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    ]
    if not args.skip_auto:
        rows.append(
            _fit_and_score(
                label="AutoSTPP",
                preset="auto_stpp",
                overrides=_auto_overrides(
                    output_dir,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    lookback=args.lookback,
                ),
                train=splits["train"],
                val=splits["val"],
                test=splits["test"],
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        )

    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "model_scores.json").write_text(json.dumps(rows, indent=2))
    comparison_png = plot_model_comparison(
        rows,
        figure_dir / "model_comparison.svg",
        title="Held-out NLL on the tutorial event dataset",
    )
    comparison_html = write_results_table_html(
        rows,
        table_dir / "model_comparison.html",
        title="Model comparison",
    )

    summary = {
        "dataset_dir": str(dataset_dir),
        "eda_panel": str(eda_path),
        "event_movie": str(movie_path),
        "model_comparison_svg": str(comparison_png),
        "model_comparison_html": str(comparison_html),
        "scores": rows,
        "best_model": min(rows, key=lambda r: np.inf if np.isnan(r["test_nll"]) else r["test_nll"])[
            "model"
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/tutorials/modeling_events_in_space_and_time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--skip-auto", action="store_true", help="Run only the fast baseline.")
    return parser


if __name__ == "__main__":
    result = run_tutorial(build_parser().parse_args())
    print(json.dumps(result, indent=2))
