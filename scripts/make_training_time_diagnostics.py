"""
Build appendix-ready wall-clock training-time diagnostics.

The reported quantity is successful-run wall-clock training time from
run_result.json (`train_time_sec`). Failed/cancelled attempts are not included.
If more than one completed run exists for a (suite, config, preset, seed) cell,
the run with the best reported test NLL is selected, matching the result-table
use case more closely than blindly taking the newest run.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unified_stpp_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOTS = [
    ROOT / "runs" / "hawkesnest_campaigns" / "suite3_entanglement",
    ROOT / "runs" / "hawkesnest_campaigns" / "suite4_heterogeneity",
    ROOT / "runs" / "exp1",
]
OUT = ROOT / "runs" / "local_eval_analysis" / "training_time_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

SUITE_LABELS = {
    "suite3_entanglement": "Suite 3",
    "suite4_heterogeneity": "Suite 4",
    "realdata": "Real Data",
}
CONFIG_ORDER = {
    "suite3_entanglement": ["L0", "L1", "L2", "L3"],
    "suite4_heterogeneity": ["H0", "H1", "H2", "H3"],
    "realdata": ["covid-stpp", "earthquakes-stpp", "citibike-stpp"],
}
CONFIG_LABELS = {
    "covid-stpp": "COVID",
    "earthquakes-stpp": "Earthquakes",
    "citibike-stpp": "Citibike",
}

PRESET_LABELS = {
    "auto_stpp": "AutoSTPP",
    "deep_stpp": "DeepSTPP",
    "diffusion_stpp": "DiffusionSTPP",
    "smash": "SMASH",
    "nsmpp": "NSMPP",
    "rmtpp_gmm": "RMTPP+GMM",
    "thp_gmm": "THP+GMM",
    "njsde": "NJSDE",
    "neural_attncnf": "Neural AttnCNF",
    "neural_jumpcnf": "Neural JumpCNF",
    "hawkes_gmm": "Hawkes+GMM",
    "hawkes_cnf": "Hawkes+CNF",
    "hawkes_tvcnf": "Hawkes+TV-CNF",
    "poisson_gmm": "Poisson+GMM",
    "poisson_cnf": "Poisson+CNF",
    "poisson_tvcnf": "Poisson+TV-CNF",
    "selfcorrecting_gmm": "SelfCorr+GMM",
    "selfcorrecting_cnf": "SelfCorr+CNF",
    "selfcorrecting_tvcnf": "SelfCorr+TV-CNF",
}

PAPER_PRESETS = [
    "auto_stpp",
    "deep_stpp",
    "diffusion_stpp",
    "smash",
    "nsmpp",
    "rmtpp_gmm",
    "thp_gmm",
    "njsde",
    "neural_attncnf",
    "neural_jumpcnf",
]

MODEL_COLORS = {
    "auto_stpp": "#1b9e77",
    "deep_stpp": "#d95f02",
    "diffusion_stpp": "#7570b3",
    "smash": "#e7298a",
    "nsmpp": "#66a61e",
    "rmtpp_gmm": "#e6ab02",
    "thp_gmm": "#a6761d",
    "njsde": "#1f78b4",
    "neural_attncnf": "#b2df8a",
    "neural_jumpcnf": "#fb9a99",
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _suite_from_path(path: Path) -> str | None:
    parts = set(path.parts)
    if "suite3_entanglement" in parts:
        return "suite3_entanglement"
    if "suite4_heterogeneity" in parts:
        return "suite4_heterogeneity"
    if "exp1" in parts:
        return "realdata"
    return None


def _load_run_results() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for root in RUN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("run_result.json"):
            suite = _suite_from_path(path)
            if suite is None:
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            train_time = _as_float(data.get("train_time_sec"))
            if train_time is None:
                continue
            cfg = str(data.get("dataset_id") or "")
            preset = str(data.get("preset") or "")
            seed = data.get("seed")
            if not cfg or not preset or seed is None:
                continue
            eff = data.get("effective_config") or {}
            training = eff.get("training") or {}
            records.append(
                {
                    "suite": suite,
                    "config_id": cfg,
                    "preset": preset,
                    "model": PRESET_LABELS.get(preset, preset),
                    "seed": int(seed),
                    "train_time_sec": train_time,
                    "train_time_min": train_time / 60.0,
                    "train_time_hr": train_time / 3600.0,
                    "test_nll": _as_float(data.get("test_nll")),
                    "n_params": _as_float(data.get("n_params")),
                    "n_epochs": _as_float(training.get("n_epochs")),
                    "batch_size": _as_float(training.get("batch_size")),
                    "device": training.get("device"),
                    "precision": training.get("precision"),
                    "lr_schedule": training.get("lr_schedule"),
                    "run_result_path": str(path.relative_to(ROOT)),
                    "run_dir": data.get("run_dir"),
                }
            )
    return pd.DataFrame.from_records(records)


def _select_cells(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    sort = raw.copy()
    sort["_test_nll_sort"] = sort["test_nll"].fillna(float("inf"))
    sort["_path_sort"] = sort["run_result_path"]
    attempts = (
        sort.groupby(["suite", "config_id", "preset", "seed"])
        .size()
        .rename("n_completed_attempts")
        .reset_index()
    )
    selected = (
        sort.sort_values(
            ["suite", "config_id", "preset", "seed", "_test_nll_sort", "_path_sort"]
        )
        .groupby(["suite", "config_id", "preset", "seed"], as_index=False)
        .first()
        .drop(columns=["_test_nll_sort", "_path_sort"])
    )
    return selected.merge(attempts, on=["suite", "config_id", "preset", "seed"], how="left")


def _summarize(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_cell = (
        selected.groupby(["suite", "config_id", "preset", "model"], as_index=False)
        .agg(
            train_time_hr_mean=("train_time_hr", "mean"),
            train_time_hr_std=("train_time_hr", "std"),
            train_time_hr_median=("train_time_hr", "median"),
            train_time_hr_min=("train_time_hr", "min"),
            train_time_hr_max=("train_time_hr", "max"),
            n_seeds=("seed", "nunique"),
            n_params_mean=("n_params", "mean"),
            n_completed_attempts=("n_completed_attempts", "sum"),
        )
    )
    by_model = (
        selected.groupby(["suite", "preset", "model"], as_index=False)
        .agg(
            train_time_hr_mean=("train_time_hr", "mean"),
            train_time_hr_std=("train_time_hr", "std"),
            train_time_hr_median=("train_time_hr", "median"),
            train_time_hr_min=("train_time_hr", "min"),
            train_time_hr_max=("train_time_hr", "max"),
            n_runs=("train_time_hr", "count"),
            n_configs=("config_id", "nunique"),
            n_seeds=("seed", "nunique"),
            n_params_mean=("n_params", "mean"),
        )
    )
    return by_cell, by_model


def _setup_mpl() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 600})]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _plot_suite_bars(by_model: pd.DataFrame, suite: str) -> pd.DataFrame:
    sub = by_model[(by_model["suite"] == suite) & (by_model["preset"].isin(PAPER_PRESETS))].copy()
    order = [p for p in PAPER_PRESETS if p in set(sub["preset"])]
    sub["preset_order"] = sub["preset"].map({p: i for i, p in enumerate(order)})
    sub = sub.sort_values("preset_order")

    fig, ax = plt.subplots(figsize=(7.0, 2.8), constrained_layout=True)
    xs = range(len(sub))
    colors = [MODEL_COLORS.get(p, "#808080") for p in sub["preset"]]
    yerr = sub["train_time_hr_std"].fillna(0.0).to_numpy()
    ax.bar(xs, sub["train_time_hr_mean"], yerr=yerr, color=colors, alpha=0.9, capsize=2)
    ax.set_yscale("log")
    ax.set_ylabel("Training time per successful run (hours, log)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(sub["model"], rotation=35, ha="right")
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.text(0.01, 0.96, SUITE_LABELS[suite], transform=ax.transAxes, ha="left", va="top")
    _save(fig, f"{suite}_training_time_by_model")
    plotted = sub[
        [
            "suite",
            "preset",
            "model",
            "train_time_hr_mean",
            "train_time_hr_std",
            "train_time_hr_median",
            "n_runs",
            "n_configs",
        ]
    ].copy()
    plotted.insert(0, "figure", f"{suite}_training_time_by_model")
    return plotted


def _plot_config_lines(by_cell: pd.DataFrame, suite: str) -> pd.DataFrame:
    sub = by_cell[(by_cell["suite"] == suite) & (by_cell["preset"].isin(PAPER_PRESETS))].copy()
    configs = CONFIG_ORDER[suite]
    order = [p for p in PAPER_PRESETS if p in set(sub["preset"])]
    fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
    plotted: list[dict[str, Any]] = []
    for preset in order:
        psub = sub[sub["preset"] == preset].copy()
        psub["x"] = psub["config_id"].map({c: i for i, c in enumerate(configs)})
        psub = psub.dropna(subset=["x"]).sort_values("x")
        if psub.empty:
            continue
        ax.errorbar(
            psub["x"],
            psub["train_time_hr_mean"],
            yerr=psub["train_time_hr_std"].fillna(0.0),
            label=PRESET_LABELS.get(preset, preset),
            color=MODEL_COLORS.get(preset, "#808080"),
            lw=1.0,
            marker="o",
            ms=3.0,
            capsize=2,
        )
        for _, row in psub.iterrows():
            plotted.append(
                {
                    "figure": f"{suite}_training_time_by_config",
                    "suite": suite,
                    "config_id": row["config_id"],
                    "preset": preset,
                    "model": PRESET_LABELS.get(preset, preset),
                    "train_time_hr_mean": row["train_time_hr_mean"],
                    "train_time_hr_std": row["train_time_hr_std"],
                    "n_seeds": row["n_seeds"],
                }
            )
    ax.set_yscale("log")
    ax.set_ylabel("Training time per successful run (hours, log)")
    ax.set_xlabel("Configuration")
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs])
    ax.grid(axis="y", lw=0.4, alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False)
    ax.text(0.01, 0.96, SUITE_LABELS[suite], transform=ax.transAxes, ha="left", va="top")
    _save(fig, f"{suite}_training_time_by_config")
    return pd.DataFrame.from_records(plotted)


def _make_markdown(by_model: pd.DataFrame, selected: pd.DataFrame) -> None:
    display = by_model[by_model["preset"].isin(PAPER_PRESETS)].copy()
    display["mean_hr"] = display["train_time_hr_mean"].map(lambda x: f"{x:.2f}")
    display["std_hr"] = display["train_time_hr_std"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    display["median_hr"] = display["train_time_hr_median"].map(lambda x: f"{x:.2f}")
    display["suite"] = display["suite"].map(SUITE_LABELS)
    display = display.sort_values(["suite", "train_time_hr_mean"])
    cols = ["suite", "model", "mean_hr", "std_hr", "median_hr", "n_runs", "n_configs"]
    table = display[cols].rename(
        columns={
            "suite": "Suite",
            "model": "Model",
            "mean_hr": "Mean h",
            "std_hr": "Std h",
            "median_hr": "Median h",
            "n_runs": "Runs",
            "n_configs": "Configs",
        }
    )
    def _markdown_table(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        rows = frame.astype(str).values.tolist()
        out = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        out.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(out)

    lines = [
        "# Training-Time Diagnostics",
        "",
        "Wall-clock training time is read from successful `run_result.json` files (`train_time_sec`).",
        "Failed/cancelled attempts are excluded. When multiple successful attempts exist for the same",
        "`suite/config/model/seed`, the attempt with the best reported test NLL is selected.",
        "",
        "These numbers are suitable as *successful-run wall-clock diagnostics*, not as total cluster-cost accounting.",
        "",
        _markdown_table(table),
        "",
        "## Coverage",
        "",
        f"- Selected successful runs: {len(selected)}",
        f"- Suites: {', '.join(sorted(selected['suite'].unique()))}",
        "- Output figures use a log y-axis because runtimes span seconds to many hours.",
    ]
    (OUT / "training_time_summary.md").write_text("\n".join(lines) + "\n")


def _make_inventory(plotted_frames: list[pd.DataFrame]) -> None:
    lines = [
        "# Figure Inventory",
        "",
        "Generated by `scripts/make_training_time_diagnostics.py`.",
        "",
        "## Figures",
    ]
    for stem in [
        "suite3_entanglement_training_time_by_model",
        "suite3_entanglement_training_time_by_config",
        "suite4_heterogeneity_training_time_by_model",
        "suite4_heterogeneity_training_time_by_config",
        "realdata_training_time_by_model",
        "realdata_training_time_by_config",
    ]:
        lines.append(f"- `{stem}.pdf/.svg/.png`")
    lines += [
        "",
        "## CSVs",
        "",
        "- `training_time_by_run_selected.csv`",
        "- `training_time_by_cell.csv`",
        "- `training_time_by_model.csv`",
        "- `training_time_plotted.csv`",
        "",
        "## Scope Note",
        "",
        "The figures compare successful completed training runs only. They do not include failed attempts,",
        "queueing time, evaluation time, or total time spent across restarted jobs.",
    ]
    (OUT / "figure_inventory.md").write_text("\n".join(lines) + "\n")
    plotted = pd.concat([p for p in plotted_frames if not p.empty], ignore_index=True)
    plotted.to_csv(OUT / "training_time_plotted.csv", index=False)


def main() -> int:
    _setup_mpl()
    raw = _load_run_results()
    if raw.empty:
        raise SystemExit("No run_result.json files with train_time_sec found.")
    selected = _select_cells(raw)
    by_cell, by_model = _summarize(selected)

    raw.to_csv(OUT / "training_time_by_run_all_completed.csv", index=False)
    selected.to_csv(OUT / "training_time_by_run_selected.csv", index=False)
    by_cell.to_csv(OUT / "training_time_by_cell.csv", index=False)
    by_model.to_csv(OUT / "training_time_by_model.csv", index=False)

    plotted_frames = []
    for suite in CONFIG_ORDER:
        plotted_frames.append(_plot_suite_bars(by_model, suite))
        plotted_frames.append(_plot_config_lines(by_cell, suite))
    _make_markdown(by_model, selected)
    _make_inventory(plotted_frames)
    print(f"Wrote training-time diagnostics to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
