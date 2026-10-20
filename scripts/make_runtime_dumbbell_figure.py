"""Create appendix runtime dumbbell figure from successful training runs."""

from __future__ import annotations

import os
import pathlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unified_stpp_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "local_eval_analysis" / "training_time_diagnostics"
INPUT = OUT_DIR / "training_time_by_run_selected.csv"

OUT_STEM = OUT_DIR / "appendix_runtime_dumbbell"
PLOTTED = OUT_DIR / "appendix_runtime_dumbbell_plotted.csv"
CAPTION = OUT_DIR / "appendix_runtime_dumbbell_caption.txt"

SUITES = {
    "realdata": "Real data",
    "suite3_entanglement": "Suite 3",
}

NON_FACTORIZED = [
    ("thp_gmm", "THP"),
    ("rmtpp_gmm", "RMTPP"),
    ("diffusion_stpp", "DSTPP"),
    ("smash", "SMASH"),
    ("deep_stpp", "DeepSTPP"),
    ("nsmpp", "NSMPP"),
    ("auto_stpp", "AutoSTPP"),
    ("neural_cond_gmm", "NJSDE"),
    ("neural_attncnf", "Attn-CNF"),
    ("neural_jumpcnf", "Jump-CNF"),
]

FACTORIZED = {
    "poisson_family": {
        "label": "Poisson",
        "presets": ["poisson_gmm", "poisson_cnf", "poisson_tvcnf"],
    },
    "hawkes_family": {
        "label": "Hawkes",
        "presets": ["hawkes_gmm", "hawkes_cnf", "hawkes_tvcnf"],
    },
    "selfcorrecting_family": {
        "label": "Self-correcting",
        "presets": [
            "selfcorrecting_gmm",
            "selfcorrecting_cnf",
            "selfcorrecting_tvcnf",
        ],
    },
}


def _median_or_nan(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.median())


def _summarize_nonfactorized(df: pd.DataFrame, preset: str, label: str) -> list[dict]:
    rows = []
    for suite, suite_label in SUITES.items():
        sub = df[(df["suite"] == suite) & (df["preset"] == preset)]
        rows.append(
            {
                "row_id": preset,
                "model": label,
                "suite": suite,
                "suite_label": suite_label,
                "median_hr": _median_or_nan(sub["train_time_hr"]),
                "n_runs": int(len(sub)),
                "aggregation": "preset median over successful runs",
            }
        )
    return rows


def _summarize_factorized(df: pd.DataFrame, family_id: str, spec: dict) -> list[dict]:
    rows = []
    presets = spec["presets"]
    for suite, suite_label in SUITES.items():
        head_medians = []
        n_runs = 0
        for preset in presets:
            sub = df[(df["suite"] == suite) & (df["preset"] == preset)]
            n_runs += int(len(sub))
            med = _median_or_nan(sub["train_time_hr"])
            if np.isfinite(med):
                head_medians.append(med)
        rows.append(
            {
                "row_id": family_id,
                "model": spec["label"],
                "suite": suite,
                "suite_label": suite_label,
                "median_hr": float(np.mean(head_medians)) if head_medians else float("nan"),
                "n_runs": n_runs,
                "aggregation": "mean of spatial-head medians",
            }
        )
    return rows


def build_plot_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["suite"].isin(SUITES)].copy()
    rows: list[dict] = []
    for preset, label in NON_FACTORIZED:
        rows.extend(_summarize_nonfactorized(df, preset, label))
    for family_id, spec in FACTORIZED.items():
        rows.extend(_summarize_factorized(df, family_id, spec))

    long = pd.DataFrame(rows)
    long["include_in_figure"] = long.groupby("row_id")["median_hr"].transform(
        lambda s: bool(s.notna().any())
    )
    long["availability_note"] = np.where(
        long["include_in_figure"],
        "",
        "unavailable: no successful timing rows",
    )
    figure_long = long[long["include_in_figure"]].copy()

    order_rows = []
    for row_id, sub in figure_long.groupby("row_id", sort=False):
        vals = sub["median_hr"].dropna().to_numpy(dtype=float)
        overall = float(np.median(vals)) if len(vals) else float("inf")
        order_rows.append(
            {
                "row_id": row_id,
                "overall_median_hr": overall,
                "model": str(sub["model"].iloc[0]),
            }
        )
    order = pd.DataFrame(order_rows).sort_values(
        ["overall_median_hr", "model"], na_position="last"
    )
    long["plot_order"] = long["row_id"].map(
        {row_id: idx for idx, row_id in enumerate(order["row_id"])}
    )
    long = long.sort_values(["plot_order", "suite"]).reset_index(drop=True)
    return long, order


def make_figure(long: pd.DataFrame, order: pd.DataFrame) -> None:
    plot_long = long[long["include_in_figure"]].copy()
    labels = order["model"].tolist()
    row_ids = order["row_id"].tolist()
    y = np.arange(len(row_ids))

    colors = {
        "realdata": "#4C78A8",
        "suite3_entanglement": "#D55E00",
    }
    markers = {
        "realdata": "o",
        "suite3_entanglement": "s",
    }

    finite = plot_long["median_hr"].replace([np.inf, -np.inf], np.nan).dropna()
    xmin = max(1e-3, float(finite.min()) * 0.62)
    xmax = float(finite.max()) * 1.45

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig_h = max(4.0, 0.32 * len(row_ids) + 0.95)
    fig, ax = plt.subplots(figsize=(5.4, fig_h))

    for yi, row_id in enumerate(row_ids):
        sub = plot_long[plot_long["row_id"] == row_id].set_index("suite")
        real = sub.loc["realdata", "median_hr"] if "realdata" in sub.index else np.nan
        suite3 = (
            sub.loc["suite3_entanglement", "median_hr"]
            if "suite3_entanglement" in sub.index
            else np.nan
        )
        if np.isfinite(real) and np.isfinite(suite3):
            ax.plot([real, suite3], [yi, yi], color="#BDBDBD", lw=1.1, zorder=1)

        for suite in SUITES:
            if suite not in sub.index:
                continue
            value = sub.loc[suite, "median_hr"]
            if not np.isfinite(value):
                continue
            ax.scatter(
                value,
                yi,
                s=42,
                marker=markers[suite],
                color=colors[suite],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label=SUITES[suite] if yi == 0 else None,
            )

    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.grid(axis="x", which="major", color="#E2E2E2", lw=0.7)
    ax.grid(axis="x", which="minor", color="#EFEFEF", lw=0.35)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncols=2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.34, right=0.985, top=0.91, bottom=0.08)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_caption() -> None:
    CAPTION.write_text(
        "Median successful-run wall-clock training time on real datasets and "
        "HawkesNest Suite 3. The x-axis is logarithmic. Factorized temporal "
        "baselines are aggregated into Poisson, Hawkes, and self-correcting "
        "families by averaging the median runtimes over the GMM, CNF, and TV-CNF "
        "spatial heads. Run counts are omitted from the figure body and retained "
        "in the plotted-data CSV.\n",
        encoding="utf-8",
    )


def main() -> None:
    df = pd.read_csv(INPUT)
    long, order = build_plot_data(df)
    long.to_csv(PLOTTED, index=False)
    make_figure(long, order)
    write_caption()
    print(f"wrote {OUT_STEM}.pdf/.svg/.png")
    print(f"wrote {PLOTTED}")
    print(f"wrote {CAPTION}")


if __name__ == "__main__":
    main()
