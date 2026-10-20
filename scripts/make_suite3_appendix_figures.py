"""
Appendix figure generator for Suite 3 training-budget summary.

Source files (both from suite3_paper_artifacts/):
  suite3_training_budget_summary_table.csv   — primary; all three figures draw from here
  suite3_budget_recovery_core_plotted.csv    — not used directly; referenced for context

Outputs to suite3_paper_artifacts/:
  appendix_budget_stability_heatmap.{svg,png}
  appendix_budget_delta_barchart.{svg,png}
  appendix_best_checkpoint_location.{svg,png}
"""

from __future__ import annotations

import os
import pathlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unified_stpp_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "runs" / "local_eval_analysis" / "suite3_paper_artifacts"
CSV = ARTIFACTS / "suite3_training_budget_summary_table.csv"

MODELS_OF_INTEREST = [
    "auto_stpp", "deep_stpp", "nsmpp",
    "neural_cond_gmm", "neural_jumpcnf", "neural_attncnf",
]
MODEL_LABELS: dict[str, str] = {
    "auto_stpp":       "AutoSTPP",
    "deep_stpp":       "DeepSTPP",
    "nsmpp":           "NSMPP",
    "neural_cond_gmm": "NJSDE",
    "neural_jumpcnf":  "Jump-CNF",
    "neural_attncnf":  "Attn-CNF",
}
LEVELS = ["L0", "L1", "L2", "L3"]
LEVEL_LABELS: dict[str, str] = {
    "L0": "L0 (easy)", "L1": "L1", "L2": "L2", "L3": "L3 (hard)",
}

# Muted publication palette — one colour per model
MODEL_COLORS: dict[str, str] = {
    "auto_stpp":       "#5778a4",
    "deep_stpp":       "#e49444",
    "nsmpp":           "#d1615d",
    "neural_cond_gmm": "#85b6b2",
    "neural_jumpcnf":  "#6a9f68",
    "neural_attncnf":  "#b07aa1",
}

# Muted palette — one colour per level
LEVEL_COLORS: dict[str, str] = {
    "L0": "#5778a4",
    "L1": "#85b6b2",
    "L2": "#e49444",
    "L3": "#d1615d",
}

RC = {
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
}


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(ARTIFACTS / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    return df[df["preset"].isin(MODELS_OF_INTEREST)].copy()


# ── Figure 1: Heatmap — delta_final_minus_best across models × levels ────────

def fig_stability_heatmap(df: pd.DataFrame) -> None:
    pivot = (
        df.pivot(index="preset", columns="level", values="delta_final_minus_best")
        .reindex(index=MODELS_OF_INTEREST, columns=LEVELS)
    )
    mat = pivot.values.astype(float)
    row_labels = [MODEL_LABELS[p] for p in MODELS_OF_INTEREST]
    col_labels = [LEVEL_LABELS[l] for l in LEVELS]

    # cap colormap at 95th-percentile so a single outlier doesn't wash out detail
    vmax = max(0.04, float(np.nanpercentile(mat, 95)))
    cmap = matplotlib.colormaps["YlOrRd"]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

        for i in range(len(MODELS_OF_INTEREST)):
            for j in range(len(LEVELS)):
                val = mat[i, j]
                if np.isnan(val):
                    ax.add_patch(mpatches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="////", edgecolor="#cccccc", linewidth=0,
                    ))
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=12, color="#999999")
                else:
                    text_color = "white" if val > 0.62 * vmax else "#1a1a1a"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, color=text_color, fontweight="semibold")

        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(MODELS_OF_INTEREST)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Entanglement level")

        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.025)
        cbar.set_label("Δ NLL  (final − best checkpoint)", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        fig.tight_layout()

    _save(fig, "appendix_budget_stability_heatmap")
    print("  [1] appendix_budget_stability_heatmap.{svg,png}")


# ── Figure 2: Grouped bar chart — delta_final_minus_best per model per level ─

def fig_delta_barchart(df: pd.DataFrame) -> None:
    n_models = len(MODELS_OF_INTEREST)
    n_levels = len(LEVELS)
    group_w = 0.78
    bar_w = group_w / n_levels
    x = np.arange(n_models)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        for j, level in enumerate(LEVELS):
            sub = df[df["level"] == level].set_index("preset")
            vals = [
                float(sub.loc[p, "delta_final_minus_best"]) if p in sub.index else float("nan")
                for p in MODELS_OF_INTEREST
            ]
            offset = (j - (n_levels - 1) / 2) * bar_w
            ax.bar(
                x + offset, vals, width=bar_w * 0.88,
                color=LEVEL_COLORS[level], label=LEVEL_LABELS[level],
                edgecolor="white", linewidth=0.6,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[p] for p in MODELS_OF_INTEREST],
            rotation=22, ha="right",
        )
        ax.set_ylabel("Δ NLL  (final − best checkpoint)")
        ax.set_ylim(bottom=0)
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.legend(title="Level", loc="upper right", framealpha=0.92, edgecolor="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

    _save(fig, "appendix_budget_delta_barchart")
    print("  [2] appendix_budget_delta_barchart.{svg,png}")


# ── Figure 3: Dot plot — best_at_pct (early-stopping benefit) ────────────────

def fig_best_checkpoint(df: pd.DataFrame) -> None:
    n_models = len(MODELS_OF_INTEREST)
    n_levels = len(LEVELS)
    group_w = 0.70
    spacing = group_w / n_levels
    x = np.arange(n_models)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        for j, level in enumerate(LEVELS):
            sub = df[df["level"] == level].set_index("preset")
            vals = [
                float(sub.loc[p, "best_at_pct"]) if p in sub.index else float("nan")
                for p in MODELS_OF_INTEREST
            ]
            offset = (j - (n_levels - 1) / 2) * spacing
            ax.scatter(
                x + offset, vals,
                color=LEVEL_COLORS[level], label=LEVEL_LABELS[level],
                s=90, zorder=3, edgecolors="white", linewidths=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[p] for p in MODELS_OF_INTEREST],
            rotation=22, ha="right",
        )
        ax.set_ylabel("Best checkpoint location (% of training budget)")
        ax.set_ylim(0, 112)
        ax.axhline(100, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.legend(title="Level", loc="lower right", framealpha=0.92, edgecolor="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

    _save(fig, "appendix_best_checkpoint_location")
    print("  [3] appendix_best_checkpoint_location.{svg,png}")


if __name__ == "__main__":
    df = _load()
    print(f"Loaded {len(df)} rows for {df['preset'].nunique()} models across levels: "
          f"{sorted(df['level'].unique())}")
    fig_stability_heatmap(df)
    fig_delta_barchart(df)
    fig_best_checkpoint(df)
    print(f"\nAll figures written to:\n  {ARTIFACTS}")
