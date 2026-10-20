"""
Two-panel appendix heatmap for Suite 3 training-budget summary.

Panel A — best_at_pct: where in training the best checkpoint was found.
Panel B — delta_final_minus_best: NLL degradation from best to final epoch.

Source file:
  runs/local_eval_analysis/suite3_paper_artifacts/suite3_training_budget_summary_table.csv

Outputs to the same directory:
  appendix_two_panel_heatmap.{pdf,svg,png}
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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "runs" / "local_eval_analysis" / "suite3_paper_artifacts"
CSV = ARTIFACTS / "suite3_training_budget_summary_table.csv"

# ── display config ─────────────────────────────────────────────────────────────
PRESET_ORDER = [
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
LEVEL_LABELS = ["L0", "L1", "L2", "L3"]

RC = {
    "font.family":     "sans-serif",
    "font.size":       13,
    "axes.labelsize":  13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi":      150,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _pivot(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return (n_models, n_levels) float array; NaN where data is absent."""
    p = (
        df.pivot(index="preset", columns="level", values=col)
        .reindex(index=PRESET_ORDER, columns=LEVELS)
    )
    return p.values.astype(float)


def _draw_heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt_fn,           # (float) -> str
    xlabel: str,
    show_yticklabels: bool,
    panel_tag: str,
) -> matplotlib.image.AxesImage:
    n_rows, n_cols = mat.shape
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    # thin white cell borders
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # major ticks → labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(LEVEL_LABELS, fontsize=12)
    ax.set_yticks(range(n_rows))
    row_labels = [MODEL_LABELS[p] for p in PRESET_ORDER]
    ax.set_yticklabels(row_labels if show_yticklabels else [""] * n_rows, fontsize=12)
    ax.tick_params(which="major", length=0)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=6)

    # panel tag (A / B) in top-left corner
    ax.text(-0.18 if show_yticklabels else -0.06, 1.04, panel_tag,
            transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")

    # cell annotations
    norm_val = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                # hatching + em-dash for missing cells
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, facecolor="#e8e8e8",
                    hatch="////", edgecolor="#c0c0c0", linewidth=0,
                ))
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=13, color="#999999")
            else:
                # choose white or dark text based on luminance of background
                bg_rgba = matplotlib.colormaps[cmap](norm_val(val))
                luminance = 0.299 * bg_rgba[0] + 0.587 * bg_rgba[1] + 0.114 * bg_rgba[2]
                text_color = "#1a1a1a" if luminance > 0.50 else "white"
                ax.text(j, i, fmt_fn(val), ha="center", va="center",
                        fontsize=11, color=text_color, fontweight="semibold")

    # remove outer spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    return im


def _colorbar(fig: plt.Figure, im, ax: plt.Axes, label: str) -> None:
    cbar = fig.colorbar(im, ax=ax, shrink=0.90, pad=0.03, aspect=20)
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_visible(False)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = pd.read_csv(CSV)
    df = df[df["preset"].isin(PRESET_ORDER)].copy()

    mat_pct   = _pivot(df, "best_at_pct")
    mat_delta = _pivot(df, "delta_final_minus_best")

    # colormap choices
    # Panel A: sequential blue — 100% (used full budget) is darkest
    cmap_a = "Blues"
    vmin_a, vmax_a = 0, 100

    # Panel B: YlOrRd — 0 (no drift) near white/yellow, larger gaps orange→red
    cmap_b = "YlOrRd"
    vmin_b = 0
    vmax_b = max(0.04, float(np.nanpercentile(mat_delta, 95)))

    with plt.rc_context(RC):
        fig, axes = plt.subplots(
            1, 2,
            figsize=(12.5, 4.6),
            constrained_layout=True,
            gridspec_kw={"wspace": 0.38},
        )

        im_a = _draw_heatmap(
            axes[0], mat_pct, cmap_a, vmin_a, vmax_a,
            fmt_fn=lambda v: f"{int(round(v))}",
            xlabel="Best checkpoint (% training budget)",
            show_yticklabels=True,
            panel_tag="A",
        )
        _colorbar(fig, im_a, axes[0], "% of budget")

        im_b = _draw_heatmap(
            axes[1], mat_delta, cmap_b, vmin_b, vmax_b,
            fmt_fn=lambda v: f"{v:.3f}",
            xlabel="Final-minus-best test NLL",
            show_yticklabels=False,
            panel_tag="B",
        )
        _colorbar(fig, im_b, axes[1], "ΔNLL")

        for ext in ("pdf", "svg", "png"):
            out = ARTIFACTS / f"appendix_two_panel_heatmap.{ext}"
            fig.savefig(out, bbox_inches="tight")
            print(f"  wrote {out.name}")

        plt.close(fig)


if __name__ == "__main__":
    main()
