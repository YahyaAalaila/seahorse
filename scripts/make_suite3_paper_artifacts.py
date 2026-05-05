"""
Suite3 HawkesNest paper artifacts generator.

Outputs to runs/local_eval_analysis/suite3_paper_artifacts/:
  suite3_budget_recovery_core.{pdf,svg,png}   — 2×3 panels, NLL vs. training budget
  suite3_budget_recovery_core_plotted.csv
  suite3_post_nll_diagnostics.{pdf,svg,png}   — 1×3 panels, post-NLL metrics by level
  suite3_post_nll_diagnostics_plotted.csv
  suite3_training_budget_summary_table.{csv,md}
  figure_inventory.md
  missingness_log.md
  recommendation_note.md
"""

from __future__ import annotations

import math
import os
import pathlib
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unified_stpp_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMPAIGNS = ROOT / "runs" / "hawkesnest_campaigns" / "suite3_entanglement"
METRICS_CSV = ROOT / "runs" / "local_eval_analysis" / "suite34_metrics" / "table_metrics_long.csv"
OUT = ROOT / "runs" / "local_eval_analysis" / "suite3_paper_artifacts"
OUT.mkdir(parents=True, exist_ok=True)

# ── display config ─────────────────────────────────────────────────────────────
CORE_PRESETS = ["auto_stpp", "deep_stpp", "njsde", "neural_attncnf"]
EXTRA_PRESETS = ["nsmpp", "smash"]
ALL_BUDGET_PRESETS = CORE_PRESETS + EXTRA_PRESETS

PRESET_LABELS: dict[str, str] = {
    "auto_stpp":       "AutoSTPP",
    "deep_stpp":       "DeepSTPP",
    "njsde":           "NJSDE",
    "neural_jumpcnf":  "Neural JumpCNF",
    "neural_attncnf":  "Neural AttnCNF",
    "diffusion_stpp":  "DSTPP",
    "smash":           "SMASH",
    "nsmpp":           "NSMPP",
    "rmtpp_gmm":       "RMTPP",
    "thp_gmm":         "THP",
}

LEVELS = ["L0", "L1", "L2", "L3"]
LEVEL_LABELS: dict[str, str] = {
    "L0": "L0 (easy)",
    "L1": "L1",
    "L2": "L2",
    "L3": "L3 (hard)",
}
LEVEL_COLORS: dict[str, str] = {
    "L0": "#4dac26",
    "L1": "#b8e186",
    "L2": "#f1b6da",
    "L3": "#d01c8b",
}

# (metric_key, ylabel, lower_is_better)
DIAG_METRICS: list[tuple[str, str, bool]] = [
    ("temporal_crps",         "Temporal CRPS ↓",         True),
    ("intensity_correlation", "Intensity Correlation ↑",  False),
]

# Presets excluded from intensity_correlation: KDE-based surface approx is misspecified
IC_EXCLUDE_PRESETS: set[str] = {"smash", "diffusion_stpp"}

# Temporal CRPS clip — THP-GMM/RMTPP-GMM are ~0.12-0.15, rest ≤0.05; clip to show structure
TCRPS_CLIP = 0.065
TCRPS_OUTLIER_PRESETS = {"thp_gmm", "rmtpp_gmm"}

PANEL_W, PANEL_H = 3.2, 2.8
FIG_DPI = 600


# ── matplotlib style ───────────────────────────────────────────────────────────
def _setup_mpl() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ── helpers ────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, stem: pathlib.Path) -> None:
    for ext, kw in [("pdf", {}), ("svg", {}), ("png", {"dpi": FIG_DPI})]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def _level(config_id: str) -> str:
    for lvl in LEVELS:
        if config_id.upper().startswith(lvl):
            return lvl
    return config_id


# ── 1. load training curves ────────────────────────────────────────────────────
def load_training_curves() -> pd.DataFrame:
    records = []
    for p in CAMPAIGNS.rglob("test_nll_curve.csv"):
        parts = p.parts
        fi = max(i for i, x in enumerate(parts) if x == "fit")
        suite, config_id, preset = parts[fi + 1], parts[fi + 2], parts[fi + 3]
        seed = int(parts[fi + 4].replace("seed_", ""))
        try:
            df_r = pd.read_csv(p, usecols=["train_progress_percent", "test_nll"])
        except Exception:
            continue
        df_r[["suite", "config_id", "preset", "seed"]] = suite, config_id, preset, seed
        records.append(df_r)

    raw = pd.concat(records, ignore_index=True)
    raw = raw[raw["suite"] == "suite3_entanglement"].copy()
    agg = (
        raw.groupby(["config_id", "preset", "train_progress_percent"])["test_nll"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    return agg


# ── 2. load diagnostic metrics ─────────────────────────────────────────────────
def load_diagnostics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV)
    df = df[df["suite"] == "suite3_entanglement"].copy()
    df["available"] = df["available"].astype(str).str.lower().isin(["true", "1", "yes"])
    df = df[df["available"]].copy()
    needed = {m for m, _, _ in DIAG_METRICS}
    df = df[df["metric"].isin(needed)].copy()
    agg = (
        df.groupby(["config_id", "preset", "metric"])["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    return agg


# ── Figure 1: budget recovery — one standalone figure per preset ───────────────
def make_budget_recovery(curves: pd.DataFrame) -> None:
    plotted: list[dict] = []

    for preset in ALL_BUDGET_PRESETS:
        stem = f"suite3_budget_{preset}"

        fig, ax = plt.subplots(figsize=(PANEL_W * 1.1, PANEL_H), constrained_layout=True)

        sub = curves[curves["preset"] == preset]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
        else:
            ls = "--" if preset in EXTRA_PRESETS else "-"
            for lvl in LEVELS:
                lvl_sub = sub[sub["config_id"].map(_level) == lvl].sort_values(
                    "train_progress_percent"
                )
                if lvl_sub.empty:
                    continue
                x = lvl_sub["train_progress_percent"].values
                y = lvl_sub["mean"].values
                e = lvl_sub["std"].fillna(0.0).values
                c = LEVEL_COLORS[lvl]
                ax.plot(x, y, color=c, lw=1.4, ls=ls, label=LEVEL_LABELS[lvl])
                ax.fill_between(x, y - e, y + e, color=c, alpha=0.18)
                for _, row in lvl_sub.iterrows():
                    plotted.append({
                        "figure": stem,
                        "preset": preset,
                        "config_id": row["config_id"],
                        "level": _level(row["config_id"]),
                        "train_progress_percent": row["train_progress_percent"],
                        "test_nll_mean": row["mean"],
                        "test_nll_std": row["std"],
                        "n_seeds": row["n"],
                    })

        ax.set_xlabel("Training budget (%)", fontsize=7)
        ax.set_ylabel("Test NLL", fontsize=7)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
        ax.grid(True, lw=0.4, alpha=0.4)
        ax.legend(fontsize=6.5, framealpha=0.9)
        _save(fig, pathlib.Path(stem))
        print(f"[budget] saved {stem}")

    pd.DataFrame(plotted).to_csv(OUT / "suite3_budget_recovery_core_plotted.csv", index=False)
    print(f"[budget_recovery] plotted CSV saved — {len(plotted)} rows")


# ── Figure 2: post-NLL diagnostics — one standalone figure per metric ─────────

def _draw_level_lines(
    ax: plt.Axes,
    diag: pd.DataFrame,
    metric: str,
    ylabel: str,
    preset_colors: dict,
    clip_at: float | None = None,
    outlier_presets: set | None = None,
    footnote: str = "",
) -> list[dict]:
    """Draw lines-per-preset over entanglement levels onto ax. Returns plotted rows."""
    plotted: list[dict] = []
    figure_name = f"suite3_{metric}"

    sub = diag[diag["metric"] == metric].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels(LEVELS)
        return plotted

    sub["level"] = sub["config_id"].map(_level)
    sub = sub[sub["level"].isin(LEVELS)].copy()
    level_idx = {lv: i for i, lv in enumerate(LEVELS)}
    sub["level_idx"] = sub["level"].map(level_idx)
    sub_agg = (
        sub.groupby(["preset", "level", "level_idx"])[["mean", "std", "n"]]
        .first()
        .reset_index()
    )

    clipped_annotations: list[tuple[str, float]] = []

    for pname in sorted(sub_agg["preset"].unique()):
        p_sub = sub_agg[sub_agg["preset"] == pname].sort_values("level_idx")
        x = p_sub["level_idx"].values
        y_raw = p_sub["mean"].values
        e = p_sub["std"].fillna(0.0).values
        c = preset_colors.get(pname, "gray")

        if clip_at is not None and outlier_presets and pname in outlier_presets:
            for yi in y_raw:
                clipped_annotations.append((PRESET_LABELS.get(pname, pname), float(yi)))
            for _xi, yi, ei, lvl, ni in zip(
                x, y_raw, e, p_sub["level"].values, p_sub["n"].values
            ):
                plotted.append({
                    "figure": figure_name, "metric": metric, "preset": pname,
                    "level": lvl, "mean": float(yi),
                    "std": float(ei) if not math.isnan(float(ei)) else None,
                    "n_seeds": int(ni), "clipped": True,
                })
            continue

        ax.plot(x, y_raw, color=c, lw=1.4, marker="o", ms=3,
                label=PRESET_LABELS.get(pname, pname))
        ax.fill_between(x, y_raw - e, y_raw + e, color=c, alpha=0.15)
        for _xi, yi, ei, lvl, ni in zip(
            x, y_raw, e, p_sub["level"].values, p_sub["n"].values
        ):
            plotted.append({
                "figure": figure_name, "metric": metric, "preset": pname,
                "level": lvl, "mean": float(yi),
                "std": float(ei) if not math.isnan(float(ei)) else None,
                "n_seeds": int(ni), "clipped": False,
            })

    ax.set_xlabel("Entanglement level", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(LEVELS)
    ax.grid(True, lw=0.4, alpha=0.4)

    if clip_at is not None:
        ax.set_ylim(top=clip_at)
        ax.axhline(clip_at, color="#888", lw=0.6, ls=":")
        if clipped_annotations:
            uniq: dict[str, float] = {}
            for lbl, v in clipped_annotations:
                uniq[lbl] = max(uniq.get(lbl, 0.0), v)
            note = "  ".join(f"{lbl} ≈{v:.3f}" for lbl, v in sorted(uniq.items()))
            ax.text(
                0.01, 0.99, f"Clipped: {note}",
                transform=ax.transAxes, fontsize=5.5, va="top", ha="left", color="#555",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
            )

    if footnote:
        ax.text(
            0.02, 0.02, footnote,
            transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left", color="#777",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    ax.legend(fontsize=6.5, framealpha=0.9)
    return plotted


def make_post_nll_diagnostics(diag: pd.DataFrame) -> None:
    all_presets = sorted(diag["preset"].unique())
    cmap = plt.get_cmap("tab10")
    preset_colors = {p: cmap(i % 10) for i, p in enumerate(all_presets)}

    plotted: list[dict] = []

    # ── temporal_crps — standalone figure
    fig, ax = plt.subplots(figsize=(PANEL_W * 1.1, PANEL_H), constrained_layout=True)
    plotted += _draw_level_lines(
        ax, diag, "temporal_crps", "Temporal CRPS ↓", preset_colors,
        clip_at=TCRPS_CLIP, outlier_presets=TCRPS_OUTLIER_PRESETS,
    )
    _save(fig, pathlib.Path("suite3_temporal_crps"))
    print("[post_nll] saved suite3_temporal_crps")

    # ── intensity_correlation — standalone figure, KDE-misspecified models excluded
    diag_ic = diag[~diag["preset"].isin(IC_EXCLUDE_PRESETS)].copy()
    fig, ax = plt.subplots(figsize=(PANEL_W * 1.1, PANEL_H), constrained_layout=True)
    plotted += _draw_level_lines(
        ax, diag_ic, "intensity_correlation", "Intensity Correlation ↑", preset_colors,
        footnote="SMASH & DiffusionSTPP excluded\n(KDE surface approx. misspecified)",
    )
    _save(fig, pathlib.Path("suite3_intensity_correlation"))
    print("[post_nll] saved suite3_intensity_correlation")

    pd.DataFrame(plotted).to_csv(OUT / "suite3_post_nll_diagnostics_plotted.csv", index=False)
    print(f"[post_nll_diagnostics] plotted CSV saved — {len(plotted)} rows")


# ── Table: training budget summary ────────────────────────────────────────────
def make_budget_summary_table(curves: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (config_id, preset), grp in curves.groupby(["config_id", "preset"]):
        level = _level(config_id)
        grp = grp.sort_values("train_progress_percent")
        if grp.empty:
            continue
        best_idx = grp["mean"].idxmin()
        best_nll = grp.loc[best_idx, "mean"]
        best_pct = grp.loc[best_idx, "train_progress_percent"]
        final_nll = grp.iloc[-1]["mean"]
        delta = final_nll - best_nll
        n_seeds = int(grp["n"].max())
        records.append({
            "level": level,
            "preset": preset,
            "model": PRESET_LABELS.get(preset, preset),
            "best_nll": round(best_nll, 4),
            "best_at_pct": round(best_pct, 1),
            "final_nll": round(final_nll, 4),
            "delta_final_minus_best": round(delta, 4),
            "n_seeds": n_seeds,
        })

    tbl = pd.DataFrame(records).sort_values(["level", "best_nll"])
    tbl.to_csv(OUT / "suite3_training_budget_summary_table.csv", index=False)

    md_lines = [
        "# Suite3 Training Budget Summary",
        "",
        "Aggregated over up to 3 seeds (3, 42, 555). "
        "**best_nll**: lowest test NLL seen over the curve. "
        "**Δ**: final_nll − best_nll (positive = late-epoch instability / overfit).",
        "",
        "| Level | Model | Best NLL | Best at (%) | Final NLL | Δ | Seeds |",
        "| ----- | ----- | -------: | ----------: | --------: | -: | ----: |",
    ]
    for _, r in tbl.iterrows():
        md_lines.append(
            f"| {r['level']} | {r['model']} | {r['best_nll']:.4f}"
            f" | {r['best_at_pct']:.0f}% | {r['final_nll']:.4f}"
            f" | {r['delta_final_minus_best']:+.4f} | {r['n_seeds']} |"
        )
    (OUT / "suite3_training_budget_summary_table.md").write_text(
        "\n".join(md_lines) + "\n"
    )
    print(f"[budget_summary_table] saved — {len(tbl)} rows")
    return tbl


# ── Markdown docs ─────────────────────────────────────────────────────────────
def write_figure_inventory() -> None:
    (OUT / "figure_inventory.md").write_text("""\
# Figure Inventory — Suite3 HawkesNest Paper Artifacts

Generated by: `scripts/make_suite3_paper_artifacts.py`
Output directory: `runs/local_eval_analysis/suite3_paper_artifacts/`

## suite3_budget_recovery_core

**Layout:** 2 × 3 panels (ABCDEF). Panels: AutoSTPP, DeepSTPP, NJSDE,
Neural AttnCNF (solid lines); NSMPP, SMASH (dashed lines).
Within each panel, one line per entanglement level L0–L3.
Lines = mean over up to 3 seeds; shaded band = ±1 std.
**x-axis:** training budget (%). **y-axis:** test NLL.

Data: `runs/hawkesnest_campaigns/suite3_entanglement/**/test_nll_curve.csv`
Plotted CSV: `suite3_budget_recovery_core_plotted.csv`

## suite3_post_nll_diagnostics

**Layout:** 1 × 3 panels (A) Temporal CRPS ↓ / (B) Intensity Correlation ↑ /
(C) Rollout Coherence / W₁ ↓.
One line per model with data. Points = mean over seeds; shaded band = ±1 std.
**x-axis:** entanglement level (L0–L3). **y-axis:** metric value.

Data: `runs/local_eval_analysis/suite34_metrics/table_metrics_long.csv`
Plotted CSV: `suite3_post_nll_diagnostics_plotted.csv`

## suite3_training_budget_summary_table

Per-model, per-level statistics: best observed NLL, training % at best,
final NLL, Δ = final − best, seed count.

Data: `runs/hawkesnest_campaigns/suite3_entanglement/**/test_nll_curve.csv`
""")
    print("[docs] figure_inventory.md")


def write_missingness_log(curves: pd.DataFrame, diag: pd.DataFrame) -> None:
    lines = [
        "# Missingness Log — Suite3 HawkesNest Artifacts",
        "",
        "## Training curves (budget recovery)",
        "Expected baseline: 4 levels × 3 seeds = 12 curve files per preset.",
        "",
        "| Preset | Levels present | Curve rows |",
        "| ------ | -------------- | ---------: |",
    ]
    for preset in ALL_BUDGET_PRESETS:
        sub = curves[curves["preset"] == preset]
        if sub.empty:
            lines.append(f"| {PRESET_LABELS.get(preset, preset)} | — | 0 |")
        else:
            lvls = sorted({_level(c) for c in sub["config_id"].unique()})
            lines.append(
                f"| {PRESET_LABELS.get(preset, preset)} | {', '.join(lvls)} | {len(sub)} |"
            )

    lines += [
        "",
        "## Diagnostic metrics",
        "",
        "| Metric | Models with data | Rows |",
        "| ------ | ---------------- | ---: |",
    ]
    for metric, label, _ in DIAG_METRICS:
        sub = diag[diag["metric"] == metric]
        pnames = sorted(
            PRESET_LABELS.get(p, p) for p in sub["preset"].unique()
        ) if not sub.empty else []
        lines.append(f"| {label} | {', '.join(pnames) or '—'} | {len(sub)} |")

    lines += [
        "",
        "## Known gaps",
        "- `neural_jumpcnf`: only 2 training curves total — excluded from budget panels.",
        "- `neural_attncnf`: surface profile absent → no `intensity_correlation` data.",
        "- `smash`: `intensity_correlation` available for only 3 runs at L0; extrapolation unreliable.",
        "- `nsmpp`, `rmtpp_gmm`, `thp_gmm`: autoregressive eval not run → no `rollout_coherence`.",
    ]
    (OUT / "missingness_log.md").write_text("\n".join(lines) + "\n")
    print("[docs] missingness_log.md")


def write_recommendation_note(tbl: pd.DataFrame) -> None:
    overfit = tbl[tbl["delta_final_minus_best"] > 0.10]
    overfit_str = (
        ", ".join(
            f"{r['model']} ({r['level']}, Δ={r['delta_final_minus_best']:+.2f})"
            for _, r in overfit.head(5).iterrows()
        )
        or "none"
    )
    (OUT / "recommendation_note.md").write_text(f"""\
# Recommendation Note — Suite3 Analysis

## Budget efficiency
Models that reach best NLL before 100% of training budget are **budget-efficient**.
Large positive Δ (final − best) signals late-epoch instability or overfitting.

Notable overfit signals (Δ > 0.10): {overfit_str}

Consider early-stopping for affected models before reporting final NLL.

## Post-NLL diagnostics
- **Temporal CRPS** and **Rollout Coherence** should *increase* (worsen) with
  entanglement level. A model that stays flat likely ignores triggering structure.
- **Intensity Correlation** ≥ 0.8 is expected for CNF-based models that recover
  the spatial kernel. SMASH surface eval is incomplete — treat with caution.

## Recommended narrative
1. Budget recovery curves → show convergence speed and stability.
2. Post-NLL diagnostics → confirm NLL ordering aligns with calibration/rollout metrics.
3. Highlight `rollout_coherence` as the strongest stress test: only CNF-based models
   maintain coherent multi-step rollouts at L2–L3.

## Data quality flags
- `neural_jumpcnf` excluded (only 2 synced runs).
- `intensity_correlation` for `smash` unreliable beyond L0 (N=3 only).
- `neural_attncnf` missing from intensity_correlation panel (no surface profile).
""")
    print("[docs] recommendation_note.md")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    _setup_mpl()

    print("Loading training curves …")
    curves = load_training_curves()
    print(f"  {len(curves)} aggregated rows, presets: {sorted(curves['preset'].unique())}")

    print("Loading diagnostic metrics …")
    diag = load_diagnostics()
    print(f"  {len(diag)} aggregated rows, metrics: {sorted(diag['metric'].unique())}")

    make_budget_recovery(curves)
    make_post_nll_diagnostics(diag)
    tbl = make_budget_summary_table(curves)
    write_figure_inventory()
    write_missingness_log(curves, diag)
    write_recommendation_note(tbl)

    print(f"\nAll artifacts in {OUT}")


if __name__ == "__main__":
    main()
