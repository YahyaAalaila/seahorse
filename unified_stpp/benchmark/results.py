"""
BenchmarkTable — aggregate RunResults across presets × datasets × seeds.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unified_stpp.runner.results import RunResult


# ---------------------------------------------------------------------------
# BenchmarkTable
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkTable:
    """Container for all RunResults from a Benchmark run."""

    runs: list[RunResult]
    surfaces_by_dataset: dict = field(default_factory=dict)
    """Nested dict: ``{dataset_id: {preset: list[SurfaceResult]}}``."""

    # ------------------------------------------------------------------
    # Tabular views
    # ------------------------------------------------------------------

    def to_dataframe(self, metric: str = "test_nll"):
        """Return a pivot DataFrame: presets (rows) × datasets (cols).

        Cells contain ``mean ± std`` across seeds (NaN if only one seed).
        Requires ``pandas``.
        """
        import pandas as pd

        records = []
        for r in self.runs:
            val = getattr(r, metric, r.extra_metrics.get(metric, math.nan))
            records.append(
                {"preset": r.preset, "dataset": r.dataset_id, "seed": r.seed, "value": val}
            )
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        agg = (
            df.groupby(["preset", "dataset"])["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg["cell"] = agg.apply(
            lambda row: f"{row['mean']:.4f}"
            if math.isnan(row["std"]) or row["std"] == 0
            else f"{row['mean']:.4f} ± {row['std']:.4f}",
            axis=1,
        )
        pivot = agg.pivot(index="preset", columns="dataset", values="cell")
        return pivot

    def to_latex(self, metric: str = "test_nll") -> str:
        """Return a LaTeX table string with bold minimum per dataset column."""
        import pandas as pd

        records = []
        for r in self.runs:
            val = getattr(r, metric, r.extra_metrics.get(metric, math.nan))
            records.append(
                {"preset": r.preset, "dataset": r.dataset_id, "value": val}
            )
        if not records:
            return ""

        df = pd.DataFrame(records)
        agg = (
            df.groupby(["preset", "dataset"])["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        datasets = sorted(agg["dataset"].unique())
        presets = sorted(agg["preset"].unique())

        # Best (min mean) per dataset
        best_mean: dict[str, float] = {}
        for ds in datasets:
            sub = agg[agg["dataset"] == ds]["mean"]
            best_mean[ds] = float(sub.min()) if len(sub) else math.nan

        header = " & ".join(["Preset"] + datasets) + r" \\"
        rows = [r"\begin{tabular}{l" + "c" * len(datasets) + "}", r"\toprule", header, r"\midrule"]
        for preset in presets:
            cells = [preset]
            for ds in datasets:
                sub = agg[(agg["preset"] == preset) & (agg["dataset"] == ds)]
                if sub.empty:
                    cells.append("—")
                else:
                    mean = float(sub["mean"].iloc[0])
                    std = float(sub["std"].iloc[0])
                    cell = f"{mean:.4f}" if math.isnan(std) or std == 0 else f"{mean:.4f}{{\\scriptsize ±{std:.4f}}}"
                    if abs(mean - best_mean[ds]) < 1e-8:
                        cell = r"\textbf{" + cell + "}"
                    cells.append(cell)
            rows.append(" & ".join(cells) + r" \\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, path) -> None:
        """Write all runs to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump([r.to_dict() for r in self.runs], f, indent=2, default=str)

    @classmethod
    def from_json(cls, path) -> "BenchmarkTable":
        """Load from a JSON file written by :meth:`to_json`."""
        with open(path) as f:
            data = json.load(f)
        return cls(runs=[RunResult.from_dict(d) for d in data])

    # ------------------------------------------------------------------
    # Rich report
    # ------------------------------------------------------------------

    def plot_intensities(
        self,
        splits: dict,
        out_dir,
        fmt: str = "gif",
        n_frames: int = 24,
        n_grid: int = 40,
        fps: int = 8,
        device: str = "cpu",
    ) -> list:
        """Render animated intensity plots; one GIF/MP4/PNG per dataset.

        Parameters
        ----------
        splits  : ``{dataset_id: (train_seqs, val_seqs, test_seqs)}`` —
                  same dict that was passed to :class:`Benchmark`.
        out_dir : directory to write output files into.
        fmt     : ``"gif"`` | ``"mp4"`` | ``"png"`` (static 4-snapshot grid).
        n_frames: number of animation frames (gif/mp4 only).
        n_grid  : spatial grid resolution per axis.
        fps     : animation speed.
        device  : torch device for model inference.

        Returns
        -------
        List of :class:`pathlib.Path` objects for the produced files.
        """
        from unified_stpp.benchmark.intensity_plot import plot_bench_intensities
        return plot_bench_intensities(
            self, splits=splits, out_dir=out_dir,
            fmt=fmt, n_frames=n_frames, n_grid=n_grid, fps=fps, device=device,
        )

    def save_surface_comparison(
        self,
        out_dir,
        render_mode: str = "3d",
        animate: bool = False,
        fps: int = 4,
    ) -> dict:
        """Save benchmark-level cross-model comparison panels (one per dataset).

        Requires ``surfaces_by_dataset`` to be populated (i.e. ``Benchmark.run()``
        was called with ``surface_viz`` enabled).

        Only datasets with ≥ 2 models produce a comparison figure.

        Parameters
        ----------
        out_dir     : directory to write comparison figures into.
        render_mode : ``"3d"`` (default) or ``"2d"``.
        animate     : also produce a .gif animation per dataset.
        fps         : animation frames per second (when ``animate=True``).

        Returns
        -------
        dict mapping artifact name → Path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from unified_stpp.viz.multi_plot import plot_model_comparison

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifacts: dict = {}

        for dataset_id, preset_surfaces in self.surfaces_by_dataset.items():
            if len(preset_surfaces) < 2:
                continue

            fig = plot_model_comparison(preset_surfaces, render_mode=render_mode)
            png_path = out / f"comparison_{dataset_id}.png"
            fig.savefig(png_path, bbox_inches="tight")
            plt.close(fig)
            artifacts[f"comparison_{dataset_id}"] = png_path

            if animate:
                from unified_stpp.viz.animation import animate_surface_sequence
                gif_path = out / f"comparison_{dataset_id}.gif"
                actual_path = animate_surface_sequence(
                    preset_surfaces,
                    output_path=gif_path,
                    render_mode=render_mode,
                    fps=fps,
                )
                artifacts[f"comparison_anim_{dataset_id}"] = actual_path

        return artifacts

    def report(self, out_dir) -> None:
        """Write a self-contained HTML report + CSV + JSON to *out_dir*.

        Files produced:
        - ``results.json``
        - ``table_test_nll.csv``
        - ``report.html`` (figures embedded as base64)
        - ``comparison_{dataset_id}.png`` (when surface_viz was used)
        """
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        self.to_json(out / "results.json")

        # CSV pivot
        pivot = self.to_dataframe(metric="test_nll")
        if not isinstance(pivot, pd.DataFrame) or pivot.empty:
            pivot_csv = ""
        else:
            pivot_csv = pivot.to_csv()
            (out / "table_test_nll.csv").write_text(pivot_csv)

        # Figures
        fig_nll_b64 = self._make_nll_bar_chart()
        fig_time_b64 = self._make_time_bar_chart()

        # Surface comparison (if surfaces were collected)
        surface_b64s: dict = {}
        if self.surfaces_by_dataset:
            surface_artifacts = self.save_surface_comparison(out, render_mode="3d")
            for key, path in surface_artifacts.items():
                if Path(path).suffix == ".png":
                    with open(path, "rb") as f:
                        surface_b64s[key] = base64.b64encode(f.read()).decode()

        # HTML
        html = _render_html(
            runs=self.runs,
            pivot_html=pivot.to_html() if not pivot.empty else "<p>No data.</p>",
            latex=self.to_latex(),
            fig_nll_b64=fig_nll_b64,
            fig_time_b64=fig_time_b64,
            surface_b64s=surface_b64s,
        )
        (out / "report.html").write_text(html, encoding="utf-8")

    def _make_nll_bar_chart(self) -> str:
        """Bar chart of test NLL per preset grouped by dataset. Returns base64 PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            return ""

        records = [
            {"preset": r.preset, "dataset": r.dataset_id, "test_nll": r.test_nll}
            for r in self.runs
            if not math.isnan(r.test_nll)
        ]
        if not records:
            return ""

        df = pd.DataFrame(records)
        agg = df.groupby(["dataset", "preset"])["test_nll"].mean().reset_index()
        datasets = sorted(agg["dataset"].unique())
        presets = sorted(agg["preset"].unique())

        fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), squeeze=False)
        for i, ds in enumerate(datasets):
            ax = axes[0][i]
            sub = agg[agg["dataset"] == ds]
            vals = [float(sub[sub["preset"] == p]["test_nll"].iloc[0]) if p in sub["preset"].values else 0.0 for p in presets]
            ax.bar(presets, vals)
            ax.set_title(ds)
            ax.set_ylabel("Test NLL" if i == 0 else "")
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        return _fig_to_b64(fig)

    def _make_time_bar_chart(self) -> str:
        """Bar chart of training time per preset. Returns base64 PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            return ""

        records = [{"preset": r.preset, "train_time_sec": r.train_time_sec} for r in self.runs]
        if not records:
            return ""
        df = pd.DataFrame(records)
        agg = df.groupby("preset")["train_time_sec"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(agg["preset"], agg["train_time_sec"])
        ax.set_ylabel("Avg training time (s)")
        ax.set_title("Training time by preset")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _render_html(
    runs: list[RunResult],
    pivot_html: str,
    latex: str,
    fig_nll_b64: str,
    fig_time_b64: str,
    surface_b64s: dict | None = None,
) -> str:
    """Produce a self-contained HTML string."""
    rows_html = "\n".join(
        f"<tr><td>{r.preset}</td><td>{r.dataset_id}</td><td>{r.seed}</td>"
        f"<td>{r.val_nll:.4f}</td>"
        f"<td>{'n/a' if math.isnan(r.test_nll) else f'{r.test_nll:.4f}'}</td>"
        f"<td>{r.train_time_sec:.1f}s</td><td>{r.n_params:,}</td></tr>"
        for r in runs
    )
    nll_img = f'<img src="data:image/png;base64,{fig_nll_b64}" alt="NLL chart" style="max-width:100%">' if fig_nll_b64 else ""
    time_img = f'<img src="data:image/png;base64,{fig_time_b64}" alt="Time chart" style="max-width:100%">' if fig_time_b64 else ""

    surface_section = ""
    if surface_b64s:
        imgs = "\n".join(
            f'<p><b>{key}</b></p>'
            f'<img src="data:image/png;base64,{b64}" alt="{key}" style="max-width:100%">'
            for key, b64 in surface_b64s.items()
        )
        surface_section = f"<h2>Surface Comparison</h2>\n{imgs}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>STPP Benchmark Report</title>
<style>
  body {{ font-family: monospace; margin: 2em; }}
  table {{ border-collapse: collapse; margin-bottom: 2em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  pre {{ background: #f8f8f8; padding: 1em; overflow-x: auto; }}
  h2 {{ margin-top: 2em; }}
</style>
</head>
<body>
<h1>STPP Benchmark Report</h1>

<h2>Pivot Table (Test NLL)</h2>
{pivot_html}

<h2>All Runs</h2>
<table>
<thead>
  <tr><th>Preset</th><th>Dataset</th><th>Seed</th><th>Val NLL</th><th>Test NLL</th><th>Train Time</th><th>Params</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>

<h2>Figures</h2>
{nll_img}
{time_img}

{surface_section}

<h2>LaTeX Table</h2>
<pre>{latex}</pre>
</body>
</html>"""
