"""
BenchmarkTable — aggregate RunResults across presets × datasets × seeds.
"""

from __future__ import annotations

import base64
import io
import json
import math
from dataclasses import dataclass
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

    def report(self, out_dir) -> None:
        """Write a self-contained HTML report + CSV + JSON to *out_dir*.

        Files produced:
        - ``results.json``
        - ``table_test_nll.csv``
        - ``report.html`` (figures embedded as base64)
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

        # HTML
        html = _render_html(
            runs=self.runs,
            pivot_html=pivot.to_html() if not pivot.empty else "<p>No data.</p>",
            latex=self.to_latex(),
            fig_nll_b64=fig_nll_b64,
            fig_time_b64=fig_time_b64,
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

<h2>LaTeX Table</h2>
<pre>{latex}</pre>
</body>
</html>"""
