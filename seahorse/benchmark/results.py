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

from seahorse.runner.results import RunResult


@dataclass
class BenchmarkTable:
    """Container for all RunResults from a Benchmark run."""

    runs: list[RunResult]

    # ------------------------------------------------------------------
    # Primary tabular views
    # ------------------------------------------------------------------

    def to_dataframe(self, metric: str = "test_nll", group: str = "all"):
        """Return a pivot DataFrame: presets (rows) × datasets (cols).

        Parameters
        ----------
        metric : str
            Attribute name on RunResult (e.g. ``"test_nll"``, ``"temporal_nll"``).
        group : str
            ``"exact"``  — only runs where ``nll_kind == "exact"`` and
                           ``nll_report_space == "raw"``.
            ``"approx"`` — only runs where ``nll_kind != "exact"``.
            ``"all"``    — all runs (default).
        """
        import pandas as pd

        records = []
        for r in self.runs:
            nll_kind = getattr(r, "nll_kind", "exact")
            report_space = getattr(r, "nll_report_space", "native")
            if group == "exact" and (nll_kind != "exact" or report_space != "raw"):
                continue
            if group == "approx" and nll_kind == "exact":
                continue
            val = getattr(r, metric, r.extra_metrics.get(metric, math.nan))
            footnote = getattr(r, "nll_footnote", "")
            records.append(
                {
                    "preset": r.preset,
                    "dataset": r.dataset_id,
                    "seed": r.seed,
                    "value": val,
                    "footnote": footnote,
                }
            )
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        agg = df.groupby(["preset", "dataset"]).agg(
            mean=("value", "mean"), std=("value", "std"), footnote=("footnote", "first")
        ).reset_index()
        agg["cell"] = agg.apply(
            lambda row: (
                f"{row['mean']:.4f}"
                if math.isnan(row["std"]) or row["std"] == 0
                else f"{row['mean']:.4f} ± {row['std']:.4f}"
            ) + (row["footnote"] if row["footnote"] else ""),
            axis=1,
        )
        return agg.pivot(index="preset", columns="dataset", values="cell")

    def to_latex(self, metric: str = "test_nll") -> str:
        """Return LaTeX tables: Table 1 (exact-NLL models), Table 2 (all models).

        Non-exact models receive superscript footnotes (†, ‡, …).
        """
        import pandas as pd

        records = []
        footnote_map: dict[str, str] = {}
        for r in self.runs:
            val = getattr(r, metric, r.extra_metrics.get(metric, math.nan))
            fn = getattr(r, "nll_footnote", "")
            if fn:
                footnote_map[r.preset] = fn
            records.append(
                {"preset": r.preset, "dataset": r.dataset_id, "value": val, "footnote": fn}
            )
        if not records:
            return ""

        df = pd.DataFrame(records)
        agg = df.groupby(["preset", "dataset"]).agg(
            mean=("value", "mean"), std=("value", "std"), footnote=("footnote", "first")
        ).reset_index()

        def _make_table(presets, title):
            sub_agg = agg[agg["preset"].isin(presets)]
            if sub_agg.empty:
                return ""
            datasets = sorted(sub_agg["dataset"].unique())
            best_mean: dict[str, float] = {}
            for ds in datasets:
                vals = sub_agg[sub_agg["dataset"] == ds]["mean"]
                best_mean[ds] = float(vals.min()) if len(vals) else math.nan

            header = " & ".join(["Preset"] + datasets) + r" \\"
            lines = [
                f"% {title}",
                r"\begin{tabular}{l" + "c" * len(datasets) + "}",
                r"\toprule", header, r"\midrule",
            ]
            for preset in sorted(presets):
                cells = [preset + footnote_map.get(preset, "")]
                for ds in datasets:
                    row = sub_agg[(sub_agg["preset"] == preset) & (sub_agg["dataset"] == ds)]
                    if row.empty:
                        cells.append("—")
                        continue
                    mean = float(row["mean"].iloc[0])
                    std = float(row["std"].iloc[0])
                    cell = (
                        f"{mean:.4f}"
                        if math.isnan(std) or std == 0
                        else f"{mean:.4f}{{\\scriptsize ±{std:.4f}}}"
                    )
                    if abs(mean - best_mean[ds]) < 1e-8:
                        cell = r"\textbf{" + cell + "}"
                    cells.append(cell)
                lines.append(" & ".join(cells) + r" \\")
            lines += [r"\bottomrule", r"\end{tabular}"]
            if footnote_map:
                for fn_text in sorted(set(footnote_map.values())):
                    lines.append(f"% Footnote: {fn_text}")
            return "\n".join(lines)

        exact_presets = {
            r.preset for r in self.runs
            if (
                getattr(r, "nll_kind", "exact") == "exact"
                and getattr(r, "nll_report_space", "native") == "raw"
            )
        }
        all_presets = {r.preset for r in self.runs}

        parts = []
        if exact_presets:
            parts.append(_make_table(exact_presets, "Table 1 — exact NLL models only"))
        if all_presets - exact_presets:
            parts.append(_make_table(all_presets, "Table 2 — all models"))
        return "\n\n".join(parts)

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
    # HTML report
    # ------------------------------------------------------------------

    def report(self, out_dir, metric: str = "test_nll") -> None:
        """Write a self-contained HTML report + CSV + JSON to *out_dir*."""
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.to_json(out / "results.json")

        metric_slug = _metric_slug(metric)

        # Primary pivot: exact-NLL models only
        pivot_exact = self.to_dataframe(metric=metric, group="exact")
        # Full pivot: all models
        pivot_all = self.to_dataframe(metric=metric, group="all")

        for pivot, slug in [(pivot_exact, f"table_{metric_slug}_exact"),
                            (pivot_all, f"table_{metric_slug}_all")]:
            if isinstance(pivot, pd.DataFrame) and not pivot.empty:
                (out / f"{slug}.csv").write_text(pivot.to_csv())

        fig_metric_b64 = self._make_metric_bar_chart(metric)
        fig_time_b64 = self._make_time_bar_chart()

        html = _render_html(
            runs=self.runs,
            metric=metric,
            pivot_exact_html=pivot_exact.to_html() if isinstance(pivot_exact, pd.DataFrame) and not pivot_exact.empty else "<p>No exact-NLL runs.</p>",
            pivot_all_html=pivot_all.to_html() if isinstance(pivot_all, pd.DataFrame) and not pivot_all.empty else "<p>No data.</p>",
            latex=self.to_latex(metric=metric),
            fig_metric_b64=fig_metric_b64,
            fig_time_b64=fig_time_b64,
        )
        (out / "report.html").write_text(html, encoding="utf-8")

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    def _make_metric_bar_chart(self, metric: str) -> str:
        """Bar chart of the selected benchmark metric by preset/dataset."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            return ""

        records = []
        for r in self.runs:
            val = getattr(r, metric, r.extra_metrics.get(metric, math.nan))
            if math.isnan(val):
                continue
            records.append({"preset": r.preset, "dataset": r.dataset_id, metric: val})
        if not records:
            return ""

        df = pd.DataFrame(records)
        agg = df.groupby(["dataset", "preset"])[metric].mean().reset_index()
        datasets = sorted(agg["dataset"].unique())
        presets = sorted(agg["preset"].unique())

        fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), squeeze=False)
        for i, ds in enumerate(datasets):
            ax = axes[0][i]
            sub = agg[agg["dataset"] == ds]
            vals = [
                float(sub[sub["preset"] == p][metric].iloc[0]) if p in sub["preset"].values else 0.0
                for p in presets
            ]
            ax.bar(presets, vals)
            ax.set_title(ds)
            ax.set_ylabel(metric if i == 0 else "")
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
# Module-level helpers
# ---------------------------------------------------------------------------

def _metric_slug(metric: str) -> str:
    return metric.replace("/", "_").replace(" ", "_")


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _render_html(
    runs: list[RunResult],
    metric: str,
    pivot_exact_html: str,
    pivot_all_html: str,
    latex: str,
    fig_metric_b64: str,
    fig_time_b64: str,
) -> str:
    """Produce a self-contained HTML string with full metric-semantics annotations."""
    # All-runs table rows — includes objective, nll_kind, footnote
    rows_html = "\n".join(
        f"<tr>"
        f"<td>{r.preset}</td>"
        f"<td>{r.dataset_id}</td>"
        f"<td>{r.seed}</td>"
        f"<td>{r.val_objective:.4f} ({r.val_metric_key})</td>"
        f"<td>{'n/a' if math.isnan(r.test_nll) else f'{r.test_nll:.4f}'}</td>"
        f"<td>{r.train_time_sec:.1f}s</td>"
        f"<td>{r.n_params:,}</td>"
        f"<td>{getattr(r, 'nll_kind', '—')}</td>"
        f"<td>{getattr(r, 'nll_footnote', '') or '—'}</td>"
        f"</tr>"
        for r in runs
    )

    # Metric metadata table — one row per unique preset
    seen_presets: set[str] = set()
    meta_rows: list[str] = []
    asymmetry_presets: list[str] = []
    for r in runs:
        if r.preset in seen_presets:
            continue
        seen_presets.add(r.preset)
        nll_kind  = getattr(r, "nll_kind", "exact")
        val_key   = getattr(r, "val_metric_key", "nll")
        obj       = getattr(r, "training_objective", "—")
        obj_desc  = getattr(r, "objective_description", "") or obj
        desc      = getattr(r, "nll_description", "—")
        fn        = getattr(r, "nll_footnote", "") or "—"
        report_space = getattr(r, "nll_report_space", "native")
        group     = "A – exact" if nll_kind == "exact" else ("B – approx" if nll_kind == "approx" else "C – none")
        # Flag asymmetry: training objective is not NLL (val metric ≠ test metric family)
        asymmetry = val_key != "nll"
        asym_flag = " ⚠" if asymmetry else ""
        meta_rows.append(
            f"<tr><td>{r.preset}{asym_flag}</td><td>{obj_desc}</td>"
            f"<td>val/{val_key} → test/nll [{nll_kind}]</td><td>{report_space}</td>"
            f"<td>{desc}</td><td>{fn}</td><td>{group}</td></tr>"
        )
        if asymmetry:
            asymmetry_presets.append(r.preset)

    # Asymmetry notice — shown when training objective is not NLL
    asymmetry_section = ""
    if asymmetry_presets:
        asym_rows = []
        seen_asym: set[str] = set()
        for r in runs:
            if r.preset not in asymmetry_presets or r.preset in seen_asym:
                continue
            seen_asym.add(r.preset)
            val_key  = getattr(r, "val_metric_key", "nll")
            nll_kind = getattr(r, "nll_kind", "exact")
            obj_desc = getattr(r, "objective_description", "") or getattr(r, "training_objective", "—")
            desc     = getattr(r, "nll_description", "—")
            asym_rows.append(
                f"<tr><td>{r.preset}</td><td>{obj_desc} (val/{val_key})</td>"
                f"<td>test/nll [{nll_kind}]</td><td>{desc}</td></tr>"
            )
        if asym_rows:
            asymmetry_section = f"""
<h2>⚠ Metric Asymmetry Notice</h2>
<p class="note">For the following presets, <strong>val_objective</strong> (used for checkpoint
selection and HPO) and <strong>test_nll</strong> (used for benchmark reporting) are
<em>different quantities</em>. val_objective is the model's native training objective;
test_nll is the benchmark-facing held-out next-event NLL and may be exact or
approximate depending on <code>nll_kind</code>. Do not compare val_objective across
model families.</p>
<table>
<thead>
  <tr><th>Preset</th><th>Training objective (val)</th><th>Test NLL kind</th><th>test_nll description</th></tr>
</thead>
<tbody>
{"".join(asym_rows)}
</tbody>
</table>"""

    # Temporal/spatial breakdown table (only rows where data is available)
    breakdown_rows: list[str] = []
    for r in runs:
        t_nll = getattr(r, "temporal_nll", math.nan)
        s_nll = getattr(r, "spatial_nll", math.nan)
        if math.isnan(t_nll) and math.isnan(s_nll):
            continue
        t_str = "n/a" if math.isnan(t_nll) else f"{t_nll:.4f}"
        s_str = "n/a" if math.isnan(s_nll) else f"{s_nll:.4f}"
        breakdown_rows.append(
            f"<tr><td>{r.preset}</td><td>{r.dataset_id}</td><td>{r.seed}</td>"
            f"<td>{t_str}</td><td>{s_str}</td></tr>"
        )
    breakdown_section = ""
    if breakdown_rows:
        breakdown_section = f"""
<h2>Temporal / Spatial NLL Breakdown</h2>
<table>
<thead>
  <tr><th>Preset</th><th>Dataset</th><th>Seed</th><th>Temporal NLL</th><th>Spatial NLL</th></tr>
</thead>
<tbody>
{"".join(breakdown_rows)}
</tbody>
</table>"""

    extra_metric_keys = sorted(
        {
            key
            for r in runs
            for key, value in (getattr(r, "extra_metrics", {}) or {}).items()
            if isinstance(value, (int, float)) and not math.isnan(float(value))
        }
    )
    extra_metrics_section = ""
    if extra_metric_keys:
        extra_rows = []
        for r in runs:
            values = []
            for key in extra_metric_keys:
                value = (getattr(r, "extra_metrics", {}) or {}).get(key, math.nan)
                values.append("n/a" if math.isnan(float(value)) else f"{float(value):.4f}")
            extra_rows.append(
                "<tr>"
                f"<td>{r.preset}</td><td>{r.dataset_id}</td><td>{r.seed}</td>"
                + "".join(f"<td>{v}</td>" for v in values)
                + "</tr>"
            )
        headers = "".join(f"<th>{key}</th>" for key in extra_metric_keys)
        extra_metrics_section = f"""
<h2>Auxiliary NLL Diagnostics</h2>
<p class="note">Verification-only metrics saved in <code>extra_metrics</code>. For diffusion, this includes upstream-style per-dim quantities alongside the benchmark-facing <code>test_nll</code>.</p>
<table>
<thead>
  <tr><th>Preset</th><th>Dataset</th><th>Seed</th>{headers}</tr>
</thead>
<tbody>
{"".join(extra_rows)}
</tbody>
</table>"""

    metric_img = (
        f'<img src="data:image/png;base64,{fig_metric_b64}" alt="{metric} chart" '
        'style="max-width:100%">'
        if fig_metric_b64
        else ""
    )
    time_img = (
        f'<img src="data:image/png;base64,{fig_time_b64}" alt="Time chart" style="max-width:100%">'
        if fig_time_b64
        else ""
    )

    footnote_legend = (
        "<p><strong>Footnotes:</strong> "
        "‡ approximate NLL via model-native mechanics (e.g., variational ELBO, intensity "
        "quadrature + Tweedie spatial). Biased vs. true NLL; not directly comparable with "
        "exact-NLL models. Exact-NLL models are directly comparable. "
        "Approximate-NLL models are shown for reference.</p>"
    )

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
  .note {{ color: #666; font-size: 0.9em; margin-bottom: 1em; }}
</style>
</head>
<body>
<h1>STPP Benchmark Report</h1>

<h2>Section 1 — Exact-NLL Models ({metric})</h2>
<p class="note">Only models reporting exact joint NLL/event in the benchmark's canonical raw space.
Directly comparable across presets.</p>
{pivot_exact_html}
{footnote_legend}

<h2>Section 2 — All Models ({metric})</h2>
<p class="note">Includes approximate and proxy objectives. See Metric Metadata table for semantics.</p>
{pivot_all_html}

<h2>Section 3 — Metric Metadata</h2>
<table>
<thead>
  <tr><th>Preset</th><th>Objective</th><th>Val → Test (NLL kind)</th><th>Report space</th><th>NLL description</th><th>Footnote</th><th>Group</th></tr>
</thead>
<tbody>
{"".join(meta_rows)}
</tbody>
</table>
{asymmetry_section}

<h2>Section 4 — All Runs</h2>
<table>
<thead>
  <tr><th>Preset</th><th>Dataset</th><th>Seed</th><th>Val objective (metric)</th><th>Test NLL</th>
      <th>Train Time</th><th>Params</th><th>NLL kind</th><th>Note</th></tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
{breakdown_section}
{extra_metrics_section}

<h2>Figures</h2>
{metric_img}
{time_img}

<h2>LaTeX Tables</h2>
<pre>{latex}</pre>
</body>
</html>"""
