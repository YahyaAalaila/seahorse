"""Plotting helpers for the lightweight estimator API."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


def _dump_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class STPPPlotter:
    """Small plotting facade around existing evaluation and rendering helpers."""

    def __init__(self, runner, run_dir: Path | None = None) -> None:
        self._runner = runner
        self._run_dir = run_dir

    def plot_intensity(
        self,
        context: dict,
        *,
        x_nstep: int = 81,
        y_nstep: int = 81,
        t_nstep: int = 41,
        future_horizon: float | None = None,
        frame_index: int = -1,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Render a surface diagnostic for one context sequence."""
        if self._run_dir is None:
            raise RuntimeError("plot_intensity requires a fitted or loaded runner with a run directory.")

        from seahorse.evaluation.runtime import HistoryQuery, RunTarget
        from seahorse.evaluation.surface import SurfaceDiagnosticEvaluator, SurfaceDiagnosticSpec
        from seahorse.viz import SurfaceRenderConfig, render_surface_bundle

        out_dir = Path(output_path) if output_path is not None else Path(tempfile.mkdtemp())
        context_path = out_dir / "context.jsonl"
        _dump_jsonl([context], context_path)

        preset = self._runner.config.model.preset
        profile = (
            "future_exact"
            if preset in {"njsde", "neural_cond_gmm", "neural_jumpcnf", "neural_attncnf"}
            else "history_frame"
        )
        spec = SurfaceDiagnosticSpec(
            profile=profile,
            x_nstep=x_nstep,
            y_nstep=y_nstep,
            t_nstep=t_nstep,
            future_horizon=future_horizon,
            frame_index=frame_index,
            round_time=True,
            trunc=None,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            spatial_chunk_size=None,
            device="auto",
        )
        result = SurfaceDiagnosticEvaluator().evaluate(
            RunTarget(run=Path(self._run_dir)),
            HistoryQuery(
                history_path=context_path,
                split="test",
                seq_idx=0,
                history_length=0,
            ),
            spec,
        )
        artifacts = render_surface_bundle(
            result,
            out_dir,
            SurfaceRenderConfig(interactive=True),
        )
        return {
            "html": str(artifacts.get("interactive_html", out_dir / "surface.html")),
            "run_dir": str(out_dir),
        }

    def plot_kde_surface(
        self,
        context: dict,
        *,
        n_samples: int = 128,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Render a simple next-event predictive-sample summary as HTML."""
        from seahorse.evaluation.predictive.sampling import compute_predictive_samples

        device = next(self._runner.model.parameters()).device
        out_dir = Path(output_path) if output_path is not None else Path(tempfile.mkdtemp())
        out_dir.mkdir(parents=True, exist_ok=True)

        samples = compute_predictive_samples(
            self._runner,
            [context],
            k=int(n_samples),
            seed=0,
            device=device,
        )
        if samples.next_times.size == 0:
            raise ValueError("No held-out next-event context was available for plotting.")

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise RuntimeError(
                "plot_kde_surface requires plotly. Install plotly or use predictive arrays directly."
            ) from exc

        times = samples.next_times[0]
        locs = samples.next_locs[0]
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Sampled next times", "Sampled next locations"),
        )
        fig.add_trace(go.Histogram(x=times, name="next_times"), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=locs[:, 0],
                y=locs[:, 1],
                mode="markers",
                name="next_locations",
            ),
            row=1,
            col=2,
        )
        fig.update_layout(
            title_text=f"Next-event samples ({samples.sampling_backend})",
            showlegend=True,
        )
        html_path = out_dir / "kde_surface.html"
        fig.write_html(html_path)
        return {
            "html": str(html_path),
            "run_dir": str(out_dir),
            "sampling_backend": samples.sampling_backend,
        }
