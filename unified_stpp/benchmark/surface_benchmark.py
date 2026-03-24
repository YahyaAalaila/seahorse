"""Surface benchmark: compare multiple fitted models on a shared evaluation spec.

Usage
-----
    from unified_stpp.evaluation.surface import SurfaceEvalSpec, SurfaceEvaluator
    from unified_stpp.benchmark.surface_benchmark import SurfaceBenchmark

    spec = SurfaceEvalSpec(split="val", seq_idx=0, n_grid=50, n_time_steps=3)
    bench = SurfaceBenchmark(spec)

    runners = {"deep_stpp": runner_a, "auto_stpp": runner_b}
    results = bench.run(runners)          # dict[str, list[SurfaceResult]]
    fig = bench.plot(results)             # matplotlib Figure
    fig.savefig("comparison.png")

Design
------
All models receive **exactly the same** ``SurfaceEvalSpec``.  The benchmark
enforces this by passing a single spec to every ``SurfaceEvaluator`` — no
per-model configuration leakage.  Models that return ``proxy_kde`` surfaces
are shown with a beige background in the comparison figure (via the
``comparable=False`` flag on their ``SurfaceResult``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from unified_stpp.evaluation.surface import SurfaceEvalSpec, SurfaceResult
    from unified_stpp.runner.runner import STPPRunner
    from unified_stpp.viz.reference import ReferenceSurfaceProvider

_LOG = logging.getLogger(__name__)


class SurfaceBenchmark:
    """Evaluate surface visualization across multiple model families.

    All models receive the same ``SurfaceEvalSpec``, ensuring that
    - the same sequence (``split``, ``seq_idx``) is used,
    - the same history window is applied,
    - the same ``t_queries`` are evaluated,
    - the same spatial domain and grid resolution are used.

    Parameters
    ----------
    spec : SurfaceEvalSpec
        Shared evaluation spec.  Every runner passed to ``run()`` uses this
        exact spec — no per-model overrides.
    """

    def __init__(self, spec: "SurfaceEvalSpec"):
        self.spec = spec

    def run(
        self,
        runners: dict[str, "STPPRunner"],
        reference_provider: Optional["ReferenceSurfaceProvider"] = None,
    ) -> dict[str, list["SurfaceResult"]]:
        """Evaluate each runner and return per-model surface frames.

        Parameters
        ----------
        runners : dict[model_name → STPPRunner]
            Fitted runners to compare.  All must have been fitted before calling.
        reference_provider : optional
            If provided, a reference surface is computed for each frame and
            stored under the key ``"__reference__"`` in the returned dict.

        Returns
        -------
        dict[str, list[SurfaceResult]]
            Keys are model names (from ``runners``), plus ``"__reference__"``
            if ``reference_provider`` is given.  Each list contains one
            ``SurfaceResult`` per evaluated time step.
        """
        from unified_stpp.evaluation.surface import SurfaceEvaluator

        results: dict[str, list] = {}

        for name, runner in runners.items():
            _LOG.info("SurfaceBenchmark: evaluating '%s'", name)
            evaluator = SurfaceEvaluator(runner)
            frames = evaluator.evaluate_sequence(self.spec)
            for f in frames:
                f.model_name = name
            results[name] = frames

        if reference_provider is not None and results:
            import numpy as np
            # Use the first model's frames as geometry anchors
            anchor_frames = next(iter(results.values()))
            ref_frames = []
            for frame in anchor_frames:
                ht = frame.history_times if frame.history_times is not None else np.zeros(0)
                hl = frame.history_locs if frame.history_locs is not None else np.zeros((0, 2))
                ref = reference_provider.compute(
                    history_times=ht,
                    history_locs=hl,
                    t_query=frame.t_query,
                    xs=frame.xs,
                    ys=frame.ys,
                )
                ref_frames.append(ref)
            results["__reference__"] = ref_frames

        return results

    def plot(
        self,
        results: dict[str, list["SurfaceResult"]],
        render_mode: str = "2d",
        cmap: str = "viridis",
        suptitle: Optional[str] = None,
        n_steps: int = 5,
    ) -> "Figure":
        """Render a model-comparison figure.

        Parameters
        ----------
        results : output of ``run()``
        render_mode : ``"2d"`` (heatmap, default) or ``"3d"`` (surface plot)
        cmap : matplotlib colormap name
        suptitle : optional figure super-title
        n_steps : cap on time steps per model (default 5; ``plot_model_comparison``
                  requires ≤ 5)

        Returns
        -------
        matplotlib.figure.Figure
        """
        from unified_stpp.viz.multi_plot import plot_model_comparison

        references = results.pop("__reference__", None)

        # Cap time steps per model
        trimmed = {name: frames[:n_steps] for name, frames in results.items()}
        if references is not None:
            references = references[:n_steps]

        fig = plot_model_comparison(
            surfaces_by_model=trimmed,
            references=references,
            cmap=cmap,
            share_colorscale_within_type=True,
            suptitle=suptitle,
            render_mode=render_mode,
        )
        return fig
