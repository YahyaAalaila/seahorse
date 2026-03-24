"""Surface visualization workflow — config + orchestration.

Architecture
------------
``SurfaceEvalSpec``  (defined in evaluation/surface.py)
    Pure evaluation parameters.  No rendering fields.  Used by benchmark too.

``SurfaceVizConfig``
    Visualization config.  Wraps ``eval_spec: SurfaceEvalSpec`` (composition)
    and adds rendering/output fields.

``SurfaceVisualizationWorkflow``
    Orchestrates evaluation via ``SurfaceEvaluator`` and renders artifacts.

Usage (post-hoc)
----------------
    runner = STPPRunner.load("/path/to/run")
    spec = SurfaceEvalSpec(n_grid=50)
    config = SurfaceVizConfig(eval_spec=spec)
    wf = SurfaceVisualizationWorkflow(config)
    artifacts = wf.run(runner, run_dir=runner._run_dir)

Usage (fit-time)
----------------
    runner.fit(train_seqs, val_seqs, surface_viz=SurfaceVizConfig(enabled=True))

After run() the workflow exposes:
    wf.surfaces_    — list[SurfaceResult] queried at each time step
    wf.references_  — list[SurfaceResult] from reference_provider (or None)

Artifact layout
---------------
    {run_dir}/surfaces/
    ├── intensity_t00_t=1.23.png
    ├── intensity_t01_t=2.47.png
    ├── panel_t00-t02.png
    ├── panel_t00-t02_with_ref.png
    └── animation_t00-t02.gif
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

_LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    from unified_stpp.evaluation.surface import SurfaceEvalSpec, SurfaceResult
    from unified_stpp.runner.runner import STPPRunner
    from unified_stpp.viz.reference import ReferenceSurfaceProvider


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SurfaceVizConfig:
    """Configuration for the surface visualization workflow.

    Wraps a ``SurfaceEvalSpec`` (evaluation parameters) via composition and
    adds rendering/output parameters.

    Pass to ``runner.fit(surface_viz=...)`` or use standalone with
    ``SurfaceVisualizationWorkflow``.

    ``reference_provider`` is an optional runtime-injected object
    (not YAML-serializable).
    """

    eval_spec: "SurfaceEvalSpec" = field(default_factory=lambda: _make_default_eval_spec())

    enabled: bool = False

    # --- Rendering ---
    render_mode: Literal["2d", "3d"] = "3d"

    # --- Output ---
    formats:         list = field(default_factory=lambda: ["png"])
    cmap:            str = "viridis"
    save_individual: bool = True     # one file per time step
    save_panel:      bool = True     # multi-panel figure (all time steps)
    animate:         bool = False
    animation_fps:   int = 4
    animate_share_colorscale: bool = True
    # When True (default), the animation uses a single fixed vmin/vmax per
    # surface_type across all frames so colors are comparable over time.

    # --- Layout ---
    reference_first: bool = False  # when True, reference appears before model

    # --- Reference surface ---
    reference_mode: Literal["none", "empirical_kde", "sthp_gt"] = "none"
    # "none"         : no reference unless reference_provider is explicitly set
    # "empirical_kde": auto-create EmpiricalKDEProvider from the sequence events
    # "sthp_gt"      : use an STHPGroundTruthProvider (must be passed via reference_provider)
    reference_provider: Optional["ReferenceSurfaceProvider"] = dataclasses.field(
        default=None, repr=False
    )


def _make_default_eval_spec() -> "SurfaceEvalSpec":
    from unified_stpp.evaluation.surface import SurfaceEvalSpec
    return SurfaceEvalSpec()


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class SurfaceVisualizationWorkflow:
    """Execute the surface visualization workflow given a fitted runner.

    Steps
    -----
    1. Delegate evaluation to ``SurfaceEvaluator.evaluate_sequence(cfg.eval_spec)``.
    2. If ``reference_provider``: compute reference surfaces on same grid.
    3. Expose queried surfaces via ``self.surfaces_`` and ``self.references_``.
    4. Save individual panels (if ``save_individual``).
    5. Save multi-panel figure (if ``save_panel``).
    6. Save animation (if ``animate``).
    7. Return artifact dict ``{name: path}``.

    Attributes set after run()
    --------------------------
    surfaces_    : list[SurfaceResult] — one per queried time step
    references_  : list[SurfaceResult] | None — from reference_provider
    """

    surfaces_: Optional[list] = None
    references_: Optional[list] = None

    def __init__(self, config: SurfaceVizConfig):
        self._cfg = config
        self.surfaces_ = None
        self.references_ = None

    def run(self, runner: "STPPRunner", run_dir: Path) -> dict[str, Path]:
        """Execute and return ``{artifact_name: path}``."""
        import matplotlib
        matplotlib.use("Agg")

        from unified_stpp.evaluation.surface import SurfaceEvalSpec, SurfaceEvaluator

        cfg = self._cfg
        spec = cfg.eval_spec
        surfaces_dir = Path(run_dir) / "surfaces"
        surfaces_dir.mkdir(parents=True, exist_ok=True)

        rolling = spec.history_mode == "rolling"

        if cfg.animate and not rolling:
            import warnings as _w
            _w.warn(
                "animate=True with history_mode='fixed' uses the same history for all "
                "frames.  Set history_mode='rolling' for rolling-history semantics where "
                "each frame conditions on events before its query time.",
                UserWarning, stacklevel=2,
            )

        # Enforce minimum frame count for useful animations
        if cfg.animate and spec.n_time_steps < 10:
            import warnings as _w
            _w.warn(
                f"animate=True with n_time_steps={spec.n_time_steps} is too few for a "
                "useful animation; raising to 10.  Set n_time_steps >= 10 to suppress.",
                UserWarning, stacklevel=2,
            )
            spec = dataclasses.replace(spec, n_time_steps=10)

        # ---- Pre-flight: verify model supports surface queries -------------
        caps = runner.model.event_model.capabilities
        if not (caps.has_intensity or caps.has_density or caps.has_native_sampler):
            model_cls = type(runner.model.event_model).__name__
            raise ValueError(
                f"Surface visualization is not supported for {model_cls} "
                f"(has_intensity={caps.has_intensity}, has_density={caps.has_density}, "
                f"has_native_sampler={caps.has_native_sampler}). "
                "The EventModel must implement query_surface()."
            )

        # ---- 1. Evaluate model surfaces via SurfaceEvaluator ---------------
        evaluator = SurfaceEvaluator(runner)

        # reference_provider and empirical_kde need the full sequence locs
        dm = runner._data_module
        seq = dm.get_original_sequence(spec.split, spec.seq_idx)
        all_locs = seq["locations"]

        eff_reference_provider = cfg.reference_provider
        if eff_reference_provider is None and cfg.reference_mode == "empirical_kde":
            from unified_stpp.viz.reference import EmpiricalKDEProvider
            eff_reference_provider = EmpiricalKDEProvider(event_locs=all_locs)

        surfaces = evaluator.evaluate_sequence(spec)

        # ---- 1b. Log per-frame diagnostic --------------------------------
        for i, s in enumerate(surfaces):
            ht = s.history_times if s.history_times is not None else np.zeros(0)
            _LOG.info(
                "[surface_viz] Frame %2d | t_query=%.4f | hist_len=%d | hist_t=[%.4f…%.4f]"
                " | surface min=%.4g max=%.4g mean=%.4g",
                i, s.t_query, len(ht),
                float(ht[0]) if len(ht) > 0 else float("nan"),
                float(ht[-1]) if len(ht) > 0 else float("nan"),
                float(s.values.min()),
                float(s.values.max()),
                float(s.values.mean()),
            )

        # ---- 2. Reference surfaces ----------------------------------------
        references = None
        if eff_reference_provider is not None:
            references = []
            for s in surfaces:
                ref = eff_reference_provider.compute(
                    history_times=s.history_times if s.history_times is not None else np.zeros(0),
                    history_locs=s.history_locs if s.history_locs is not None else np.zeros((0, all_locs.shape[-1])),
                    t_query=s.t_query,
                    xs=s.xs,
                    ys=s.ys,
                )
                references.append(ref)

        # ---- 3. Expose surfaces -------------------------------------------
        self.surfaces_ = surfaces
        self.references_ = references

        # ---- 4. Individual panels ----------------------------------------
        artifacts: dict[str, Path] = {}
        n = len(surfaces)
        step_tags = [f"t{i:02d}" for i in range(n)]

        if cfg.save_individual:
            import matplotlib.pyplot as plt
            from unified_stpp.viz.surface_plot import plot_surface
            for i, (s, tag) in enumerate(zip(surfaces, step_tags)):
                primary_prefix = "model_" if (cfg.reference_first and references is not None) else ""
                fname = f"{primary_prefix}{s.surface_type}_{tag}_t={s.t_query:.3f}.{cfg.formats[0]}"
                out_path = surfaces_dir / fname
                fig = plt.figure(figsize=(6, 5))
                ax = (fig.add_subplot(111, projection="3d")
                      if cfg.render_mode == "3d" and s.ys.size > 0
                      else fig.add_subplot(111))
                plot_surface(s, ax=ax, cmap=cfg.cmap,
                             history_locs=s.history_locs,
                             render_mode=cfg.render_mode)
                fig.savefig(out_path, bbox_inches="tight")
                plt.close(fig)
                artifacts[f"viz_surface_{tag}"] = out_path

                if references is not None:
                    ref = references[i]
                    ref_prefix = "" if (cfg.reference_first and references is not None) else "ref_"
                    ref_fname = f"{ref_prefix}{ref.surface_type}_{tag}_t={ref.t_query:.3f}.{cfg.formats[0]}"
                    ref_path = surfaces_dir / ref_fname
                    fig = plt.figure(figsize=(6, 5))
                    ax = (fig.add_subplot(111, projection="3d")
                          if cfg.render_mode == "3d" and ref.ys.size > 0
                          else fig.add_subplot(111))
                    plot_surface(ref, ax=ax, cmap=cfg.cmap, render_mode=cfg.render_mode)
                    fig.savefig(ref_path, bbox_inches="tight")
                    plt.close(fig)
                    artifacts[f"viz_ref_{tag}"] = ref_path

        # ---- 5. Multi-panel figure ----------------------------------------
        if cfg.save_panel:
            import matplotlib.pyplot as plt
            from unified_stpp.viz.multi_plot import plot_surface_panel
            step_range = f"{step_tags[0]}-{step_tags[-1]}" if n > 1 else step_tags[0]
            panel_surfaces = surfaces[:5]   # cap: animation may have 10+ steps
            hlocs_panel = [s.history_locs for s in panel_surfaces]

            panel_fig = plot_surface_panel(
                panel_surfaces,
                references=None,
                cmap=cfg.cmap,
                history_locs=hlocs_panel,
                share_colorscale=True,
                render_mode=cfg.render_mode,
            )
            panel_fname = f"panel_{step_range}.{cfg.formats[0]}"
            panel_path = surfaces_dir / panel_fname
            panel_fig.savefig(panel_path, bbox_inches="tight")
            plt.close(panel_fig)
            artifacts["viz_panel"] = panel_path

            if references is not None:
                if cfg.reference_first:
                    panel_s   = references[:5]
                    panel_ref = panel_surfaces
                else:
                    panel_s   = panel_surfaces
                    panel_ref = references[:5]
                panel_ref_fig = plot_surface_panel(
                    panel_s,
                    references=panel_ref,
                    cmap=cfg.cmap,
                    history_locs=hlocs_panel,
                    share_colorscale=True,
                    render_mode=cfg.render_mode,
                )
                panel_ref_fname = f"panel_{step_range}_with_ref.{cfg.formats[0]}"
                panel_ref_path = surfaces_dir / panel_ref_fname
                panel_ref_fig.savefig(panel_ref_path, bbox_inches="tight")
                plt.close(panel_ref_fig)
                artifacts["viz_panel_ref"] = panel_ref_path

        # ---- 6. Animation ------------------------------------------------
        if cfg.animate:
            from unified_stpp.viz.animation import animate_surface_sequence
            step_range = f"{step_tags[0]}-{step_tags[-1]}" if n > 1 else step_tags[0]
            anim_path = surfaces_dir / f"animation_{step_range}.gif"
            actual_path = animate_surface_sequence(
                surfaces,
                output_path=anim_path,
                references=references,
                history_locs=[s.history_locs for s in surfaces],
                cmap=cfg.cmap,
                fps=cfg.animation_fps,
                render_mode=cfg.render_mode,
                reference_first=cfg.reference_first,
                share_colorscale=cfg.animate_share_colorscale,
            )
            artifacts["viz_animation"] = actual_path

        return artifacts
