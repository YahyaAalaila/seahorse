"""Runner for the registered NumPy/SciPy statistical STPP baselines.

The class mirrors the shared public lifecycle (fit, save, load and evaluate)
without constructing a Lightning module or a Torch data module. Statistical
models receive already frozen sequence dictionaries directly.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from seahorse.models.configs import ConfigRegistry
from seahorse.models.statistical import build_statistical_model, model_from_state

from .artifacts import make_run_id, save_run_artifacts, update_latest_symlink
from .results import RunResult


def _scipy_version() -> str | None:
    """Return the installed SciPy version without making it a new import-time requirement."""
    try:
        import scipy
    except ImportError:
        return None
    return scipy.__version__


class StatisticalSTPPRunner:
    """Fit and evaluate one registered non-neural statistical STPP preset."""

    runner_name = "statstpp"

    def __init__(self, config: Any) -> None:
        if ConfigRegistry.runner_kind(config.model.preset) != "statstpp":
            raise ValueError("StatisticalSTPPRunner requires a statstpp preset.")
        self.config = config
        self.statistical_model: Any | None = None
        self._run_dir: Path | None = None
        self._norm_stats: dict[str, Any] = {}

    @property
    def model(self) -> Any:
        """The fitted source-model adapter."""
        if self.statistical_model is None:
            raise RuntimeError("Model is not built yet; call fit() or load() first.")
        return self.statistical_model

    @property
    def norm_stats(self) -> dict[str, Any]:
        return self._norm_stats

    @property
    def capabilities(self) -> SimpleNamespace:
        """Evaluation capabilities without pretending this is a Torch model."""
        preset = self.config.model.preset
        exact = preset in {"homogeneous_poisson", "spatial_poisson", "seasonal_poisson", "hawkes_exp_gaussian"}
        return SimpleNamespace(
            nll_kind="exact" if exact else "approx",
            nll_footnote=(
                "" if exact else "Omori/Pareto temporal and spatial integrals use declared source quadrature."
            ),
            has_density=False,
            has_intensity=False,
            has_native_sampler=True,
        )

    def fit(
        self,
        train_seqs: Sequence[Mapping[str, Any]],
        val_seqs: Sequence[Mapping[str, Any]] | None = None,
        test_seqs: Sequence[Mapping[str, Any]] | None = None,
        *,
        dataset_id: str = "unknown",
        data_module: Any | None = None,
        extra_callbacks: Sequence[Any] | None = None,
        conditional_contexts: Any | None = None,
        **unexpected: Any,
    ) -> RunResult:
        """Fit complete observation windows and persist a portable run.

        Validation is optional for direct API use. Without it the objective is
        ``NaN`` rather than a training-set proxy, so HPO cannot treat it as a
        selected validation score.
        """
        if unexpected:
            raise TypeError(f"Unsupported statistical fit arguments: {sorted(unexpected)}")
        if data_module is not None:
            raise TypeError(
                "Statistical runners consume frozen sequences directly, not a Torch data module."
            )
        if extra_callbacks:
            raise TypeError("Statistical runners do not support Lightning callbacks.")
        options = dict(self.config.model.build_overrides)
        inference = dict(getattr(self.config, "statistical", {}) or {})
        self.statistical_model = build_statistical_model(
            self.config.model.preset, options
        )
        fit_started = time.perf_counter()
        if conditional_contexts is not None:
            if not hasattr(self.model, "fit_conditional"):
                raise TypeError("This statistical preset does not implement conditional-context fitting.")
            diagnostics = self.model.fit_conditional(
                conditional_contexts, seed=int(self.config.data.seed), inference=inference
            )
        else:
            diagnostics = self.model.fit(
                list(train_seqs), seed=int(self.config.data.seed), inference=inference
            )
        fit_elapsed = time.perf_counter() - fit_started

        run_dir = self.config.logging.run_dir(self.config.model.preset, make_run_id())
        run_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir
        val_values = [self.model.full_window_nll(dict(seq)) for seq in (val_seqs or [])]
        val_objective = float(np.mean(val_values)) if val_values else float("nan")

        from seahorse.evaluation.statistical import compute_statistical_next_event_nll

        score = compute_statistical_next_event_nll(self, list(test_seqs or []))
        result = RunResult(
            preset=self.config.model.preset,
            dataset_id=dataset_id,
            seed=int(self.config.data.seed),
            val_objective=val_objective,
            val_metric_key="full_window_nll",
            test_nll=float(score["mean_nll"]),
            train_time_sec=fit_elapsed,
            n_params=int(self.model.n_params),
            effective_config=self.config.model_dump(mode="json"),
            run_dir=run_dir,
            checkpoint_path=run_dir / "statistical_checkpoint.json",
            training_objective=str(diagnostics.get("method", "statistical_fit")),
            objective_description=(
                "statistical fit; objective and conditioning are recorded in diagnostics.method"
            ),
            nll_kind=str(score["kind"]),
            nll_description=str(score["description"]),
            nll_footnote=str(score.get("footnote", "")),
            nll_report_space=str(score["report_space"]),
            test_nll_method=str(score["method"]),
            test_nll_contexts=int(score["n_contexts"]),
            test_nll_scored_contexts=int(score["n_scored_contexts"]),
            test_nll_missing_contexts=int(score["n_missing_contexts"]),
            norm_stats={},
            extra_metrics={
                "inference": inference,
                "diagnostics": diagnostics,
                "environment": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "scipy": _scipy_version(),
                },
            },
        )
        self.save(run_dir)
        save_run_artifacts(run_dir, result, self.config)
        update_latest_symlink(run_dir)
        return result

    def evaluate(
        self, test_seqs: Sequence[Mapping[str, Any]], **unexpected: Any
    ) -> dict[str, Any]:
        """Return the standard frozen-prefix next-event likelihood summary."""
        if unexpected:
            raise TypeError(f"Unsupported statistical evaluation arguments: {sorted(unexpected)}")
        from seahorse.evaluation.statistical import compute_statistical_next_event_nll

        return compute_statistical_next_event_nll(self, list(test_seqs))

    def save(self, path: str | Path) -> Path:
        """Write a self-contained configuration and JSON model state."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(out / "config.yaml")
        payload = {
            "format_version": 1,
            "runner": self.runner_name,
            "preset": self.config.model.preset,
            "state": self.model.to_state(),
        }
        (out / "statistical_checkpoint.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        return out

    @classmethod
    def load(cls, path: str | Path) -> "StatisticalSTPPRunner":
        """Restore a runner saved by :meth:`save` without a Torch dependency."""
        from seahorse.config import STPPConfig

        directory = Path(path).resolve()
        config_path = directory / "resolved_config.yaml"
        if not config_path.exists():
            config_path = directory / "config.yaml"
        config = STPPConfig.from_yaml(config_path, sanitize=False)
        payload = json.loads((directory / "statistical_checkpoint.json").read_text())
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported statistical checkpoint format version.")
        if payload.get("runner") != cls.runner_name:
            raise ValueError(f"Not a statistical runner checkpoint: {directory}")
        if payload.get("preset") != config.model.preset:
            raise ValueError("Checkpoint preset and saved configuration disagree.")
        runner = cls(config)
        runner.statistical_model = model_from_state(payload["state"])
        runner.statistical_model.preset = config.model.preset
        runner._run_dir = directory
        return runner
