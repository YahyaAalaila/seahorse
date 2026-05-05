"""Sklearn-style estimator wrapper over the existing STPP runner."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from unified_stpp.config import STPPConfig
from unified_stpp.evaluation.likelihood import compute_next_event_test_nll, compute_seq_nlls
from unified_stpp.evaluation.predictive.sampling import compute_predictive_samples
from unified_stpp.runner import RunResult, STPPRunner

from .model_class_map import friendly_name_for_preset, resolve_preset

if TYPE_CHECKING:
    from .viz import STPPPlotter


class STPPEstimator:
    """Python-first estimator for spatio-temporal point process presets.

    The estimator is intentionally thin: it resolves a friendly model name to an
    existing preset, builds an ``STPPConfig``, and delegates fitting,
    persistence, sampling, and likelihood evaluation to the existing runner and
    evaluation modules.
    """

    def __init__(
        self,
        model_class: str,
        config_overrides: dict[str, Any] | None = None,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.model_class = model_class
        self.config_overrides = copy.deepcopy(config_overrides or {})
        self.device = device
        self.seed = int(seed)

        self._preset = resolve_preset(model_class)
        self._runner: STPPRunner | None = None
        self._fit_result: RunResult | None = None
        self._is_fitted = False

    def __repr__(self) -> str:
        return (
            f"STPPEstimator(model_class={self.model_class!r}, "
            f"preset={self._preset!r}, is_fitted={self._is_fitted})"
        )

    @property
    def preset(self) -> str:
        """Resolved internal preset name."""
        return self._preset

    @property
    def runner(self) -> STPPRunner:
        """Underlying fitted runner."""
        if self._runner is None:
            raise RuntimeError("Model is not initialized. Call fit() or load() first.")
        return self._runner

    @property
    def model(self):
        """Underlying fitted ``UnifiedSTPP`` model."""
        return self.runner.model

    @property
    def norm_stats(self) -> dict:
        """Normalization statistics captured by the fitted runner."""
        return self.runner.norm_stats

    @property
    def fit_result(self) -> RunResult | None:
        """Result returned by the most recent fit call, when available."""
        return self._fit_result

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_cli_values(
        self,
        *,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        cli_values = copy.deepcopy(self.config_overrides)
        training = cli_values.setdefault("training", {})
        training["device"] = self.device
        training["seed"] = self.seed
        if epochs is not None:
            training["n_epochs"] = int(epochs)
        if lr is not None:
            training["lr"] = float(lr)
        if batch_size is not None:
            training["batch_size"] = int(batch_size)

        data = cli_values.setdefault("data", {})
        data["seed"] = self.seed
        if batch_size is not None:
            data["batch_size"] = int(batch_size)
        return cli_values

    def fit(
        self,
        train_seqs: list[dict],
        val_seqs: list[dict] | None = None,
        test_seqs: list[dict] | None = None,
        *,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        dataset_id: str = "api_fit",
    ) -> "STPPEstimator":
        """Fit a preset from in-memory train/validation/test sequences."""
        if val_seqs is None:
            raise ValueError("val_seqs is required; pass an explicit validation split.")

        config = STPPConfig.from_source(
            preset=self._preset,
            cli_values=self._build_cli_values(
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            ),
        )
        runner = STPPRunner(config)
        self._fit_result = runner.fit(
            train_seqs,
            val_seqs,
            test_seqs,
            dataset_id=dataset_id,
        )
        self._runner = runner
        self._is_fitted = True
        return self

    def tune(
        self,
        train_seqs: list[dict],
        val_seqs: list[dict],
        *,
        n_trials: int = 10,
        max_epochs: int | None = None,
        patience: int | None = None,
        **tuning_overrides: Any,
    ) -> dict[str, Any]:
        """Run HPO through the existing Ray Tune path and store best overrides."""
        from unified_stpp.benchmark.hpo import run_hpo
        from unified_stpp.config.tuning import TuningConfig

        cli_values = self._build_cli_values(epochs=max_epochs)
        if patience is not None:
            cli_values.setdefault("training", {})["patience"] = int(patience)
        config = STPPConfig.from_source(preset=self._preset, cli_values=cli_values)
        tuning = TuningConfig.from_sources(
            cli_values={
                "n_trials": int(n_trials),
                "seed": self.seed,
                **tuning_overrides,
            }
        )
        best_config = run_hpo(
            config_dict=config.model_dump(mode="json"),
            tuning=tuning,
            train_seqs=train_seqs,
            val_seqs=val_seqs,
        )
        best_dict = best_config.model_dump(mode="json")
        self.config_overrides = best_dict
        return best_dict

    def predict_next(
        self,
        sequences: list[dict],
        *,
        n_samples: int = 32,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Sample next events for held-out prefixes in ``sequences``.

        The returned arrays follow ``PredictiveSamples`` semantics: one row per
        held-out next-event context, not necessarily one row per input sequence.
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        caps = self.model.event_model.capabilities
        if not caps.has_native_sampler and caps.nll_kind != "exact":
            raise NotImplementedError(
                f"Predictive sampling is not available for preset {self._preset!r}."
            )

        samples = compute_predictive_samples(
            self.runner,
            sequences,
            k=int(n_samples),
            seed=self.seed if seed is None else int(seed),
            device=self._model_device(),
        )
        return {
            "next_times": samples.next_times,
            "next_locations": samples.next_locs,
            "true_next_times": samples.true_next_times,
            "true_next_locations": samples.true_next_locs,
            "history_end_times": samples.history_end_times,
            "sequence_index": samples.sequence_index,
            "target_event_index": samples.target_event_index,
            "history_length": samples.history_length,
            "sampling_succeeded": samples.sampling_succeeded,
            "sampling_backend": samples.sampling_backend,
        }

    def evaluate(
        self,
        test_seqs: list[dict],
        *,
        metrics: list[str] | None = None,
        metric_profile: str = "core",
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate implemented likelihood metrics on test sequences."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if metrics is None:
            if metric_profile == "core":
                metrics = ["test_nll", "mean_seq_nll"]
            elif metric_profile == "predictive":
                metrics = ["test_nll"]
            else:
                raise ValueError(f"Unknown metric_profile {metric_profile!r}.")

        supported = {"test_nll", "mean_seq_nll"}
        unsupported = sorted(set(metrics) - supported)
        if unsupported:
            raise NotImplementedError(
                f"Unsupported estimator metrics: {unsupported}. "
                f"Supported metrics are: {sorted(supported)}."
            )

        results: dict[str, Any] = {}
        device = self._model_device()
        if "test_nll" in metrics:
            summary = compute_next_event_test_nll(
                self.runner,
                test_seqs,
                device=device,
                predictive_samples=32 if metric_profile == "predictive" else 8,
                seed=self.seed if seed is None else int(seed),
            )
            results["test_nll"] = float(summary.get("mean_nll", float("nan")))
            backend = summary.get("sampling_backend")
            if backend is not None:
                results["sampling_backend"] = backend

        if "mean_seq_nll" in metrics:
            seq_nlls = compute_seq_nlls(self.runner, test_seqs, device=device)
            results["mean_seq_nll"] = float(np.nanmean(seq_nlls)) if len(seq_nlls) else float("nan")
        return results

    def plot_intensity(self, context: dict, **kwargs: Any) -> dict[str, Any]:
        """Render an intensity diagnostic via ``STPPPlotter``."""
        return self.plotter.plot_intensity(context, **kwargs)

    def plot_kde_surface(self, context: dict, **kwargs: Any) -> dict[str, Any]:
        """Render predictive samples via ``STPPPlotter``."""
        return self.plotter.plot_kde_surface(context, **kwargs)

    def save(self, path: str | Path) -> Path:
        """Save the fitted runner through ``STPPRunner.save``."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.runner.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "STPPEstimator":
        """Load an estimator from an existing runner save/run directory."""
        runner = STPPRunner.load(path)
        preset = runner.config.model.preset
        estimator = cls(model_class=friendly_name_for_preset(preset))
        estimator._preset = preset
        estimator._runner = runner
        estimator._is_fitted = True
        return estimator

    @property
    def plotter(self) -> "STPPPlotter":
        """Plotting helper bound to this fitted estimator."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        from .viz import STPPPlotter

        return STPPPlotter(self.runner, self.runner._run_dir)
