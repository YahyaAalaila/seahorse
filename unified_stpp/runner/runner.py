"""
STPPRunner — orchestrates fit → evaluate → save/load for any STPP preset.
"""

from __future__ import annotations

import contextlib
import logging
import random
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch

from unified_stpp.config import STPPConfig
from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.training.lightning_module import STPPLightningModule
from unified_stpp.training.data_module import STPPDataModule
from unified_stpp.utils import deep_update
from .artifacts import (
    make_run_id, update_latest_symlink, save_run_artifacts,
    checkpoint_file, load_state_dict, load_norm_stats,
)
from .results import RunResult



# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _require(value, msg: str):
    """Return *value* unchanged, or raise ``ValueError(msg)`` if it is ``None``."""
    if value is None:
        raise ValueError(msg)
    return value


def _seed_fit(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before model construction."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Also seed Lightning worker processes when num_workers > 0.
    pl.seed_everything(seed, workers=True, verbose=False)


# ---------------------------------------------------------------------------
# Terminal UX helpers
# ---------------------------------------------------------------------------

def _print_run_header(
    preset: str, dataset_id: str, run_dir: Path, n_params: int, device: str
) -> None:
    """Print a concise run header before training starts."""
    print(f"\n  preset:   {preset}   dataset: {dataset_id}   params: {n_params:,}")
    print(f"  run dir:  {run_dir}")
    print(f"  device:   {device}\n")


@contextlib.contextmanager
def _quiet_lightning():
    """Suppress known-harmless Lightning / DataLoader verbosity during training.

    Suppressed:
    - Lightning INFO-level hardware-detection banners
    - DataLoader worker-count hint
    - float32 matmul-precision hint
    All other warnings and errors pass through unchanged.
    """
    _loggers = [
        logging.getLogger("pytorch_lightning"),
        logging.getLogger("lightning.pytorch"),
        logging.getLogger("lightning.fabric"),
    ]
    _saved = [(lg, lg.level) for lg in _loggers]
    for lg, _ in _saved:
        lg.setLevel(logging.WARNING)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings("ignore", ".*set_float32_matmul_precision.*")
        warnings.filterwarnings("ignore", ".*exists and is not empty.*")
        warnings.filterwarnings("ignore", ".*MPS detected.*")
        warnings.filterwarnings("ignore", ".*GPU available but not used.*")
        try:
            yield
        finally:
            for lg, lvl in _saved:
                lg.setLevel(lvl)


class STPPRunner:
    """Orchestrates the full fit → evaluate → save/load lifecycle.

    Example
    -------
    >>> runner = STPPRunner.from_preset("auto_stpp")
    >>> result = runner.fit(train_seqs, val_seqs, test_seqs)
    >>> print(result.val_objective)
    >>> runner.save("/tmp/my_run/")
    """

    def __init__(self, config: STPPConfig):
        self.config = config
        self._lightning_module: Optional[STPPLightningModule] = None
        self._data_module: Optional[STPPDataModule] = None
        self._run_dir: Optional[Path] = None
        self._norm_stats: Optional[dict] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, preset: str) -> "STPPRunner":
        """Build a runner from a bundled preset YAML (e.g. ``"auto_stpp"``)."""
        return cls(STPPConfig.from_preset(preset))

    @classmethod
    def from_yaml(cls, path) -> "STPPRunner":
        """Build a runner from a custom YAML config file."""
        return cls(STPPConfig.from_yaml(path))

    @classmethod
    def from_config_source(
        cls,
        preset: "str | None",
        config: "str | None",
        *,
        cli_values: Optional[dict] = None,
        override_list: Optional[list[str]] = None,
    ) -> "STPPRunner":
        """Build a runner from a preset/YAML source plus CLI-layer overrides.

        ``config`` takes precedence when both parameters are provided.
        Programmatic callers must supply at least one source.
        """
        return cls(
            STPPConfig.from_source(
                preset=preset,
                config=config,
                cli_values=cli_values,
                override_list=override_list,
            )
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The trained ``UnifiedSTPP`` model (raises if ``fit()`` not called)."""
        if self._lightning_module is None:
            raise RuntimeError("Model not built yet — call fit() first.")
        return self._lightning_module.model

    @property
    def norm_stats(self) -> dict:
        """Normalization statistics from training (raises if not fitted)."""
        return _require(
            self._norm_stats,
            "norm_stats not available — call fit() or load() before accessing norm_stats.",
        )

    @property
    def data_module(self) -> STPPDataModule:
        """The fitted ``STPPDataModule`` (raises if ``fit()`` not called)."""
        if self._data_module is None:
            raise RuntimeError("DataModule not built yet — call fit() first.")
        return self._data_module

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_seqs: list[dict],
        val_seqs: list[dict],
        test_seqs: Optional[list[dict]] = None,
        data_module: Optional[STPPDataModule] = None,
        dataset_id: str = "unknown",
        extra_callbacks: Optional[list[pl.Callback]] = None,
    ) -> RunResult:
        """Train the model and return a ``RunResult``.

        Parameters
        ----------
        train_seqs:   Training sequences — list of ``{"times": ..., "locations": ...}``.
        val_seqs:     Validation sequences.
        test_seqs:    Optional test sequences; ``result.test_nll`` is ``nan`` if omitted.
        data_module:     Override the auto-built data module (researcher escape hatch).
        dataset_id:      Human-readable name stored in the returned ``RunResult``.
        extra_callbacks: Optional Lightning callbacks for orchestration layers
                         such as HPO. Normal fit/bench callers leave this unset.
        """
        _seed_fit(self.config.data.seed)
        dm = self._prepare_data_module(train_seqs, val_seqs, test_seqs, data_module)
        self._sync_model_spatial_dim_from_data(dm)
        resume_ckpt_path = self._resume_checkpoint_path()
        run_dir = self._prepare_run_dir(self.config.model.preset)
        for callback in extra_callbacks or []:
            bind = getattr(callback, "bind_run_context", None)
            if callable(bind):
                bind(run_dir=run_dir)
        with _quiet_lightning():
            model, lm, trainer = self._build_training_stack(
                dm,
                run_dir,
                extra_callbacks=extra_callbacks,
            )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _print_run_header(
            self.config.model.preset, dataset_id, run_dir, n_params,
            self.config.training.device,
        )
        t0 = time.perf_counter()
        with _quiet_lightning():
            trainer.fit(
                lm,
                datamodule=dm,
                ckpt_path=None if resume_ckpt_path is None else str(resume_ckpt_path),
            )
        return self._finalize_fit(trainer, lm, dm, dataset_id, t0, run_dir, test_seqs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_data_module(
        self,
        train_seqs: list[dict],
        val_seqs: list[dict],
        test_seqs: Optional[list[dict]],
        data_module: Optional[STPPDataModule],
    ) -> STPPDataModule:
        """Build (or accept) the data module."""
        if data_module is not None:
            return data_module
        bundle = self.config.build_data_bundle(train_seqs, val_seqs, test_seqs)
        return STPPDataModule(
            bundle,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            seed=self.config.data.seed,
        )

    def _resume_checkpoint_path(self) -> Optional[Path]:
        raw = self.config.training.resume_from_checkpoint
        if not raw:
            return None
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path

    def _prepare_run_dir(self, preset_name: str) -> Path:
        """Create a unique run directory and save the original config (crash-safe)."""
        run_dir = self.config.logging.run_dir(preset_name, make_run_id())
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(run_dir / "config.yaml")
        return run_dir

    def _build_training_stack(
        self,
        dm: STPPDataModule,
        run_dir: Path,
        *,
        extra_callbacks: Optional[list[pl.Callback]] = None,
    ):
        """Build model, LightningModule, and Trainer."""
        model = self._build_model(dm)
        lm = STPPLightningModule(model=model, tc=self.config.training)
        accelerator = ConfigRegistry.resolve_accelerator(
            self.config.model.preset, self.config.training.device
        )
        loggers = self.config.logging.build_loggers(run_dir)
        trainer = self.config.training.build_trainer(
            run_dir,
            accelerator,
            loggers,
            monitor_key=lm.val_monitor_key,
            extra_callbacks=extra_callbacks,
        )
        return model, lm, trainer

    def _finalize_fit(
        self,
        trainer,
        lm: STPPLightningModule,
        dm: STPPDataModule,
        dataset_id: str,
        start_time: float,
        run_dir: Path,
        test_seqs: Optional[list[dict]] = None,
    ) -> RunResult:
        """Stash trained state, collect result, update symlink."""
        self._lightning_module = lm
        self._data_module = dm
        self._run_dir = run_dir
        self._restore_selected_checkpoint(lm, run_dir)
        result = self._collect_result(
            trainer,
            lm,
            dm,
            dataset_id,
            time.perf_counter() - start_time,
            run_dir,
            test_seqs,
        )
        self._norm_stats = result.norm_stats
        update_latest_symlink(run_dir)
        return result

    def _build_model(self, dm: STPPDataModule):
        """Construct and return the UnifiedSTPP model."""
        data_overrides = ConfigRegistry.data_init_overrides(self.config.model.preset, dm)
        self._effective_model_overrides = data_overrides or {}
        return self.config.model.build_model(extra_overrides=data_overrides or None)

    def _sync_model_spatial_dim_from_data(self, dm: STPPDataModule) -> None:
        """Make model construction follow the loaded dataset location dimension."""
        inferred = self._infer_spatial_dim(dm)
        if inferred is None:
            return
        configured = int(self.config.model.spatial_dim)
        if configured != inferred:
            self.config.model.spatial_dim = inferred

    @staticmethod
    def _infer_spatial_dim(dm: STPPDataModule) -> Optional[int]:
        dataset = getattr(dm, "train_dataset", None)
        sequences = getattr(dataset, "sequences", None)
        if not sequences:
            return None
        locations = np.asarray(sequences[0].get("locations"))
        if locations.ndim < 2:
            return None
        return int(locations.shape[-1])

    def _extract_fit_metrics(
        self,
        trainer: pl.Trainer,
        lm: STPPLightningModule,
        dm: STPPDataModule,
    ) -> tuple[float, float, float, float, dict[str, float]]:
        """Return fit metrics and test-time extra diagnostics after training.

        val_objective is taken from the best checkpoint score when available.
        test_nll / temporal_nll / spatial_nll are nan when no test dataset is present.
        """
        from pytorch_lightning.callbacks import ModelCheckpoint
        ckpt_callback = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))

        raw_val = trainer.callback_metrics.get(lm.val_monitor_key)
        val_objective = float(raw_val) if raw_val is not None else float("nan")
        if ckpt_callback.best_model_score is not None:
            val_objective = float(ckpt_callback.best_model_score)

        test_nll = float("nan")
        temporal_nll = float("nan")
        spatial_nll = float("nan")
        test_extra_metrics: dict[str, float] = {}
        if dm._bundle.test_dataset is not None:
            test_results = trainer.test(
                lm,
                datamodule=dm,
                verbose=False,
                ckpt_path=self.config.training.checkpoint_select,
            )
            if test_results:
                r = test_results[0]
                test_nll = float(r.get("test/nll", float("nan")))
                temporal_nll = float(r.get("test/temporal_nll", float("nan")))
                spatial_nll = float(r.get("test/spatial_nll", float("nan")))
                for key, value in r.items():
                    if not key.startswith("test/"):
                        continue
                    metric_name = key[len("test/") :]
                    if metric_name in {"nll", "temporal_nll", "spatial_nll"}:
                        continue
                    test_extra_metrics[metric_name] = float(value)

        return val_objective, test_nll, temporal_nll, spatial_nll, test_extra_metrics

    def _extract_checkpoint_path(self, trainer: pl.Trainer) -> "Optional[Path]":
        """Return the selected checkpoint path if save_checkpoints is enabled."""
        from pytorch_lightning.callbacks import ModelCheckpoint
        ckpt_callback = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))
        if not self.config.logging.save_checkpoints:
            return None
        selected = self.config.training.checkpoint_select
        if selected == "best" and ckpt_callback.best_model_path:
            return Path(ckpt_callback.best_model_path)
        if selected == "last":
            last_ckpt = checkpoint_file(self._run_dir or Path(ckpt_callback.dirpath), selection="last")
            if last_ckpt.exists():
                return last_ckpt
        return None

    def _restore_selected_checkpoint(self, lm: STPPLightningModule, run_dir: Path) -> None:
        """Keep the in-memory runner aligned with the configured checkpoint policy."""
        try:
            state = load_state_dict(run_dir, selection=self.config.training.checkpoint_select)
        except FileNotFoundError:
            return
        lm.model.load_state_dict(state)
    

    def _collect_result(
        self,
        trainer: pl.Trainer,
        lm: STPPLightningModule,
        dm: STPPDataModule,
        dataset_id: str,
        train_time: float,
        run_dir: Path,
        test_seqs: Optional[list[dict]] = None,
    ) -> RunResult:
        """Assemble and persist a RunResult from the completed training run."""
        from unified_stpp.evaluation.likelihood import compute_next_event_test_nll

        cfg = self.config
        val_objective, reported_test_nll, reported_temporal_nll, reported_spatial_nll, test_extra_metrics = self._extract_fit_metrics(
            trainer, lm, dm
        )
        ckpt_path = self._extract_checkpoint_path(trainer)
        norm_stats = dm.get_norm_stats(cfg.data.normalize)
        self._norm_stats = norm_stats
        caps = lm.model.event_model.capabilities
        native_test_nll = float(test_extra_metrics.get("native_nll", reported_test_nll))
        native_temporal_nll = float(
            test_extra_metrics.get("native_temporal_nll", reported_temporal_nll)
        )
        native_spatial_nll = float(
            test_extra_metrics.get("native_spatial_nll", reported_spatial_nll)
        )
        benchmark_test_nll = {
            "mean_nll": float("nan"),
            "method": "unavailable_no_test_data",
            "kind": caps.nll_kind,
            "report_space": "native",
            "description": "held-out next-event test NLL unavailable (no test split)",
            "footnote": caps.nll_footnote,
            "per_context_nll": np.zeros((0,), dtype=np.float32),
            "n_contexts": 0,
            "n_scored_contexts": 0,
            "n_missing_contexts": 0,
            "sampling_backend": None,
        }
        if test_seqs:
            model_device = next(lm.model.parameters()).device
            benchmark_test_nll = compute_next_event_test_nll(
                self,
                test_seqs,
                device=model_device,
                predictive_samples=cfg.training.predictive_test_nll_samples,
                seed=cfg.data.seed,
            )
            test_extra_metrics["test_nll_sampling_backend"] = (
                benchmark_test_nll.get("sampling_backend")
            )
        test_extra_metrics["native_nll_description"] = caps.nll_description
        test_extra_metrics["native_nll_kind"] = caps.nll_kind
        test_extra_metrics["native_nll_report_space"] = "native"
        result = RunResult(
            preset=cfg.model.preset,
            preset_status=ConfigRegistry.canonical_status(cfg.model.preset),
            dataset_id=dataset_id,
            seed=cfg.data.seed,
            val_objective=val_objective,
            val_metric_key=caps.metric_key,
            test_nll=benchmark_test_nll["mean_nll"],
            train_time_sec=train_time,
            n_params=sum(p.numel() for p in lm.model.parameters() if p.requires_grad),
            effective_config=self._effective_config_dump(),
            checkpoint_path=ckpt_path,
            norm_stats=norm_stats,
            run_dir=run_dir,
            training_objective=caps.training_objective,
            objective_description=caps.objective_description,
            nll_kind=benchmark_test_nll["kind"],
            nll_description=benchmark_test_nll["description"],
            nll_footnote=benchmark_test_nll["footnote"],
            nll_report_space=benchmark_test_nll["report_space"],
            test_nll_method=benchmark_test_nll["method"],
            test_nll_contexts=benchmark_test_nll["n_contexts"],
            test_nll_scored_contexts=benchmark_test_nll["n_scored_contexts"],
            test_nll_missing_contexts=benchmark_test_nll["n_missing_contexts"],
            native_test_nll=native_test_nll,
            native_temporal_nll=native_temporal_nll,
            native_spatial_nll=native_spatial_nll,
            temporal_nll=native_temporal_nll,
            spatial_nll=native_spatial_nll,
            extra_metrics=test_extra_metrics,
        )
        save_run_artifacts(run_dir, result, cfg)
        return result

    def _effective_config_dump(self) -> dict:
        """Return the config dict actually used to build the fitted model."""
        raw = self.config.model_dump()
        model_overrides = getattr(self, "_effective_model_overrides", None) or {}
        if model_overrides:
            deep_update(raw.setdefault("model", {}), model_overrides)
        return raw

    # ------------------------------------------------------------------
    # Post-fit evaluation
    # ------------------------------------------------------------------

    def evaluate(self, *args, **kwargs) -> dict[str, Path]:
        """Deprecated compatibility shim for the retired runner-owned eval path."""
        raise RuntimeError(
            "STPPRunner.evaluate() has been retired. Use "
            "'python -m unified_stpp evaluate metrics ...', "
            "'python -m unified_stpp evaluate predictive-compare ...', or "
            "'python -m unified_stpp evaluate surface ...', or call "
            "PredictiveComparator / SurfaceDiagnosticEvaluator directly."
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> Path:
        """Save config YAML + model weights to *path* directory.

        Creates two files: ``config.yaml`` and ``model.ckpt`` (state_dict).
        """
        if self._lightning_module is None:
            raise RuntimeError("Call fit() before save().")

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(out / "config.yaml")
        torch.save(self.model.state_dict(), out / "model.ckpt")
        return out

    @classmethod
    def load(cls, path) -> "STPPRunner":
        """Restore a runner from a saved directory.

        Looks for ``config.yaml`` + (in order of preference):
        1. ``checkpoints/{training.checkpoint_select}.ckpt`` — Lightning checkpoint written by ``fit()``
        2. ``model.ckpt``           — legacy plain state-dict written by ``save()``
        """
        p = Path(path).resolve()
        cfg_path = p / "resolved_config.yaml"
        if not cfg_path.exists():
            cfg_path = p / "config.yaml"
        config = STPPConfig.from_yaml(cfg_path, sanitize=False)
        runner = cls(config)
        model = config.model.build_model()
        model.load_state_dict(load_state_dict(p, selection=config.training.checkpoint_select))
        runner._lightning_module = STPPLightningModule(model=model, tc=config.training)
        runner._run_dir = p
        runner._norm_stats = load_norm_stats(p)
        return runner
