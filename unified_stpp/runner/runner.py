"""
STPPRunner — orchestrates fit → evaluate → save/load for any STPP preset.
"""

from __future__ import annotations

import contextlib
import logging
import time
import warnings
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch

from unified_stpp.config import STPPConfig
from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.training.lightning_module import STPPLightningModule
from unified_stpp.training.data_module import STPPDataModule
from unified_stpp.utils import deep_update
from .artifacts import (
    make_run_id, update_latest_symlink, save_run_artifacts, _extend_viz_manifest,
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
    ) -> RunResult:
        """Train the model and return a ``RunResult``.

        Parameters
        ----------
        train_seqs:   Training sequences — list of ``{"times": ..., "locations": ...}``.
        val_seqs:     Validation sequences.
        test_seqs:    Optional test sequences; ``result.test_nll`` is ``nan`` if omitted.
        data_module:  Override the auto-built data module (researcher escape hatch).
        dataset_id:   Human-readable name stored in the returned ``RunResult``.
        """
        dm = self._prepare_data_module(train_seqs, val_seqs, test_seqs, data_module)
        run_dir = self._prepare_run_dir(self.config.model.preset)
        with _quiet_lightning():
            model, lm, trainer = self._build_training_stack(dm, run_dir)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _print_run_header(
            self.config.model.preset, dataset_id, run_dir, n_params,
            self.config.training.device,
        )
        t0 = time.perf_counter()
        with _quiet_lightning():
            trainer.fit(lm, datamodule=dm)
        return self._finalize_fit(trainer, lm, dm, dataset_id, t0, run_dir)

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
    ):
        """Build model, LightningModule, and Trainer."""
        model = self._build_model(dm)
        lm = STPPLightningModule(model=model, tc=self.config.training)
        accelerator = ConfigRegistry.resolve_accelerator(
            self.config.model.preset, self.config.training.device
        )
        loggers = self.config.logging.build_loggers(run_dir)
        trainer = self.config.training.build_trainer(
            run_dir, accelerator, loggers, monitor_key=lm.val_monitor_key
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
    ) -> RunResult:
        """Stash trained state, collect result, update symlink."""
        self._lightning_module = lm
        self._data_module = dm
        self._run_dir = run_dir
        result = self._collect_result(
            trainer, lm, dm, dataset_id, time.perf_counter() - start_time, run_dir
        )
        self._restore_selected_checkpoint(lm, run_dir)
        self._norm_stats = result.norm_stats
        update_latest_symlink(run_dir)
        return result

    def _build_model(self, dm: STPPDataModule):
        """Construct and return the UnifiedSTPP model."""
        data_overrides = ConfigRegistry.data_init_overrides(self.config.model.preset, dm)
        self._effective_model_overrides = data_overrides or {}
        return self.config.model.build_model(extra_overrides=data_overrides or None)

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
    ) -> RunResult:
        """Assemble and persist a RunResult from the completed training run."""
        cfg = self.config
        val_objective, test_nll, temporal_nll, spatial_nll, test_extra_metrics = self._extract_fit_metrics(
            trainer, lm, dm
        )
        ckpt_path = self._extract_checkpoint_path(trainer)
        norm_stats = dm.get_norm_stats(cfg.data.normalize)
        caps = lm.model.event_model.capabilities
        nll_report_space = "native"
        nll_description = caps.nll_description
        if (
            cfg.training.test_nll_space == "raw"
            and "native_nll" in test_extra_metrics
            and caps.supports_raw_reporting
        ):
            nll_report_space = "raw"
            nll_description = caps.raw_nll_description or caps.nll_description
        result = RunResult(
            preset=cfg.model.preset,
            dataset_id=dataset_id,
            seed=cfg.data.seed,
            val_objective=val_objective,
            val_metric_key=caps.metric_key,
            test_nll=test_nll,
            train_time_sec=train_time,
            n_params=sum(p.numel() for p in lm.model.parameters() if p.requires_grad),
            effective_config=self._effective_config_dump(),
            checkpoint_path=ckpt_path,
            norm_stats=norm_stats,
            run_dir=run_dir,
            training_objective=caps.training_objective,
            objective_description=caps.objective_description,
            nll_kind=caps.nll_kind,
            nll_description=nll_description,
            nll_footnote=caps.nll_footnote,
            nll_report_space=nll_report_space,
            temporal_nll=temporal_nll,
            spatial_nll=spatial_nll,
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
    # Post-fit evaluation (primary analysis path)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        val_seqs: Optional[list[dict]] = None,
        *,
        surface_viz=None,
        run_dir: Optional[Path] = None,
    ) -> dict[str, Path]:
        """Post-fit evaluation: run analysis workflows on a fitted runner.

        Works both after ``fit()`` (data module is in memory) and after
        ``load()`` (pass ``val_seqs`` to rebuild the data module).

        Parameters
        ----------
        val_seqs    : event sequences for history queries. Required after
                      ``load()`` when ``_data_module`` is not held in memory.
                      Ignored if ``_data_module`` is already set.
        surface_viz : :class:`~unified_stpp.viz.workflow.SurfaceVizConfig`.
                      When ``enabled=True``, queries intensity/density surfaces
                      and saves artifacts.
        run_dir     : artifact output directory.  Defaults to ``self._run_dir``
                      (set by ``fit()`` or ``load()``).

        Returns
        -------
        dict[str, Path]
            Mapping from artifact name to path.  Empty if no workflows are enabled.
        """
        run_dir = self._resolve_run_dir(run_dir)
        self._ensure_data_module(val_seqs)

        artifacts: dict[str, Path] = {}
        for wf_cfg in [surface_viz]:   # extend this list as new workflows are added
            if wf_cfg is not None and getattr(wf_cfg, "enabled", False):
                from unified_stpp.viz.workflow import SurfaceVisualizationWorkflow
                wf = SurfaceVisualizationWorkflow(wf_cfg)
                wf_artifacts = wf.run(self, run_dir)
                _extend_viz_manifest(run_dir, wf_artifacts)
                artifacts.update(wf_artifacts)
        return artifacts

    def _resolve_run_dir(self, run_dir: Optional[Path]) -> Path:
        """Return a resolved run directory, raising if none is available."""
        if run_dir is not None:
            return Path(run_dir)
        return _require(
            self._run_dir,
            "run_dir is required when the runner has no associated run "
            "directory.  Pass run_dir= explicitly, or call fit() / "
            "load() before evaluate().",
        )

    def _ensure_data_module(self, val_seqs: Optional[list[dict]]) -> None:
        """Ensure ``_data_module`` is set, building from val_seqs if needed."""
        if self._data_module is not None:
            return
        self._data_module = self._build_eval_data_module(
            _require(
                val_seqs,
                "val_seqs is required when calling evaluate() after load() "
                "(the data module is not held in memory).",
            )
        )

    def _build_eval_data_module(self, val_seqs: list[dict]) -> STPPDataModule:
        """Build a minimal eval-only data module from val_seqs.

        Used only to support ``get_original_sequence()`` access in the
        visualization workflow.  No DataLoader is iterated in the evaluate path.
        Normalization stats are held by the evaluator (via ``runner.norm_stats``),
        not by this data module.
        """
        from unified_stpp.data import STPPDataset, collate_fn as _canonical_collate
        from unified_stpp.data.registry import DataBundle

        ds = STPPDataset(val_seqs, normalize_time=False, normalize_space=False)
        bundle = DataBundle(
            train_dataset=ds, val_dataset=ds, test_dataset=None,
            collate_fn=_canonical_collate, train_batch_sampler=None,
        )
        return STPPDataModule(bundle, batch_size=1)

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
