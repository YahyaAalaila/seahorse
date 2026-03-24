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

import numpy as np
import pytorch_lightning as pl
import torch

from unified_stpp.config import STPPConfig
from unified_stpp.models.configs import ConfigRegistry
from unified_stpp.utils import deep_update
from unified_stpp.training.lightning_module import STPPLightningModule
from unified_stpp.training.data_module import STPPDataModule
from unified_stpp.models.sampling import IntensityEvaluator
from .artifacts import make_run_id, update_latest_symlink, save_run_artifacts, _extend_viz_manifest
from .results import RunResult


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
    >>> print(result.val_nll)
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
        surface_viz=None,
    ) -> RunResult:
        """Train the model and return a ``RunResult``.

        Parameters
        ----------
        train_seqs:   Training sequences — list of ``{"times": ..., "locations": ...}``.
        val_seqs:     Validation sequences.
        test_seqs:    Optional test sequences; ``result.test_nll`` is ``nan`` if omitted.
        data_module:  Override the auto-built data module (researcher escape hatch).
        dataset_id:   Human-readable name stored in the returned ``RunResult``.
        surface_viz:  Optional ``SurfaceVizConfig``; if ``enabled=True``, runs the
                      surface visualization workflow after training and saves artifacts.
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
        return self._finalize_fit(trainer, lm, dm, dataset_id, t0, run_dir, surface_viz)

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
        """Build (or accept) and set up the data module."""
        dm = data_module or STPPDataModule.from_splits(
            self.config.data, train_seqs, val_seqs, test_seqs,
            model_preset=self.config.model.preset,
        )
        dm.setup()
        return dm

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
        trainer = self.config.training.build_trainer(run_dir, accelerator, loggers)
        return model, lm, trainer

    def _finalize_fit(
        self,
        trainer,
        lm: STPPLightningModule,
        dm: STPPDataModule,
        dataset_id: str,
        start_time: float,
        run_dir: Path,
        surface_viz=None,
    ) -> RunResult:
        """Stash trained state, collect result, update symlink."""
        self._lightning_module = lm
        self._data_module = dm
        self._run_dir = run_dir
        result = self._collect_result(
            trainer, lm, dm, dataset_id, time.perf_counter() - start_time, run_dir
        )

        self._norm_stats = result.norm_stats

        if surface_viz is not None and getattr(surface_viz, "enabled", False):
            from unified_stpp.viz.workflow import SurfaceVisualizationWorkflow
            viz_artifacts = SurfaceVisualizationWorkflow(surface_viz).run(self, run_dir)
            _extend_viz_manifest(run_dir, viz_artifacts)

        update_latest_symlink(run_dir)
        return result

    def _build_model(self, dm: STPPDataModule):
        """Construct and return the UnifiedSTPP model."""
        mc = self.config.model
        overrides = dict(mc.build_overrides)
        deep_update(overrides, ConfigRegistry.data_init_overrides(mc.preset, dm))
        return ConfigRegistry.build(
            mc.preset,
            overrides=overrides,
            hidden_dim=mc.hidden_dim,
            spatial_dim=mc.spatial_dim,
            n_marks=mc.n_marks,
            event_cov_dim=mc.event_cov_dim,
            field_cov_dim=mc.field_cov_dim,
        )

    def _collect_result(
        self,
        trainer: pl.Trainer,
        lm: STPPLightningModule,
        dm: STPPDataModule,
        dataset_id: str,
        train_time: float,
        run_dir: Path,
    ) -> RunResult:
        """Extract metrics and normalization stats; return RunResult."""
        from pytorch_lightning.callbacks import ModelCheckpoint
        cfg = self.config
        ckpt_callback = next(c for c in trainer.callbacks if isinstance(c, ModelCheckpoint))

        raw_val = trainer.callback_metrics.get("val/nll")
        val_nll = float(raw_val) if raw_val is not None else float("nan")
        if ckpt_callback.best_model_score is not None:
            val_nll = float(ckpt_callback.best_model_score)

        test_nll = float("nan")
        if dm._test_dataset is not None:
            test_results = trainer.test(lm, datamodule=dm, verbose=False)
            if test_results:
                test_nll = float(test_results[0].get("test/nll", float("nan")))

        ckpt_path: Optional[Path] = None
        if cfg.logging.save_checkpoints and ckpt_callback.best_model_path:
            ckpt_path = Path(ckpt_callback.best_model_path)

        ds = dm._train_dataset
        norm_stats = {
            "normalize": cfg.data.normalize,
            "time_mean": float(getattr(ds, "time_mean", 0.0)),
            "time_std":  float(getattr(ds, "time_std",  1.0)),
            "loc_mean":  list(np.asarray(getattr(ds, "loc_mean", [0.0, 0.0])).tolist()),
            "loc_std":   list(np.asarray(getattr(ds, "loc_std",  [1.0, 1.0])).tolist()),
        }

        result = RunResult(
            preset=cfg.model.preset,
            dataset_id=dataset_id,
            seed=cfg.data.seed,
            val_nll=val_nll,
            test_nll=test_nll,
            train_time_sec=train_time,
            n_params=sum(p.numel() for p in lm.model.parameters() if p.requires_grad),
            effective_config=cfg.model_dump(),
            checkpoint_path=ckpt_path,
            norm_stats=norm_stats,
            run_dir=run_dir,
        )
        save_run_artifacts(run_dir, result, cfg)
        return result

    # ------------------------------------------------------------------
    # Intensity grid (post-training visualization)
    # ------------------------------------------------------------------

    def intensity_grid(
        self,
        history_times: np.ndarray,
        history_locs: np.ndarray,
        t_query: float,
        n_grid: int = 50,
        x_range: Optional[tuple] = None,
        y_range: Optional[tuple] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate λ*(t_query, · | history) on a spatial grid.

        Returns (xs, ys, intensity_grid) all as numpy arrays in *original*
        (un-normalised) coordinates.

        Only supports the ``"unified"`` protocol.  For the paper protocol,
        use ``IntensityEvaluator`` directly with your own normalisation.
        """
        if self._lightning_module is None or self._data_module is None:
            raise RuntimeError("Call fit() before intensity_grid().")

        protocol = self.config.data.protocol
        if protocol != "standard":
            raise NotImplementedError(
                f"intensity_grid() only supports protocol='standard'; "
                f"got {protocol!r}.  Use IntensityEvaluator directly."
            )

        model = self.model
        model.eval()
        dm = self._data_module
        ds = dm._train_dataset  # provides normalization stats

        device = next(model.parameters()).device

        # ---- Normalize history -----------------------------------------------
        t_arr = np.asarray(history_times, dtype=np.float64)
        s_arr = np.asarray(history_locs, dtype=np.float64)

        t_norm = (t_arr - ds.time_mean) / max(ds.time_std, 1e-8)
        s_norm = (s_arr - ds.loc_mean) / np.maximum(ds.loc_std, 1e-8)

        # ---- Build evaluator -------------------------------------------------
        N = len(t_norm)
        evaluator = IntensityEvaluator(
            model=model,
            history_times=torch.tensor(
                t_norm, dtype=torch.float32, device=device
            ).unsqueeze(0),
            history_locations=torch.tensor(
                s_norm, dtype=torch.float32, device=device
            ).unsqueeze(0),
            history_lengths=torch.tensor([N], dtype=torch.long, device=device),
        )

        # ---- Grid bounds in normalized space ---------------------------------
        d = s_norm.shape[-1]
        if d == 2:
            if x_range is None:
                x_lo = float(s_norm[:, 0].min() - 0.5)
                x_hi = float(s_norm[:, 0].max() + 0.5)
            else:
                x_lo = (x_range[0] - ds.loc_mean[0]) / max(ds.loc_std[0], 1e-8)
                x_hi = (x_range[1] - ds.loc_mean[0]) / max(ds.loc_std[0], 1e-8)

            if y_range is None:
                y_lo = float(s_norm[:, 1].min() - 0.5)
                y_hi = float(s_norm[:, 1].max() + 0.5)
            else:
                y_lo = (y_range[0] - ds.loc_mean[1]) / max(ds.loc_std[1], 1e-8)
                y_hi = (y_range[1] - ds.loc_mean[1]) / max(ds.loc_std[1], 1e-8)
        else:
            x_lo = float(s_norm[:, 0].min() - 0.5)
            x_hi = float(s_norm[:, 0].max() + 0.5)
            y_lo, y_hi = 0.0, 0.0  # unused for d=1

        t_query_norm = (t_query - ds.time_mean) / max(ds.time_std, 1e-8)

        s_min = torch.tensor([x_lo, y_lo][:d], dtype=torch.float32, device=device)
        s_max = torch.tensor([x_hi, y_hi][:d], dtype=torch.float32, device=device)

        # ---- Evaluate -------------------------------------------------------
        with torch.no_grad():
            xs_norm, ys_norm, lam = evaluator.intensity_grid(
                t=t_query_norm,
                s_min=s_min,
                s_max=s_max,
                n_grid=n_grid,
            )

        # ---- Denormalise grid axes ------------------------------------------
        xs = xs_norm.cpu().numpy() * ds.loc_std[0] + ds.loc_mean[0]
        if ys_norm is not None:
            ys = ys_norm.cpu().numpy() * ds.loc_std[1] + ds.loc_mean[1]
        else:
            ys = np.zeros(0)

        return xs, ys, lam.cpu().numpy()

    # ------------------------------------------------------------------
    # Post-fit evaluation (primary analysis path)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        val_seqs: Optional[list[dict]] = None,
        surface_viz=None,
        run_dir: Optional[Path] = None,
    ) -> dict[str, Path]:
        """Post-fit evaluation: run analysis workflows on a fitted runner.

        This is the primary path for post-fit analysis. It delegates to the
        same workflow objects used internally by ``fit()`` — there is no
        duplicated logic. Works both after ``fit()`` (data module is in memory)
        and after ``load()`` (pass ``val_seqs`` to rebuild the data module).

        Parameters
        ----------
        val_seqs    : event sequences for history queries. Required after
                      ``load()`` when ``_data_module`` is not held in memory.
                      Ignored if ``_data_module`` is already set (i.e. called
                      directly after ``fit()``).
        surface_viz : :class:`~unified_stpp.viz.workflow.SurfaceVizConfig`.
                      When ``enabled=True``, queries intensity/density surfaces
                      and saves artifacts.  Delegates entirely to
                      :class:`~unified_stpp.viz.workflow.SurfaceVisualizationWorkflow`.
        run_dir     : artifact output directory.  Defaults to ``self._run_dir``
                      (the run's own directory, set by ``fit()`` or ``load()``).

        Returns
        -------
        dict[str, Path]
            Mapping from artifact name to path.  Empty if no workflows are
            enabled.
        """
        if run_dir is None:
            if self._run_dir is None:
                raise ValueError(
                    "run_dir is required when the runner has no associated run "
                    "directory.  Pass run_dir= explicitly, or call fit() / "
                    "load() before evaluate()."
                )
            run_dir = self._run_dir
        run_dir = Path(run_dir)

        if self._data_module is None:
            if val_seqs is None:
                raise ValueError(
                    "val_seqs is required when calling evaluate() after load() "
                    "(the data module is not held in memory)."
                )
            self._data_module = self._build_eval_data_module(val_seqs)

        artifacts: dict[str, Path] = {}

        if surface_viz is not None and getattr(surface_viz, "enabled", False):
            from unified_stpp.viz.workflow import SurfaceVisualizationWorkflow
            wf = SurfaceVisualizationWorkflow(surface_viz)
            viz_artifacts = wf.run(self, run_dir)
            _extend_viz_manifest(run_dir, viz_artifacts)
            artifacts.update(viz_artifacts)

        return artifacts

    def _build_eval_data_module(self, val_seqs: list[dict]) -> STPPDataModule:
        """Build a minimal eval-only data module from val_seqs.

        This is an evaluation-only reconstruction: ``val_seqs`` is passed as
        both ``train_seqs`` and ``val_seqs`` solely to:

        (a) Give :class:`~unified_stpp.viz.workflow.SurfaceVisualizationWorkflow`
            access to raw sequences via ``get_original_sequence()``.
        (b) Provide a ``_train_dataset`` object whose normalization attributes
            can be overridden with the saved training-set stats.

        No DataLoader is ever iterated in the evaluate path; ``batch_size`` is
        set to 1 to make any accidental use obvious.  The normalization stats on
        ``_train_dataset`` are overridden with ``self._norm_stats`` (loaded from
        ``run_result.json`` during ``load()``) so the model sees the same
        coordinate system it was trained with.
        """
        dm = STPPDataModule(
            train_seqs=val_seqs,
            val_seqs=val_seqs,
            batch_size=1,           # eval-only: DataLoader is never iterated
            normalize=self.config.data.normalize,
        )
        dm.setup()

        if self.config.data.normalize:
            if not self._norm_stats or not self._norm_stats.get("normalize", False):
                raise ValueError(
                    "Cannot rebuild the eval data module: saved norm_stats are "
                    "missing or indicate no normalization was used, but "
                    "config.data.normalize=True.  Ensure the run directory "
                    "contains run_result.json with norm_stats populated."
                )
            ds = dm._train_dataset
            ds.time_mean = float(self._norm_stats["time_mean"])
            ds.time_std  = float(self._norm_stats["time_std"])
            ds.loc_mean  = np.array(self._norm_stats["loc_mean"], dtype=np.float64)
            ds.loc_std   = np.array(self._norm_stats["loc_std"],  dtype=np.float64)

        return dm

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
        1. ``checkpoints/best.ckpt`` — Lightning checkpoint written by ``fit()``
        2. ``model.ckpt``           — legacy plain state-dict written by ``save()``
        """
        p = Path(path).resolve()
        config = STPPConfig.from_yaml(p / "config.yaml", sanitize=False)
        runner = cls(config)

        mc = config.model
        model = ConfigRegistry.build(
            mc.preset,
            overrides=mc.build_overrides,
            hidden_dim=mc.hidden_dim,
            spatial_dim=mc.spatial_dim,
            n_marks=mc.n_marks,
            event_cov_dim=mc.event_cov_dim,
            field_cov_dim=mc.field_cov_dim,
        )

        pl_ckpt = p / "checkpoints" / "best.ckpt"
        if pl_ckpt.exists():
            # Lightning checkpoint: state_dict keys are prefixed with "model."
            ckpt = torch.load(pl_ckpt, map_location="cpu", weights_only=False)
            pl_state = ckpt["state_dict"]
            state = {
                k[len("model."):]: v
                for k, v in pl_state.items()
                if k.startswith("model.")
            }
        else:
            # Legacy: plain state-dict saved by runner.save()
            state = torch.load(p / "model.ckpt", map_location="cpu", weights_only=False)

        model.load_state_dict(state)
        lm = STPPLightningModule(model=model, tc=config.training)
        runner._lightning_module = lm
        runner._run_dir = p
        result_json = p / "run_result.json"
        if result_json.exists():
            import json
            with open(result_json) as f:
                runner._norm_stats = json.load(f).get("norm_stats")
        return runner
