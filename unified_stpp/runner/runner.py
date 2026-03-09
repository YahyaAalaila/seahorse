"""
STPPRunner — orchestrates fit → evaluate → save/load for any STPP preset.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from unified_stpp.config import STPPConfig
from unified_stpp.registry import build_model
from unified_stpp.training.lightning_module import STPPLightningModule
from unified_stpp.training.data_module import STPPDataModule
from unified_stpp.models.sampling import IntensityEvaluator
from .results import RunResult


def _autoint_bbox_from_dm(dm: "STPPDataModule", margin: float = 0.5) -> dict:
    """Compute spatial bounding box from normalized training data.

    Samples up to 20 batches from the training dataloader, collects all valid
    location values (masked by sequence lengths), and returns the 1st/99th
    percentile expanded by *margin* in normalized (z-scored) coordinates.
    """
    locs = []
    loader = dm.train_dataloader()
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        t = batch["locations"]  # (B, T, 2)
        lens = batch["lengths"]  # (B,)
        for b in range(t.shape[0]):
            valid = t[b, : int(lens[b].item())]
            locs.append(valid.detach().cpu().numpy())

    if not locs:
        return {"x_lo": -2.0, "x_hi": 2.0, "y_lo": -2.0, "y_hi": 2.0}

    pts = np.concatenate(locs, axis=0)  # (N, 2)
    x_lo = float(np.percentile(pts[:, 0], 1)) - margin
    x_hi = float(np.percentile(pts[:, 0], 99)) + margin
    y_lo = float(np.percentile(pts[:, 1], 1)) - margin
    y_hi = float(np.percentile(pts[:, 1], 99)) + margin
    return {"x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi}


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
    # Data module construction
    # ------------------------------------------------------------------

    def build_data_module(
        self,
        train_seqs: list[dict],
        val_seqs: list[dict],
        test_seqs: Optional[list[dict]] = None,
    ) -> STPPDataModule:
        """Build an ``STPPDataModule`` from sequences using ``self.config.data``.

        Dispatches on ``config.data.protocol``:
        - ``"unified"``: passes the three splits directly to the data module.
        - ``"paper_autostpp_sthp"``: concatenates all splits into a single
          ``raw_seq`` and lets the data module build sliding windows + internal
          train/val/test split (matching the AutoSTPP paper pipeline).
        """
        cfg = self.config.data
        mc = self.config.model

        if cfg.protocol == "paper_autostpp_sthp":
            # Concatenate all sequences to form one raw_seq
            all_seqs = list(train_seqs) + list(val_seqs) + (list(test_seqs) if test_seqs else [])
            all_times = np.concatenate([np.asarray(s["times"]) for s in all_seqs])
            all_locs = np.concatenate([np.asarray(s["locations"]) for s in all_seqs])
            raw_seq = {"times": all_times, "locations": all_locs}

            lookback = cfg.paper_lookback or 10
            dm = STPPDataModule(
                train_seqs=[],
                val_seqs=[],
                test_seqs=None,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                normalize=cfg.normalize,
                seed=cfg.seed,
                protocol=cfg.protocol,
                raw_seq=raw_seq,
                paper_lookback=lookback,
                paper_lookahead=cfg.paper_lookahead,
                paper_split_ratio=cfg.paper_split_ratio,
            )
        else:
            dm = STPPDataModule(
                train_seqs=train_seqs,
                val_seqs=val_seqs,
                test_seqs=test_seqs,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                normalize=cfg.normalize,
                seed=cfg.seed,
                protocol="unified",
            )
        return dm

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
        cfg = self.config
        tcfg = cfg.training
        lcfg = cfg.logging

        # -- 1. Data module ---------------------------------------------------
        dm = data_module or self.build_data_module(train_seqs, val_seqs, test_seqs)
        dm.setup()

        # -- 2. Model ---------------------------------------------------------
        build_overrides = dict(cfg.model.build_overrides)
        if cfg.model.preset == "auto_stpp":
            bbox = _autoint_bbox_from_dm(dm)
            decoder_ov = dict(build_overrides.get("decoder", {}))
            decoder_ov.update(bbox)
            build_overrides["decoder"] = decoder_ov
            print(
                f"  AutoSTPP bbox (auto): "
                f"x=[{bbox['x_lo']:.2f}, {bbox['x_hi']:.2f}], "
                f"y=[{bbox['y_lo']:.2f}, {bbox['y_hi']:.2f}]"
            )

        model = build_model(
            config=build_overrides,
            preset=cfg.model.preset,
            hidden_dim=cfg.model.hidden_dim,
            spatial_dim=cfg.model.spatial_dim,
            n_marks=cfg.model.n_marks,
            event_cov_dim=cfg.model.event_cov_dim,
            field_cov_dim=cfg.model.field_cov_dim,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        lm = STPPLightningModule(
            model=model,
            lr=tcfg.lr,
            weight_decay=tcfg.weight_decay,
            grad_clip=tcfg.grad_clip,
            adam_beta1=tcfg.adam_beta1,
            adam_beta2=tcfg.adam_beta2,
            lr_schedule=tcfg.lr_schedule,
            lr_warmup_epochs=tcfg.lr_warmup_epochs,
            lr_step_size=tcfg.lr_step_size,
            lr_step_gamma=tcfg.lr_step_gamma,
            vae_beta=tcfg.vae_beta,
        )

        # -- 3. Callbacks & Loggers -------------------------------------------
        run_dir = Path(lcfg.out_dir) / (lcfg.experiment_name or cfg.model.preset)
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_callback = ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="best",
            monitor="val/nll",
            mode="min",
            save_top_k=1,
        )
        callbacks = [ckpt_callback, LearningRateMonitor()]
        if tcfg.patience is not None:
            callbacks.append(
                EarlyStopping(monitor="val/nll", patience=tcfg.patience, mode="min")
            )

        loggers: list = [CSVLogger(save_dir=str(lcfg.out_dir), name=lcfg.experiment_name or cfg.model.preset)]
        try:
            if lcfg.wandb:
                from pytorch_lightning.loggers import WandbLogger
                loggers.append(WandbLogger(**lcfg.wandb))
        except ImportError:
            pass

        # -- 4. Trainer -------------------------------------------------------
        # torchdiffeq uses float64 internally which MPS doesn't support, and
        # its adjoint backward stores ODE states that would be on a different
        # device than the MPS gradients. Force CPU for ODE-based models on MPS.
        accelerator = tcfg.device
        if accelerator == "auto" and torch.backends.mps.is_available():
            from unified_stpp.models.dynamics.neural_ode import NeuralODEDynamics
            from unified_stpp.models.decoders.spatial import CNFSpatial
            if any(isinstance(m, (NeuralODEDynamics, CNFSpatial))
                   for m in model.modules()):
                import warnings
                warnings.warn(
                    "MPS detected but torchdiffeq doesn't support MPS (float64). "
                    "Falling back to CPU for this model.",
                    stacklevel=2,
                )
                accelerator = "cpu"

        # inference_mode=False: fall back to torch.no_grad() for val/test so
        # that decoders using torch.enable_grad() internally (e.g. AutoInt)
        # can still compute autograd-based quantities during evaluation.
        trainer = pl.Trainer(
            max_epochs=tcfg.n_epochs,
            accelerator=accelerator,
            callbacks=callbacks,
            logger=loggers,
            enable_progress_bar=True,
            log_every_n_steps=1,
            inference_mode=False,
        )

        # -- 5. Fit -----------------------------------------------------------
        t0 = time.perf_counter()
        trainer.fit(lm, datamodule=dm)
        train_time = time.perf_counter() - t0

        # -- 6. Evaluate val / test NLL ---------------------------------------
        raw_val = trainer.callback_metrics.get("val/nll")
        val_nll = float(raw_val) if raw_val is not None else float("nan")

        # Use best checkpoint score if available
        if ckpt_callback.best_model_score is not None:
            val_nll = float(ckpt_callback.best_model_score)

        test_nll = float("nan")
        if test_seqs is not None or (data_module is not None and dm._test_dataset is not None):
            test_results = trainer.test(lm, datamodule=dm, verbose=False)
            if test_results:
                test_nll = float(test_results[0].get("test/nll", float("nan")))

        # -- 7. Checkpoint ----------------------------------------------------
        ckpt_path: Optional[Path] = None
        if lcfg.save_checkpoints and ckpt_callback.best_model_path:
            ckpt_path = Path(ckpt_callback.best_model_path)

        # -- 8. Stash for later use -------------------------------------------
        self._lightning_module = lm
        self._data_module = dm

        # Extract normalization stats from the training dataset so downstream
        # code (sampling-based metrics, intensity plots) can convert model
        # outputs from normalized to original coordinates without re-reading
        # the raw sequences.
        ds = dm._train_dataset
        norm_stats = {
            "normalize": cfg.data.normalize,
            "time_mean": float(getattr(ds, "time_mean", 0.0)),
            "time_std":  float(getattr(ds, "time_std",  1.0)),
            "loc_mean":  list(np.asarray(getattr(ds, "loc_mean", [0.0, 0.0])).tolist()),
            "loc_std":   list(np.asarray(getattr(ds, "loc_std",  [1.0, 1.0])).tolist()),
        }

        return RunResult(
            preset=cfg.model.preset,
            dataset_id=dataset_id,
            seed=cfg.data.seed,
            val_nll=val_nll,
            test_nll=test_nll,
            train_time_sec=train_time,
            n_params=n_params,
            effective_config=cfg.model_dump(),
            checkpoint_path=ckpt_path,
            norm_stats=norm_stats,
        )

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
        if protocol != "unified":
            raise NotImplementedError(
                f"intensity_grid() only supports protocol='unified'; "
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

        # ---- Encode history → z ---------------------------------------------
        N = len(t_norm)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        s_tensor = torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0)               # (1, N, d)
        lengths = torch.tensor([N], device=device)

        events = torch.cat([t_tensor, s_tensor], dim=-1)  # (1, N, 1+d)
        with torch.no_grad():
            z_final, _ = model.encode(events, lengths)

        z = z_final  # (1, h)
        t_prev = torch.tensor([[t_norm[-1]]], dtype=torch.float32, device=device)  # (1, 1)
        history_locs_norm_tensor = torch.tensor(s_norm, dtype=torch.float32, device=device)  # (N, d)

        # ---- Build evaluator -------------------------------------------------
        evaluator = IntensityEvaluator(
            model=model,
            z=z,
            t_prev=t_prev,
            history_locs_norm=history_locs_norm_tensor,
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
        """Restore a runner from a saved directory (``config.yaml`` + ``model.ckpt``)."""
        p = Path(path)
        config = STPPConfig.from_yaml(p / "config.yaml")
        runner = cls(config)

        mc = config.model
        model = build_model(
            config=mc.build_overrides,
            preset=mc.preset,
            hidden_dim=mc.hidden_dim,
            spatial_dim=mc.spatial_dim,
            n_marks=mc.n_marks,
            event_cov_dim=mc.event_cov_dim,
            field_cov_dim=mc.field_cov_dim,
        )
        state = torch.load(p / "model.ckpt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)

        tc = config.training
        lm = STPPLightningModule(
            model=model,
            lr=tc.lr,
            weight_decay=tc.weight_decay,
            grad_clip=tc.grad_clip,
            adam_beta1=tc.adam_beta1,
            adam_beta2=tc.adam_beta2,
            lr_schedule=tc.lr_schedule,
            lr_warmup_epochs=tc.lr_warmup_epochs,
            lr_step_size=tc.lr_step_size,
            lr_step_gamma=tc.lr_step_gamma,
            vae_beta=tc.vae_beta,
        )
        runner._lightning_module = lm
        return runner
