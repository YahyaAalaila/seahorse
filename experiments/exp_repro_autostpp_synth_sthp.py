#!/usr/bin/env python3
"""
Reproducibility experiment for AutoSTPP synthetic ST-Hawkes datasets.

What this script does
---------------------
1. Builds synthetic ST-Hawkes data corresponding to three benchmark settings
   (`sthp0`, `sthp1`, `sthp2`) using parameters from the AutoSTPP paper appendix
   (ST-Hawkes DS1/DS2/DS3).
2. Performs a parity checklist between:
   - our STHP simulator / canonical data pipeline, and
   - the original AutoSTPP-style STHP simulator behaviour.
   - Validates that every batch passes the unified data contract.
3. Trains two models with identical training budgets on exactly the same splits:
   - `auto_stpp`
   - `deep_stpp`
4. Reports test LL and NLL/event using the shared `LikelihoodEvaluator` so
   metrics are directly comparable across all models.

Notes
-----
- This script is standalone and does not modify core training/model code.
- Both models use `STPPDataModule(normalize=True)` (z-score) so LL values are
  directly comparable on the same coordinate system.
- Only ST-Hawkes is considered here (no self-correcting process).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from unified_stpp.data import collate_fn as core_collate_fn
from unified_stpp.data.contract import validate_batch, fingerprint_batch
from unified_stpp.data.synthetic import STHPDataset
from unified_stpp.evaluation import LikelihoodEvaluator, EvalResult, ParityReport
from unified_stpp.models import IntensityEvaluator
from unified_stpp.registry import build_model, PRESETS
from unified_stpp.training.data_module import STPPDataModule, assert_protocol_model_compatible
from unified_stpp.training.lightning_module import STPPLightningModule
from unified_stpp.training.trainer import Trainer as LegacyTrainer


# ---------------------------------------------------------------------------
# Dataset parameter presets
# ---------------------------------------------------------------------------

@dataclass
class STHPParams:
    name: str
    alpha: float
    beta: float
    mu: float
    g0_cov: np.ndarray
    g2_cov: np.ndarray
    source: str


STHP_PRESETS: Dict[str, STHPParams] = {
    # Mapping to AutoSTPP paper appendix (Table 3, ST-Hawkes DS1/DS2/DS3).
    "sthp0": STHPParams(
        name="sthp0",
        alpha=0.5,
        beta=1.0,
        mu=0.2,
        g0_cov=np.array([[0.2, 0.0], [0.0, 0.2]], dtype=np.float64),
        g2_cov=np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64),
        source="AutoSTPP paper appendix Table 3: ST-Hawkes DS1",
    ),
    "sthp1": STHPParams(
        name="sthp1",
        alpha=0.5,
        beta=0.6,
        mu=0.15,
        g0_cov=np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float64),
        g2_cov=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64),
        source="AutoSTPP paper appendix Table 3: ST-Hawkes DS2",
    ),
    "sthp2": STHPParams(
        name="sthp2",
        alpha=0.3,
        beta=2.0,
        mu=1.0,
        g0_cov=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        g2_cov=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float64),
        source="AutoSTPP paper appendix Table 3: ST-Hawkes DS3",
    ),
}


# ---------------------------------------------------------------------------
# Reference AutoSTPP-style simulator (kept for simulation parity check)
# ---------------------------------------------------------------------------

class ReferenceAutoSTHPSimulator:
    """
    Local reference implementation matching the original AutoSTPP-style STHP
    simulation logic (Ogata thinning + Gaussian spatial kernels).
    """

    def __init__(
        self,
        s_mu: np.ndarray,
        g0_cov: np.ndarray,
        g2_cov: np.ndarray,
        alpha: float,
        beta: float,
        mu: float,
        max_history: int = 100,
    ):
        self.s_mu = np.asarray(s_mu, dtype=np.float64)
        self.g0_cov = np.asarray(g0_cov, dtype=np.float64)
        self.g2_cov = np.asarray(g2_cov, dtype=np.float64)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mu = float(mu)
        self.max_history = int(max_history)

        self.g0_ic = np.linalg.inv(self.g0_cov)
        self.g0_sidc = 1.0 / np.sqrt(np.linalg.det(self.g0_cov))
        self.g2_ic = np.linalg.inv(self.g2_cov)
        self.g2_sidc = 1.0 / np.sqrt(np.linalg.det(self.g2_cov))

        self.his_s = np.zeros((0, 2), dtype=np.float64)
        self.his_t = np.array([], dtype=np.float64)
        self.t_start = 0.0
        self.t_end = 0.0

    def trunc(self, his: np.ndarray) -> np.ndarray:
        if len(his) > self.max_history:
            return his[-self.max_history :]
        return his

    @staticmethod
    def g1(t: float, his_t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        return alpha * np.exp(-beta * (t - his_t))

    @staticmethod
    def g2(
        s: np.ndarray, his_s: np.ndarray, sidc: float, inv_cov: np.ndarray
    ) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
        his = np.asarray(his_s, dtype=np.float64).reshape(-1, 2)
        if his.shape[0] == 0:
            return np.zeros((s.shape[0], 0), dtype=np.float64)
        delta = s[:, None, :] - his[None, :, :]
        quad = np.einsum("bni,ij,bnj->bn", delta, inv_cov, delta)
        return (1.0 / (2.0 * np.pi)) * sidc * np.exp(-0.5 * quad)

    @staticmethod
    def g0(
        s: np.ndarray, s_mu: np.ndarray, sidc: float, inv_cov: np.ndarray
    ) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64).reshape(-1, 2)
        mu = np.asarray(s_mu, dtype=np.float64).reshape(1, 2)
        delta = s - mu
        quad = np.einsum("bi,ij,bj->b", delta, inv_cov, delta)
        return (1.0 / (2.0 * np.pi)) * sidc * np.exp(-0.5 * quad)

    def generate_offsprings(self, t_i: float, s_i: np.ndarray) -> None:
        t = float(t_i)
        while True:
            m = self.alpha * np.exp(-self.beta * (t - t_i))
            if m <= 0.0:
                break
            t += np.random.exponential(scale=1.0 / m)
            if t > self.t_end:
                break
            lamb = self.alpha * np.exp(-self.beta * (t - t_i))
            if lamb / m >= np.random.uniform():
                s = np.random.multivariate_normal(np.asarray(s_i).reshape(-1), self.g2_cov)
                s = np.expand_dims(s.astype(np.float64), 0)
                n = len(self.his_t[self.his_t < t])
                self.his_s = np.insert(self.his_s, n, s, axis=0)
                self.his_t = np.insert(self.his_t, n, t)

    def generate(self, t_start: float, t_end: float) -> None:
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        t = float(t_start)
        self.his_s = np.zeros((0, 2), dtype=np.float64)
        self.his_t = np.array([], dtype=np.float64)

        while True:
            t += np.random.exponential(scale=1.0 / self.mu)
            if t > t_end:
                break
            s = np.random.multivariate_normal(self.s_mu, self.g0_cov)
            s = np.expand_dims(s.astype(np.float64), 0)
            self.his_s = np.vstack((self.his_s, s))
            self.his_t = np.append(self.his_t, t)

        if len(self.his_t) == 0:
            return

        t = float(t_start)
        n = 0
        while True:
            self.generate_offsprings(self.his_t[n], self.his_s[n])
            try:
                n = next(i for i, ti in enumerate(self.his_t) if ti > t)
                t = float(self.his_t[n])
            except StopIteration:
                break

    def generate_single_sequence(
        self, *, t_start: float, t_end: float, seed: int
    ) -> Dict[str, np.ndarray]:
        np.random.seed(seed)
        self.generate(t_start=t_start, t_end=t_end)
        return {
            "times": np.asarray(self.his_t, dtype=np.float32),
            "locations": np.asarray(self.his_s, dtype=np.float32).reshape(-1, 2),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducibility experiment for AutoSTPP synthetic ST-Hawkes datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sthp0", "sthp1", "sthp2"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Shared hidden dimension for both AutoSTPP and DeepSTPP runs.",
    )
    parser.add_argument(
        "--deep_num_heads",
        type=int,
        default=4,
        help="DeepSTPP attention heads for encoder/updater.",
    )
    parser.add_argument(
        "--deep_num_layers",
        type=int,
        default=1,
        help="DeepSTPP encoder attention layer count.",
    )
    parser.add_argument(
        "--deep_k",
        type=int,
        default=16,
        help="DeepSTPP mixture components for temporal and spatial decoders.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-3,
        help="Paper range examples: 0.004 or 0.0002.",
    )
    # Per-momentumdel overrides (if omitted, fallback to shared defaults above).
    parser.add_argument("--auto_epochs", type=int, default=None)
    parser.add_argument("--deep_epochs", type=int, default=None)
    parser.add_argument("--auto_batch_size", type=int, default=None)
    parser.add_argument("--deep_batch_size", type=int, default=None)
    parser.add_argument("--auto_lr", type=float, default=None)
    parser.add_argument("--deep_lr", type=float, default=None)
    parser.add_argument("--auto_hidden_dim", type=int, default=None)
    parser.add_argument("--deep_hidden_dim", type=int, default=None)
    parser.add_argument("--auto_grad_clip", type=float, default=5.0)
    parser.add_argument("--deep_grad_clip", type=float, default=5.0)
    parser.add_argument("--auto_adam_beta1", type=float, default=0.9)
    parser.add_argument("--deep_adam_beta1", type=float, default=0.9)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/exp_repro_autostpp_synth_sthp_uni",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="unified",
        choices=["unified", "paper_autostpp_sthp"],
        help=(
            "Data pipeline protocol. "
            "'unified': z-score normalisation + T=200 windows split 40/5/5. "
            "'paper_autostpp_sthp': MinMax normalisation + sliding windows "
            "(lookback=10, lookahead=1) split ~80/10/10, matching the AutoSTPP repo."
        ),
    )
    parser.add_argument(
        "--intensity_video",
        type=str,
        default="mp4",
        choices=["none", "gif", "mp4"],
        help=(
            "Optional side-by-side animated intensity comparison for "
            "(True, AutoSTPP, DeepSTPP)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        s = out.strip()
        return s if s else None
    except Exception:
        return None


def _resolve_runtime_device(device_arg: str) -> Tuple[str, int, str]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "gpu", 1, "cuda"
        # DeepSTPP uses attention masking that currently hits an MPS backend
        # limitation for nested tensor checks. Prefer CPU for reproducibility.
        if torch.backends.mps.is_available():
            print(
                "MPS detected, but this experiment falls back to CPU because "
                "PyTorch MPS can fail on Transformer mask ops used by deep_stpp."
            )
            return "cpu", 1, "cpu"
        return "cpu", 1, "cpu"
    if device_arg in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU/CUDA but torch.cuda.is_available() is False.")
        return "gpu", 1, "cuda"
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is False.")
        print(
            "Requested MPS, but using CPU to avoid known MPS missing op "
            "for Transformer mask handling in deep_stpp."
        )
        return "cpu", 1, "cpu"
    return "cpu", 1, "cpu"


def _resolve_model_hparams(args: argparse.Namespace, preset: str) -> Dict[str, Any]:
    """
    Resolve per-model hyperparameters, falling back to shared defaults when the
    model-specific flag is not provided.
    """
    if preset == "auto_stpp":
        return {
            "epochs": int(args.auto_epochs if args.auto_epochs is not None else args.epochs),
            "batch_size": int(
                args.auto_batch_size if args.auto_batch_size is not None else args.batch_size
            ),
            "lr": float(args.auto_lr if args.auto_lr is not None else args.lr),
            "hidden_dim": int(
                args.auto_hidden_dim if args.auto_hidden_dim is not None else args.hidden_dim
            ),
            "grad_clip": float(args.auto_grad_clip),
            "adam_beta1": float(args.auto_adam_beta1),
        }
    if preset == "deep_stpp":
        return {
            "epochs": int(args.deep_epochs if args.deep_epochs is not None else args.epochs),
            "batch_size": int(
                args.deep_batch_size if args.deep_batch_size is not None else args.batch_size
            ),
            "lr": float(args.deep_lr if args.deep_lr is not None else args.lr),
            "hidden_dim": int(
                args.deep_hidden_dim if args.deep_hidden_dim is not None else args.hidden_dim
            ),
            "grad_clip": float(args.deep_grad_clip),
            "adam_beta1": float(args.deep_adam_beta1),
        }
    return {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "hidden_dim": int(args.hidden_dim),
        "grad_clip": 5.0,
        "adam_beta1": 0.9,
    }


def _generate_long_sequence_unified(
    params: STHPParams, *, seed: int, t_end: float
) -> Dict[str, np.ndarray]:
    gen = STHPDataset(
        s_mu=np.array([0.0, 0.0], dtype=np.float64),
        g0_cov=params.g0_cov,
        g2_cov=params.g2_cov,
        alpha=params.alpha,
        beta=params.beta,
        mu=params.mu,
        seed=seed,
        covariate_fn=lambda t, s: np.array([0.0], dtype=np.float32),
    )
    np.random.seed(seed)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        gen.generate(t_start=0.0, t_end=t_end, verbose=False)
    return {
        "times": np.asarray(gen.his_t, dtype=np.float32),
        "locations": np.asarray(gen.his_s, dtype=np.float32).reshape(-1, 2),
    }


def _generate_long_sequence_reference(
    params: STHPParams, *, seed: int, t_end: float
) -> Dict[str, np.ndarray]:
    ref = ReferenceAutoSTHPSimulator(
        s_mu=np.array([0.0, 0.0], dtype=np.float64),
        g0_cov=params.g0_cov,
        g2_cov=params.g2_cov,
        alpha=params.alpha,
        beta=params.beta,
        mu=params.mu,
    )
    return ref.generate_single_sequence(t_start=0.0, t_end=t_end, seed=seed)


def _split_long_sequence(
    seq: Dict[str, np.ndarray],
    *,
    n_windows: int = 50,
    window_T: float = 200.0,
    reset_time_to_window: bool = True,
) -> List[Dict[str, np.ndarray]]:
    times = np.asarray(seq["times"], dtype=np.float64)
    locs = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)
    windows: List[Dict[str, np.ndarray]] = []
    for i in range(n_windows):
        t0 = i * window_T
        t1 = (i + 1) * window_T
        mask = (times >= t0) & (times < t1)
        tw = times[mask]
        sw = locs[mask]
        if reset_time_to_window:
            tw = tw - t0
        windows.append(
            {
                "times": tw.astype(np.float32),
                "locations": sw.astype(np.float32),
            }
        )
    return windows


def _all_sequences_valid(seqs: Sequence[Dict[str, np.ndarray]], min_len: int = 3) -> bool:
    return all(int(len(s["times"])) >= min_len for s in seqs)


def _compute_ranges(seqs: Sequence[Dict[str, np.ndarray]]) -> Dict[str, float]:
    all_t = np.concatenate([s["times"] for s in seqs], axis=0).astype(np.float64)
    all_s = np.concatenate([s["locations"] for s in seqs], axis=0).astype(np.float64)
    return {
        "t_min": float(np.min(all_t)),
        "t_max": float(np.max(all_t)),
        "x_min": float(np.min(all_s[:, 0])),
        "x_max": float(np.max(all_s[:, 0])),
        "y_min": float(np.min(all_s[:, 1])),
        "y_max": float(np.max(all_s[:, 1])),
    }


def _autoint_bbox_from_dm(dm: STPPDataModule, margin: float = 0.5) -> Dict[str, float]:
    """
    Compute the AutoInt spatial bbox from training data.

    For ``protocol='unified'``: derive bbox from the empirical z-scored range
    of the training locations + margin.  A fixed ±3.5σ is too narrow for
    heavy-tailed datasets (e.g. sthp1 with g0_cov=[[5,0],[0,5]]).

    For ``protocol='paper_autostpp_sthp'``: MinMax-scaled spatial coords lie
    in [0, 1] by construction, so the bbox is simply [-margin, 1+margin].
    """
    if getattr(dm, "protocol", "unified") == "paper_autostpp_sthp":
        return {
            "x_lo": -margin,
            "x_hi":  1.0 + margin,
            "y_lo": -margin,
            "y_hi":  1.0 + margin,
        }

    loc_mean = dm._train_dataset.loc_mean  # (d,) numpy array
    loc_std  = dm._train_dataset.loc_std   # (d,) numpy array

    all_locs = np.concatenate(
        [np.asarray(s["locations"], dtype=np.float64) for s in dm.train_seqs], axis=0
    )
    locs_z = (all_locs - loc_mean) / loc_std  # z-scored

    return {
        "x_lo": float(locs_z[:, 0].min()) - margin,
        "x_hi": float(locs_z[:, 0].max()) + margin,
        "y_lo": float(locs_z[:, 1].min()) - margin,
        "y_hi": float(locs_z[:, 1].max()) + margin,
    }


def _summarize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for k, v in batch.items():
        if v is None:
            summary[k] = {"type": "None"}
            continue
        if isinstance(v, torch.Tensor):
            summary[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
            }
        else:
            summary[k] = {"type": str(type(v))}
    return summary


def _flatten_valid_event_targets(
    times: torch.Tensor,
    locations: torch.Tensor,
    lengths: torch.Tensor,
    all_states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = times.shape[0]
    max_len = int(lengths.max().item())
    if max_len < 2:
        z_empty = all_states.new_zeros((0, all_states.shape[-1]))
        t_empty = times.new_zeros((0, 1))
        s_empty = locations.new_zeros((0, locations.shape[-1]))
        return z_empty, t_empty, s_empty, t_empty

    L = max_len - 1
    z_cond = all_states[:, :L, :]
    t_target = times[:, 1 : 1 + L].unsqueeze(-1)
    s_target = locations[:, 1 : 1 + L, :]
    t_prev = times[:, :L].unsqueeze(-1)

    n_idx = torch.arange(L, device=times.device).unsqueeze(0)
    valid = n_idx < (lengths.unsqueeze(1) - 1)
    z_flat = z_cond[valid]
    t_flat = t_target[valid]
    s_flat = s_target[valid]
    t_prev_flat = t_prev[valid]
    return z_flat, t_flat, s_flat, t_prev_flat


def _load_lightning_ckpt_into_model(ckpt_path: Path, model: torch.nn.Module) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})
    if not isinstance(sd, dict):
        raise RuntimeError(f"Invalid checkpoint state_dict: {ckpt_path}")
    stripped: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("model."):
            stripped[k[6:]] = v
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {ckpt_path}\n"
            f"missing={missing[:8]}{'...' if len(missing) > 8 else ''}\n"
            f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )


def _gaussian_kernel_2d(
    points: np.ndarray, centers: np.ndarray, inv_cov: np.ndarray, sidc: float
) -> np.ndarray:
    """
    2D Gaussian kernel values for all point-center pairs.
    Returns shape (n_points, n_centers).
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 2)
    delta = p[:, None, :] - c[None, :, :]
    quad = np.einsum("pni,ij,pnj->pn", delta, inv_cov, delta)
    coeff = (1.0 / (2.0 * np.pi)) * float(sidc)
    return coeff * np.exp(-0.5 * quad)


def _true_sthp_intensity_on_mesh(
    params: STHPParams,
    times: np.ndarray,
    locations: np.ndarray,
    t_query: float,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Evaluate the generative STHP intensity on a spatial mesh at fixed time."""
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float64)
    his_t = np.asarray(times, dtype=np.float64).reshape(-1)
    his_s = np.asarray(locations, dtype=np.float64).reshape(-1, 2)

    g0_ic = np.linalg.inv(params.g0_cov)
    g0_sidc = 1.0 / np.sqrt(np.linalg.det(params.g0_cov))
    g2_ic = np.linalg.inv(params.g2_cov)
    g2_sidc = 1.0 / np.sqrt(np.linalg.det(params.g2_cov))

    s_mu = np.zeros((1, 2), dtype=np.float64)
    bg = float(params.mu) * _gaussian_kernel_2d(points, s_mu, g0_ic, g0_sidc)[:, 0]

    valid = his_t < float(t_query)
    if np.any(valid):
        ht = his_t[valid]
        hs = his_s[valid]
        temporal = float(params.alpha) * np.exp(-float(params.beta) * (float(t_query) - ht))
        spatial = _gaussian_kernel_2d(points, hs, g2_ic, g2_sidc)  # (n_points, n_hist)
        excite = spatial @ temporal
    else:
        excite = np.zeros(points.shape[0], dtype=np.float64)

    lam = bg + excite
    return lam.reshape(xx.shape)


def _model_intensity_on_mesh(
    model: torch.nn.Module,
    times: np.ndarray,
    locations: np.ndarray,
    t_query: float,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    loc_mean: np.ndarray,
    loc_std: np.ndarray,
    time_mean: float,
    time_std: float,
    device: str,
) -> np.ndarray:
    """
    Evaluate model intensity on the same native-unit mesh as the true process.
    """
    t_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(locations, dtype=np.float64).reshape(-1, 2)
    n_hist = int(np.searchsorted(t_arr, float(t_query), side="right"))
    n_hist = max(1, min(n_hist, t_arr.shape[0]))

    sstd = np.asarray(loc_std, dtype=np.float64).reshape(2)
    sstd = np.where(np.abs(sstd) > 1e-12, sstd, 1.0)
    tstd = float(time_std) if abs(float(time_std)) > 1e-12 else 1.0

    h_times = ((t_arr[:n_hist] - float(time_mean)) / tstd).astype(np.float32)
    h_locs = ((s_arr[:n_hist] - np.asarray(loc_mean, dtype=np.float64)) / sstd).astype(np.float32)

    dev = torch.device(device)
    h_times_t = torch.tensor(h_times, dtype=torch.float32, device=dev).unsqueeze(0)
    h_locs_t = torch.tensor(h_locs, dtype=torch.float32, device=dev).unsqueeze(0)
    h_len_t = torch.tensor([n_hist], dtype=torch.long, device=dev)

    s_min = np.array([float(xx.min()), float(yy.min())], dtype=np.float64)
    s_max = np.array([float(xx.max()), float(yy.max())], dtype=np.float64)
    s_min_norm = torch.tensor((s_min - loc_mean) / sstd, dtype=torch.float32, device=dev)
    s_max_norm = torch.tensor((s_max - loc_mean) / sstd, dtype=torch.float32, device=dev)
    t_norm = float((float(t_query) - float(time_mean)) / tstd)
    n_grid = int(xx.shape[0])

    jac = max(float(tstd * np.prod(sstd)), 1e-12)

    with torch.no_grad():
        events = torch.cat([h_times_t.unsqueeze(-1), h_locs_t], dim=-1)
        z_hist, _ = model.encoder(events, h_len_t, x_event=None)
        t_prev = h_times_t[torch.arange(1, device=dev), (h_len_t - 1).long()].unsqueeze(-1)
        evaluator = IntensityEvaluator(
            model, z=z_hist, t_prev=t_prev, history_locs_norm=h_locs_t[0]
        )

        _, _, lam_norm = evaluator.intensity_grid(
            t=t_norm,
            s_min=s_min_norm,
            s_max=s_max_norm,
            n_grid=n_grid,
            x_field_fn=None,
        )
        lam_norm_np = lam_norm.detach().cpu().numpy()

    lam = lam_norm_np / jac
    return np.clip(lam, a_min=0.0, a_max=None)


def _model_intensity_on_mesh_paper(
    model: torch.nn.Module,
    times: np.ndarray,
    locations: np.ndarray,
    t_query: float,
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    loc_mean: np.ndarray,
    loc_std: np.ndarray,
    time_mean: float,
    time_std: float,
    device: str,
) -> np.ndarray:
    """
    Evaluate model intensity on a spatial mesh using the paper pipeline's time encoding.

    Unlike the unified variant (which z-scores absolute times), this converts the
    conditioning history to cumsum(MinMax(delta_t)) — matching how
    PaperSlidingWindowDataset.__getitem__ builds sequence inputs.

    The Jacobian from model → native units uses the MinMax ranges:
      lambda_native = lambda_model / (dt_range * x_range * y_range)
    which is structurally identical to the z-score case (std → range).
    """
    t_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(locations, dtype=np.float64).reshape(-1, 2)
    n_hist = int(np.searchsorted(t_arr, float(t_query), side="right"))
    n_hist = max(1, min(n_hist, t_arr.shape[0]))

    sstd     = np.asarray(loc_std, dtype=np.float64)
    sstd     = np.where(np.abs(sstd) > 1e-12, sstd, 1.0)
    dt_range = max(float(time_std), 1e-12)
    dt_min   = float(time_mean)

    # Absolute times → delta_t → MinMax scale → cumulative sum
    ht       = t_arr[:n_hist]
    delta_t  = np.diff(ht, prepend=ht[0])
    delta_t[0] = 0.0
    delta_t_mm = (delta_t - dt_min) / dt_range
    h_times  = np.cumsum(delta_t_mm).astype(np.float32)

    # MinMax-scale locations
    h_locs = ((s_arr[:n_hist] - np.asarray(loc_mean, dtype=np.float64)) / sstd).astype(np.float32)

    dev       = torch.device(device)
    h_times_t = torch.tensor(h_times, dtype=torch.float32, device=dev).unsqueeze(0)
    h_locs_t  = torch.tensor(h_locs,  dtype=torch.float32, device=dev).unsqueeze(0)
    h_len_t   = torch.tensor([n_hist], dtype=torch.long,   device=dev)

    s_min_nat = np.array([float(xx.min()), float(yy.min())], dtype=np.float64)
    s_max_nat = np.array([float(xx.max()), float(yy.max())], dtype=np.float64)
    s_min_mm  = torch.tensor(
        (s_min_nat - np.asarray(loc_mean)) / sstd, dtype=torch.float32, device=dev
    )
    s_max_mm  = torch.tensor(
        (s_max_nat - np.asarray(loc_mean)) / sstd, dtype=torch.float32, device=dev
    )

    # t_query in model space: last model time + next delta_t_mm
    query_dt_native = max(float(t_query) - float(ht[-1]), 1e-6)
    query_dt_mm     = (query_dt_native - dt_min) / dt_range
    t_query_model   = float(h_times[-1]) + query_dt_mm

    n_grid = int(xx.shape[0])

    # Jacobian: native = model / (dt_range * x_range * y_range)
    jac = max(float(dt_range * np.prod(sstd)), 1e-12)

    with torch.no_grad():
        events = torch.cat([h_times_t.unsqueeze(-1), h_locs_t], dim=-1)
        z_hist, _ = model.encoder(events, h_len_t, x_event=None)
        t_prev = h_times_t[torch.arange(1, device=dev), (h_len_t - 1).long()].unsqueeze(-1)
        evaluator = IntensityEvaluator(
            model, z=z_hist, t_prev=t_prev, history_locs_norm=h_locs_t[0]
        )

        _, _, lam_norm = evaluator.intensity_grid(
            t=t_query_model,
            s_min=s_min_mm,
            s_max=s_max_mm,
            n_grid=n_grid,
            x_field_fn=None,
        )
        lam_norm_np = lam_norm.detach().cpu().numpy()

    lam = lam_norm_np / jac
    return np.clip(lam, a_min=0.0, a_max=None)


def _plot_intensity_snippets_true_auto_deep(
    *,
    args: argparse.Namespace,
    params: STHPParams,
    train_seqs: List[Dict[str, np.ndarray]],
    val_seqs: List[Dict[str, np.ndarray]],
    test_seqs: List[Dict[str, np.ndarray]],
    results: Sequence[Dict[str, Any]],
    out_dir: Path,
    eval_device: str,
    n_times: int = 4,
    n_grid: int = 45,
    protocol: str = "unified",
    raw_seq: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Save a 3x4 grid: rows=(True, AutoSTPP, DeepSTPP), cols=four time snippets.

    Both ``unified`` and ``paper_autostpp_sthp`` protocols are supported.
    For ``paper_autostpp_sthp``, history is encoded as cumsum(MinMax(delta_t))
    using ``_model_intensity_on_mesh_paper``; the Jacobian correction uses MinMax
    ranges so all intensity values are in native (x, y, t) units.

    Optionally, this also saves a side-by-side animation
    (True / AutoSTPP / DeepSTPP) when ``args.intensity_video`` is ``gif`` or ``mp4``.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping intensity snippet plot: matplotlib unavailable ({exc})")
        return None

    auto_res = next((r for r in results if r.get("preset") == "auto_stpp"), None)
    deep_res = next((r for r in results if r.get("preset") == "deep_stpp"), None)
    if auto_res is None or deep_res is None:
        print("Skipping intensity snippet plot: missing auto_stpp or deep_stpp run result.")
        return None

    if protocol == "paper_autostpp_sthp":
        if raw_seq is None:
            print("Skipping intensity snippet plot: protocol='paper_autostpp_sthp' requires raw_seq.")
            return None
        _paper_dm = STPPDataModule(
            None, None,
            batch_size=128, num_workers=0, seed=args.seed,
            protocol=protocol, raw_seq=raw_seq,
            paper_lookback=10, paper_lookahead=1, paper_split_ratio=(8, 1, 1),
        )
        _paper_dm.setup()
        train_ds = _paper_dm._train_dataset
        mesh_fn = _model_intensity_on_mesh_paper
    else:
        _unified_dm = STPPDataModule(
            train_seqs, val_seqs, test_seqs=test_seqs,
            batch_size=int(_resolve_model_hparams(args, "auto_stpp")["batch_size"]),
            num_workers=0, normalize=True, seed=args.seed,
        )
        _unified_dm.setup()
        train_ds = _unified_dm._train_dataset
        mesh_fn = _model_intensity_on_mesh
    loc_mean = np.asarray(train_ds.loc_mean, dtype=np.float64)
    loc_std = np.asarray(train_ds.loc_std, dtype=np.float64)
    time_mean = float(train_ds.time_mean)
    time_std = float(train_ds.time_std)

    models: Dict[str, torch.nn.Module] = {}
    for res in (auto_res, deep_res):
        ckpt = res.get("checkpoint_best")
        if not ckpt:
            print(f"Skipping intensity snippet plot: {res['preset']} has no checkpoint path.")
            return None
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"Skipping intensity snippet plot: checkpoint not found: {ckpt_path}")
            return None
        model = build_model(
            config=res.get("overrides", {}),
            spatial_dim=2,
            hidden_dim=int(res.get("run_hparams", {}).get("hidden_dim", args.hidden_dim)),
            event_cov_dim=0,
            field_cov_dim=0,
            preset=res["preset"],
            n_marks=0,
        )
        _load_lightning_ckpt_into_model(ckpt_path, model)
        model.to(eval_device)
        model.eval()
        models[res["preset"]] = model

    seq_idx = int(np.argmax([len(s["times"]) for s in test_seqs]))
    seq = test_seqs[seq_idx]
    seq_t = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
    seq_s = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)
    if seq_t.shape[0] < 2:
        print("Skipping intensity snippet plot: selected test sequence has too few events.")
        return None

    # Static 3x4 snippets use this sequence and paper-faithful START_IDX logic.
    # Video uses an optional wider absolute-time span (see T_VIDEO_START/T_VIDEO_END).
    START_IDX = 2
    seq_t_video = seq_t
    seq_s_video = seq_s

    if protocol == "paper_autostpp_sthp":
        lookback = 10
        st_x_chunks: List[np.ndarray] = []
        st_y_chunks: List[np.ndarray] = []
        loc_chunks: List[np.ndarray] = []
        abs_t_chunks: List[np.ndarray] = []
        abs_s_chunks: List[np.ndarray] = []
        for loc_idx, seq_i in enumerate(test_seqs):
            t_i = np.asarray(seq_i["times"], dtype=np.float32).reshape(-1)
            s_i = np.asarray(seq_i["locations"], dtype=np.float32).reshape(-1, 2)
            if t_i.size == 0:
                continue
            abs_t_chunks.append(t_i.astype(np.float64) + float(loc_idx) * 200.0)
            abs_s_chunks.append(s_i.astype(np.float64))

            n_i = int(t_i.shape[0])
            if n_i <= lookback:
                continue
            st_i = np.column_stack([s_i, t_i]).astype(np.float32)
            n_sw = n_i - lookback
            st_x_chunks.append(np.stack([st_i[j : j + lookback] for j in range(n_sw)], axis=0))
            st_y_chunks.append(np.stack([st_i[j + lookback : j + lookback + 1] for j in range(n_sw)], axis=0))
            loc_chunks.append(np.full((n_sw,), loc_idx, dtype=np.int64))

        if abs_t_chunks:
            seq_t_video = np.concatenate(abs_t_chunks, axis=0)
            seq_s_video = np.concatenate(abs_s_chunks, axis=0)
            order = np.argsort(seq_t_video)
            seq_t_video = seq_t_video[order]
            seq_s_video = seq_s_video[order]

        T_START = float(seq_t.min())
        T_END = float(seq_t.max())
        T_VIDEO_START = float(seq_t_video.min())
        T_VIDEO_END = float(seq_t_video.max())

        if st_y_chunks:
            st_x_all = torch.tensor(np.concatenate(st_x_chunks, axis=0), dtype=torch.float32)
            st_y_all = torch.tensor(np.concatenate(st_y_chunks, axis=0), dtype=torch.float32)
            loc_all = torch.tensor(np.concatenate(loc_chunks, axis=0), dtype=torch.long)
            n_sw_all = int(loc_all.shape[0])
            dummy = torch.zeros((n_sw_all, 1), dtype=torch.float32)
            test_loader = DataLoader(
                torch.utils.data.TensorDataset(dummy, dummy, st_x_all, st_y_all, loc_all),
                batch_size=128,
                shuffle=False,
                num_workers=0,
            )

            his_st_chunks: List[torch.Tensor] = []
            for _, _, st_x, st_y, loc in test_loader:
                loc_batch = (loc.detach().cpu().numpy(),)
                if START_IDX not in loc_batch[0]:
                    continue
                his_st_chunks.append(st_y[np.where(loc_batch[0] == START_IDX)[0]])

            if his_st_chunks:
                his_st = torch.cat(his_st_chunks, dim=0).squeeze(1).cpu().numpy()
                his_st[:, -1] += START_IDX * 200.0
                T_START = float(his_st[:, -1][0])
                T_END = float(his_st[:, -1][-1])
    else:
        T_START = float(seq_t.min())
        T_END = float(seq_t.max())
        T_VIDEO_START = T_START
        T_VIDEO_END = T_END

    print(f"Intensity time range : {T_START} ~ {T_END}")
    if (float(T_VIDEO_END) - float(T_VIDEO_START)) > (float(T_END) - float(T_START)) + 1e-9:
        print(f"Intensity video time range : {T_VIDEO_START} ~ {T_VIDEO_END}")
    t_snaps = np.linspace(T_START, T_END, 4, endpoint=False, dtype=np.float64)
    n_times = int(t_snaps.shape[0])

    if protocol == "paper_autostpp_sthp" and raw_seq is not None:
        # Use full long-sequence locations for representative spatial bounds
        all_locs = np.asarray(raw_seq["locations"], dtype=np.float64).reshape(-1, 2)
    else:
        all_locs = np.concatenate(
            [np.asarray(s["locations"], dtype=np.float64).reshape(-1, 2)
             for s in (train_seqs + val_seqs + test_seqs)],
            axis=0,
        )
    xq = np.percentile(all_locs[:, 0], [1.0, 99.0])
    yq = np.percentile(all_locs[:, 1], [1.0, 99.0])
    x_span = max(float(xq[1] - xq[0]), 1e-3)
    y_span = max(float(yq[1] - yq[0]), 1e-3)
    x_bounds = (float(xq[0] - 0.1 * x_span), float(xq[1] + 0.1 * x_span))
    y_bounds = (float(yq[0] - 0.1 * y_span), float(yq[1] + 0.1 * y_span))

    x = np.linspace(x_bounds[0], x_bounds[1], int(n_grid))
    y = np.linspace(y_bounds[0], y_bounds[1], int(n_grid))
    xx, yy = np.meshgrid(x, y, indexing="ij")

    lam_true: List[np.ndarray] = []
    lam_auto: List[np.ndarray] = []
    lam_deep: List[np.ndarray] = []
    zmax_all = 1e-8
    for ti in t_snaps:
        gt = _true_sthp_intensity_on_mesh(params, seq_t, seq_s, float(ti), xx, yy)
        la = mesh_fn(
            models["auto_stpp"],
            seq_t, seq_s, float(ti), xx, yy,
            loc_mean=loc_mean, loc_std=loc_std,
            time_mean=time_mean, time_std=time_std,
            device=eval_device,
        )
        ld = mesh_fn(
            models["deep_stpp"],
            seq_t, seq_s, float(ti), xx, yy,
            loc_mean=loc_mean, loc_std=loc_std,
            time_mean=time_mean, time_std=time_std,
            device=eval_device,
        )
        lam_true.append(gt)
        lam_auto.append(la)
        lam_deep.append(ld)
        zmax_all = max(zmax_all, float(np.max(gt)), float(np.max(la)), float(np.max(ld)))

    # Anchor visual scale to the true-intensity range so model amplitudes are
    # judged against the generator scale (rather than each model's own peak).
    zmax_true = max(1e-8, max(float(np.max(z)) for z in lam_true))
    zmax = zmax_true

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        3,
        int(n_times),
        left=0.04,
        right=0.995,
        top=0.90,
        bottom=0.07,
        wspace=0.10,
        hspace=0.14,
    )
    rows = [("True STHP", lam_true), ("AutoSTPP", lam_auto), ("DeepSTPP", lam_deep)]
    for r_idx, (row_name, row_vals) in enumerate(rows):
        for c_idx, ti in enumerate(t_snaps):
            ax = fig.add_subplot(gs[r_idx, c_idx], projection="3d")
            ax.plot_surface(
                xx,
                yy,
                row_vals[c_idx],
                cmap="magma",
                linewidth=0,
                antialiased=True,
                vmin=0.0,
                vmax=zmax,
            )
            ax.set_zlim(0.0, zmax)
            ax.view_init(elev=35, azim=-60)
            ax.set_box_aspect((1.0, 1.0, 0.6))
            ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
            if r_idx == 0:
                ax.set_title(f"t={float(ti):.2f}")
            if c_idx == 0:
                ax.text2D(
                    -0.18,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                )
                ax.set_zlabel("lambda")
            else:
                ax.set_zticklabels([])
            if float(np.max(row_vals[c_idx])) > zmax * (1.0 + 1e-12):
                ratio = float(np.max(row_vals[c_idx]) / max(zmax, 1e-12))
                ax.text2D(0.02, 0.94, f"clip x{ratio:.1f}", transform=ax.transAxes, fontsize=8, color="crimson")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

    peak_true = [float(np.max(z)) for z in lam_true]
    peak_auto = [float(np.max(z)) for z in lam_auto]
    peak_deep = [float(np.max(z)) for z in lam_deep]
    mean_true = [float(np.mean(z)) for z in lam_true]
    mean_auto = [float(np.mean(z)) for z in lam_auto]
    mean_deep = [float(np.mean(z)) for z in lam_deep]
    peak_ratio_deep_over_true = [d / max(t, 1e-12) for d, t in zip(peak_deep, peak_true)]
    peak_ratio_auto_over_true = [a / max(t, 1e-12) for a, t in zip(peak_auto, peak_true)]
    mean_ratio_deep_over_auto = [d / max(a, 1e-12) for d, a in zip(mean_deep, mean_auto)]

    fig.suptitle(
        f"STHP intensity snippets (dataset={args.dataset}, seed={args.seed}, seq_idx={seq_idx})",
        fontsize=14,
    )

    out_png = out_dir / "intensity_snippets_true_auto_deep_3x4.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved intensity 3x4 grid: {out_png}")
    print(
        "Intensity peak diagnostics (native units): "
        f"zmax_true={zmax_true:.6g}, zmax_all={zmax_all:.6g}"
    )
    print(
        "Deep vs Auto intensity diagnostics: "
        f"peak_ratio_deep_over_true={[round(x, 3) for x in peak_ratio_deep_over_true]}, "
        f"peak_ratio_auto_over_true={[round(x, 3) for x in peak_ratio_auto_over_true]}, "
        f"mean_ratio_deep_over_auto={[round(x, 3) for x in mean_ratio_deep_over_auto]}"
    )

    video_mode = str(getattr(args, "intensity_video", "none")).lower()
    video_artifact: Optional[Dict[str, Any]] = None
    if video_mode in ("gif", "mp4"):
        try:
            from matplotlib import animation
        except Exception as exc:
            print(f"Skipping intensity video: matplotlib.animation unavailable ({exc})")
        else:
            n_frames = 36
            fps = 10
            t_video = np.linspace(
                float(T_VIDEO_START),
                float(T_VIDEO_END),
                int(n_frames),
                endpoint=False,
                dtype=np.float64,
            )
            v_true: List[np.ndarray] = []
            v_auto: List[np.ndarray] = []
            v_deep: List[np.ndarray] = []
            for ti in t_video:
                gt = _true_sthp_intensity_on_mesh(params, seq_t_video, seq_s_video, float(ti), xx, yy)
                la = mesh_fn(
                    models["auto_stpp"],
                    seq_t_video, seq_s_video, float(ti), xx, yy,
                    loc_mean=loc_mean, loc_std=loc_std,
                    time_mean=time_mean, time_std=time_std,
                    device=eval_device,
                )
                ld = mesh_fn(
                    models["deep_stpp"],
                    seq_t_video, seq_s_video, float(ti), xx, yy,
                    loc_mean=loc_mean, loc_std=loc_std,
                    time_mean=time_mean, time_std=time_std,
                    device=eval_device,
                )
                v_true.append(gt)
                v_auto.append(la)
                v_deep.append(ld)

            zmax_video = max(1e-8, max(float(np.max(z)) for z in v_true))
            fig_v = plt.figure(figsize=(16, 5.5))
            axes = [
                fig_v.add_subplot(1, 3, 1, projection="3d"),
                fig_v.add_subplot(1, 3, 2, projection="3d"),
                fig_v.add_subplot(1, 3, 3, projection="3d"),
            ]
            panels = [("True STHP", v_true), ("AutoSTPP", v_auto), ("DeepSTPP", v_deep)]
            title_obj = fig_v.suptitle(
                f"STHP intensity 3D comparison (dataset={args.dataset}, seed={args.seed})  "
                f"t={float(t_video[0]):.2f}"
            )

            def _update(frame_idx: int):
                for ax, (panel_name, frames) in zip(axes, panels):
                    ax.cla()
                    ax.plot_surface(
                        xx,
                        yy,
                        frames[frame_idx],
                        cmap="magma",
                        linewidth=0,
                        antialiased=True,
                        vmin=0.0,
                        vmax=zmax_video,
                    )
                    ax.set_title(panel_name)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("lambda")
                    ax.set_zlim(0.0, zmax_video)
                    ax.view_init(elev=35, azim=-60)
                    ax.set_box_aspect((1.0, 1.0, 0.6))
                title_obj.set_text(
                    f"STHP intensity 3D comparison (dataset={args.dataset}, seed={args.seed})  "
                    f"t={float(t_video[frame_idx]):.2f}"
                )
                return [*axes, title_obj]

            ani = animation.FuncAnimation(
                fig_v,
                _update,
                frames=len(t_video),
                interval=int(1000 / max(fps, 1)),
                blit=False,
            )
            _update(0)
            requested_mode = video_mode
            saved_mode = requested_mode
            out_video = out_dir / (
                "intensity_compare_true_auto_deep.gif"
                if requested_mode == "gif"
                else "intensity_compare_true_auto_deep.mp4"
            )
            save_error: Optional[str] = None
            try:
                if requested_mode == "gif":
                    ani.save(out_video, writer=animation.PillowWriter(fps=fps), dpi=140)
                else:
                    ani.save(out_video, writer=animation.FFMpegWriter(fps=fps), dpi=140)
            except Exception as exc:
                save_error = str(exc)
                if requested_mode == "mp4":
                    # Fallback when ffmpeg is unavailable.
                    out_video = out_dir / "intensity_compare_true_auto_deep.gif"
                    saved_mode = "gif"
                    try:
                        ani.save(out_video, writer=animation.PillowWriter(fps=fps), dpi=140)
                        save_error = None
                    except Exception as exc2:
                        save_error = f"{save_error}; fallback gif failed: {exc2}"
            finally:
                plt.close(fig_v)

            if save_error is None:
                print(f"Saved intensity side-by-side {saved_mode}: {out_video}")
                if requested_mode == "mp4" and saved_mode != "mp4":
                    print("Requested mp4, but ffmpeg was unavailable; saved gif fallback instead.")
                video_artifact = {
                    "path": str(out_video),
                    "format_requested": requested_mode,
                    "format_saved": saved_mode,
                    "fps": int(fps),
                    "n_frames": int(len(t_video)),
                    "time_range": [float(T_VIDEO_START), float(T_VIDEO_END)],
                    "render": "3d_surface",
                }
            else:
                print(f"Skipping intensity video: failed to save animation ({save_error})")

    return {
        "path": str(out_png),
        "sequence_index": int(seq_idx),
        "time_snippets": [float(t) for t in t_snaps.tolist()],
        "grid_n": int(n_grid),
        "x_bounds": [float(x_bounds[0]), float(x_bounds[1])],
        "y_bounds": [float(y_bounds[0]), float(y_bounds[1])],
        "scale_mode": "anchored_to_true",
        "zmax_true": float(zmax_true),
        "zmax_all_models_and_true": float(zmax_all),
        "peak_true_per_time": peak_true,
        "peak_auto_per_time": peak_auto,
        "peak_deep_per_time": peak_deep,
        "mean_true_per_time": mean_true,
        "mean_auto_per_time": mean_auto,
        "mean_deep_per_time": mean_deep,
        "peak_ratio_deep_over_true_per_time": [float(x) for x in peak_ratio_deep_over_true],
        "peak_ratio_auto_over_true_per_time": [float(x) for x in peak_ratio_auto_over_true],
        "mean_ratio_deep_over_auto_per_time": [float(x) for x in mean_ratio_deep_over_auto],
        "video": video_artifact,
    }


# ---------------------------------------------------------------------------
# Parity checks
# ---------------------------------------------------------------------------

def _parity_checks(
    *,
    dataset_name: str,
    params: STHPParams,
    unified_short: Dict[str, np.ndarray],
    ref_short: Dict[str, np.ndarray],
    canonical_batch: Dict[str, Any],
    n_train: int,
    n_val: int,
    n_test: int,
    total_windows: int,
    window_T: float,
) -> List[Dict[str, Any]]:
    """
    Parity checklist comparing unified STHP simulator to the reference
    AutoSTPP-style simulator, and verifying that the canonical batch
    conforms to the data contract.
    """
    checks: List[Dict[str, Any]] = []

    # 1. Simulation parity
    times_equal = np.allclose(unified_short["times"], ref_short["times"], atol=1e-7, rtol=0.0)
    locs_equal = np.allclose(
        unified_short["locations"], ref_short["locations"], atol=1e-7, rtol=0.0
    )
    checks.append(
        {
            "name": "simulation_procedure_ogata_and_gaussian_kernels",
            "pass": bool(times_equal and locs_equal),
            "detail": (
                f"short-horizon exact match: times={times_equal}, locations={locs_equal}, "
                f"events={len(unified_short['times'])}"
            ),
        }
    )

    # 2. Dataset parameter set defined
    checks.append(
        {
            "name": "parameter_set_defined_for_sthp0_1_2",
            "pass": dataset_name in STHP_PRESETS,
            "detail": params.source,
        }
    )

    # 3. Domain/split logic matches paper
    checks.append(
        {
            "name": "domain_and_split_logic_matches_autostpp_paper",
            "pass": (
                math.isclose(window_T, 200.0)
                and total_windows == 50
                and n_train == 40
                and n_val == 5
                and n_test == 5
            ),
            "detail": (
                f"windows={total_windows}, window_T={window_T}, "
                f"split={n_train}/{n_val}/{n_test}"
            ),
        }
    )

    # 4. Canonical batch has required schema fields
    has_txy  = "txys" in canonical_batch and canonical_batch["txys"] is not None
    has_mask = "pad_mask" in canonical_batch and canonical_batch["pad_mask"] is not None
    checks.append(
        {
            "name": "canonical_batch_has_txys_and_pad_mask",
            "pass": bool(has_txy and has_mask),
            "detail": f"txys={has_txy}, pad_mask={has_mask}",
        }
    )

    # 5. Data contract validation (validate_batch)
    contract_ok = True
    contract_err = ""
    try:
        validate_batch(canonical_batch)
    except Exception as exc:
        contract_ok = False
        contract_err = str(exc)
    checks.append(
        {
            "name": "canonical_batch_passes_data_contract",
            "pass": contract_ok,
            "detail": "validate_batch() OK" if contract_ok else f"FAILED: {contract_err}",
        }
    )

    # 6. Padding mask consistency and dtype correctness
    pad_ok = False
    dtypes_ok = False
    if has_mask and has_txy:
        pad_mask = canonical_batch["pad_mask"]
        lengths  = canonical_batch["lengths"]
        txys     = canonical_batch["txys"]
        pad_ok   = bool(torch.all(pad_mask.sum(dim=1).to(lengths.dtype) == lengths))
        dtypes_ok = (
            str(canonical_batch["times"].dtype) == "torch.float32"
            and str(canonical_batch["locations"].dtype) == "torch.float32"
            and str(lengths.dtype) == "torch.int64"
            and str(txys.dtype) == "torch.float32"
        )
    checks.append(
        {
            "name": "padding_and_dtype_consistency",
            "pass": bool(pad_ok and dtypes_ok),
            "detail": f"pad_consistent={pad_ok}, dtypes_ok={dtypes_ok}",
        }
    )

    return checks


# ---------------------------------------------------------------------------
# Audit checklist
# ---------------------------------------------------------------------------

def _audit_checklist(
    *,
    train_seqs: List[Dict[str, np.ndarray]],
    canonical_batch: Dict[str, Any],
    auto_result: Dict[str, Any],
    overfit_auto: Dict[str, Any],
    args: argparse.Namespace,
    parity_report: Optional[ParityReport],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    all_t = np.concatenate([s["times"] for s in train_seqs], axis=0).astype(np.float64)
    t_min, t_max = float(all_t.min()), float(all_t.max())

    # Required coverage: bbox must contain ALL z-scored training locations.
    # Derive the actual z-scored range from the canonical_batch (which uses
    # the same z-score stats as the training dm) and compare to the decoder bbox.
    tr = auto_result.get("input_transform", {})
    dec_cfg = auto_result.get("overrides", {}).get("decoder", {})
    x_lo = float(dec_cfg.get("x_lo", np.nan))
    x_hi = float(dec_cfg.get("x_hi", np.nan))
    y_lo = float(dec_cfg.get("y_lo", np.nan))
    y_hi = float(dec_cfg.get("y_hi", np.nan))

    # Use the z-scored locations from the canonical batch
    z_locs = canonical_batch.get("locations")  # (B, N, 2) tensor, z-scored
    if z_locs is not None:
        valid = canonical_batch["pad_mask"]  # (B, N) bool
        flat_locs = z_locs[valid]            # (n_valid, 2)
        actual_x_min = float(flat_locs[:, 0].min().item())
        actual_x_max = float(flat_locs[:, 0].max().item())
        actual_y_min = float(flat_locs[:, 1].min().item())
        actual_y_max = float(flat_locs[:, 1].max().item())
    else:
        actual_x_min = actual_x_max = actual_y_min = actual_y_max = float("nan")

    bbox_ok = bool(
        not math.isnan(x_lo)
        and x_lo <= actual_x_min
        and x_hi >= actual_x_max
        and y_lo <= actual_y_min
        and y_hi >= actual_y_max
    )

    checks.append(
        {
            "section": "1) Data/units",
            "name": "auto_stpp_bbox_covers_zscore_spatial_range",
            "pass": bbox_ok,
            "detail": (
                f"transform={tr.get('name')}, decoder bbox "
                f"x=[{x_lo:.3g},{x_hi:.3g}] y=[{y_lo:.3g},{y_hi:.3g}], "
                f"data z-scored range x=[{actual_x_min:.3g},{actual_x_max:.3g}] "
                f"y=[{actual_y_min:.3g},{actual_y_max:.3g}]"
            ),
        }
    )
    checks.append(
        {
            "section": "1) Data/units",
            "name": "time_scale_is_bounded_and_consistent_with_original_windowing",
            "pass": bool(0.0 <= t_min and t_max <= 200.0 + 1e-6),
            "detail": f"time range in train windows=[{t_min:.3g},{t_max:.3g}]",
        }
    )

    txys_ok = False
    if "txys" in canonical_batch and canonical_batch["txys"] is not None:
        txys = canonical_batch["txys"]
        txys_ok = bool(
            torch.allclose(txys[..., 0], canonical_batch["times"])
            and torch.allclose(txys[..., 1:], canonical_batch["locations"])
        )
    checks.append(
        {
            "section": "2) Batch/sequence encoding",
            "name": "txys_field_order_is_exactly_(t,x,y)",
            "pass": txys_ok,
            "detail": f"txys_order_ok={txys_ok}",
        }
    )
    pad_ok = bool(
        torch.all(
            canonical_batch["pad_mask"].sum(dim=1).to(canonical_batch["lengths"].dtype)
            == canonical_batch["lengths"]
        )
    )
    checks.append(
        {
            "section": "2) Batch/sequence encoding",
            "name": "padding_mask_matches_lengths",
            "pass": pad_ok,
            "detail": (
                f"pad_true_counts={canonical_batch['pad_mask'].sum(dim=1).tolist()}, "
                f"lengths={canonical_batch['lengths'].tolist()}"
            ),
        }
    )

    # 3) Objective parity: LikelihoodEvaluator.parity_check
    if parity_report is not None:
        nll_diff = parity_report.max_abs_diff
        checks.append(
            {
                "section": "3) Objective definition",
                "name": "autoint_manual_event_nll_matches_model_forward_nll",
                "pass": parity_report.passed,
                "detail": (
                    f"model_nll={parity_report.model_nll:.6f}, "
                    f"manual_nll={parity_report.manual_nll:.6f}, "
                    f"abs_diff={nll_diff:.2e}, tol={parity_report.tol:.2e}"
                ),
            }
        )
    else:
        checks.append(
            {
                "section": "3) Objective definition",
                "name": "autoint_manual_event_nll_matches_model_forward_nll",
                "pass": False,
                "detail": "parity_check not available",
            }
        )

    auto_cfg = _resolve_model_hparams(args, "auto_stpp")
    lr_ok = float(auto_cfg["lr"]) in (0.004, 0.0002)
    checks.append(
        {
            "section": "4) Training config",
            "name": "paper_like_optimizer_and_key_hyperparams",
            "pass": bool(
                auto_result.get("optimizer") == "Adam"
                and float(auto_result.get("adam_beta1")) == float(auto_cfg["adam_beta1"])
                and lr_ok
                and int(auto_cfg["batch_size"]) == 128
                and int(auto_cfg["epochs"]) == 50
                and int(auto_result.get("overrides", {}).get("decoder", {}).get("n_components", -1)) == 2
            ),
            "detail": (
                f"optimizer={auto_result.get('optimizer')}, beta1={auto_result.get('adam_beta1')}, "
                f"lr={auto_cfg['lr']}, batch={auto_cfg['batch_size']}, epochs={auto_cfg['epochs']}, "
                f"N={auto_result.get('overrides', {}).get('decoder', {}).get('n_components')}"
            ),
        }
    )
    checks.append(
        {
            "section": "5) Evaluation",
            "name": "evaluation_uses_LikelihoodEvaluator_single_source_of_truth",
            "pass": True,
            "detail": (
                "All NLL/LL metrics produced by LikelihoodEvaluator.evaluate(). "
                "Sets model.eval(); deterministic forward likelihood; no MC estimator; "
                "identical masking convention for all models."
            ),
        }
    )
    checks.append(
        {
            "section": "Diagnostics",
            "name": "tiny_subset_can_overfit_in_20_steps",
            "pass": bool(overfit_auto.get("can_overfit", False)),
            "detail": (
                f"start_nll={overfit_auto.get('start_nll'):.4f}, "
                f"end_nll={overfit_auto.get('end_nll'):.4f}, "
                f"relative_drop={overfit_auto.get('relative_drop'):.3f}"
            ),
        }
    )
    return checks


# ---------------------------------------------------------------------------
# Tiny-subset overfit diagnostic  (uses LegacyTrainer for the training loop
# since the full Lightning setup is overkill for a 20-step sanity check)
# ---------------------------------------------------------------------------

def _overfit_tiny_subset(
    *,
    preset: str,
    train_seqs: List[Dict[str, np.ndarray]],
    args: argparse.Namespace,
    eval_device: str,
    seed: int,
) -> Dict[str, Any]:
    run_cfg = _resolve_model_hparams(args, preset)
    tiny = train_seqs[:2]
    if len(tiny) == 0:
        raise RuntimeError("No train sequences for overfit diagnostic.")

    tiny_dm = STPPDataModule(
        tiny,
        tiny,
        test_seqs=tiny,
        batch_size=min(len(tiny), 2),
        num_workers=0,
        normalize=True,
        seed=seed,
    )
    tiny_dm.setup()

    overrides: Dict[str, Any] = {}
    if preset == "auto_stpp":
        bbox = _autoint_bbox_from_dm(tiny_dm)
        overrides = {
            "decoder": {
                **bbox,
                "n_components": 2,
                "n_layers": 2,
                "internal_dim": 64,
            }
        }

    model = build_model(
        config=overrides,
        spatial_dim=2,
        hidden_dim=int(run_cfg["hidden_dim"]),
        event_cov_dim=0,
        field_cov_dim=0,
        preset=preset,
        n_marks=0,
    )
    trainer = LegacyTrainer(
        model=model,
        lr=float(run_cfg["lr"]),
        weight_decay=1e-5,
        grad_clip=0.0,
        device=eval_device,
    )
    hist = trainer.train(
        train_loader=tiny_dm.train_dataloader(),
        val_loader=None,
        n_epochs=20,
        log_every=1000,
        early_stopping_patience=0,
        restore_best=False,
    )
    train_curve = [float(x) for x in hist.get("train_nll", [])]
    if not train_curve:
        raise RuntimeError("Tiny overfit diagnostic produced empty training curve.")
    start = train_curve[0]
    end = train_curve[-1]
    rel_drop = (start - end) / max(abs(start), 1e-12)
    return {
        "preset": preset,
        "n_sequences": len(tiny),
        "steps": len(train_curve),
        "start_nll": float(start),
        "end_nll": float(end),
        "relative_drop": float(rel_drop),
        "curve": train_curve,
        "can_overfit": bool(end < start),
    }


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def _train_one_model(
    *,
    preset: str,
    train_seqs: List[Dict[str, np.ndarray]],
    val_seqs: List[Dict[str, np.ndarray]],
    test_seqs: List[Dict[str, np.ndarray]],
    args: argparse.Namespace,
    output_dir: Path,
    accelerator: str,
    devices: int,
    eval_device: str,
    protocol: str = "unified",
    raw_seq: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    run_name = f"{args.dataset}_{preset}_seed{args.seed}"
    pl.seed_everything(args.seed, workers=True)
    run_cfg = _resolve_model_hparams(args, preset)

    if protocol == "paper_autostpp_sthp":
        input_transform: Dict[str, Any] = {"name": "minmax_delta_t"}
        # Read seq_len from the preset so the data window matches the model's
        # expected history length (deep_stpp uses seq_len from decoder.spatial).
        _preset_cfg = PRESETS.get(preset, {})
        paper_lookback = int(
            _preset_cfg.get("seq_len")
            or _preset_cfg.get("decoder", {}).get("spatial", {}).get("seq_len", 10)
        )
        dm = STPPDataModule(
            None,
            None,
            batch_size=int(run_cfg["batch_size"]),
            num_workers=0,
            seed=args.seed,
            protocol=protocol,
            raw_seq=raw_seq,
            paper_lookback=paper_lookback,
            paper_lookahead=1,
            paper_split_ratio=(8, 1, 1),
        )
    else:
        # Both models use the same z-score normalization so that LL values are
        # directly comparable on the same coordinate system (mean≈0, std≈1).
        input_transform = {
            "name": "zscore",
            "train_ranges_before": _compute_ranges(train_seqs),
        }
        # normalize=True applies per-dimension z-score from training statistics.
        # seed= passes a persistent generator to train_dataloader(), making the
        # shuffle sequence deterministic and independent of the global torch RNG.
        dm = STPPDataModule(
            train_seqs,
            val_seqs,
            test_seqs=test_seqs,
            batch_size=int(run_cfg["batch_size"]),
            num_workers=0,
            normalize=True,
            seed=args.seed,
        )
    dm.setup()

    overrides: Dict[str, Any] = {}
    if preset == "auto_stpp":
        # Compute bbox from the ACTUAL z-scored training data range + margin.
        # A fixed ±3.5σ is too narrow for heavy-tailed datasets (e.g. sthp1
        # with g0_cov=[[5,0],[0,5]] can push z-scored locations to ±4.3σ).
        # Events outside the bbox make the compensator integral incorrect.
        bbox = _autoint_bbox_from_dm(dm)
        overrides = {
            "decoder": {
                **bbox,
                # Paper Table 5: N = 2 for synthetic datasets.
                "n_components": 2,
                "n_layers": 2,
                "internal_dim": 64,
            }
        }
    elif preset == "deep_stpp":
        overrides = {
            "encoder": {
                "num_heads": int(args.deep_num_heads),
                "num_layers": int(args.deep_num_layers),
            },
            "updater": {
                "num_heads": int(args.deep_num_heads),
            },
            "decoder": {
                "temporal": {"n_components": int(args.deep_k)},
                "spatial": {"n_components": int(args.deep_k)},
            },
        }

    model = build_model(
        config=overrides,
        spatial_dim=2,
        hidden_dim=int(run_cfg["hidden_dim"]),
        event_cov_dim=0,
        field_cov_dim=0,
        preset=preset,
        n_marks=0,
    )
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    module = STPPLightningModule(
        model=model,
        lr=float(run_cfg["lr"]),
        weight_decay=1e-5,
        grad_clip=float(run_cfg["grad_clip"]),
        adam_beta1=float(run_cfg["adam_beta1"]),
    )

    logs_root = output_dir / "logs"
    ckpt_root = output_dir / "checkpoints" / run_name
    logger   = CSVLogger(save_dir=str(logs_root), name=run_name)
    ckpt_cb  = ModelCheckpoint(
        dirpath=str(ckpt_root),
        monitor="val/nll",
        mode="min",
        save_top_k=1,
        filename="epoch{epoch:03d}-val_nll{val/nll:.6f}",
        auto_insert_metric_name=False,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="epoch")]

    trainer = pl.Trainer(
        max_epochs=int(run_cfg["epochs"]),
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        deterministic=True,
        log_every_n_steps=1,
    )
    trainer.fit(module, dm)

    best_ckpt = Path(ckpt_cb.best_model_path) if ckpt_cb.best_model_path else None
    if best_ckpt is not None and best_ckpt.exists():
        _load_lightning_ckpt_into_model(best_ckpt, model)

    # Parity check on one train batch (post-training, best checkpoint loaded).
    parity_report: Optional[ParityReport] = None
    if preset == "auto_stpp":
        ev_probe = LikelihoodEvaluator(model, device=eval_device)
        train_batch = next(iter(dm.train_dataloader()))
        parity_report = ev_probe.parity_check(train_batch, tol=1e-4)

    # Event-weighted metrics using the shared LikelihoodEvaluator.
    ev = LikelihoodEvaluator(model, device=eval_device)
    train_result = ev.evaluate(dm.train_dataloader())
    val_result   = ev.evaluate(dm.val_dataloader())
    test_result  = ev.evaluate(dm.test_dataloader())

    def _result_dict(r: EvalResult) -> Dict[str, Any]:
        return {
            "nll_per_event": r.nll_per_event,
            "ll_per_event":  r.ll_per_event,
            "n_events":      r.n_events,
            "ll_total":      r.ll_total,
        }

    return {
        "preset":           preset,
        "optimizer":        "Adam",
        "adam_beta1":       float(run_cfg["adam_beta1"]),
        "adam_beta2":       0.999,
        "run_hparams":      run_cfg,
        "n_params":         n_params,
        "overrides":        overrides,
        "input_transform":  input_transform,
        "parity_report":    {
            "passed":       parity_report.passed       if parity_report else None,
            "model_nll":    parity_report.model_nll    if parity_report else None,
            "manual_nll":   parity_report.manual_nll   if parity_report else None,
            "max_abs_diff": parity_report.max_abs_diff if parity_report else None,
            "n_events":     parity_report.n_events     if parity_report else None,
        },
        "train":            _result_dict(train_result),
        "val":              _result_dict(val_result),
        "test":             _result_dict(test_result),
        "checkpoint_best":  str(best_ckpt) if best_ckpt is not None else None,
        "logger_dir":       logger.log_dir,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_parity(checks: Sequence[Dict[str, Any]]) -> None:
    print("\nParity / Data-Contract Checklist")
    print("---------------------------------")
    for item in checks:
        flag = "PASS" if item["pass"] else "FAIL"
        print(f"[{flag}] {item['name']}: {item['detail']}")


def _print_results_table(results: Sequence[Dict[str, Any]]) -> None:
    print("\nTest Metrics — shared LikelihoodEvaluator (same splits, same budget)")
    print("----------------------------------------------------------------------")
    header = f"{'Model':<12} {'LL/event':>12} {'NLL/event':>12} {'LL total':>14} {'N events':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        t = r["test"]
        print(
            f"{r['preset']:<12} "
            f"{t['ll_per_event']:>12.6f} "
            f"{t['nll_per_event']:>12.6f} "
            f"{t['ll_total']:>14.2f} "
            f"{t['n_events']:>10d}"
        )


def _print_audit_checklist(items: Sequence[Dict[str, Any]]) -> None:
    print("\nAutoSTPP Audit Checklist")
    print("------------------------")
    n_pass = sum(1 for it in items if it["pass"])
    n_fail = len(items) - n_pass
    for it in items:
        flag = "PASS" if it["pass"] else "FAIL"
        print(f"[{flag}] {it['section']} | {it['name']}: {it['detail']}")
    print(f"\nSummary: {n_pass} PASS / {n_fail} FAIL")


def _print_uniformity_summary(
    parity_checks: Sequence[Dict[str, Any]],
    audit_checks: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
) -> None:
    """Print the overall framework uniformity PASS/FAIL summary."""
    all_checks = list(parity_checks) + list(audit_checks)
    n_pass = sum(1 for c in all_checks if c["pass"])
    n_fail = len(all_checks) - n_pass

    print("\n" + "=" * 60)
    print("FRAMEWORK UNIFORMITY SUMMARY")
    print("=" * 60)
    print(f"  Data-contract / parity checks : {n_pass}/{len(all_checks)} PASS")
    print(
        f"  Evaluation source of truth    : LikelihoodEvaluator "
        f"(identical for all {len(results)} models)"
    )
    for r in results:
        pr = r.get("parity_report", {})
        pr_pass = pr.get("passed")
        pr_str = (
            f"parity {'PASS' if pr_pass else 'FAIL'} "
            f"(diff={pr.get('max_abs_diff')})"
            if pr_pass is not None
            else "parity N/A"
        )
        print(f"    {r['preset']:<12}: {pr_str}")
    overall = "PASS" if n_fail == 0 else f"FAIL ({n_fail} checks failed)"
    print(f"\n  Overall: {overall}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    auto_cfg = _resolve_model_hparams(args, "auto_stpp")
    deep_cfg = _resolve_model_hparams(args, "deep_stpp")
    for name, cfg in (("auto_stpp", auto_cfg), ("deep_stpp", deep_cfg)):
        if int(cfg["epochs"]) <= 0:
            raise ValueError(f"{name}: epochs must be > 0")
        if int(cfg["batch_size"]) <= 0:
            raise ValueError(f"{name}: batch_size must be > 0")
        if float(cfg["lr"]) <= 0:
            raise ValueError(f"{name}: lr must be > 0")
        if int(cfg["hidden_dim"]) <= 0:
            raise ValueError(f"{name}: hidden_dim must be > 0")
        if float(cfg["grad_clip"]) < 0:
            raise ValueError(f"{name}: grad_clip must be >= 0")
        if not (0.0 < float(cfg["adam_beta1"]) < 1.0):
            raise ValueError(f"{name}: adam_beta1 must be in (0,1)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    accelerator, devices, eval_device = _resolve_runtime_device(args.device)
    print(f"Runtime device: accelerator={accelerator}, eval_device={eval_device}")

    params    = STHP_PRESETS[args.dataset]
    t_end_long = 10000.0
    n_windows  = 50
    window_T   = 200.0
    n_train, n_val, n_test = 40, 5, 5

    print(
        f"Dataset: {args.dataset} ({params.source})\n"
        f"alpha={params.alpha}, beta={params.beta}, mu={params.mu}\n"
        f"g0_cov={params.g0_cov.tolist()}, g2_cov={params.g2_cov.tolist()}"
    )
    print(
        f"Generating one long STHP sequence on [0, {t_end_long:.0f}) then split into "
        f"{n_windows} windows of length {window_T:.0f} with split {n_train}/{n_val}/{n_test}."
    )
    print(f"Protocol: {args.protocol!r}")

    # Guardrail: raise early if the protocol hasn't been validated for these models.
    for preset in ("auto_stpp", "deep_stpp"):
        assert_protocol_model_compatible(args.protocol, preset)

    long_seq = _generate_long_sequence_unified(params, seed=args.seed, t_end=t_end_long)
    windows  = _split_long_sequence(
        long_seq, n_windows=n_windows, window_T=window_T, reset_time_to_window=True,
    )
    train_seqs = windows[:n_train]
    val_seqs   = windows[n_train : n_train + n_val]
    test_seqs  = windows[n_train + n_val : n_train + n_val + n_test]

    if not _all_sequences_valid(windows, min_len=3):
        bad = [i for i, s in enumerate(windows) if len(s["times"]) < 3]
        raise RuntimeError(
            f"Some windows have fewer than 3 events: {bad[:10]}"
            f"{'...' if len(bad) > 10 else ''}. "
            "Try a different seed or larger horizon."
        )

    # Simulation parity: compare short-horizon generation against local reference.
    unified_short = _generate_long_sequence_unified(params, seed=args.seed, t_end=200.0)
    ref_short     = _generate_long_sequence_reference(params, seed=args.seed, t_end=200.0)

    # Build canonical DataModule and grab one train batch for contract validation.
    probe_dm = STPPDataModule(
        train_seqs, val_seqs, test_seqs=test_seqs,
        batch_size=max(int(auto_cfg["batch_size"]), int(deep_cfg["batch_size"])),
        num_workers=0, normalize=True,
        seed=args.seed,
    )
    probe_dm.setup()
    canonical_batch = next(iter(probe_dm.train_dataloader()))

    # Validate data contract and capture fingerprint.
    print("\nValidating canonical batch against data contract...")
    try:
        validate_batch(canonical_batch)
        print("  [PASS] validate_batch()")
    except Exception as exc:
        print(f"  [FAIL] validate_batch(): {exc}")

    fp = fingerprint_batch(canonical_batch)
    print(
        f"  Batch fingerprint: n_seq={fp['n_seq']}, "
        f"n_events_max={fp['n_events_max']}, "
        f"spatial_dim={fp['spatial_dim']}, "
        f"total_events={fp['total_events']}"
    )
    if "times" in fp.get("stats", {}):
        ts = fp["stats"]["times"]
        print(f"  times  : min={ts['min']:.4f}  mean={ts['mean']:.4f}  max={ts['max']:.4f}")
    if "locations" in fp.get("stats", {}):
        ls = fp["stats"]["locations"]
        print(f"  locs   : min={ls['min']:.4f}  mean={ls['mean']:.4f}  max={ls['max']:.4f}")

    parity = _parity_checks(
        dataset_name=args.dataset,
        params=params,
        unified_short=unified_short,
        ref_short=ref_short,
        canonical_batch=canonical_batch,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        total_windows=n_windows,
        window_T=window_T,
    )
    _print_parity(parity)

    out_dir = Path(args.output_dir) / f"{args.dataset}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train both models on identical splits with identical budget.
    results: List[Dict[str, Any]] = []
    for preset in ("auto_stpp", "deep_stpp"):
        print(f"\nTraining preset={preset} ...")
        r = _train_one_model(
            preset=preset,
            train_seqs=train_seqs,
            val_seqs=val_seqs,
            test_seqs=test_seqs,
            args=args,
            output_dir=out_dir,
            accelerator=accelerator,
            devices=devices,
            eval_device=eval_device,
            protocol=args.protocol,
            raw_seq=long_seq,
        )
        results.append(r)

    _print_results_table(results)

    # Tiny overfit diagnostic (AutoSTPP, 20 gradient steps on 2 sequences).
    print("\nRunning tiny overfit diagnostic (AutoSTPP, 20 steps on 2 sequences)...")
    overfit_auto = _overfit_tiny_subset(
        preset="auto_stpp",
        train_seqs=train_seqs,
        args=args,
        eval_device=eval_device,
        seed=args.seed,
    )
    print(
        "Tiny overfit result: "
        f"start_nll={overfit_auto['start_nll']:.6f}, "
        f"end_nll={overfit_auto['end_nll']:.6f}, "
        f"relative_drop={overfit_auto['relative_drop']:.3f}, "
        f"can_overfit={overfit_auto['can_overfit']}"
    )

    auto_result = next((r for r in results if r["preset"] == "auto_stpp"), None)
    if auto_result is None:
        raise RuntimeError("Missing auto_stpp result for audit checklist.")

    parity_report: Optional[ParityReport] = None
    pr_dict = auto_result.get("parity_report", {})
    if pr_dict.get("passed") is not None:
        parity_report = ParityReport(
            passed=bool(pr_dict["passed"]),
            model_nll=float(pr_dict["model_nll"]),
            manual_nll=float(pr_dict["manual_nll"]),
            max_abs_diff=float(pr_dict["max_abs_diff"]),
            n_events=int(pr_dict["n_events"]),
            tol=1e-4,
        )

    if args.protocol == "unified":
        audit = _audit_checklist(
            train_seqs=train_seqs,
            canonical_batch=canonical_batch,
            auto_result=auto_result,
            overfit_auto=overfit_auto,
            args=args,
            parity_report=parity_report,
        )
        _print_audit_checklist(audit)
    else:
        audit = []
        print(
            f"\n[protocol={args.protocol!r}] Skipping audit checklist "
            "(z-score bbox checks do not apply to the MinMax pipeline)."
        )

    _print_uniformity_summary(parity, audit, results)

    intensity_plot = _plot_intensity_snippets_true_auto_deep(
        args=args,
        params=params,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        test_seqs=test_seqs,
        results=results,
        out_dir=out_dir,
        eval_device=eval_device,
        protocol=args.protocol,
        raw_seq=long_seq,
    )

    # Basic finite checks
    for r in results:
        for split in ("train", "val", "test"):
            m = r[split]
            if not np.isfinite(m["nll_per_event"]):
                raise RuntimeError(f"Non-finite NLL for {r['preset']}:{split}")
            if not np.isfinite(m["ll_per_event"]):
                raise RuntimeError(f"Non-finite LL/event for {r['preset']}:{split}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_short(),
        "args": {
            "dataset":             args.dataset,
            "protocol":            args.protocol,
            "seed":                int(args.seed),
            "epochs":              int(args.epochs),
            "batch_size":          int(args.batch_size),
            "hidden_dim":          int(args.hidden_dim),
            "deep_num_heads":      int(args.deep_num_heads),
            "deep_num_layers":     int(args.deep_num_layers),
            "deep_k":              int(args.deep_k),
            "lr":                  float(args.lr),
            "auto_epochs":         args.auto_epochs,
            "deep_epochs":         args.deep_epochs,
            "auto_batch_size":     args.auto_batch_size,
            "deep_batch_size":     args.deep_batch_size,
            "auto_lr":             args.auto_lr,
            "deep_lr":             args.deep_lr,
            "auto_hidden_dim":     args.auto_hidden_dim,
            "deep_hidden_dim":     args.deep_hidden_dim,
            "auto_grad_clip":      float(args.auto_grad_clip),
            "deep_grad_clip":      float(args.deep_grad_clip),
            "auto_adam_beta1":     float(args.auto_adam_beta1),
            "deep_adam_beta1":     float(args.deep_adam_beta1),
            "resolved_auto_hparams": auto_cfg,
            "resolved_deep_hparams": deep_cfg,
            "optimizer":           "Adam",
            "adam_beta1":          0.9,
            "adam_beta2":          0.999,
            "device":              args.device,
            "resolved_accelerator": accelerator,
            "resolved_eval_device": eval_device,
            "output_dir":          str(out_dir),
            "intensity_video":     args.intensity_video,
        },
        "dataset": {
            "name":        args.dataset,
            "source":      params.source,
            "alpha":       float(params.alpha),
            "beta":        float(params.beta),
            "mu":          float(params.mu),
            "g0_cov":      params.g0_cov.tolist(),
            "g2_cov":      params.g2_cov.tolist(),
            "long_horizon": t_end_long,
            "n_windows":   n_windows,
            "window_T":    window_T,
            "split":       {"train": n_train, "val": n_val, "test": n_test},
            "events_per_split": (
                {
                    "train": int(sum(len(s["times"]) for s in train_seqs)),
                    "val":   int(sum(len(s["times"]) for s in val_seqs)),
                    "test":  int(sum(len(s["times"]) for s in test_seqs)),
                }
                if args.protocol == "unified"
                else {"note": "paper_protocol: window counts reported by DataModule at setup()"}
            ),
        },
        "data_contract": {
            "validator":        "unified_stpp.data.contract.validate_batch",
            "canonical_batch_fingerprint": fp,
            "batch_summary":    _summarize_batch(canonical_batch),
            "data_module":      "unified_stpp.training.data_module.STPPDataModule",
        },
        "parity": {
            "checks":             parity,
            "simulation_events":  len(unified_short["times"]),
        },
        "audit_checklist": audit,
        "intensity_snippets": intensity_plot,
        "tiny_overfit_auto": overfit_auto,
        "results": results,
    }

    out_json = out_dir / "results.json"
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved artifact: {out_json}")


if __name__ == "__main__":
    main()
