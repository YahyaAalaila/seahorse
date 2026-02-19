"""
Train a unified STPP model.

Usage:
    python train.py --config configs/deep_stpp.yaml
    python train.py --preset neural_stpp
    python train.py --preset deep_stpp --field_cov_dim 1 --data inhomogeneous
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime
import numpy as np
import yaml
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from unified_stpp.registry import build_model
from unified_stpp.data import STPPDataset, collate_fn
from unified_stpp.data.synthetic import (
    generate_hawkes_stpp,
    generate_inhomogeneous_stpp,
    generate_moving_hotspot_stpp,
    moving_hotspot_intensity,
    moving_hotspot_covariates,
    InhomogeneousPoissonSyntheticDataset,
    STHPDataset,
)
from unified_stpp.data.regime_gated_hawkes import (
    generate_regime_gated_hawkes_stpp,
    covariates_at as regime_gated_covariates_at,
    intensity_from_history as regime_gated_intensity_from_history,
)
from unified_stpp.training import Trainer
from unified_stpp.models import IntensityEvaluator


def _default_covariates(t: np.ndarray, s: np.ndarray, covariate_dim: int) -> np.ndarray:
    base = [
        np.sin(s[..., 0]) + np.cos(t),
        np.cos(s[..., 1] if s.shape[-1] > 1 else s[..., 0]) + np.sin(0.5 * t),
        np.sin(s[..., 0] + t),
        np.cos((s[..., 1] if s.shape[-1] > 1 else s[..., 0]) - t),
    ]
    if covariate_dim <= len(base):
        vals = base[:covariate_dim]
    else:
        vals = base + [np.sin((k + 1) * t) for k in range(covariate_dim - len(base))]
    return np.stack(vals, axis=-1).astype(np.float32)


def _moving_hotspot_covariates(
    t: np.ndarray,
    s: np.ndarray,
    t_end: float,
    switch_frac: float,
    covariate_dim: int,
    switch_time: float = None,
    move_duration: float = None,
    spatial_bounds: tuple = (-5.0, 5.0),
    sigma: float = 0.9,
    t1_frac: float = 0.32,
    t2_frac: float = 0.46,
    jitter_radius: float = 0.55,
    jitter_f1: float = 0.8,
    jitter_f2: float = 1.25,
    amp0: float = 0.0,
    amp1: float = 0.55,
    amp_noise: float = 0.10,
    seed: int = 42,
    n_noise_knots: int = 16,
) -> np.ndarray:
    t_arr = np.asarray(t).reshape(-1)
    s_arr = np.asarray(s).reshape(-1, s.shape[-1])
    covs = [
        moving_hotspot_covariates(
            t=float(ti),
            s=si,
            T=t_end,
            spatial_bounds=spatial_bounds,
            sigma=sigma,
            covariate_dim=max(1, covariate_dim),
            t1_frac=t1_frac,
            t2_frac=t2_frac,
            jitter_radius=jitter_radius,
            jitter_f1=jitter_f1,
            jitter_f2=jitter_f2,
            amp0=amp0,
            amp1=amp1,
            amp_noise=amp_noise,
            seed=seed,
            n_noise_knots=n_noise_knots,
        )
        for ti, si in zip(t_arr, s_arr)
    ]
    return np.asarray(covs, dtype=np.float32)


def _true_intensity(
    data_type,
    seq,
    t,
    s,
    *,
    t_end,
    spatial_bounds,
    base_rate,
    covariate_dim,
    sthp_alpha,
    sthp_beta,
    sthp_mu,
    hotspot_weight,
    hotspot_sigma,
    hotspot_switch_frac,
    hotspot_switch_time,
    hotspot_move_duration,
    hotspot_t1_frac,
    hotspot_t2_frac,
    hotspot_jitter_radius,
    hotspot_jitter_f1,
    hotspot_jitter_f2,
    hotspot_amp0,
    hotspot_amp1,
    hotspot_amp_noise,
    hotspot_noise_knots,
    data_seed,
    hotspot_interaction_weight,
    hotspot_tod_weight,
):
    # s shape: (..., 2), t shape: (...)
    if data_type in ("inhomogeneous", "inhomogeneous_class"):
        x = _default_covariates(t, s, covariate_dim=max(1, covariate_dim))
        return base_rate * np.exp(np.clip(x.sum(axis=-1), -5, 5))

    if data_type == "regime_gated_hawkes":
        return regime_gated_intensity_from_history(seq, t=float(t), s=s)

    if data_type == "moving_hotspot":
        return moving_hotspot_intensity(
            t=float(t),
            s=s,
            T=t_end,
            spatial_bounds=spatial_bounds,
            base_rate=base_rate,
            hotspot_weight=hotspot_weight,
            sigma=hotspot_sigma,
            t1_frac=hotspot_t1_frac,
            t2_frac=hotspot_t2_frac,
            jitter_radius=hotspot_jitter_radius,
            jitter_f1=hotspot_jitter_f1,
            jitter_f2=hotspot_jitter_f2,
            amp0=hotspot_amp0,
            amp1=hotspot_amp1,
            amp_noise=hotspot_amp_noise,
            seed=data_seed,
            n_noise_knots=hotspot_noise_knots,
        )

    if data_type == "hawkes":
        mu = 1.0
        alpha = 0.5
        beta = 1.0
        sigma_s = 1.0
        hist_mask = seq["times"] < t
        his_t = seq["times"][hist_mask]
        his_s = seq["locations"][hist_mask]
        lam = np.full(s.shape[:-1], mu, dtype=np.float64)
        if len(his_t) > 0:
            for ti, si in zip(his_t, his_s):
                temporal = alpha * beta * np.exp(-beta * (t - ti))
                spatial = np.exp(-np.sum((s - si) ** 2, axis=-1) / (2 * sigma_s ** 2))
                spatial /= (2 * np.pi * sigma_s ** 2)
                lam += temporal * spatial
        return lam

    if data_type == "sthp_class":
        s_mu = np.array([0.0, 0.0], dtype=np.float64)
        g0_cov = np.eye(2, dtype=np.float64)
        g2_cov = np.eye(2, dtype=np.float64)
        g0_ic = np.linalg.inv(g0_cov)
        g0_sidc = 1 / np.sqrt(np.linalg.det(g0_cov))
        g2_ic = np.linalg.inv(g2_cov)
        g2_sidc = 1 / np.sqrt(np.linalg.det(g2_cov))
        hist_mask = seq["times"] < t
        his_t = seq["times"][hist_mask]
        his_s = seq["locations"][hist_mask]
        delta0 = s - s_mu
        g0 = 1 / (2 * np.pi) * g0_sidc * np.exp(
            -np.einsum("...i,ij,...j->...", delta0, g0_ic, delta0) / 2
        )
        lam = sthp_mu * g0
        if len(his_t) > 0:
            g1 = sthp_alpha * np.exp(-sthp_beta * (t - his_t))
            for gi, si in zip(g1, his_s):
                delta = s - si
                g2 = 1 / (2 * np.pi) * g2_sidc * np.exp(
                    -np.einsum("...i,ij,...j->...", delta, g2_ic, delta) / 2
                )
                lam += gi * g2
        return lam

    raise ValueError(f"True intensity not implemented for data type: {data_type}")


def _compute_config_hash(payload: dict) -> str:
    serial = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(serial.encode("utf-8")).hexdigest()[:12]


def _save_epoch_metrics(
    *,
    history: dict,
    metrics_dir: str,
    run_name: str,
    covariates_enabled: bool,
    seed: int,
    config_hash: str,
    data_type: str,
):
    os.makedirs(metrics_dir, exist_ok=True)
    run_dir = os.path.join(metrics_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    suffix = "with_cov" if covariates_enabled else "no_cov"
    csv_path = os.path.join(run_dir, f"metrics_{suffix}.csv")
    jsonl_path = os.path.join(run_dir, f"metrics_{suffix}.jsonl")

    train_nll = history.get("train_nll", [])
    val_nll = history.get("val_nll", [])
    epoch_time = history.get("epoch_time_sec", [])

    rows = []
    for i, tr in enumerate(train_nll, start=1):
        rows.append(
            {
                "epoch": i,
                "train_nll": float(tr),
                "val_nll": float(val_nll[i - 1]) if i - 1 < len(val_nll) else "",
                "train_time_sec": float(epoch_time[i - 1]) if i - 1 < len(epoch_time) else "",
                "seed": int(seed),
                "config_hash": config_hash,
                "covariates_enabled": bool(covariates_enabled),
                "data_type": data_type,
            }
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_nll",
                "val_nll",
                "train_time_sec",
                "seed",
                "config_hash",
                "covariates_enabled",
                "data_type",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return csv_path, jsonl_path, run_dir


def _resolve_t_end(data_cfg: dict, fallback_t_end: float) -> float:
    """Resolve time horizon from modern `t_end` or legacy `T` config keys."""
    t_end_cfg = data_cfg.get("t_end", None)
    legacy_t = data_cfg.get("T", None)

    if t_end_cfg is not None:
        if legacy_t is not None and not np.isclose(float(t_end_cfg), float(legacy_t)):
            print(
                "Warning: both data.t_end and legacy data.T are set with different values; "
                f"using data.t_end={float(t_end_cfg):.6g} over data.T={float(legacy_t):.6g}."
            )
        return float(t_end_cfg)

    if legacy_t is not None:
        return float(legacy_t)

    return float(fallback_t_end)


def _resolve_optimizer_hparams(
    training_cfg: dict,
    *,
    lr_default: float,
    weight_decay_default: float,
    grad_clip_default: float,
):
    """Resolve optimizer/training scalars with CLI defaults as fallback."""
    lr = float(training_cfg.get("lr", lr_default))
    weight_decay = float(training_cfg.get("weight_decay", weight_decay_default))
    grad_clip = float(training_cfg.get("grad_clip", grad_clip_default))
    return lr, weight_decay, grad_clip


def _cli_arg_provided(flag: str) -> bool:
    """Return True if a CLI flag is explicitly present (e.g. --n_epochs or --n_epochs=2)."""
    argv = sys.argv[1:]
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def main():
    parser = argparse.ArgumentParser(description="Train unified STPP model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["neural_stpp", "deep_stpp", "dstpp"],
                        help="Use a preset configuration")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--spatial_dim", type=int, default=2)
    parser.add_argument("--event_cov_dim", type=int, default=0)
    parser.add_argument("--field_cov_dim", type=int, default=0)
    parser.add_argument("--data", type=str, default="hawkes",
                        choices=["hawkes", "inhomogeneous", "inhomogeneous_class", "sthp_class", "moving_hotspot", "regime_gated_hawkes"])
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--t_end", type=float, default=5.0)
    parser.add_argument("--base_rate", type=float, default=2.0)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--dataset_cache", type=str, default=None,
                        help="Optional path to cache/load generated sequences so multiple runs share identical data.")
    parser.add_argument("--data_covariate_dim", type=int, default=None,
                        help="Covariate dimension used by data generation (independent from model field_cov_dim).")
    parser.add_argument("--spatial_min", type=float, default=-5.0)
    parser.add_argument("--spatial_max", type=float, default=5.0)
    parser.add_argument("--sthp_alpha", type=float, default=0.5)
    parser.add_argument("--sthp_beta", type=float, default=1.0)
    parser.add_argument("--sthp_mu", type=float, default=1.0)
    parser.add_argument("--hotspot_sigma", type=float, default=0.9)
    parser.add_argument("--hotspot_switch_frac", type=float, default=0.5)
    parser.add_argument("--hotspot_switch_time", type=float, default=None)
    parser.add_argument("--hotspot_move_duration", type=float, default=None)
    parser.add_argument("--hotspot_t1_frac", type=float, default=0.32)
    parser.add_argument("--hotspot_t2_frac", type=float, default=0.46)
    parser.add_argument("--hotspot_jitter_radius", type=float, default=0.55)
    parser.add_argument("--hotspot_jitter_f1", type=float, default=0.8)
    parser.add_argument("--hotspot_jitter_f2", type=float, default=1.25)
    parser.add_argument("--hotspot_amp0", type=float, default=0.0)
    parser.add_argument("--hotspot_amp1", type=float, default=0.55)
    parser.add_argument("--hotspot_amp_noise", type=float, default=0.10)
    parser.add_argument("--hotspot_noise_knots", type=int, default=16)
    parser.add_argument("--hotspot_weight", type=float, default=3.0)
    parser.add_argument("--hotspot_interaction_weight", type=float, default=1.2)
    parser.add_argument("--hotspot_tod_weight", type=float, default=0.6)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--plot_intensity_3d", action="store_true",
                        help="After training, save 3D plot of estimated vs true intensity.")
    parser.add_argument("--plot_intensity_gif", action="store_true",
                        help="After training, save animated GIF of intensity over time.")
    parser.add_argument("--plot_grid_n", type=int, default=40)
    parser.add_argument("--plot_time", type=float, default=None,
                        help="Time slice (original time scale) for intensity plot.")
    parser.add_argument("--plot_seq_idx", type=int, default=0,
                        help="Validation sequence index used as conditioning history.")
    parser.add_argument("--plot_out", type=str, default="intensity_compare.png")
    parser.add_argument("--plot_n_times", type=int, default=50,
                        help="Number of representative time slices for GIF.")
    parser.add_argument("--plot_gif_fps", type=int, default=5,
                        help="Frames per second for GIF output.")
    parser.add_argument("--metrics_dir", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_metrics", dest="save_metrics", action="store_true")
    parser.add_argument("--no_save_metrics", dest="save_metrics", action="store_false")
    parser.set_defaults(save_metrics=True)
    parser.add_argument("--rg_num_marks", type=int, default=3)
    parser.add_argument("--rg_num_regimes", type=int, default=3)
    parser.add_argument("--rg_env_dim", type=int, default=4)
    parser.add_argument("--rg_sigma_reg", type=float, default=0.15)
    parser.add_argument("--rg_switch_rate", type=float, default=1.2)
    parser.add_argument("--rg_sigma_hotspot", type=float, default=1.0)
    parser.add_argument("--rg_lambda_bg", type=float, default=0.02)
    parser.add_argument("--rg_tau_t", type=float, default=0.8)
    parser.add_argument("--rg_sigma_exc", type=float, default=0.9)
    parser.add_argument("--rg_max_events_per_seq", type=int, default=250)
    args = parser.parse_args()

    # ========================================================================
    # Load config
    # ========================================================================
    training_cfg = {}
    data_cfg = {}
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        training_cfg = cfg.get("training", {})
        data_cfg = cfg.get("data", {})

        preset = cfg.get("model", {}).get("preset", args.preset)
        hidden_dim = cfg.get("model", {}).get("hidden_dim", args.hidden_dim)
        spatial_dim = cfg.get("model", {}).get("spatial_dim", args.spatial_dim)
        event_cov_dim = cfg.get("model", {}).get("event_cov_dim", args.event_cov_dim)
        field_cov_dim = cfg.get("model", {}).get("field_cov_dim", args.field_cov_dim)
        overrides = cfg.get("model", {}).get("overrides", {})
        n_epochs = args.n_epochs if _cli_arg_provided("--n_epochs") else training_cfg.get("n_epochs", args.n_epochs)
        batch_size = args.batch_size if _cli_arg_provided("--batch_size") else training_cfg.get("batch_size", args.batch_size)
        lr, weight_decay, grad_clip = _resolve_optimizer_hparams(
            training_cfg,
            lr_default=args.lr,
            weight_decay_default=args.weight_decay,
            grad_clip_default=args.grad_clip,
        )
        if _cli_arg_provided("--lr"):
            lr = float(args.lr)
        if _cli_arg_provided("--weight_decay"):
            weight_decay = float(args.weight_decay)
        if _cli_arg_provided("--grad_clip"):
            grad_clip = float(args.grad_clip)

        data_type = data_cfg.get("type", args.data)
        n_train = data_cfg.get("n_train", args.n_train)
        n_val = data_cfg.get("n_val", args.n_val)
        t_end = _resolve_t_end(data_cfg, args.t_end)
        if _cli_arg_provided("--t_end"):
            t_end = float(args.t_end)
        base_rate = data_cfg.get("base_rate", args.base_rate)
        data_seed = data_cfg.get("seed", args.data_seed)
        dataset_cache = data_cfg.get("dataset_cache", args.dataset_cache)
        data_covariate_dim = data_cfg.get("covariate_dim", args.data_covariate_dim)
        spatial_min = data_cfg.get("spatial_min", args.spatial_min)
        spatial_max = data_cfg.get("spatial_max", args.spatial_max)
        sthp_alpha = data_cfg.get("sthp_alpha", args.sthp_alpha)
        sthp_beta = data_cfg.get("sthp_beta", args.sthp_beta)
        sthp_mu = data_cfg.get("sthp_mu", args.sthp_mu)
        hotspot_sigma = data_cfg.get("hotspot_sigma", args.hotspot_sigma)
        hotspot_switch_frac = data_cfg.get("hotspot_switch_frac", args.hotspot_switch_frac)
        hotspot_switch_time = data_cfg.get("hotspot_switch_time", args.hotspot_switch_time)
        hotspot_move_duration = data_cfg.get("hotspot_move_duration", args.hotspot_move_duration)
        hotspot_t1_frac = data_cfg.get("hotspot_t1_frac", args.hotspot_t1_frac)
        hotspot_t2_frac = data_cfg.get("hotspot_t2_frac", args.hotspot_t2_frac)
        hotspot_jitter_radius = data_cfg.get("hotspot_jitter_radius", args.hotspot_jitter_radius)
        hotspot_jitter_f1 = data_cfg.get("hotspot_jitter_f1", args.hotspot_jitter_f1)
        hotspot_jitter_f2 = data_cfg.get("hotspot_jitter_f2", args.hotspot_jitter_f2)
        hotspot_amp0 = data_cfg.get("hotspot_amp0", args.hotspot_amp0)
        hotspot_amp1 = data_cfg.get("hotspot_amp1", args.hotspot_amp1)
        hotspot_amp_noise = data_cfg.get("hotspot_amp_noise", args.hotspot_amp_noise)
        hotspot_noise_knots = data_cfg.get("hotspot_noise_knots", args.hotspot_noise_knots)
        hotspot_weight = data_cfg.get("hotspot_weight", args.hotspot_weight)
        hotspot_interaction_weight = data_cfg.get(
            "hotspot_interaction_weight", args.hotspot_interaction_weight
        )
        hotspot_tod_weight = data_cfg.get("hotspot_tod_weight", args.hotspot_tod_weight)
        rg_num_marks = data_cfg.get("rg_num_marks", args.rg_num_marks)
        rg_num_regimes = data_cfg.get("rg_num_regimes", args.rg_num_regimes)
        rg_env_dim = data_cfg.get("rg_env_dim", args.rg_env_dim)
        rg_sigma_reg = data_cfg.get("rg_sigma_reg", args.rg_sigma_reg)
        rg_switch_rate = data_cfg.get("rg_switch_rate", args.rg_switch_rate)
        rg_sigma_hotspot = data_cfg.get("rg_sigma_hotspot", args.rg_sigma_hotspot)
        rg_lambda_bg = data_cfg.get("rg_lambda_bg", args.rg_lambda_bg)
        rg_tau_t = data_cfg.get("rg_tau_t", args.rg_tau_t)
        rg_sigma_exc = data_cfg.get("rg_sigma_exc", args.rg_sigma_exc)
        rg_max_events_per_seq = data_cfg.get("rg_max_events_per_seq", args.rg_max_events_per_seq)
    else:
        preset = args.preset or "deep_stpp"
        hidden_dim = args.hidden_dim
        spatial_dim = args.spatial_dim
        event_cov_dim = args.event_cov_dim
        field_cov_dim = args.field_cov_dim
        overrides = {}
        n_epochs = args.n_epochs
        batch_size = args.batch_size
        lr = args.lr
        weight_decay = args.weight_decay
        grad_clip = args.grad_clip
        data_type = args.data
        n_train = args.n_train
        n_val = args.n_val
        t_end = args.t_end
        base_rate = args.base_rate
        data_seed = args.data_seed
        dataset_cache = args.dataset_cache
        data_covariate_dim = args.data_covariate_dim
        spatial_min = args.spatial_min
        spatial_max = args.spatial_max
        sthp_alpha = args.sthp_alpha
        sthp_beta = args.sthp_beta
        sthp_mu = args.sthp_mu
        hotspot_sigma = args.hotspot_sigma
        hotspot_switch_frac = args.hotspot_switch_frac
        hotspot_switch_time = args.hotspot_switch_time
        hotspot_move_duration = args.hotspot_move_duration
        hotspot_t1_frac = args.hotspot_t1_frac
        hotspot_t2_frac = args.hotspot_t2_frac
        hotspot_jitter_radius = args.hotspot_jitter_radius
        hotspot_jitter_f1 = args.hotspot_jitter_f1
        hotspot_jitter_f2 = args.hotspot_jitter_f2
        hotspot_amp0 = args.hotspot_amp0
        hotspot_amp1 = args.hotspot_amp1
        hotspot_amp_noise = args.hotspot_amp_noise
        hotspot_noise_knots = args.hotspot_noise_knots
        hotspot_weight = args.hotspot_weight
        hotspot_interaction_weight = args.hotspot_interaction_weight
        hotspot_tod_weight = args.hotspot_tod_weight
        rg_num_marks = args.rg_num_marks
        rg_num_regimes = args.rg_num_regimes
        rg_env_dim = args.rg_env_dim
        rg_sigma_reg = args.rg_sigma_reg
        rg_switch_rate = args.rg_switch_rate
        rg_sigma_hotspot = args.rg_sigma_hotspot
        rg_lambda_bg = args.rg_lambda_bg
        rg_tau_t = args.rg_tau_t
        rg_sigma_exc = args.rg_sigma_exc
        rg_max_events_per_seq = args.rg_max_events_per_seq

    if args.config is not None:
        metrics_dir = training_cfg.get("metrics_dir", args.metrics_dir)
        run_name = training_cfg.get("run_name", args.run_name)
        save_metrics = training_cfg.get("save_metrics", args.save_metrics)
    else:
        metrics_dir = args.metrics_dir
        run_name = args.run_name
        save_metrics = args.save_metrics

    config_payload = cfg if args.config is not None else vars(args)
    config_hash = _compute_config_hash(config_payload)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    print(
        "Resolved settings: "
        f"t_end={float(t_end):.6g}, weight_decay={float(weight_decay):.6g}, grad_clip={float(grad_clip):.6g}"
    )

    # ========================================================================
    # Generate data
    # ========================================================================
    print(f"Generating {data_type} data: {n_train} train + {n_val} val sequences...")
    if data_type == "moving_hotspot":
        default_data_cov_dim = 4
    elif data_type == "regime_gated_hawkes":
        default_data_cov_dim = rg_num_regimes + 2 + rg_env_dim
    else:
        default_data_cov_dim = max(1, field_cov_dim)
    covariate_dim = int(
        default_data_cov_dim if data_covariate_dim is None else max(1, data_covariate_dim)
    )

    all_seqs = None
    if dataset_cache and os.path.exists(dataset_cache):
        with open(dataset_cache, "rb") as f:
            all_seqs = pickle.load(f)
        print(f"Loaded dataset from cache: {dataset_cache}")
        required = n_train + n_val
        if len(all_seqs) < required:
            raise ValueError(
                f"Cached dataset has {len(all_seqs)} sequences but {required} are required."
            )
    else:
        if data_type == "hawkes":
            all_seqs = generate_hawkes_stpp(
                n_sequences=n_train + n_val,
                T=t_end,
                spatial_bounds=(spatial_min, spatial_max),
                spatial_dim=spatial_dim,
                seed=data_seed,
            )
        elif data_type == "inhomogeneous":
            all_seqs = generate_inhomogeneous_stpp(
                n_sequences=n_train + n_val,
                T=t_end,
                spatial_dim=spatial_dim,
                base_rate=base_rate,
                covariate_dim=covariate_dim,
                seed=data_seed,
            )
        elif data_type == "inhomogeneous_class":
            generator = InhomogeneousPoissonSyntheticDataset(
                spatial_dim=spatial_dim,
                base_rate=base_rate,
                covariate_dim=covariate_dim,
                seed=data_seed,
            )
            all_seqs = generator.generate_sequences(
                n_sequences=n_train + n_val,
                t_start=0.0,
                t_end=t_end,
            )
        elif data_type == "sthp_class":
            if spatial_dim != 2:
                raise ValueError("sthp_class currently supports only spatial_dim=2")
            generator = STHPDataset(
                s_mu=[0.0, 0.0],
                g0_cov=[[1.0, 0.0], [0.0, 1.0]],
                g2_cov=[[1.0, 0.0], [0.0, 1.0]],
                alpha=sthp_alpha,
                beta=sthp_beta,
                mu=sthp_mu,
                seed=data_seed,
            )
            all_seqs = generator.generate_sequences(
                n_sequences=n_train + n_val,
                t_start=0.0,
                t_end=t_end,
            )
        elif data_type == "moving_hotspot":
            all_seqs = generate_moving_hotspot_stpp(
                n_sequences=n_train + n_val,
                T=t_end,
                spatial_bounds=(spatial_min, spatial_max),
                spatial_dim=spatial_dim,
                base_rate=base_rate,
                sigma=hotspot_sigma,
                switch_frac=hotspot_switch_frac,
                switch_time=hotspot_switch_time,
                move_duration=hotspot_move_duration,
                t1_frac=hotspot_t1_frac,
                t2_frac=hotspot_t2_frac,
                jitter_radius=hotspot_jitter_radius,
                jitter_f1=hotspot_jitter_f1,
                jitter_f2=hotspot_jitter_f2,
                amp0=hotspot_amp0,
                amp1=hotspot_amp1,
                amp_noise=hotspot_amp_noise,
                n_noise_knots=hotspot_noise_knots,
                hotspot_weight=hotspot_weight,
                interaction_weight=hotspot_interaction_weight,
                tod_weight=hotspot_tod_weight,
                covariate_dim=covariate_dim,
                seed=data_seed,
            )
        elif data_type == "regime_gated_hawkes":
            print("Generating regime-gated Hawkes data with parameters:")
            all_seqs = generate_regime_gated_hawkes_stpp(
                n_sequences=n_train + n_val,
                T=t_end,
                spatial_bounds=(spatial_min, spatial_max),
                n_marks=rg_num_marks,
                n_regimes=rg_num_regimes,
                env_dim=rg_env_dim,
                sigma_reg=rg_sigma_reg,
                switch_rate=rg_switch_rate,
                sigma_hotspot=rg_sigma_hotspot,
                lambda_bg=rg_lambda_bg,
                tau_t=rg_tau_t,
                sigma_exc=rg_sigma_exc,
                max_events_per_seq=rg_max_events_per_seq,
                seed=data_seed,
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if dataset_cache:
            os.makedirs(os.path.dirname(dataset_cache) or ".", exist_ok=True)
            with open(dataset_cache, "wb") as f:
                pickle.dump(all_seqs, f)
            print(f"Saved dataset cache to: {dataset_cache}")

    # If the model is configured without field covariates, drop them from data.
    # This enables fair "with vs without covariates" comparisons on the same generator.
    if field_cov_dim == 0:
        for seq in all_seqs:
            seq.pop("field_covariates", None)

    train_seqs = all_seqs[:n_train]
    val_seqs = all_seqs[n_train:]

    train_dataset = STPPDataset(train_seqs)
    # Val dataset reuses train normalization stats so the model sees the same
    # feature scale at evaluation time as during training.
    val_dataset = STPPDataset(
        val_seqs,
        cov_mean=train_dataset.cov_mean,
        cov_std=train_dataset.cov_std,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # ========================================================================
    # Build model
    # ========================================================================
    print(f"Building model: preset={preset}, hidden={hidden_dim}, "
          f"spatial={spatial_dim}, event_cov={event_cov_dim}, field_cov={field_cov_dim}")

    model = build_model(
        config=overrides,
        spatial_dim=spatial_dim,
        hidden_dim=hidden_dim,
        event_cov_dim=event_cov_dim,
        field_cov_dim=field_cov_dim,
        preset=preset,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Components:")
    print(f"  Encoder:  {model.encoder.__class__.__name__}")
    print(f"  Dynamics: {model.dynamics.__class__.__name__}")
    print(f"  Updater:  {model.updater.__class__.__name__}")
    print(f"  Decoder:  {model.decoder.__class__.__name__}")

    # ========================================================================
    # Train
    # ========================================================================
    trainer = Trainer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        device=device,
    )
    print(f"\nTraining for {n_epochs} epochs...")
    history = trainer.train(
        train_loader, val_loader, n_epochs=n_epochs, log_every=5
    )

    covariates_enabled = field_cov_dim > 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"{data_type}_{preset}_{timestamp}_{config_hash[:8]}"
    csv_metrics_path, jsonl_metrics_path, _ = _save_epoch_metrics(
        history=history,
        metrics_dir=metrics_dir,
        run_name=run_id,
        covariates_enabled=covariates_enabled,
        seed=data_seed,
        config_hash=config_hash,
        data_type=data_type,
    )

    if save_metrics:
        print(f"Saved per-epoch metrics CSV:   {csv_metrics_path}")
        print(f"Saved per-epoch metrics JSONL: {jsonl_metrics_path}")
    else:
        print(
            "Saved per-epoch metrics (always on for reproducibility): "
            f"{csv_metrics_path}"
        )

    print("\nDone!")
    print(f"Final train NLL: {history['train_nll'][-1]:.4f}")
    if history['val_nll']:
        print(f"Final val NLL:   {history['val_nll'][-1]:.4f}")

    if args.plot_intensity_3d or args.plot_intensity_gif:
        if spatial_dim != 2:
            print("Skipping intensity plot: only spatial_dim=2 is supported.")
            return
        if len(val_dataset) == 0:
            print("Skipping intensity plot: validation dataset is empty.")
            return

        seq_idx = min(max(args.plot_seq_idx, 0), len(val_dataset) - 1)
        seq_item = val_dataset[seq_idx]
        raw_seq = val_dataset.sequences[seq_idx]

        history_times = seq_item["times"].unsqueeze(0).to(device)
        history_locs = seq_item["locations"].unsqueeze(0).to(device)
        history_lengths = torch.tensor([seq_item["length"]], device=device)

        model.eval()

        def build_evaluator_with_history(n_hist: int) -> IntensityEvaluator:
            n_hist = max(1, min(int(n_hist), int(seq_item["length"])))
            h_times = seq_item["times"][:n_hist].unsqueeze(0).to(device)
            h_locs = seq_item["locations"][:n_hist].unsqueeze(0).to(device)
            h_lengths = torch.tensor([n_hist], device=device)
            with torch.no_grad():
                events = torch.cat([h_times.unsqueeze(-1), h_locs], dim=-1)
                z_hist, _ = model.encoder(events, h_lengths, x_event=None)
                t_prev_hist = h_times[
                    torch.arange(1, device=device), (h_lengths - 1).long()
                ].unsqueeze(-1)
            return IntensityEvaluator(model, z=z_hist, t_prev=t_prev_hist)

        # Default evaluator uses full available history.
        evaluator = build_evaluator_with_history(seq_item["length"])

        loc_mean = train_dataset.loc_mean
        loc_std = train_dataset.loc_std
        time_mean = train_dataset.time_mean
        time_std = train_dataset.time_std
        # Covariate normalization stats from the training split — used to
        # standardise the covariate function output at inference time so that
        # the model sees the same feature scale it was trained on.
        cov_mean = train_dataset.cov_mean   # np.ndarray or None
        cov_std  = train_dataset.cov_std    # np.ndarray or None

        all_locs = np.concatenate([s["locations"] for s in train_dataset.sequences + val_dataset.sequences], axis=0)
        s_min = all_locs.min(axis=0)
        s_max = all_locs.max(axis=0)

        last_t_orig = float(raw_seq["times"][-1]) if len(raw_seq["times"]) > 0 else 0.0
        t_orig = args.plot_time if args.plot_time is not None else min(
            t_end, last_t_orig + 0.5 * max(t_end - last_t_orig, 1e-3)
        )

        n_grid = max(10, args.plot_grid_n)
        x = np.linspace(s_min[0], s_max[0], n_grid)
        y = np.linspace(s_min[1], s_max[1], n_grid)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        s_orig = np.stack([xx, yy], axis=-1)

        # Model grid in normalized coordinates
        s_min_norm = torch.tensor((s_min - loc_mean) / loc_std, dtype=torch.float32, device=device)
        s_max_norm = torch.tensor((s_max - loc_mean) / loc_std, dtype=torch.float32, device=device)
        def x_field_fn_norm(t_norm_tensor, s_norm_tensor):
            # Denormalise model inputs back to original scale before computing covariates.
            t_np = (t_norm_tensor.squeeze(-1).detach().cpu().numpy() * time_std + time_mean)
            s_np = (s_norm_tensor.detach().cpu().numpy() * loc_std + loc_mean)
            if data_type == "moving_hotspot":
                x_np = _moving_hotspot_covariates(
                    t_np,
                    s_np,
                    t_end=t_end,
                    switch_frac=hotspot_switch_frac,
                    covariate_dim=max(1, field_cov_dim),
                    switch_time=hotspot_switch_time,
                    move_duration=hotspot_move_duration,
                    spatial_bounds=(spatial_min, spatial_max),
                    sigma=hotspot_sigma,
                    t1_frac=hotspot_t1_frac,
                    t2_frac=hotspot_t2_frac,
                    jitter_radius=hotspot_jitter_radius,
                    jitter_f1=hotspot_jitter_f1,
                    jitter_f2=hotspot_jitter_f2,
                    amp0=hotspot_amp0,
                    amp1=hotspot_amp1,
                    amp_noise=hotspot_amp_noise,
                    seed=data_seed,
                    n_noise_knots=hotspot_noise_knots,
                )
            elif data_type == "regime_gated_hawkes":
                x_np = regime_gated_covariates_at(
                    t=t_np,
                    x=s_np,
                    T=t_end,
                    n_regimes=rg_num_regimes,
                    regime_change_times=np.asarray(raw_seq["regime_change_times"], dtype=np.float64),
                    regime_states=np.asarray(raw_seq["regime_states"], dtype=np.int64),
                    env_dim=rg_env_dim,
                    spatial_bounds=(spatial_min, spatial_max),
                    sigma_reg=0.0,
                    apply_tanh=True,
                    rng=None,
                )
            else:
                x_np = _default_covariates(t_np, s_np, covariate_dim=max(1, field_cov_dim))
            # Apply the same z-score normalisation used on the training data so
            # that the model receives covariates in the exact same scale as
            # during training.
            if cov_mean is not None and cov_std is not None:
                x_np = (x_np - cov_mean) / cov_std
            return torch.tensor(x_np, dtype=torch.float32, device=t_norm_tensor.device)

        x_field_fn = x_field_fn_norm if field_cov_dim > 0 else None

        def compute_intensities_for_time(t_query: float, evaluator_local: IntensityEvaluator = None):
            evaluator_use = evaluator if evaluator_local is None else evaluator_local
            t_norm_local = float((t_query - time_mean) / time_std)
            _, _, lam_model_norm_local = evaluator_use.intensity_grid(
                t=t_norm_local,
                s_min=s_min_norm,
                s_max=s_max_norm,
                n_grid=n_grid,
                x_field_fn=x_field_fn,
            )
            lam_model_local = (
                lam_model_norm_local.detach().cpu().numpy() / (time_std * np.prod(loc_std))
            )
            lam_true_local = _true_intensity(
                data_type,
                raw_seq,
                t_query,
                s_orig,
                t_end=t_end,
                spatial_bounds=(spatial_min, spatial_max),
                base_rate=base_rate,
                covariate_dim=field_cov_dim,
                sthp_alpha=sthp_alpha,
                sthp_beta=sthp_beta,
                sthp_mu=sthp_mu,
                hotspot_weight=hotspot_weight,
                hotspot_sigma=hotspot_sigma,
                hotspot_switch_frac=hotspot_switch_frac,
                hotspot_switch_time=hotspot_switch_time,
                hotspot_move_duration=hotspot_move_duration,
                hotspot_t1_frac=hotspot_t1_frac,
                hotspot_t2_frac=hotspot_t2_frac,
                hotspot_jitter_radius=hotspot_jitter_radius,
                hotspot_jitter_f1=hotspot_jitter_f1,
                hotspot_jitter_f2=hotspot_jitter_f2,
                hotspot_amp0=hotspot_amp0,
                hotspot_amp1=hotspot_amp1,
                hotspot_amp_noise=hotspot_amp_noise,
                hotspot_noise_knots=hotspot_noise_knots,
                data_seed=data_seed,
                hotspot_interaction_weight=hotspot_interaction_weight,
                hotspot_tod_weight=hotspot_tod_weight,
            )
            return lam_true_local, lam_model_local

        # Single 3D plot
        if args.plot_intensity_3d:
            lam_true, lam_model = compute_intensities_for_time(t_orig)
            fig = plt.figure(figsize=(14, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")

            ax1.plot_surface(xx, yy, lam_true, cmap="viridis", linewidth=0, antialiased=True)
            ax1.set_title(f"True Intensity at t={t_orig:.3f}")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("lambda")

            ax2.plot_surface(xx, yy, lam_model, cmap="plasma", linewidth=0, antialiased=True)
            ax2.set_title(f"Estimated Intensity at t={t_orig:.3f}")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("lambda")

            plt.tight_layout()
            plt.savefig(args.plot_out, dpi=150)
            plt.close(fig)
            print(f"Saved 3D intensity comparison to {args.plot_out}")

        # Animated GIF over representative times
        if args.plot_intensity_gif:
            try:
                from PIL import Image
            except ImportError:
                print("Skipping GIF: Pillow is not installed.")
                return

            raw_times = raw_seq["times"]
            if len(raw_times) == 0:
                print("Skipping GIF: selected sequence has no events.")
                return

            # Animate from the first observed event to t_end, and use
            # frame-specific history up to each frame time.
            start_t = min(max(float(raw_times[0]) + 1e-4, 1e-4), float(t_end))
            end_t = max(start_t + 1e-3, float(t_end))
            n_times = max(2, args.plot_n_times)
            times = np.linspace(start_t, end_t, n_times)

            frames = []
            frame_data = []
            for ti in times:
                n_hist = int(np.searchsorted(raw_times, float(ti), side="right"))
                evaluator_t = build_evaluator_with_history(n_hist)
                lam_true_i, lam_model_i = compute_intensities_for_time(float(ti), evaluator_local=evaluator_t)
                frame_data.append((float(ti), lam_true_i, lam_model_i))

            global_zmax = 1e-8
            for _, lam_true_i, lam_model_i in frame_data:
                global_zmax = max(global_zmax, float(np.max(lam_true_i)), float(np.max(lam_model_i)))

            for ti, lam_true_i, lam_model_i in frame_data:
                fig = plt.figure(figsize=(14, 6))
                ax1 = fig.add_subplot(1, 2, 1, projection="3d")
                ax2 = fig.add_subplot(1, 2, 2, projection="3d")

                ax1.plot_surface(
                    xx, yy, lam_true_i, cmap="viridis", linewidth=0, antialiased=True, vmin=0.0, vmax=global_zmax
                )
                ax1.set_title(f"True Intensity at t={ti:.3f}")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.set_zlabel("lambda")
                ax1.set_zlim(0.0, global_zmax)

                ax2.plot_surface(
                    xx, yy, lam_model_i, cmap="plasma", linewidth=0, antialiased=True, vmin=0.0, vmax=global_zmax
                )
                ax2.set_title(f"Estimated Intensity at t={ti:.3f}")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                ax2.set_zlabel("lambda")
                ax2.set_zlim(0.0, global_zmax)

                plt.tight_layout()
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                frames.append(Image.fromarray(buf[..., :3]))
                plt.close(fig)

            out_path = args.plot_out
            if not out_path.lower().endswith(".gif"):
                out_path = os.path.splitext(out_path)[0] + ".gif"

            duration_ms = int(1000 / max(1, args.plot_gif_fps))
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                disposal=2,
            )
            print(f"Saved intensity animation to {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
