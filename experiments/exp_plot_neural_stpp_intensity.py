#!/usr/bin/env python3
"""
Plot NeuralSTPP intensity vs. true STHP on a spatial mesh.

Usage
-----
    python experiments/exp_plot_neural_stpp_intensity.py \
        --ckpt checkpoints/hawkes_neural_stpp_20260225_104351_bd42c1cf/epoch004-val_nll2.9157.ckpt \
        --out out_neural_stpp_intensity.png

The script auto-detects the checkpoint architecture (GRU/attention, MLP/ConcatSquash)
and builds the matching model, so old and new checkpoints both work.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLBACKEND", "Agg")

from unified_stpp.data.synthetic import STHPDataset
from unified_stpp.models import IntensityEvaluator
from unified_stpp.registry import build_model

# ---------------------------------------------------------------------------
# STHP parameters (DS1 from AutoSTPP paper appendix Table 3)
# ---------------------------------------------------------------------------
ALPHA = 0.5
BETA  = 1.0
MU    = 0.2
G0_COV = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
G2_COV = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)


# ---------------------------------------------------------------------------
# True STHP intensity helpers
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(points: np.ndarray, centers: np.ndarray,
                         inv_cov: np.ndarray, sidc: float) -> np.ndarray:
    p = np.asarray(points).reshape(-1, 2)
    c = np.asarray(centers).reshape(-1, 2)
    delta = p[:, None, :] - c[None, :, :]
    quad = np.einsum("pni,ij,pnj->pn", delta, inv_cov, delta)
    return (1.0 / (2.0 * math.pi)) * sidc * np.exp(-0.5 * quad)


def true_sthp_intensity(times: np.ndarray, locs: np.ndarray,
                         t_query: float, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    g0_ic = np.linalg.inv(G0_COV)
    g0_sd = 1.0 / np.sqrt(np.linalg.det(G0_COV))
    g2_ic = np.linalg.inv(G2_COV)
    g2_sd = 1.0 / np.sqrt(np.linalg.det(G2_COV))
    bg = MU * _gaussian_kernel_2d(pts, np.zeros((1, 2)), g0_ic, g0_sd)[:, 0]
    mask = times < t_query
    if mask.any():
        dt = t_query - times[mask]
        excite = (ALPHA * np.exp(-BETA * dt)[:, None]
                  * _gaussian_kernel_2d(pts, locs[mask], g2_ic, g2_sd).T).sum(0)
    else:
        excite = np.zeros(pts.shape[0])
    return (bg + excite).reshape(xx.shape)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _detect_arch(ckpt_path: Path) -> Dict:
    """Infer model overrides from checkpoint key names + shapes."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})

    encoder_type = "attention" if "model.encoder.attn_layers" in " ".join(sd) else "gru"
    updater_type  = "attention" if "model.updater.cross_attn" in " ".join(sd) else "gru_jump"

    # Detect spatial velocity field
    has_concat = any("lin_z" in k for k in sd)
    layer_type = "concat" if has_concat else "mlp"

    # Count hidden layers for MLP
    if layer_type == "mlp":
        n_hidden = sum(1 for k in sd if "velocity.net." in k and k.endswith(".weight")
                       and "net.0" not in k.replace("net.0.", ""))
        # net.0, net.2, net.4 → 2 hidden layers
        vel_weights = [k for k in sd if "velocity.net." in k and k.endswith(".weight")]
        n_hidden_layers = max(0, len(vel_weights) - 1)
    else:
        n_hidden_layers = 3

    hidden_dim = sd["model.encoder.embed.weight"].shape[0]

    # Detect base_type
    base_type = "self_attentive" if any("query_proj" in k for k in sd) else "standard"

    # history_k from query_proj if self_attentive
    history_k = 20

    return {
        "hidden_dim": hidden_dim,
        "overrides": {
            "encoder": {"type": encoder_type},
            "updater": {"type": updater_type},
            "decoder": {
                "spatial": {
                    "layer_type": layer_type,
                    "n_hidden_layers": n_hidden_layers,
                    "base_type": base_type,
                    "history_k": history_k,
                },
            },
        },
    }


def _load_ckpt(ckpt_path: Path, model: nn.Module) -> None:
    """Load Lightning checkpoint with compatibility shims for renamed keys."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", {})
    stripped: Dict = {}
    for k, v in sd.items():
        if not k.startswith("model."):
            continue
        key = k[6:]
        # Old checkpoints used velocity.net, new code uses velocity.mlp for MLP mode
        key = key.replace("decoder.spatial.velocity.net.", "decoder.spatial.velocity.mlp.")
        stripped[key] = v

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    # intensity_module is just an alias for decoder.temporal — already loaded via that path
    real_missing = [k for k in missing
                    if not k.startswith("dynamics.aug_func.intensity_module.")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch:\n  missing={real_missing}\n  unexpected={unexpected}"
        )


# ---------------------------------------------------------------------------
# Intensity on mesh
# ---------------------------------------------------------------------------

def model_intensity_on_mesh(
    model: nn.Module,
    times: np.ndarray,
    locs: np.ndarray,
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
    t_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(locs,  dtype=np.float64).reshape(-1, 2)
    n_hist = max(1, int(np.searchsorted(t_arr, t_query, side="right")))
    n_hist = min(n_hist, t_arr.shape[0])

    sstd = np.where(np.abs(loc_std) > 1e-12, loc_std, 1.0)
    tstd = float(time_std) if abs(float(time_std)) > 1e-12 else 1.0

    h_times = ((t_arr[:n_hist] - float(time_mean)) / tstd).astype(np.float32)
    h_locs  = ((s_arr[:n_hist] - loc_mean) / sstd).astype(np.float32)

    dev = torch.device(device)
    h_times_t = torch.tensor(h_times, device=dev).unsqueeze(0)   # (1, N, 1)
    h_locs_t  = torch.tensor(h_locs,  device=dev).unsqueeze(0)   # (1, N, 2)
    h_len_t   = torch.tensor([n_hist], device=dev)

    s_min_norm = torch.tensor(
        (np.array([float(xx.min()), float(yy.min())]) - loc_mean) / sstd,
        dtype=torch.float32, device=dev,
    )
    s_max_norm = torch.tensor(
        (np.array([float(xx.max()), float(yy.max())]) - loc_mean) / sstd,
        dtype=torch.float32, device=dev,
    )
    t_norm = float((t_query - float(time_mean)) / tstd)
    n_grid = int(xx.shape[0])
    jac = max(float(tstd * np.prod(sstd)), 1e-12)

    with torch.no_grad():
        events = torch.cat([h_times_t.unsqueeze(-1), h_locs_t], dim=-1)
        z_hist, _ = model.encoder(events, h_len_t, x_event=None)
        t_prev = h_times_t[0, (h_len_t[0] - 1).long()].reshape(1, 1)
        evaluator = IntensityEvaluator(
            model, z=z_hist, t_prev=t_prev, history_locs_norm=h_locs_t[0]
        )
        _, _, lam_norm = evaluator.intensity_grid(
            t=t_norm, s_min=s_min_norm, s_max=s_max_norm, n_grid=n_grid,
        )
    lam = lam_norm.detach().cpu().numpy() / jac
    return np.clip(lam, 0.0, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint path")
    parser.add_argument("--out",  default="neural_stpp_intensity.png")
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_test",  type=int, default=20)
    parser.add_argument("--n_grid",  type=int, default=45)
    parser.add_argument("--n_snaps", type=int, default=4)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    print(f"Checkpoint : {ckpt_path}")

    # -- 1. Data ---------------------------------------------------------
    print("Generating synthetic STHP data...")

    def _gen_seq(seed: int, t_end: float = 200.0) -> Dict:
        gen = STHPDataset(
            s_mu=np.zeros(2, dtype=np.float64),
            g0_cov=G0_COV, g2_cov=G2_COV,
            alpha=ALPHA, beta=BETA, mu=MU,
            seed=seed,
        )
        np.random.seed(seed)
        gen.generate(t_start=0.0, t_end=t_end, verbose=False)
        return {
            "times":     np.asarray(gen.his_t, dtype=np.float32),
            "locations": np.asarray(gen.his_s, dtype=np.float32).reshape(-1, 2),
        }

    train_seqs = [_gen_seq(args.seed + i) for i in range(args.n_train)]
    test_seqs  = [_gen_seq(args.seed + args.n_train + i) for i in range(args.n_test)]

    # Normalization stats from training set
    all_t = np.concatenate([s["times"].reshape(-1) for s in train_seqs])
    all_s = np.concatenate([s["locations"].reshape(-1, 2) for s in train_seqs])
    time_mean, time_std = float(all_t.mean()), float(all_t.std())
    loc_mean = all_s.mean(axis=0)
    loc_std  = all_s.std(axis=0)
    time_std = max(time_std, 1e-8)
    loc_std  = np.where(np.abs(loc_std) > 1e-8, loc_std, 1.0)

    # Pick longest test sequence
    seq = max(test_seqs, key=lambda s: len(s["times"]))
    seq_t = np.asarray(seq["times"],     dtype=np.float64).reshape(-1)
    seq_s = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)
    print(f"Test sequence: {len(seq_t)} events, t∈[{seq_t.min():.2f}, {seq_t.max():.2f}]")

    # -- 2. Spatial grid -------------------------------------------------
    all_locs = np.concatenate([np.asarray(s["locations"]).reshape(-1, 2)
                                for s in train_seqs + test_seqs])
    xq = np.percentile(all_locs[:, 0], [1, 99])
    yq = np.percentile(all_locs[:, 1], [1, 99])
    xs = max(float(xq[1] - xq[0]), 1e-3)
    ys = max(float(yq[1] - yq[0]), 1e-3)
    x_bounds = (float(xq[0] - 0.1 * xs), float(xq[1] + 0.1 * xs))
    y_bounds = (float(yq[0] - 0.1 * ys), float(yq[1] + 0.1 * ys))
    x = np.linspace(x_bounds[0], x_bounds[1], args.n_grid)
    y = np.linspace(y_bounds[0], y_bounds[1], args.n_grid)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    t_snaps = np.linspace(float(seq_t.min()), float(seq_t.max()),
                          args.n_snaps, endpoint=False)

    # -- 3. Load model ---------------------------------------------------
    arch = _detect_arch(ckpt_path)
    print(f"Detected arch: hidden_dim={arch['hidden_dim']}, "
          f"encoder={arch['overrides']['encoder']['type']}, "
          f"layer_type={arch['overrides']['decoder']['spatial']['layer_type']}, "
          f"base_type={arch['overrides']['decoder']['spatial']['base_type']}")

    # Use neural_stpp_jump as base preset (GRU encoder + GRU_jump updater)
    # Override any keys detected from the checkpoint.
    preset = "neural_stpp_jump" \
        if arch["overrides"]["encoder"]["type"] == "gru" else "neural_stpp"
    model = build_model(
        arch["overrides"],
        preset=preset,
        hidden_dim=arch["hidden_dim"],
        spatial_dim=2,
    )
    _load_ckpt(ckpt_path, model)
    model.to(args.device).eval()
    print("Model loaded OK")

    # -- 4. Evaluate intensity -------------------------------------------
    print("Evaluating intensity grids...")
    lam_true: List[np.ndarray] = []
    lam_model: List[np.ndarray] = []

    for ti in t_snaps:
        print(f"  t={ti:.3f}", end="", flush=True)
        gt = true_sthp_intensity(seq_t, seq_s, float(ti), xx, yy)
        lm = model_intensity_on_mesh(
            model, seq_t, seq_s, float(ti), xx, yy,
            loc_mean=loc_mean, loc_std=loc_std,
            time_mean=time_mean, time_std=time_std,
            device=args.device,
        )
        lam_true.append(gt)
        lam_model.append(lm)
        print(f"  gt_peak={float(gt.max()):.3f}  model_peak={float(lm.max()):.3f}")

    # -- 5. Plot ---------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError as e:
        print(f"matplotlib unavailable: {e}")
        return

    zmax = max(1e-8, max(float(z.max()) for z in lam_true))
    n_times = len(t_snaps)

    fig = plt.figure(figsize=(4 * n_times, 7))
    gs = fig.add_gridspec(
        2, n_times,
        left=0.04, right=0.995, top=0.90, bottom=0.07,
        wspace=0.10, hspace=0.14,
    )
    rows = [("True STHP", lam_true), ("NeuralSTPP", lam_model)]
    for r_idx, (row_label, row_vals) in enumerate(rows):
        for c_idx, ti in enumerate(t_snaps):
            ax = fig.add_subplot(gs[r_idx, c_idx], projection="3d")
            ax.plot_surface(
                xx, yy, row_vals[c_idx],
                cmap=cm.viridis,
                vmin=0.0, vmax=zmax,
                linewidth=0, antialiased=False,
            )
            ax.set_zlim(0, zmax * 1.05)
            ax.set_xlabel("x", fontsize=7, labelpad=1)
            ax.set_ylabel("y", fontsize=7, labelpad=1)
            ax.tick_params(labelsize=6)
            ax.set_title(f"t={ti:.2f}", fontsize=8, pad=2)
            if c_idx == 0:
                ax.text2D(
                    -0.08, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=9, fontweight="bold",
                    va="center", ha="right", rotation=90,
                )

    fig.suptitle(
        f"NeuralSTPP vs True STHP  |  ckpt: {ckpt_path.name}",
        fontsize=10, y=0.96,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
