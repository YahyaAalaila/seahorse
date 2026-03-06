#!/usr/bin/env python3
"""
Synthetic experiment: standard LL/NLL vs. smoothed NLL (vol-corrected).

Purpose
-------
Demonstrate, in interpretable EMS-like units, that exact pointwise likelihood can
penalize "near-miss" spatial-temporal bumps even when predictions are close, while
smoothed NLL better reflects operational tolerance.

Design
------
- True process: inhomogeneous Poisson process (IPP) on [0,T] x [0,S]^2.
- True intensity:   lambda_true(t, x, y) = exp(beta0 + beta1 * bump_true(t, x, y))
- Fitted model:     lambda_fit (t, x, y) = exp(theta0 + theta1 * bump_model(t, x, y))
  where bump_model is deliberately shifted by (dx_km, dy_km) from bump_true.
- Train/test split by time: train on [0, train_T], test on (train_T, T].
- Fit theta via exact train LL with deterministic quadrature compensator.
- Evaluate standard test NLL and smoothed test NLL (volume-corrected) on a
  small (r, tau) tolerance grid.

Dependencies
------------
Only numpy + scipy + matplotlib. No repo module imports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.optimize import minimize

# Keep matplotlib in non-interactive mode and writable cache by default.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


EPS = 1e-12
DEFAULT_R_GRID_KM = np.array([0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], dtype=np.float64)
DEFAULT_TAU_GRID_H = np.array([5 / 60, 10 / 60, 20 / 60], dtype=np.float64)


@dataclass
class Domain:
    T_hours: float
    space_km: float
    train_T: float


@dataclass
class TrueParams:
    beta0: float
    beta1: float
    sigma_km: float
    T_hours: float
    space_km: float


@dataclass
class ModelParams:
    theta0: float
    theta1: float
    sigma_km: float
    dx_km: float
    dy_km: float
    T_hours: float
    space_km: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic IPP experiment for standard NLL vs smoothed NLL."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--dx_km", type=float, default=0.5)
    parser.add_argument("--dy_km", type=float, default=0.5)
    parser.add_argument("--sigma_km", type=float, default=0.8)
    parser.add_argument("--T_hours", type=float, default=10.0)
    parser.add_argument("--space_km", type=float, default=10.0)
    parser.add_argument("--target_events", type=int, default=1000)
    parser.add_argument("--train_T", type=float, default=7.0)
    parser.add_argument("--grid_n", type=int, default=60)
    parser.add_argument("--Kt", type=int, default=11)
    parser.add_argument("--Ks", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/exp_synth_ll_vs_smoothed",
    )
    return parser.parse_args()


def make_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"synth_ll_vs_smoothed_{ts}_dx{args.dx_km:.2f}_dy{args.dy_km:.2f}"


def center_trajectory(t: np.ndarray, T_hours: float, space_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth moving center trajectory in domain interior.
    """
    t = np.asarray(t, dtype=np.float64)
    w = 2.0 * math.pi / max(T_hours, EPS)
    amp = 0.25 * space_km
    cx = 0.5 * space_km + amp * np.sin(w * t)
    cy = 0.5 * space_km + amp * np.cos(w * t + 0.5)
    return cx, cy


def bump_gaussian(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sigma_km: float,
    T_hours: float,
    space_km: float,
    dx_km: float = 0.0,
    dy_km: float = 0.0,
) -> np.ndarray:
    """
    Spatiotemporal Gaussian bump centered on a moving trajectory, with optional
    static offset (dx, dy) for model misalignment.
    """
    cx, cy = center_trajectory(t=t, T_hours=T_hours, space_km=space_km)
    cx = cx + dx_km
    cy = cy + dy_km
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-0.5 * d2 / max(sigma_km ** 2, EPS))


def lambda_true(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: TrueParams,
) -> np.ndarray:
    b = bump_gaussian(
        t=t,
        x=x,
        y=y,
        sigma_km=p.sigma_km,
        T_hours=p.T_hours,
        space_km=p.space_km,
        dx_km=0.0,
        dy_km=0.0,
    )
    return np.exp(p.beta0 + p.beta1 * b)


def lambda_model(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    theta0: float,
    theta1: float,
    sigma_km: float,
    T_hours: float,
    space_km: float,
    dx_km: float,
    dy_km: float,
) -> np.ndarray:
    b = bump_gaussian(
        t=t,
        x=x,
        y=y,
        sigma_km=sigma_km,
        T_hours=T_hours,
        space_km=space_km,
        dx_km=dx_km,
        dy_km=dy_km,
    )
    return np.exp(theta0 + theta1 * b)


def make_quadrature_grid(
    t0: float,
    t1: float,
    space_km: float,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Midpoint Riemann grid over [t0,t1] x [0,S] x [0,S].
    """
    dt = (t1 - t0) / n
    ds = space_km / n
    t = t0 + (np.arange(n, dtype=np.float64) + 0.5) * dt
    x = (np.arange(n, dtype=np.float64) + 0.5) * ds
    y = (np.arange(n, dtype=np.float64) + 0.5) * ds
    tt, xx, yy = np.meshgrid(t, x, y, indexing="ij")
    w = dt * ds * ds
    return tt.ravel(), xx.ravel(), yy.ravel(), w


def calibrate_true_beta0(
    target_events: int,
    beta1: float,
    sigma_km: float,
    T_hours: float,
    space_km: float,
    n_grid: int,
) -> float:
    """
    Set beta0 so expected event count is approximately target_events:
      E[N] = exp(beta0) * ∫ exp(beta1 * bump_true) dV
    """
    t, x, y, w = make_quadrature_grid(0.0, T_hours, space_km, n_grid)
    b = bump_gaussian(
        t=t,
        x=x,
        y=y,
        sigma_km=sigma_km,
        T_hours=T_hours,
        space_km=space_km,
        dx_km=0.0,
        dy_km=0.0,
    )
    integral = w * np.sum(np.exp(beta1 * b))
    beta0 = math.log(max(target_events, 1) / max(integral, EPS))
    return float(beta0)


def simulate_ipp_thinning(
    rng: np.random.Generator,
    p: TrueParams,
) -> np.ndarray:
    """
    IPP simulation by thinning from homogeneous PPP with upper bound lambda_max.
    """
    volume = p.T_hours * p.space_km * p.space_km
    # Since bump in [0,1], max log-intensity is beta0 + beta1.
    lambda_max = math.exp(p.beta0 + max(p.beta1, 0.0))
    n_cand = int(rng.poisson(lambda_max * volume))
    if n_cand <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    t = rng.uniform(0.0, p.T_hours, size=n_cand)
    x = rng.uniform(0.0, p.space_km, size=n_cand)
    y = rng.uniform(0.0, p.space_km, size=n_cand)
    lam = lambda_true(t=t, x=x, y=y, p=p)
    u = rng.uniform(0.0, 1.0, size=n_cand)
    keep = u < (lam / max(lambda_max, EPS))
    events = np.stack([t[keep], x[keep], y[keep]], axis=1)
    if events.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return events[np.argsort(events[:, 0])]


def split_train_test_by_time(events: np.ndarray, train_T: float) -> Tuple[np.ndarray, np.ndarray]:
    train = events[events[:, 0] <= train_T]
    test = events[events[:, 0] > train_T]
    return train, test


def ll_and_grad_train(
    theta: np.ndarray,
    train_events: np.ndarray,
    bump_ev_train: np.ndarray,
    bump_quad_train: np.ndarray,
    w_train: float,
) -> Tuple[float, np.ndarray]:
    """
    Returns train NLL and gradient wrt theta=[theta0, theta1].
    """
    theta0, theta1 = float(theta[0]), float(theta[1])

    # Event term
    log_lam_ev = theta0 + theta1 * bump_ev_train
    sum_log = float(np.sum(log_lam_ev))

    # Compensator term via quadrature
    eta_quad = theta0 + theta1 * bump_quad_train
    lam_quad = np.exp(np.clip(eta_quad, -50.0, 50.0))
    comp = w_train * float(np.sum(lam_quad))

    ll = sum_log - comp
    nll = -ll

    # Gradient of LL:
    # d/dtheta0: N_train - ∫ lambda
    # d/dtheta1: sum bump_ev - ∫ bump * lambda
    n_train = float(train_events.shape[0])
    g0_ll = n_train - comp
    g1_ll = float(np.sum(bump_ev_train)) - w_train * float(np.sum(bump_quad_train * lam_quad))
    grad_nll = -np.array([g0_ll, g1_ll], dtype=np.float64)
    return nll, grad_nll


def fit_model_theta(
    train_events: np.ndarray,
    domain: Domain,
    sigma_km: float,
    dx_km: float,
    dy_km: float,
    grid_n: int,
) -> ModelParams:
    t_e = train_events[:, 0]
    x_e = train_events[:, 1]
    y_e = train_events[:, 2]
    bump_ev = bump_gaussian(
        t=t_e,
        x=x_e,
        y=y_e,
        sigma_km=sigma_km,
        T_hours=domain.T_hours,
        space_km=domain.space_km,
        dx_km=dx_km,
        dy_km=dy_km,
    )

    t_q, x_q, y_q, w_q = make_quadrature_grid(0.0, domain.train_T, domain.space_km, grid_n)
    bump_q = bump_gaussian(
        t=t_q,
        x=x_q,
        y=y_q,
        sigma_km=sigma_km,
        T_hours=domain.T_hours,
        space_km=domain.space_km,
        dx_km=dx_km,
        dy_km=dy_km,
    )

    train_volume = domain.train_T * domain.space_km * domain.space_km
    rate0 = max(train_events.shape[0], 1) / max(train_volume, EPS)
    x0 = np.array([math.log(rate0), 1.0], dtype=np.float64)

    def f(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        return ll_and_grad_train(
            theta=theta,
            train_events=train_events,
            bump_ev_train=bump_ev,
            bump_quad_train=bump_q,
            w_train=w_q,
        )

    res = minimize(
        fun=lambda th: f(th)[0],
        x0=x0,
        jac=lambda th: f(th)[1],
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    return ModelParams(
        theta0=float(res.x[0]),
        theta1=float(res.x[1]),
        sigma_km=sigma_km,
        dx_km=dx_km,
        dy_km=dy_km,
        T_hours=domain.T_hours,
        space_km=domain.space_km,
    )


def ll_ipp(
    events: np.ndarray,
    t0: float,
    t1: float,
    space_km: float,
    theta0: float,
    theta1: float,
    sigma_km: float,
    T_hours: float,
    dx_km: float,
    dy_km: float,
    grid_n: int,
) -> Tuple[float, float]:
    """
    Exact LL (event term + compensator quadrature) and compensator for a window.
    """
    if events.shape[0] > 0:
        b_ev = bump_gaussian(
            t=events[:, 0],
            x=events[:, 1],
            y=events[:, 2],
            sigma_km=sigma_km,
            T_hours=T_hours,
            space_km=space_km,
            dx_km=dx_km,
            dy_km=dy_km,
        )
        sum_log = float(np.sum(theta0 + theta1 * b_ev))
    else:
        sum_log = 0.0

    t_q, x_q, y_q, w_q = make_quadrature_grid(t0=t0, t1=t1, space_km=space_km, n=grid_n)
    lam_q = lambda_model(
        t=t_q,
        x=x_q,
        y=y_q,
        theta0=theta0,
        theta1=theta1,
        sigma_km=sigma_km,
        T_hours=T_hours,
        space_km=space_km,
        dx_km=dx_km,
        dy_km=dy_km,
    )
    comp = w_q * float(np.sum(lam_q))
    ll = sum_log - comp
    return ll, comp


def disk_samples(radius: float, Ks: int) -> np.ndarray:
    """
    Deterministic quasi-uniform points in a disk via sunflower sampling.
    """
    if Ks <= 0:
        raise ValueError("Ks must be positive.")
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    pts = np.zeros((Ks, 2), dtype=np.float64)
    for i in range(Ks):
        rho = radius * math.sqrt((i + 0.5) / Ks)
        ang = 2.0 * math.pi * ((i * phi) % 1.0)
        pts[i, 0] = rho * math.cos(ang)
        pts[i, 1] = rho * math.sin(ang)
    return pts


def smoothed_test_nll_volcorr(
    test_events: np.ndarray,
    model: ModelParams,
    test_compensator: float,
    r_km: float,
    tau_h: float,
    Kt: int,
    Ks: int,
) -> Tuple[float, float]:
    """
    Smoothed test NLL (volume-corrected) and raw counterpart.

    Raw uses log(∫_B lambda).
    Vol-corrected uses log((1/Vol)∫_B lambda) = log(∫_B lambda) - log(Vol).
    """
    if test_events.shape[0] == 0:
        return float("nan"), float("nan")

    area = math.pi * (r_km ** 2)
    vol = (2.0 * tau_h) * area
    offsets = disk_samples(r_km, Ks)  # (Ks,2)

    log_mass_terms = []
    log_avg_terms = []
    for ev in test_events:
        ti, xi, yi = float(ev[0]), float(ev[1]), float(ev[2])
        t_grid = np.linspace(ti - tau_h, ti + tau_h, Kt, dtype=np.float64)

        # Evaluate lambda at Kt x Ks local samples.
        x_pts = xi + offsets[:, 0]  # (Ks,)
        y_pts = yi + offsets[:, 1]  # (Ks,)
        tt = np.repeat(t_grid[:, None], Ks, axis=1)         # (Kt, Ks)
        xx = np.repeat(x_pts[None, :], Kt, axis=0)          # (Kt, Ks)
        yy = np.repeat(y_pts[None, :], Kt, axis=0)          # (Kt, Ks)

        lam = lambda_model(
            t=tt.ravel(),
            x=xx.ravel(),
            y=yy.ravel(),
            theta0=model.theta0,
            theta1=model.theta1,
            sigma_km=model.sigma_km,
            T_hours=model.T_hours,
            space_km=model.space_km,
            dx_km=model.dx_km,
            dy_km=model.dy_km,
        ).reshape(Kt, Ks)

        # Spatial disk integral at each time: area * average over disk samples.
        spatial_int = area * np.mean(lam, axis=1)  # (Kt,)
        # Temporal integral over [ti-tau, ti+tau].
        mass = float(np.trapezoid(spatial_int, t_grid))
        avg_lam = mass / max(vol, EPS)

        log_mass_terms.append(math.log(max(mass, EPS)))
        log_avg_terms.append(math.log(max(avg_lam, EPS)))

    ll_raw = float(np.sum(log_mass_terms) - test_compensator)
    ll_volcorr = float(np.sum(log_avg_terms) - test_compensator)
    nll_raw = -ll_raw / test_events.shape[0]
    nll_volcorr = -ll_volcorr / test_events.shape[0]
    return nll_raw, nll_volcorr


def save_plot_test_scatter_and_trajectories(
    out_path: Path,
    test_events: np.ndarray,
    domain: Domain,
    dx_km: float,
    dy_km: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    if test_events.shape[0] > 0:
        sc = ax.scatter(
            test_events[:, 1],
            test_events[:, 2],
            c=test_events[:, 0],
            s=10,
            alpha=0.55,
            cmap="viridis",
            label="Test events",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Time (hours)")

    t_line = np.linspace(0.0, domain.T_hours, 300)
    cx_true, cy_true = center_trajectory(t_line, domain.T_hours, domain.space_km)
    cx_model = cx_true + dx_km
    cy_model = cy_true + dy_km
    ax.plot(cx_true, cy_true, lw=2.5, color="tab:blue", label="True bump center")
    ax.plot(cx_model, cy_model, lw=2.5, ls="--", color="tab:orange", label="Model bump center")

    ax.set_xlim(0.0, domain.space_km)
    ax.set_ylim(0.0, domain.space_km)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Test Events and True/Model Bump Trajectories")
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.02,
        "Operational tolerance example: 500 m, 10 min",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plot_true_vs_fitted_heatmaps(
    out_path: Path,
    true_p: TrueParams,
    model_p: ModelParams,
    t_slices: Iterable[float] = (2.0, 5.0, 8.0),
    n_grid: int = 120,
) -> None:
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, true_p.space_km, n_grid)
    y = np.linspace(0.0, true_p.space_km, n_grid)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    t_slices = list(t_slices)
    fig, axes = plt.subplots(len(t_slices), 2, figsize=(11, 4 * len(t_slices)))
    if len(t_slices) == 1:
        axes = np.array([axes])  # make 2D for uniform indexing

    for i, t in enumerate(t_slices):
        tt = np.full_like(xx, t, dtype=np.float64)
        lam_true = lambda_true(tt.ravel(), xx.ravel(), yy.ravel(), true_p).reshape(xx.shape)
        lam_fit = lambda_model(
            t=tt.ravel(),
            x=xx.ravel(),
            y=yy.ravel(),
            theta0=model_p.theta0,
            theta1=model_p.theta1,
            sigma_km=model_p.sigma_km,
            T_hours=model_p.T_hours,
            space_km=model_p.space_km,
            dx_km=model_p.dx_km,
            dy_km=model_p.dy_km,
        ).reshape(xx.shape)

        vmin = min(float(np.min(lam_true)), float(np.min(lam_fit)))
        vmax = max(float(np.max(lam_true)), float(np.max(lam_fit)))

        im0 = axes[i, 0].imshow(
            lam_true,
            origin="lower",
            extent=[0, true_p.space_km, 0, true_p.space_km],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[i, 0].set_title(f"lambda_true at t={t:.1f}h")
        axes[i, 0].set_xlabel("x (km)")
        axes[i, 0].set_ylabel("y (km)")

        im1 = axes[i, 1].imshow(
            lam_fit,
            origin="lower",
            extent=[0, true_p.space_km, 0, true_p.space_km],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[i, 1].set_title(f"lambda_fitted at t={t:.1f}h")
        axes[i, 1].set_xlabel("x (km)")
        axes[i, 1].set_ylabel("y (km)")

        cbar = fig.colorbar(im1, ax=[axes[i, 0], axes[i, 1]], fraction=0.03, pad=0.02)
        cbar.set_label("Intensity (events per h·km²)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_plot_smoothed_grid(
    out_path: Path,
    r_vals: np.ndarray,
    tau_vals: np.ndarray,
    nll_volcorr: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Heatmap: rows=tau, cols=r
    im = axes[0].imshow(nll_volcorr, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_xticks(np.arange(len(r_vals)))
    axes[0].set_yticks(np.arange(len(tau_vals)))
    axes[0].set_xticklabels([f"{r:.2f}" for r in r_vals])
    axes[0].set_yticklabels([f"{tau*60:.0f}" for tau in tau_vals])  # minutes
    axes[0].set_xlabel("r (km)")
    axes[0].set_ylabel("tau (minutes)")
    axes[0].set_title("Smoothed Test NLL (Vol-Corrected)")
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label("NLL per event")

    min_idx = np.unravel_index(np.nanargmin(nll_volcorr), nll_volcorr.shape)
    axes[0].scatter([min_idx[1]], [min_idx[0]], marker="x", s=70, c="red")
    axes[0].annotate(
        f"min={nll_volcorr[min_idx]:.4f}",
        (min_idx[1], min_idx[0]),
        xytext=(8, 6),
        textcoords="offset points",
        color="red",
    )

    # Line plot: NLL vs r for each tau.
    for i, tau in enumerate(tau_vals):
        axes[1].plot(r_vals, nll_volcorr[i], marker="o", lw=2, label=f"tau={tau*60:.0f} min")
    axes[1].axvline(0.5, color="gray", ls="--", lw=1.5, label="500 m")
    axes[1].set_xlabel("r (km)")
    axes[1].set_ylabel("Smoothed test NLL (vol-corrected)")
    axes[1].set_title("Smoothed NLL vs Spatial Tolerance")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_id = make_run_id(args)
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.T_hours <= 0 or args.space_km <= 0 or args.train_T <= 0:
        raise ValueError("T_hours, space_km, and train_T must be positive.")
    if args.train_T >= args.T_hours:
        raise ValueError("train_T must be strictly less than T_hours.")
    if args.grid_n <= 4 or args.Kt <= 2 or args.Ks <= 8:
        raise ValueError("grid_n, Kt, and Ks are too small for reliable quadrature.")
    if args.sigma_km <= 0:
        raise ValueError("sigma_km must be positive.")

    rng = np.random.default_rng(args.seed)
    domain = Domain(T_hours=args.T_hours, space_km=args.space_km, train_T=args.train_T)

    # Configure true process so expected count is close to target_events.
    beta1_true = 3.0
    beta0_true = calibrate_true_beta0(
        target_events=args.target_events,
        beta1=beta1_true,
        sigma_km=args.sigma_km,
        T_hours=args.T_hours,
        space_km=args.space_km,
        n_grid=max(40, min(120, args.grid_n)),
    )
    true_p = TrueParams(
        beta0=beta0_true,
        beta1=beta1_true,
        sigma_km=args.sigma_km,
        T_hours=args.T_hours,
        space_km=args.space_km,
    )

    # Simulate and split.
    events = simulate_ipp_thinning(rng=rng, p=true_p)
    train_events, test_events = split_train_test_by_time(events, train_T=args.train_T)
    if train_events.shape[0] < 20 or test_events.shape[0] < 20:
        raise RuntimeError(
            f"Not enough events after split (train={train_events.shape[0]}, test={test_events.shape[0]}). "
            "Increase target_events."
        )

    # Fit misaligned model on train.
    fitted = fit_model_theta(
        train_events=train_events,
        domain=domain,
        sigma_km=args.sigma_km,
        dx_km=args.dx_km,
        dy_km=args.dy_km,
        grid_n=args.grid_n,
    )

    # Standard LL/NLL on train and test.
    train_ll, train_comp = ll_ipp(
        events=train_events,
        t0=0.0,
        t1=args.train_T,
        space_km=args.space_km,
        theta0=fitted.theta0,
        theta1=fitted.theta1,
        sigma_km=fitted.sigma_km,
        T_hours=fitted.T_hours,
        dx_km=fitted.dx_km,
        dy_km=fitted.dy_km,
        grid_n=args.grid_n,
    )
    test_ll, test_comp = ll_ipp(
        events=test_events,
        t0=args.train_T,
        t1=args.T_hours,
        space_km=args.space_km,
        theta0=fitted.theta0,
        theta1=fitted.theta1,
        sigma_km=fitted.sigma_km,
        T_hours=fitted.T_hours,
        dx_km=fitted.dx_km,
        dy_km=fitted.dy_km,
        grid_n=args.grid_n,
    )

    train_nll_per_event = -train_ll / train_events.shape[0]
    test_nll_per_event = -test_ll / test_events.shape[0]

    # Smoothed test NLL across required tolerance grid.
    r_vals = DEFAULT_R_GRID_KM.copy()
    tau_vals = DEFAULT_TAU_GRID_H.copy()
    smoothed_raw = np.zeros((len(tau_vals), len(r_vals)), dtype=np.float64)
    smoothed_vol = np.zeros((len(tau_vals), len(r_vals)), dtype=np.float64)
    for i_tau, tau_h in enumerate(tau_vals):
        for j_r, r_km in enumerate(r_vals):
            nll_raw, nll_volcorr = smoothed_test_nll_volcorr(
                test_events=test_events,
                model=fitted,
                test_compensator=test_comp,
                r_km=float(r_km),
                tau_h=float(tau_h),
                Kt=args.Kt,
                Ks=args.Ks,
            )
            smoothed_raw[i_tau, j_r] = nll_raw
            smoothed_vol[i_tau, j_r] = nll_volcorr

    # Plots.
    p_scatter = out_dir / "test_events_and_trajectories.png"
    p_heatmaps = out_dir / "lambda_true_vs_fitted_slices.png"
    p_smooth = out_dir / "smoothed_nll_grid.png"
    save_plot_test_scatter_and_trajectories(
        out_path=p_scatter,
        test_events=test_events,
        domain=domain,
        dx_km=args.dx_km,
        dy_km=args.dy_km,
    )
    save_plot_true_vs_fitted_heatmaps(
        out_path=p_heatmaps,
        true_p=true_p,
        model_p=fitted,
        t_slices=(2.0, 5.0, 8.0),
        n_grid=120,
    )
    save_plot_smoothed_grid(
        out_path=p_smooth,
        r_vals=r_vals,
        tau_vals=tau_vals,
        nll_volcorr=smoothed_vol,
    )

    # Self-checks.
    lam_test = lambda_model(
        t=test_events[:, 0],
        x=test_events[:, 1],
        y=test_events[:, 2],
        theta0=fitted.theta0,
        theta1=fitted.theta1,
        sigma_km=fitted.sigma_km,
        T_hours=fitted.T_hours,
        space_km=fitted.space_km,
        dx_km=fitted.dx_km,
        dy_km=fitted.dy_km,
    )
    assert np.all(np.isfinite(lam_test)) and np.all(lam_test > 0.0), "Fitted intensity invalid."
    assert np.isfinite(train_nll_per_event) and np.isfinite(test_nll_per_event), "NLL non-finite."
    assert np.all(np.isfinite(smoothed_vol)), "Smoothed NLL contains non-finite values."
    assert p_scatter.exists() and p_heatmaps.exists() and p_smooth.exists(), "Plot save failed."

    # Summary printout.
    print("")
    print("Synthetic LL vs Smoothed NLL")
    print("----------------------------")
    print(f"run_id: {run_id}")
    print(f"shift (dx, dy) km: ({args.dx_km:.3f}, {args.dy_km:.3f})")
    print(f"events total/train/test: {events.shape[0]}/{train_events.shape[0]}/{test_events.shape[0]}")
    print(f"fitted theta: theta0={fitted.theta0:.6f}, theta1={fitted.theta1:.6f}")
    print(f"train LL total: {train_ll:.6f}, train NLL/event: {train_nll_per_event:.6f}")
    print(f"test  LL total: {test_ll:.6f}, test  NLL/event: {test_nll_per_event:.6f}")
    print("")
    print("Smoothed test NLL (per event)")
    print("rows=tau (hours), cols=r (km)")
    print("tau values:", [float(x) for x in tau_vals])
    print("r values  :", [float(x) for x in r_vals])
    print("vol-corrected:")
    print(np.array2string(smoothed_vol, precision=6, suppress_small=False))
    print("raw:")
    print(np.array2string(smoothed_raw, precision=6, suppress_small=False))
    print("")
    print("EMS reference tolerances: 500 m = 0.5 km, 10 min = 0.1667 h")
    print(f"artifacts saved to: {out_dir}")

    # Save JSON artifact.
    summary = {
        "run_id": run_id,
        "seed": int(args.seed),
        "domain": {
            "space_km": float(args.space_km),
            "T_hours": float(args.T_hours),
            "train_T_hours": float(args.train_T),
        },
        "true_params": {
            "beta0": float(true_p.beta0),
            "beta1": float(true_p.beta1),
            "sigma_km": float(true_p.sigma_km),
        },
        "model_params": {
            "theta0": float(fitted.theta0),
            "theta1": float(fitted.theta1),
            "sigma_km": float(fitted.sigma_km),
            "dx_km": float(fitted.dx_km),
            "dy_km": float(fitted.dy_km),
        },
        "counts": {
            "total": int(events.shape[0]),
            "train": int(train_events.shape[0]),
            "test": int(test_events.shape[0]),
        },
        "metrics": {
            "train_ll_total": float(train_ll),
            "train_nll_per_event": float(train_nll_per_event),
            "test_ll_total": float(test_ll),
            "test_nll_per_event": float(test_nll_per_event),
            "smoothed_nll_per_event_volcorr": smoothed_vol.tolist(),
            "smoothed_nll_per_event_raw": smoothed_raw.tolist(),
            "r_values_km": r_vals.tolist(),
            "tau_values_hours": tau_vals.tolist(),
        },
        "artifacts": {
            "test_scatter_trajectory_plot": str(p_scatter),
            "heatmap_true_vs_fitted": str(p_heatmaps),
            "smoothed_nll_plot": str(p_smooth),
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("self-check: passed")


if __name__ == "__main__":
    main()
