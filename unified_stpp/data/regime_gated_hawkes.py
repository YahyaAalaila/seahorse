"""
Synthetic marked STPP with exogenous regime switching and covariate-gated excitation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sample_regime_path(
    T: float,
    n_regimes: int,
    switch_rate: float,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample piecewise-constant CTMC-like regime path with uniform jumps among other regimes."""
    if n_regimes < 1:
        raise ValueError("n_regimes must be >= 1")
    if switch_rate <= 0.0 or n_regimes == 1:
        return np.array([0.0], dtype=np.float64), np.array([0], dtype=np.int64)

    states = [int(rng.randint(0, n_regimes))]
    change_times = [0.0]
    t = 0.0
    while True:
        t += float(rng.exponential(1.0 / switch_rate))
        if t >= T:
            break
        prev = states[-1]
        choices = [r for r in range(n_regimes) if r != prev]
        states.append(int(rng.choice(choices)))
        change_times.append(t)

    return np.asarray(change_times, dtype=np.float64), np.asarray(states, dtype=np.int64)


def regime_at_times(times: np.ndarray, change_times: np.ndarray, states: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(change_times, times, side="right") - 1
    idx = np.clip(idx, 0, len(states) - 1)
    return states[idx]


def _regime_centers(n_regimes: int, spatial_bounds: Tuple[float, float]) -> np.ndarray:
    lo, hi = spatial_bounds
    if n_regimes == 1:
        return np.array([[0.5 * (lo + hi), 0.5 * (lo + hi)]], dtype=np.float64)

    # Place regimes on corners and edge-midpoints for diversity.
    pts = np.array(
        [
            [lo, lo],
            [hi, hi],
            [lo, hi],
            [hi, lo],
            [0.5 * (lo + hi), lo],
            [0.5 * (lo + hi), hi],
            [lo, 0.5 * (lo + hi)],
            [hi, 0.5 * (lo + hi)],
        ],
        dtype=np.float64,
    )
    if n_regimes <= len(pts):
        return pts[:n_regimes]

    extra = np.linspace(lo, hi, n_regimes - len(pts) + 2)[1:-1]
    more = np.stack([extra, extra[::-1]], axis=-1)
    return np.concatenate([pts, more], axis=0)[:n_regimes]


def env_features(x: np.ndarray, env_dim: int, spatial_bounds: Tuple[float, float]) -> np.ndarray:
    """Bounded spatial field features in [-1, 1]."""
    x = np.asarray(x)
    xx = x[..., 0]
    yy = x[..., 1]
    lo, hi = spatial_bounds
    scale = max(hi - lo, 1e-6)
    x_n = (xx - lo) / scale
    y_n = (yy - lo) / scale

    feats = []
    h = max(1, (env_dim + 3) // 4)
    for j in range(h):
        w = float(j + 1)
        feats.extend(
            [
                np.sin(2.0 * np.pi * w * x_n),
                np.cos(2.0 * np.pi * w * x_n),
                np.sin(2.0 * np.pi * w * y_n),
                np.cos(2.0 * np.pi * w * y_n),
            ]
        )
    out = np.stack(feats[:env_dim], axis=-1)
    return out.astype(np.float64)


def covariates_at(
    t: np.ndarray,
    x: np.ndarray,
    *,
    T: float,
    n_regimes: int,
    regime_change_times: np.ndarray,
    regime_states: np.ndarray,
    env_dim: int,
    spatial_bounds: Tuple[float, float],
    sigma_reg: float = 0.1,
    apply_tanh: bool = True,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Predictable covariates Z(t, x) = [noisy regime one-hot, tod, env(x)]."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2)

    reg = regime_at_times(t, regime_change_times, regime_states)
    z_reg = np.zeros((t.shape[0], n_regimes), dtype=np.float64)
    z_reg[np.arange(t.shape[0]), reg] = 1.0
    if sigma_reg > 0.0 and rng is not None:
        z_reg = z_reg + rng.normal(0.0, sigma_reg, size=z_reg.shape)
    if apply_tanh:
        z_reg = np.tanh(z_reg)

    tod = np.stack(
        [
            np.sin(2.0 * np.pi * t / max(T, 1e-6)),
            np.cos(2.0 * np.pi * t / max(T, 1e-6)),
        ],
        axis=-1,
    )
    z_env = env_features(x, env_dim=env_dim, spatial_bounds=spatial_bounds)
    return np.concatenate([z_reg, tod, z_env], axis=-1).astype(np.float32)


def generate_regime_gated_hawkes_stpp(
    n_sequences: int = 100,
    T: float = 5.0,
    spatial_bounds: Tuple[float, float] = (-5.0, 5.0),
    n_marks: int = 3,
    n_regimes: int = 3,
    env_dim: int = 4,
    sigma_reg: float = 0.15,
    switch_rate: float = 1.2,
    sigma_hotspot: float = 1.0,
    lambda_bg: float = 0.02,
    tau_t: float = 0.8,
    sigma_exc: float = 0.9,
    max_events_per_seq: int = 250,
    seed: int = 42,
) -> List[Dict]:
    """Generate marked STPP with exogenous regimes and covariate-gated excitation."""
    if n_marks < 1 or n_regimes < 1:
        raise ValueError("n_marks and n_regimes must be >= 1")

    rng = np.random.RandomState(seed)
    lo, hi = spatial_bounds
    area = float((hi - lo) ** 2)

    regime_centers = _regime_centers(n_regimes, spatial_bounds)

    alpha = rng.normal(-0.6, 0.35, size=(n_marks, n_regimes))
    beta_env = rng.normal(0.0, 0.4, size=(n_marks, env_dim))

    A = np.abs(rng.normal(0.12, 0.05, size=(n_marks, n_marks))).astype(np.float64)
    np.fill_diagonal(A, np.diag(A) + 0.06)

    w_regime = rng.normal(0.9, 0.35, size=(n_regimes,))
    u_tod = rng.normal(0.35, 0.2, size=(2,))

    # Conservative global bound using bounded env features and max events cap.
    baseline_max = float(
        np.max(
            np.exp(alpha + np.sum(np.abs(beta_env), axis=-1, keepdims=True))
        )
        + lambda_bg
    )
    max_a = float(np.max(A))
    lambda_bar = float(n_marks * baseline_max + n_marks * max_events_per_seq * max_a)

    print(
        "Generating regime-gated Hawkes sequences, "
        f"seed={seed}, marks={n_marks}, regimes={n_regimes}, cov_dim={n_regimes + 2 + env_dim}"
    )

    out: List[Dict] = []
    for seq_idx in range(n_sequences):
        # Sequence-specific latent regime path.
        regime_change_times, regime_states = _sample_regime_path(
            T=T,
            n_regimes=n_regimes,
            switch_rate=switch_rate,
            rng=np.random.RandomState(seed + 10007 * seq_idx + 17),
        )

        t = 0.0
        times: List[float] = []
        locs: List[np.ndarray] = []
        marks: List[int] = []
        covs: List[np.ndarray] = []
        gates: List[float] = []

        while t < T and len(times) < max_events_per_seq:
            dt = float(rng.exponential(1.0 / max(lambda_bar * area, 1e-12)))
            t += dt
            if t >= T:
                break

            s = rng.uniform(lo, hi, size=2)
            s = s.astype(np.float64)

            r_t = int(regime_at_times(np.array([t]), regime_change_times, regime_states)[0])
            z_env = env_features(s.reshape(1, 2), env_dim=env_dim, spatial_bounds=spatial_bounds)[0]

            # Baseline intensities per mark for current regime.
            d2_reg = np.sum((s.reshape(1, 2) - regime_centers[r_t : r_t + 1]) ** 2, axis=-1)[0]
            spatial_reg = np.exp(-d2_reg / (2.0 * sigma_hotspot * sigma_hotspot))
            mu_k = np.exp(alpha[:, r_t] + np.einsum("ke,e->k", beta_env, z_env)) * spatial_reg + lambda_bg

            lam_k = mu_k.astype(np.float64)
            if times:
                t_hist = np.asarray(times, dtype=np.float64)
                x_hist = np.asarray(locs, dtype=np.float64)
                k_hist = np.asarray(marks, dtype=np.int64)
                g_hist = np.asarray(gates, dtype=np.float64)

                kt = np.exp(-(t - t_hist) / max(tau_t, 1e-6))
                d2 = np.sum((x_hist - s.reshape(1, 2)) ** 2, axis=-1)
                kx = np.exp(-d2 / (2.0 * sigma_exc * sigma_exc))
                parent_factor = g_hist * kt * kx  # (n_hist,)

                # Sum_j A[k, mark_j] * factor_j
                exc = np.zeros(n_marks, dtype=np.float64)
                for k in range(n_marks):
                    exc[k] = np.sum(A[k, k_hist] * parent_factor)
                lam_k = lam_k + exc

            lam_tot = float(np.sum(lam_k))
            if rng.uniform() >= lam_tot / max(lambda_bar, 1e-12):
                continue

            probs = lam_k / max(lam_tot, 1e-12)
            mark = int(rng.choice(np.arange(n_marks), p=probs))

            cov = covariates_at(
                np.array([t]),
                s.reshape(1, 2),
                T=T,
                n_regimes=n_regimes,
                regime_change_times=regime_change_times,
                regime_states=regime_states,
                env_dim=env_dim,
                spatial_bounds=spatial_bounds,
                sigma_reg=sigma_reg,
                apply_tanh=True,
                rng=rng,
            )[0]

            z_reg = cov[:n_regimes].astype(np.float64)
            z_tod = cov[n_regimes : n_regimes + 2].astype(np.float64)
            gate = float(_sigmoid(np.dot(w_regime, z_reg) + np.dot(u_tod, z_tod)))

            times.append(t)
            locs.append(s.astype(np.float32))
            marks.append(mark)
            covs.append(cov.astype(np.float32))
            gates.append(gate)
        print(f"Generated sequence {seq_idx + 1}/{n_sequences} with {len(times)} events.")

        out.append(
            {
                "times": np.asarray(times, dtype=np.float32),
                "locations": np.asarray(locs, dtype=np.float32).reshape(-1, 2),
                "marks": np.asarray(marks, dtype=np.int64),
                "field_covariates": (
                    np.asarray(covs, dtype=np.float32).reshape(-1, n_regimes + 2 + env_dim)
                    if covs
                    else np.zeros((0, n_regimes + 2 + env_dim), dtype=np.float32)
                ),
                "regime_change_times": regime_change_times.astype(np.float32),
                "regime_states": regime_states.astype(np.int64),
                "_rg_params": {
                    "n_marks": int(n_marks),
                    "n_regimes": int(n_regimes),
                    "env_dim": int(env_dim),
                    "sigma_hotspot": float(sigma_hotspot),
                    "lambda_bg": float(lambda_bg),
                    "tau_t": float(tau_t),
                    "sigma_exc": float(sigma_exc),
                    "spatial_bounds": (float(spatial_bounds[0]), float(spatial_bounds[1])),
                    "T": float(T),
                    "alpha": alpha.astype(np.float32),
                    "beta_env": beta_env.astype(np.float32),
                    "A": A.astype(np.float32),
                    "w_regime": w_regime.astype(np.float32),
                    "u_tod": u_tod.astype(np.float32),
                    "regime_centers": regime_centers.astype(np.float32),
                    "sigma_reg": float(sigma_reg),
                    "seed": int(seed),
                },
            }
        )

    return out


def intensity_from_history(
    seq: Dict,
    t: float,
    s: np.ndarray,
) -> np.ndarray:
    """Total intensity sum_k lambda_k(t, s) for visualization, using sequence metadata."""
    p = seq.get("_rg_params")
    if p is None:
        raise ValueError("Sequence missing '_rg_params' required for true-intensity evaluation.")

    n_marks = int(p["n_marks"])
    n_regimes = int(p["n_regimes"])
    env_dim = int(p["env_dim"])
    sigma_hotspot = float(p["sigma_hotspot"])
    lambda_bg = float(p["lambda_bg"])
    tau_t = float(p["tau_t"])
    sigma_exc = float(p["sigma_exc"])
    spatial_bounds = tuple(p["spatial_bounds"])
    T = float(p["T"])

    alpha = np.asarray(p["alpha"], dtype=np.float64)
    beta_env = np.asarray(p["beta_env"], dtype=np.float64)
    A = np.asarray(p["A"], dtype=np.float64)
    w_regime = np.asarray(p["w_regime"], dtype=np.float64)
    u_tod = np.asarray(p["u_tod"], dtype=np.float64)
    regime_centers = np.asarray(p["regime_centers"], dtype=np.float64)

    change_times = np.asarray(seq["regime_change_times"], dtype=np.float64)
    regime_states = np.asarray(seq["regime_states"], dtype=np.int64)

    s_arr = np.asarray(s, dtype=np.float64)
    flat = s_arr.reshape(-1, 2)

    r_t = int(regime_at_times(np.array([t]), change_times, regime_states)[0])
    z_env = env_features(flat, env_dim=env_dim, spatial_bounds=spatial_bounds)
    d2_reg = np.sum((flat - regime_centers[r_t : r_t + 1]) ** 2, axis=-1)
    spatial_reg = np.exp(-d2_reg / (2.0 * sigma_hotspot * sigma_hotspot))

    mu_k = np.exp(alpha[:, r_t].reshape(1, -1) + z_env @ beta_env.T) * spatial_reg.reshape(-1, 1) + lambda_bg
    lam_k = mu_k.copy()

    t_hist = np.asarray(seq["times"], dtype=np.float64)
    if len(t_hist) > 0:
        mask = t_hist < float(t)
        if np.any(mask):
            t_h = t_hist[mask]
            x_h = np.asarray(seq["locations"], dtype=np.float64)[mask]
            m_h = np.asarray(seq["marks"], dtype=np.int64)[mask]
            cov_h_raw = seq.get("field_covariates")
            if cov_h_raw is not None:
                cov_h = np.asarray(cov_h_raw, dtype=np.float64)[mask]
                z_reg_h = cov_h[:, :n_regimes]
                z_tod_h = cov_h[:, n_regimes : n_regimes + 2]
            else:
                # Support no-covariate ablations where field covariates are dropped
                # from stored sequences. The gate only needs regime + time-of-day.
                reg_h = regime_at_times(t_h, change_times, regime_states)
                z_reg_h = np.zeros((len(t_h), n_regimes), dtype=np.float64)
                z_reg_h[np.arange(len(t_h)), reg_h] = 1.0
                z_reg_h = np.tanh(z_reg_h)
                z_tod_h = np.stack(
                    [
                        np.sin(2.0 * np.pi * t_h / max(T, 1e-6)),
                        np.cos(2.0 * np.pi * t_h / max(T, 1e-6)),
                    ],
                    axis=-1,
                )
            g_h = _sigmoid(z_reg_h @ w_regime + z_tod_h @ u_tod)

            kt = np.exp(-(float(t) - t_h) / max(tau_t, 1e-6))
            for j in range(len(t_h)):
                d2 = np.sum((flat - x_h[j : j + 1]) ** 2, axis=-1)
                kx = np.exp(-d2 / (2.0 * sigma_exc * sigma_exc))
                lam_k += (A[:, m_h[j]].reshape(1, -1) * (g_h[j] * kt[j] * kx).reshape(-1, 1))

    return np.sum(lam_k, axis=-1).reshape(s_arr.shape[:-1])


if __name__ == "__main__":
    seqs = generate_regime_gated_hawkes_stpp(
        n_sequences=1,
        T=5.0,
        spatial_bounds=(-5.0, 5.0),
        n_marks=3,
        n_regimes=3,
        env_dim=4,
        seed=42,
    )
    seq = seqs[0]
    print(
        "Generated sequence:",
        f"events={len(seq['times'])},",
        f"cov_dim={seq['field_covariates'].shape[-1] if len(seq['field_covariates']) else 0}",
    )
    if len(seq["times"]) > 0:
        t_mid = float(min(4.0, max(0.1, seq["times"][-1] * 0.7)))
        grid = np.stack(
            np.meshgrid(
                np.linspace(-5.0, 5.0, 25),
                np.linspace(-5.0, 5.0, 25),
                indexing="ij",
            ),
            axis=-1,
        )
        lam = intensity_from_history(seq, t=t_mid, s=grid)
        print(
            f"Intensity snapshot at t={t_mid:.3f}:",
            f"min={float(lam.min()):.4f}, max={float(lam.max()):.4f}, mean={float(lam.mean()):.4f}",
        )
