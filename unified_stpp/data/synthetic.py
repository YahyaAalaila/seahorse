"""
Synthetic spatiotemporal point process generators for testing.

1. Spatiotemporal Hawkes process — self-exciting with spatial Gaussian kernel.
2. Inhomogeneous Poisson with covariate-dependent intensity — for testing
   covariate integration.
"""

import abc
import numpy as np
from typing import List, Dict, Optional, Tuple


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _moving_hotspot_t1_t2(T: float, t1_frac: float, t2_frac: float) -> Tuple[float, float]:
    t1 = float(np.clip(t1_frac, 0.0, 1.0) * T)
    t2 = float(np.clip(t2_frac, 0.0, 1.0) * T)
    if t2 < t1:
        t2 = t1
    return t1, t2


def _moving_hotspot_latent(seed: int, T: float, n_noise_knots: int = 16):
    rng = np.random.RandomState(seed)
    phi1 = rng.uniform(0, 2 * np.pi)
    phi2 = rng.uniform(0, 2 * np.pi)
    phiA = rng.uniform(0, 2 * np.pi)
    noise_knots_t = np.linspace(0.0, T, int(max(4, n_noise_knots)))
    noise_knots_v = rng.normal(0.0, 1.0, size=noise_knots_t.shape[0])
    return phi1, phi2, phiA, noise_knots_t, noise_knots_v


def moving_hotspot_center_and_amplitude(
    t: float,
    T: float,
    spatial_bounds: Tuple[float, float],
    *,
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
) -> Tuple[np.ndarray, float]:
    sb_min, sb_max = spatial_bounds
    a = 0.5 * (sb_max - sb_min)
    b = 0.5 * (sb_max - sb_min)
    center_shift = np.array([0.5 * (sb_min + sb_max), 0.5 * (sb_min + sb_max)], dtype=np.float64)
    corner_a = center_shift + np.array([-a, -b], dtype=np.float64)
    corner_b = center_shift + np.array([a, b], dtype=np.float64)

    t1, t2 = _moving_hotspot_t1_t2(T, t1_frac=t1_frac, t2_frac=t2_frac)
    phi1, phi2, phiA, noise_t, noise_v = _moving_hotspot_latent(seed=seed, T=T, n_noise_knots=n_noise_knots)

    if t <= t1:
        base = corner_a
    elif t <= t2:
        u = (t - t1) / max(t2 - t1, 1e-9)
        s = _smoothstep(u)
        base = (1.0 - s) * corner_a + s * corner_b
    else:
        base = corner_b

    jx = np.sin(2 * np.pi * jitter_f1 * t + phi1)
    jy = np.cos(2 * np.pi * jitter_f2 * t + phi2)
    jitter = float(jitter_radius) * np.array([jx, jy], dtype=np.float64)
    c = base + jitter
    c[0] = np.clip(c[0], sb_min, sb_max)
    c[1] = np.clip(c[1], sb_min, sb_max)

    noise_val = np.interp(t, noise_t, noise_v)
    amp_val = amp0 + amp1 * np.sin(2 * np.pi * 0.35 * t + phiA) + amp_noise * noise_val
    amp = float(np.exp(amp_val))
    return c.astype(np.float64), amp


def moving_hotspot_intensity(
    t: float,
    s: np.ndarray,
    T: float,
    spatial_bounds: Tuple[float, float],
    *,
    base_rate: float,
    hotspot_weight: float,
    sigma: float,
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
    c, amp = moving_hotspot_center_and_amplitude(
        t=t,
        T=T,
        spatial_bounds=spatial_bounds,
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
    s_arr = np.asarray(s)
    d2 = np.sum((s_arr - c) ** 2, axis=-1)
    hotspot = np.exp(-d2 / (2.0 * sigma * sigma))
    return base_rate + hotspot_weight * amp * hotspot


def generate_hawkes_stpp(
    n_sequences: int = 100,
    T: float = 10.0,
    spatial_bounds: Tuple[float, float] = (-5.0, 5.0),
    spatial_dim: int = 2,
    mu: float = 1.0,
    alpha: float = 0.5,
    beta: float = 1.0,
    sigma_s: float = 1.0,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate sequences from a spatiotemporal Hawkes process.
    
    λ*(t, s) = μ + α Σ_{t_i < t} β exp(-β(t - t_i)) · N(s; s_i, σ_s² I)
    
    Args:
        n_sequences: Number of sequences to generate.
        T: Time horizon.
        spatial_bounds: (min, max) for each spatial dimension.
        spatial_dim: Number of spatial dimensions.
        mu: Background rate.
        alpha: Excitation magnitude (must be < 1 for stability).
        beta: Temporal decay rate.
        sigma_s: Spatial influence kernel bandwidth.
        seed: Random seed.
    Returns:
        List of dicts, each with 'times', 'locations', and optionally 'covariates'.
    """
    rng = np.random.RandomState(seed)
    sequences = []

    for _ in range(n_sequences):
        times = []
        locs = []
        t = 0.0

        # Upper bound for thinning
        lambda_bar = mu + 10.0  # conservative; increase if rejection rate high

        while t < T:
            # Thinning algorithm
            dt = rng.exponential(1.0 / lambda_bar)
            t = t + dt
            if t >= T:
                break

            # Compute actual intensity at (t, s_candidate)
            s_candidate = rng.uniform(spatial_bounds[0], spatial_bounds[1], size=spatial_dim)

            lam = mu
            for ti, si in zip(times, locs):
                temporal = alpha * beta * np.exp(-beta * (t - ti))
                spatial = np.exp(-np.sum((s_candidate - si) ** 2) / (2 * sigma_s ** 2))
                spatial /= (2 * np.pi * sigma_s ** 2) ** (spatial_dim / 2)
                lam += temporal * spatial

            # Accept/reject
            if rng.uniform() < lam / lambda_bar:
                times.append(t)
                locs.append(s_candidate)

            # Update upper bound (adaptive)
            lambda_bar = max(lambda_bar, lam * 1.5)

        sequences.append({
            "times": np.array(times, dtype=np.float32),
            "locations": np.array(locs, dtype=np.float32).reshape(-1, spatial_dim),
        })

    return sequences


def generate_inhomogeneous_stpp(
    n_sequences: int = 100,
    T: float = 5.0,
    spatial_bounds: Tuple[float, float] = (-5.0, 5.0),
    spatial_dim: int = 2,
    base_rate: float = 2.0,
    covariate_dim: int = 1,
    seed: int = 42,
    covariate_fn=None,
) -> List[Dict]:
    """
    Generate sequences from an inhomogeneous Poisson process whose
    intensity depends on a spatial covariate.
    
    λ*(t, s) = base_rate · exp(w^T X(t, s))
    
    where X(t, s) is a field covariate. If covariate_fn is None,
    uses a default: X(t, s) = sin(s_1) + cos(t).
    
    This is useful for testing covariate augmentation.
    """
    print(f"Generating inhomogeneous Poisson sequences with covariate-dependent intensity, with seed = {seed} and covariate_fn = {covariate_fn}")
    rng = np.random.RandomState(seed)

    if covariate_fn is None:
        def covariate_fn(t, s):
            # Deterministic feature map with configurable output width.
            base = [
                np.sin(s[0]) + np.cos(t),
                np.cos(s[1] if len(s) > 1 else s[0]) + np.sin(0.5 * t),
                np.sin(s[0] + t),
                np.cos((s[1] if len(s) > 1 else s[0]) - t),
            ]
            if covariate_dim <= len(base):
                vals = base[:covariate_dim]
            else:
                vals = base + [np.sin((k + 1) * t) for k in range(covariate_dim - len(base))]
            return np.asarray(vals, dtype=np.float32)

    sequences = []
    sb = spatial_bounds
    vol = (sb[1] - sb[0]) ** spatial_dim

    for _ in range(n_sequences):
        times = []
        locs = []
        covs = []
        t = 0.0

        lambda_bar = base_rate * np.exp(0.5)  # upper bound

        while t < T:
            dt = rng.exponential(1.0 / (lambda_bar * vol))
            t = t + dt
            if t >= T:
                break

            s = rng.uniform(sb[0], sb[1], size=spatial_dim)
            x = covariate_fn(t, s)
            lam = base_rate * np.exp(np.clip(x.sum(), -5, 5))

            if rng.uniform() < lam / lambda_bar:
                times.append(t)
                locs.append(s.astype(np.float32))
                covs.append(x)
        #print(f"times: {len(times)}, locs: {len(locs)}, covs: {len(covs)}")
        seq = {
            "times": np.array(times, dtype=np.float32),
            "locations": np.array(locs, dtype=np.float32).reshape(-1, spatial_dim),
            "field_covariates": np.array(covs, dtype=np.float32).reshape(-1, len(covs[0])) if covs else np.zeros((0, 1), dtype=np.float32),
        }
        sequences.append(seq)

    return sequences


def moving_hotspot_covariates(
    t: float,
    s: np.ndarray,          # kept for API compatibility; NOT used in computation
    T: float,
    spatial_bounds: Tuple[float, float],
    sigma: float,           # kept for API compatibility; NOT used in computation
    covariate_dim: int,
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
    """
    Pure time-dependent contextual covariates for the moving-hotspot process.

    All features are functions of t only — no dependency on the event's own
    spatial location s.  This guarantees:
      (a) No leakage of the target location into the conditioning signal.
      (b) The spatial density f*(s | z, X(t)) remains a valid normalised
          density because X does not vary with s.
      (c) Training and inference covariate distributions are aligned.

    Feature layout (first covariate_dim entries):
      0: c_x(t)      — hotspot centre x at time t
      1: c_y(t)      — hotspot centre y at time t
      2: amp(t)      — amplitude modulation at time t
      3: sin(2πt/T)  — periodic motion-phase encoding (sin)
      4: cos(2πt/T)  — periodic motion-phase encoding (cos)
      5+: higher-order sin harmonics for extra capacity
    """
    c, amp = moving_hotspot_center_and_amplitude(
        t=t,
        T=T,
        spatial_bounds=spatial_bounds,
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
    tod = 2.0 * np.pi * t / max(T, 1e-6)

    # All time-only features — s is intentionally not used here.
    base = [
        c[0],             # hotspot centre x  (time-only)
        c[1],             # hotspot centre y  (time-only)
        amp,              # amplitude modulation  (time-only)
        np.sin(tod),      # periodic phase — sin  (time-only)
        np.cos(tod),      # periodic phase — cos  (time-only)
    ]
    if covariate_dim <= len(base):
        vals = base[:covariate_dim]
    else:
        vals = base + [np.sin((k + 1) * tod) for k in range(covariate_dim - len(base))]
    return np.asarray(vals, dtype=np.float32)


def generate_moving_hotspot_stpp(
    n_sequences: int = 100,
    T: float = 5.0,
    spatial_bounds: Tuple[float, float] = (-5.0, 5.0),
    spatial_dim: int = 2,
    base_rate: float = 0.2,
    hotspot_weight: float = 3.0,
    sigma: float = 0.9,
    switch_frac: float = 0.5,
    switch_time: Optional[float] = None,
    move_duration: Optional[float] = None,
    t1_frac: float = 0.32,
    t2_frac: float = 0.46,
    jitter_radius: float = 0.55,
    jitter_f1: float = 0.8,
    jitter_f2: float = 1.25,
    amp0: float = 0.0,
    amp1: float = 0.55,
    amp_noise: float = 0.10,
    n_noise_knots: int = 16,
    interaction_weight: float = 1.2,
    tod_weight: float = 0.6,
    covariate_dim: int = 1,
    seed: int = 42,
) -> List[Dict]:
    """
    Inhomogeneous Poisson with a hotspot that moves from one corner to the opposite.
    Field covariates expose regime/time interaction explicitly.
    """
    if spatial_dim != 2:
        raise ValueError("moving_hotspot currently supports spatial_dim=2")

    rng = np.random.RandomState(seed)
    sb_min, sb_max = spatial_bounds
    vol = (sb_max - sb_min) ** spatial_dim

    # Conservative bound for thinning proposals.
    amp_max = np.exp(amp0 + abs(amp1) + 4.0 * abs(amp_noise))
    lambda_bar = base_rate + hotspot_weight * amp_max

    print(
        "Generating moving-hotspot sequences, "
        f"seed={seed}, covariate_dim={covariate_dim}"
    )

    sequences = []
    for _ in range(n_sequences):
        times = []
        locs = []
        covs = []
        t = 0.0

        while t < T:
            dt = rng.exponential(1.0 / (lambda_bar * vol))
            t = t + dt
            if t >= T:
                break

            s = rng.uniform(sb_min, sb_max, size=2)
            lam = float(
                moving_hotspot_intensity(
                    t=t,
                    s=s,
                    T=T,
                    spatial_bounds=spatial_bounds,
                    base_rate=base_rate,
                    hotspot_weight=hotspot_weight,
                    sigma=sigma,
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
            )

            cov = moving_hotspot_covariates(
                t=t,
                s=s,
                T=T,
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

            if rng.uniform() < lam / lambda_bar:
                times.append(t)
                locs.append(s.astype(np.float32))
                covs.append(cov.astype(np.float32))
        print(f"Generated sequence with {len(times)} events.")
        sequences.append(
            {
                "times": np.array(times, dtype=np.float32),
                "locations": np.array(locs, dtype=np.float32).reshape(-1, 2),
                "field_covariates": (
                    np.array(covs, dtype=np.float32).reshape(-1, max(1, covariate_dim))
                    if covs
                    else np.zeros((0, max(1, covariate_dim)), dtype=np.float32)
                ),
            }
        )
        

    return sequences


eps = 1e-10


class SyntheticDataset(abc.ABC):
    """
    Abstract parent class for synthetic datasets.
    """

    def __init__(self, dist_only: bool = False):
        self.his_s = None
        self.his_t = None
        self.t_start = None
        self.t_end = None
        self.train = None
        self.val = None
        self.test = None
        self.st_scaler = None
        self.dist_only = dist_only

    @abc.abstractmethod
    def lamb_st(self, mu, his_s, his_t, s, t):
        pass

    @abc.abstractmethod
    def generate(self, t_start, t_end):
        pass

    @staticmethod
    def g0(s, s_mu, s_sqrt_inv_det_cov, s_inv_cov):
        return SyntheticDataset.g2(s, s_mu.reshape(1, 2), s_sqrt_inv_det_cov, s_inv_cov)

    @staticmethod
    def g1(t, his_t, alpha, beta):
        delta_t = t - his_t
        return alpha * np.exp(-beta * delta_t)

    @staticmethod
    def g2(s, his_s, s_sqrt_inv_det_cov, s_inv_cov):
        delta_s = s - his_s
        return 1 / 2 / np.pi * s_sqrt_inv_det_cov * np.exp(
            -np.einsum("ij,ij->i", delta_s.dot(s_inv_cov), delta_s) / 2
        )

    def save(self, text_path):
        np.savetxt(
            text_path,
            np.hstack((self.his_s, np.expand_dims((self.his_t), 1))),
            delimiter=",",
            fmt="%f",
        )

    def load(self, text_path, t_start, t_end):
        self.t_start = t_start
        self.t_end = t_end

        his_st = np.loadtxt(text_path, delimiter=",")
        self.his_s = his_st[:, :2]
        self.his_t = his_st[:, 2]

        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)
        self.his_t = self.his_t[idx]
        self.his_s = self.his_s[idx]

    def dataset(self, lookback=10, lookahead=1, split=None):
        """
        Optional helper retained from the original interface. Only used if you
        need TensorDataset windows; training in this repo does not use it.
        """
        import torch
        from torch.utils.data import TensorDataset
        from sklearn.preprocessing import MinMaxScaler

        if self.dist_only:
            temp = np.sum(np.square(np.diff(self.his_s, axis=0)), axis=1)
            dist = np.expand_dims(np.append(0, np.sqrt(temp)), 1)
            st_data = np.hstack((dist, np.expand_dims((self.his_t), 1)))
        else:
            st_data = np.hstack((self.his_s, np.expand_dims((self.his_t), 1)))

        st_data[:, -1][1:] = np.diff(st_data[:, -1])
        st_data[:, -1][0] = 0

        if split is None:
            split = [8, 1, 1]
        split = np.asarray(split, dtype=np.float32)
        split = split / np.sum(split)

        length = len(st_data) - lookback - lookahead

        self.st_scaler = MinMaxScaler()
        self.st_scaler.fit(st_data)
        st_data = self.st_scaler.transform(st_data)

        num_features = 2 if self.dist_only else 3
        st_input = np.zeros((length, lookback, num_features))
        st_label = np.zeros((length, lookahead, num_features))

        for i in range(length):
            st_input[i] = st_data[i : i + lookback]
            st_label[i] = st_data[i + lookback : i + lookback + lookahead]

        train_size = int(split[0] * length)
        test_size = int(split[2] * length)

        self.train = TensorDataset(
            torch.Tensor(st_input[:train_size]), torch.Tensor(st_label[:train_size])
        )
        self.val = TensorDataset(
            torch.Tensor(st_input[train_size:-test_size]),
            torch.Tensor(st_label[train_size:-test_size]),
        )
        self.test = TensorDataset(
            torch.Tensor(st_input[-test_size:]), torch.Tensor(st_label[-test_size:])
        )

        print("Finished.")


class InhomogeneousPoissonSyntheticDataset(SyntheticDataset):
    """
    Class-based inhomogeneous Poisson generator compatible with SyntheticDataset.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        spatial_bounds: Tuple[float, float] = (-5.0, 5.0),
        base_rate: float = 2.0,
        covariate_dim: int = 1,
        seed: int = 42,
        covariate_fn=None,
        dist_only: bool = False,
    ):
        super().__init__(dist_only=dist_only)
        self.spatial_dim = spatial_dim
        self.spatial_bounds = spatial_bounds
        self.base_rate = base_rate
        self.covariate_dim = int(covariate_dim)
        self.seed = seed
        self.covariate_fn = covariate_fn or self._default_covariate_fn

    def _default_covariate_fn(self, t, s):
        base = [
            np.sin(s[0]) + np.cos(t),
            np.cos(s[1] if len(s) > 1 else s[0]) + np.sin(0.5 * t),
            np.sin(s[0] + t),
            np.cos((s[1] if len(s) > 1 else s[0]) - t),
        ]
        if self.covariate_dim <= len(base):
            vals = base[: self.covariate_dim]
        else:
            vals = base + [np.sin((k + 1) * t) for k in range(self.covariate_dim - len(base))]
        return np.asarray(vals, dtype=np.float32)

    def lamb_st(self, mu, his_s, his_t, s, t):
        # Inhomogeneous Poisson here is history-independent.
        x = self.covariate_fn(t, np.asarray(s).reshape(-1))
        return float(self.base_rate * np.exp(np.clip(x.sum(), -5, 5)))

    def _sample_single_sequence(
        self, rng: np.random.RandomState, t_start: float, t_end: float
    ) -> Dict:
        times = []
        locs = []
        covs = []
        t = t_start
        sb = self.spatial_bounds
        vol = (sb[1] - sb[0]) ** self.spatial_dim
        lambda_bar = self.base_rate * np.exp(0.5)

        while t < t_end:
            dt = rng.exponential(1.0 / (lambda_bar * vol))
            t = t + dt
            if t >= t_end:
                break

            s = rng.uniform(sb[0], sb[1], size=self.spatial_dim)
            x = self.covariate_fn(t, s)
            lam = self.base_rate * np.exp(np.clip(x.sum(), -5, 5))
            if rng.uniform() < lam / lambda_bar:
                times.append(t)
                locs.append(s.astype(np.float32))
                covs.append(x)

        return {
            "times": np.array(times, dtype=np.float32),
            "locations": np.array(locs, dtype=np.float32).reshape(-1, self.spatial_dim),
            "field_covariates": (
                np.array(covs, dtype=np.float32).reshape(-1, len(covs[0]))
                if covs
                else np.zeros((0, self.covariate_dim), dtype=np.float32)
            ),
        }

    def generate(self, t_start=0.0, t_end=5.0):
        self.t_start = t_start
        self.t_end = t_end
        seq = self._sample_single_sequence(np.random.RandomState(self.seed), t_start, t_end)
        self.his_t = seq["times"]
        self.his_s = seq["locations"]
        if len(self.his_t) > 0 and abs(self.his_t[0] - t_start) < eps:
            self.his_t[0] -= eps
        return seq

    def generate_sequences(
        self, n_sequences: int = 100, t_start: float = 0.0, t_end: float = 5.0
    ) -> List[Dict]:
        print(
            "Generating inhomogeneous Poisson sequences via SyntheticDataset class, "
            f"seed = {self.seed}, covariate_fn = {self.covariate_fn}"
        )
        sequences = []
        for i in range(n_sequences):
            rng = np.random.RandomState(self.seed + i)
            sequences.append(self._sample_single_sequence(rng, t_start, t_end))
        return sequences


class STHPDataset(SyntheticDataset):
    """
    Simulate a spatio-temporal Hawkes process with Gaussian spatial kernels.
    """

    def __init__(
        self,
        s_mu,
        g0_cov,
        g2_cov,
        alpha,
        beta,
        mu,
        max_history: int = 100,
        dist_only: bool = False,
        seed: int = 42,
        covariate_fn=None,
    ):
        super().__init__(dist_only=dist_only)
        self.s_mu = np.asarray(s_mu, dtype=np.float64)
        self.g0_cov = np.asarray(g0_cov, dtype=np.float64)
        self.g2_cov = np.asarray(g2_cov, dtype=np.float64)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.mu = float(mu)
        self.max_history = int(max_history)
        self.seed = int(seed)
        self.covariate_fn = covariate_fn or self._default_covariate_fn

        self.g0_ic = np.linalg.inv(self.g0_cov)
        self.g0_sidc = 1 / np.sqrt(np.linalg.det(self.g0_cov))
        self.g2_ic = np.linalg.inv(self.g2_cov)
        self.g2_sidc = 1 / np.sqrt(np.linalg.det(self.g2_cov))

    def _default_covariate_fn(self, t, s):
        # Keep default aligned with the inhomogeneous generator for easy comparison.
        return np.array([np.sin(s[0]) + np.cos(t)], dtype=np.float32)

    def trunc(self, his):
        if len(his) > self.max_history:
            return his[-self.max_history :]
        return his

    def lamb_st(self, s, t):
        his_t = self.trunc(self.his_t[self.his_t < t])
        his_s = self.trunc(self.his_s[self.his_t < t])
        return self.mu * self.g0(s, self.s_mu, self.g0_sidc, self.g0_ic) + np.sum(
            self.g1(t, his_t, self.alpha, self.beta)
            * self.g2(s, his_s, self.g2_sidc, self.g2_ic)
        )

    def predict_next(self, i):
        from scipy.integrate import quad, quad_vec

        ti = self.his_t[i]
        c = self.alpha * np.sum(np.exp(-self.beta * (ti - self.trunc(self.his_t[: (i + 1)]))))

        int_lamb = lambda t: np.exp(
            c / self.beta * (np.exp(-self.beta * (t - ti)) - 1) - self.mu * (t - ti)
        )
        time_pdf = lambda t: t * (self.mu + c * np.exp(-self.beta * (t - ti))) * int_lamb(t)
        space_pdf = lambda t: (
            self.mu * self.s_mu
            + self.alpha
            * np.sum(
                np.exp(-self.beta * (t - self.trunc(self.his_t[: (i + 1)])))[:, np.newaxis]
                * self.trunc(self.his_s[: (i + 1)]),
                axis=0,
            )
        ) * int_lamb(t)

        return quad_vec(space_pdf, ti, np.inf)[0], quad(time_pdf, ti, np.inf)[0]

    def generate_offsprings(self, t_i, s_i, verbose: bool = False):
        t = t_i
        count = 0
        while True:
            m = self.alpha * np.exp(-self.beta * (t - t_i))
            if m <= 0:
                break
            t += np.random.exponential(scale=1 / m)
            if t > self.t_end:
                break
            lamb = self.alpha * np.exp(-self.beta * (t - t_i))
            if lamb / m >= np.random.uniform():
                s = np.random.multivariate_normal(np.asarray(s_i).reshape(-1), self.g2_cov)
                s = np.expand_dims(s.astype("float64"), 0)
                count += 1
                n = len(self.his_t[self.his_t < t])
                self.his_s = np.insert(self.his_s, n, s, axis=0)
                self.his_t = np.insert(self.his_t, n, t)
        if verbose:
            print(f"{count} offsprings generated for event at {t_i}")

    def generate(self, t_start, t_end, verbose: bool = False):
        self.t_start = t_start
        self.t_end = t_end
        t = t_start
        self.his_s = np.zeros((0, 2))
        self.his_t = np.array([])

        count = 0
        while True:
            count += 1
            t += np.random.exponential(scale=1 / self.mu)
            if t > t_end:
                break
            s = np.random.multivariate_normal(self.s_mu, self.g0_cov)
            s = np.expand_dims(s.astype("float64"), 0)
            self.his_s = np.vstack((self.his_s, s))
            self.his_t = np.append(self.his_t, t)

        if verbose:
            print(f"{count} 0-generation events generated")

        if len(self.his_t) == 0:
            return

        t = t_start
        n = 0
        while True:
            self.generate_offsprings(self.his_t[n], self.his_s[n], verbose)
            try:
                n = next(x[0] for x in enumerate(self.his_t) if x[1] > t)
                t = self.his_t[n]
            except StopIteration:
                break

    def nll(self, alpha, beta, mu, g0_cov, g2_cov):
        g0_ic = np.linalg.inv(g0_cov)
        g0_sidc = 1 / np.sqrt(np.linalg.det(g0_cov))
        g2_ic = np.linalg.inv(g2_cov)
        g2_sidc = 1 / np.sqrt(np.linalg.det(g2_cov))
        s_mu = np.mean(self.his_s, axis=0)

        term_1 = 0
        for i in range(1, len(self.his_s)):
            lamb = mu * self.g0(self.his_s[i], s_mu, g0_sidc, g0_ic) + np.sum(
                self.g1(self.his_t[i], self.trunc(self.his_t[:i]), alpha, beta)
                * self.g2(self.his_s[i], self.trunc(self.his_s[:i]), g2_sidc, g2_ic)
            )
            term_1 -= np.log(lamb)

        term_2 = mu * (self.t_end - self.t_start)
        term_2 -= alpha / beta * np.sum((np.exp(-beta * (self.t_end - self.his_t)) - 1))
        return term_1 + term_2

    def mle(self):
        from scipy.optimize import minimize

        xinit = [2, 2, 2, 2, 0, 2, 2, 0, 2]
        bnds = [
            (0, None),
            (eps, None),
            (0, None),
            (eps, None),
            (0, None),
            (eps, None),
            (eps, None),
            (0, None),
            (eps, None),
        ]
        cons = [
            {"type": "ineq", "fun": lambda x: x[3] * x[5] - x[4] * x[4]},
            {"type": "ineq", "fun": lambda x: x[6] * x[8] - x[7] * x[7]},
        ]
        obj_fun = lambda x: self.nll(
            x[0],
            x[1],
            x[2],
            np.array([[x[3], x[4]], [x[4], x[5]]]),
            np.array([[x[6], x[7]], [x[7], x[8]]]),
        )
        return minimize(obj_fun, x0=xinit, bounds=bnds, constraints=cons)

    def plot_intensity(self, s=None, t_start=None, t_end=None, color="blue"):
        import matplotlib.pyplot as plt

        if s is None:
            s = self.s_mu[np.newaxis, :]
        if t_start is None:
            t_start = self.t_start
        if t_end is None:
            t_end = self.t_end

        width, _ = plt.figaspect(0.1)
        _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(width, width / 2))

        x = np.arange(1001) / 1000
        x = eps + t_start + x * (t_end - t_start)
        y = [self.lamb_st(s, t) for t in x]
        ax1.plot(x, y, color, label=f"Intensity at ({s[0,0]}, {s[0,1]})")
        ax1.set_xlim([t_start, t_end])
        ax1.legend()

        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)
        ax2.stem(
            self.his_t[idx],
            np.sqrt(np.sum(np.square(self.his_s[idx] - s), axis=1)),
            use_line_collection=True,
            label=f"Events(height = dist to ({s[0,0]}, {s[0,1]}))",
        )
        ax2.set_xlim([t_start, t_end])
        ax2.invert_yaxis()
        ax2.legend()

    def generate_sequences(
        self, n_sequences: int = 100, t_start: float = 0.0, t_end: float = 5.0
    ) -> List[Dict]:
        sequences = []
        print(
            f"Generating Hawkes sequences via STHPDataset class, seed = {self.seed}, "
            f"alpha = {self.alpha}, beta = {self.beta}, mu = {self.mu}"
        )
        for i in range(n_sequences):
            np.random.seed(self.seed + i)
            self.generate(t_start=t_start, t_end=t_end, verbose=False)
            if len(self.his_t) > 0:
                covs = np.asarray(
                    [self.covariate_fn(t, s) for t, s in zip(self.his_t, self.his_s)],
                    dtype=np.float32,
                )
            else:
                covs = np.zeros((0, 1), dtype=np.float32)
            seq = {
                "times": np.asarray(self.his_t, dtype=np.float32),
                "locations": np.asarray(self.his_s, dtype=np.float32).reshape(-1, 2),
                "field_covariates": covs.reshape(-1, covs.shape[-1] if covs.ndim > 1 else 1),
            }
            sequences.append(seq)
        return sequences


# ============================================================================
# Marked Hawkes process generator
# ============================================================================

def generate_marked_hawkes_stpp(
    n_sequences: int = 100,
    T: float = 10.0,
    spatial_bounds: Tuple[float, float] = (-5.0, 5.0),
    spatial_dim: int = 2,
    n_marks: int = 3,
    mu: float = 1.0,
    alpha: float = 0.5,
    beta: float = 1.0,
    sigma_s: float = 1.0,
    excitation_matrix: Optional[np.ndarray] = None,
    background_probs: Optional[np.ndarray] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate sequences from a marked spatiotemporal Hawkes process.

    The ground process (t, s) is a standard Hawkes process. The mark k
    for each event is drawn from a conditional categorical distribution:

      Background event:  k ~ Categorical(background_probs)
      Triggered by event j with mark k_j:
                         k ~ Categorical(excitation_matrix[k_j])

    This produces cross-excitation patterns where marks cluster in
    predictable ways — useful for testing mark decoder generalisation.

    Args:
        n_sequences: Number of sequences to generate.
        T: Time horizon.
        spatial_bounds: (min, max) for each spatial dimension.
        spatial_dim: Number of spatial dimensions.
        n_marks: Number of discrete mark types (K).
        mu: Background rate.
        alpha: Excitation magnitude (must be < 1 for stability).
        beta: Temporal decay rate.
        sigma_s: Spatial influence kernel bandwidth.
        excitation_matrix: (K, K) row-stochastic matrix. excitation_matrix[k, j]
            is the probability that a child triggered by a type-k parent gets
            type j. Defaults to a slightly-biased-toward-self matrix.
        background_probs: (K,) probability vector for background events.
            Defaults to uniform.
        seed: Random seed.

    Returns:
        List of dicts, each with keys:
            'times':     (N,) float32
            'locations': (N, d) float32
            'marks':     (N,) int64 — mark indices in [0, K)
    """
    rng = np.random.RandomState(seed)

    # Default excitation matrix: self-exciting with small cross-excitation
    if excitation_matrix is None:
        diag_weight = 0.6
        off_diag_weight = (1.0 - diag_weight) / max(n_marks - 1, 1)
        excitation_matrix = np.full((n_marks, n_marks), off_diag_weight)
        np.fill_diagonal(excitation_matrix, diag_weight)

    excitation_matrix = np.asarray(excitation_matrix, dtype=np.float64)
    # Row-normalise to ensure valid distribution
    row_sums = excitation_matrix.sum(axis=1, keepdims=True)
    excitation_matrix = excitation_matrix / np.maximum(row_sums, 1e-12)

    if background_probs is None:
        background_probs = np.ones(n_marks, dtype=np.float64) / n_marks
    background_probs = np.asarray(background_probs, dtype=np.float64)
    background_probs = background_probs / background_probs.sum()

    sequences = []
    for _ in range(n_sequences):
        times = []
        locs = []
        marks = []
        parent_marks = []  # mark of the triggering event (None = background)

        t = 0.0
        lambda_bar = mu + 10.0

        while t < T:
            dt = rng.exponential(1.0 / lambda_bar)
            t = t + dt
            if t >= T:
                break

            s_candidate = rng.uniform(spatial_bounds[0], spatial_bounds[1], size=spatial_dim)

            lam = mu
            for ti, si in zip(times, locs):
                temporal = alpha * beta * np.exp(-beta * (t - ti))
                spatial = np.exp(-np.sum((s_candidate - si) ** 2) / (2 * sigma_s ** 2))
                spatial /= (2 * np.pi * sigma_s ** 2) ** (spatial_dim / 2)
                lam += temporal * spatial

            if rng.uniform() < lam / lambda_bar:
                # Determine which event triggered this one (background vs parent)
                # Compute contribution of each past event to lam
                contribs = []
                for ti, si in zip(times, locs):
                    temporal = alpha * beta * np.exp(-beta * (t - ti))
                    spatial = np.exp(-np.sum((s_candidate - si) ** 2) / (2 * sigma_s ** 2))
                    spatial /= (2 * np.pi * sigma_s ** 2) ** (spatial_dim / 2)
                    contribs.append(temporal * spatial)
                total_excitation = sum(contribs)
                bg_prob = mu / (mu + total_excitation + 1e-12)

                if len(contribs) == 0 or rng.uniform() < bg_prob:
                    # Background event
                    k = rng.choice(n_marks, p=background_probs)
                else:
                    # Triggered: sample parent proportional to contributions
                    contribs_arr = np.array(contribs)
                    parent_probs = contribs_arr / (contribs_arr.sum() + 1e-12)
                    parent_idx = rng.choice(len(contribs), p=parent_probs)
                    parent_mark = marks[parent_idx]
                    k = rng.choice(n_marks, p=excitation_matrix[parent_mark])

                times.append(t)
                locs.append(s_candidate)
                marks.append(k)

            lambda_bar = max(lambda_bar, lam * 1.5)

        sequences.append({
            "times": np.array(times, dtype=np.float32),
            "locations": np.array(locs, dtype=np.float32).reshape(-1, spatial_dim),
            "marks": np.array(marks, dtype=np.int64),
        })

    return sequences
