"""Synthetic data generation and reference helpers.

The maintained live CLI in ``unified_stpp.__main__`` consumes prebuilt JSONL
splits for ``fit``, ``evaluate``, ``bench``, and ``tune``. Those splits are
currently produced offline by scripts such as ``scripts/gen_sthp_splits.py``,
which depend on this module's STHP generator and plotting/reference helpers.

Older or non-STHP experimental paths are reference-only and may remain archived
elsewhere in the repo; this module is the maintained live implementation.
"""

import abc
import numpy as np
from typing import List, Dict, Optional, Tuple

import plotly







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

    def lamb_st_grid(self, xs: np.ndarray, ys: np.ndarray, t: float) -> np.ndarray:
        """Evaluate λ*(t, s | H) on a 2-D spatial grid in one vectorised pass.

        Parameters
        ----------
        xs : (Nx,)  x-axis sample points
        ys : (Ny,)  y-axis sample points
        t  : scalar query time

        Returns
        -------
        grid : (Nx, Ny)  conditional intensity at each grid cell
        """
        # Build flat grid: (Nx*Ny, 2)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")          # (Nx, Ny) each
        s_flat = np.column_stack([xx.ravel(), yy.ravel()])    # (N, 2)

        # Background term: vectorised over N — g0 calls g2 with s_mu (1,2)
        bg = self.mu * self.g0(s_flat, self.s_mu, self.g0_sidc, self.g0_ic)  # (N,)

        # History up to t
        his_t = self.trunc(self.his_t[self.his_t < t])
        his_s = self.trunc(self.his_s[self.his_t < t])

        if len(his_t) == 0:
            return bg.reshape(len(xs), len(ys))

        # Temporal kernel: (H,)
        g1_vec = self.g1(t, his_t, self.alpha, self.beta)

        # Spatial kernel: (N, H) via broadcast
        delta_s = s_flat[:, np.newaxis, :] - his_s[np.newaxis, :, :]  # (N, H, 2)
        quad = np.einsum("nhi,ij,nhj->nh", delta_s, self.g2_ic, delta_s)  # (N, H)
        g2_mat = 1.0 / (2.0 * np.pi) * self.g2_sidc * np.exp(-quad / 2.0)  # (N, H)

        excitation = (g2_mat * g1_vec[np.newaxis, :]).sum(axis=1)             # (N,)
        return (bg + excitation).reshape(len(xs), len(ys))

    def get_lamb_st(
        self,
        t_start: float,
        t_end: float,
        n_x: int = 50,
        n_y: int = 50,
        n_t: int = 20,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Compute the conditional intensity surface over a space-time grid.

        Spatial bounds are derived from events in [t_start, t_end]; an extra
        20 % margin is added so the boundary region is visible.

        Parameters
        ----------
        t_start, t_end : time range to visualise
        n_x, n_y       : spatial grid resolution (points per axis)
        n_t            : number of time slices

        Returns
        -------
        lambs   : list of n_t arrays each shaped (n_x, n_y)
        x_range : (n_x,)
        y_range : (n_y,)
        t_range : (n_t,)
        """
        from tqdm import tqdm

        idx = np.logical_and(self.his_t >= t_start, self.his_t < t_end)
        his_s_win = self.his_s[idx]
        if len(his_s_win) > 0:
            x_min, y_min = his_s_win.min(axis=0)
            x_max, y_max = his_s_win.max(axis=0)
        else:
            x_min, y_min = self.s_mu - 2.0
            x_max, y_max = self.s_mu + 2.0

        x_range = np.linspace(x_min, x_max, n_x)
        y_range = np.linspace(y_min, y_max, n_y)
        t_range = np.linspace(t_start, t_end, n_t)

        lambs = [
            self.lamb_st_grid(x_range, y_range, float(t))
            for t in tqdm(t_range, desc="intensity grid", leave=False)
        ]
        return lambs, x_range, y_range, t_range

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
# Interactive intensity surface visualisation
# ============================================================================

def plot_intensity_surface(
    lambs: list[np.ndarray],
    x_range: np.ndarray,
    y_range: np.ndarray,
    t_range: np.ndarray,
    *,
    mode: str = "heatmap",
    title: str = "Spatio-temporal Conditional Intensity λ*(t,s|H)",
    colorscale: str = "Viridis",
    save_path: Optional[str] = None,
    show: bool = False,
) -> "plotly.graph_objects.Figure":
    """Build an animated plotly figure of the spatiotemporal intensity surface.

    Parameters
    ----------
    lambs     : sequence of (n_x, n_y) arrays, one per time step
    x_range   : (n_x,) x-axis coordinates
    y_range   : (n_y,) y-axis coordinates
    t_range   : (n_t,) time values
    mode      : ``"heatmap"`` (2-D colour) or ``"surface"`` (3-D mesh)
    title     : figure title
    colorscale: plotly colorscale name (default ``"Viridis"``)
    save_path : if given, write an interactive HTML file to this path
    show      : call ``fig.show()`` (opens browser)

    Returns
    -------
    plotly Figure with a time-slider and play/pause controls
    """
    import plotly.graph_objects as go

    lambs_arr = np.asarray(lambs)            # (n_t, n_x, n_y)
    cmin = float(lambs_arr.min())
    cmax = float(lambs_arr.max())
    trace_kw = dict(x=x_range, y=y_range, colorscale=colorscale,
                    cmin=cmin, cmax=cmax, cauto=False)

    def _trace(z):
        z_T = z.T    # plotly heatmap/surface: rows = y, cols = x
        if mode == "surface":
            return go.Surface(z=z_T, **trace_kw)
        return go.Heatmap(z=z_T, **trace_kw)

    frames = [
        go.Frame(data=_trace(lambs_arr[i]),
                 name=f"{t_range[i]:.2f}")
        for i in range(len(t_range))
    ]
    fig = go.Figure(data=_trace(lambs_arr[0]), frames=frames)

    sliders = [{
        "pad": {"b": 10, "t": 60},
        "len": 0.9, "x": 0.1, "y": 0,
        "steps": [
            {"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate",
                                 "transition": {"duration": 0}}],
             "label": f.name, "method": "animate"}
            for f in frames
        ],
    }]
    fig.update_layout(
        title=title,
        width=700, height=600,
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 200}, "fromcurrent": True}],
                 "label": "▶", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                 "label": "⏸", "method": "animate"},
            ],
            "direction": "left", "type": "buttons",
            "pad": {"r": 10, "t": 70}, "x": 0.1, "y": 0,
        }],
        sliders=sliders,
        xaxis_title="x", yaxis_title="y",
    )
    if mode == "surface":
        fig.update_scenes(
            aspectmode="cube",
            zaxis=dict(title="λ", range=[cmin, cmax], autorange=False),
        )

    if save_path is not None:
        fig.write_html(save_path, include_plotlyjs="cdn")
    if show:
        fig.show()
    return fig


# ============================================================================
# Pinwheel-Hawkes dataset — replicates the PinwheelHawkes benchmark from
# Chen et al. (2021) "Neural Spatio-Temporal Point Processes" (ICLR 2021).
#
# Data generation parameters exactly match the original (toy_datasets.py):
#   num_classes = 10, mu = 0.05/class, alpha circular 0.6, omega = 10, T = 30
# ============================================================================

def _pinwheel_locations(n_per_arm: int, num_arms: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate 2-D pinwheel spatial samples.

    Replicates the ``pinwheel`` function from the original repository
    (toy_datasets.py, MIT license, Facebook Research).

    Args:
        n_per_arm: number of samples per arm.
        num_arms: number of pinwheel arms.
        rng: numpy RandomState for reproducibility.

    Returns:
        locations: (num_arms * n_per_arm, 2)
    """
    radial_std = 0.3
    tangential_std = 0.1
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_arms, endpoint=False)

    features = rng.randn(num_arms * n_per_arm, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_arms), n_per_arm)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    rotations = np.stack([cos_a, -sin_a, sin_a, cos_a], axis=-1).reshape(-1, 2, 2)  # (N, 2, 2)
    return 2.0 * np.einsum("ti,tij->tj", features, rotations)  # (N, 2)


def _generate_mhp_sequence(
    T: float,
    mu: np.ndarray,
    alpha: np.ndarray,
    omega: float,
    rng: np.random.RandomState,
) -> tuple:
    """Ogata thinning for a multivariate Hawkes process.

    Adapted from MHP.py in the original repository (MIT license, Steve Morse).

    Returns:
        event_times: list of float
        event_classes: list of int
    """
    dim = len(mu)
    data = []
    Istar = np.sum(mu)
    s = rng.exponential(1.0 / Istar)
    n0 = rng.choice(dim, p=mu / Istar)
    data.append((s, n0))

    lastrates = mu.copy()
    dec_Istar = False

    while True:
        tj, uj = data[-1]
        if dec_Istar:
            Istar = np.sum(rates)
            dec_Istar = False
        else:
            Istar = np.sum(lastrates) + omega * np.sum(alpha[:, uj])

        s += rng.exponential(1.0 / max(Istar, 1e-10))
        rates = mu + np.exp(-omega * (s - tj)) * (alpha[:, uj].flatten() * omega + lastrates - mu)

        diff = Istar - np.sum(rates)
        probs = np.append(np.maximum(rates, 0.0), max(diff, 0.0))
        probs_sum = probs.sum()
        if probs_sum <= 0:
            break
        n0 = rng.choice(dim + 1, p=probs / probs_sum)

        if n0 < dim:
            data.append((s, n0))
            lastrates = rates.copy()
        else:
            dec_Istar = True

        if s >= T:
            break

    event_times = [t for t, _ in data if t < T]
    event_classes = [c for t, c in data if t < T]
    return event_times, event_classes


def generate_pinwheel_hawkes_stpp(
    n_sequences: int = 2000,
    T: float = 30.0,
    num_arms: int = 10,
    mu_per_arm: float = 0.05,
    alpha_offdiag: float = 0.6,
    omega: float = 10.0,
    seed: int = 13579,
) -> List[Dict]:
    """Generate the PinwheelHawkes benchmark dataset.

    Exactly replicates ``PinwheelHawkes`` from Chen et al. (2021)
    ``toy_datasets.py``:
      - Multivariate Hawkes process with circular excitation matrix
      - Spatial locations drawn from the pinwheel distribution per class
      - 10 arms, mu=0.05, alpha_offdiag=0.6, omega=10, T=30

    Args:
        n_sequences: number of sequences to generate.
        T: time horizon.
        num_arms: number of pinwheel arms / Hawkes classes.
        mu_per_arm: background rate per class.
        alpha_offdiag: off-diagonal excitation strength (circular).
        omega: temporal decay rate.
        seed: random seed for reproducibility (original uses 13579).

    Returns:
        List of dicts with keys ``times`` (N,), ``locations`` (N, 2),
        ``marks`` (N,) containing the arm index of each event.
    """
    mu = np.array([mu_per_arm] * num_arms)
    # Circular excitation: each class excites the next (and wraps around)
    alpha = (
        np.diag([alpha_offdiag] * (num_arms - 1), k=-1)
        + np.diag([alpha_offdiag], k=num_arms - 1)
        + np.diag([0.0] * num_arms, k=0)
    )

    rng = np.random.RandomState(seed)
    sequences = []

    for _ in range(n_sequences):
        event_times, event_classes = _generate_mhp_sequence(T, mu, alpha, omega, rng)
        n = len(event_times)

        if n == 0:
            sequences.append({
                "times": np.array([], dtype=np.float32),
                "locations": np.zeros((0, 2), dtype=np.float32),
                "marks": np.array([], dtype=np.int64),
            })
            continue

        # Generate n spatial samples per arm, then pick the sample for each
        # event's arm (matching the original's generate() function exactly).
        all_locs = _pinwheel_locations(n_per_arm=n, num_arms=num_arms, rng=rng)
        # all_locs shape: (num_arms * n, 2); arm k occupies rows [k*n, (k+1)*n)
        locs = np.zeros((n, 2), dtype=np.float64)
        for i, k in enumerate(event_classes):
            locs[i] = all_locs[k * n + i]

        sequences.append({
            "times": np.array(event_times, dtype=np.float32),
            "locations": locs.astype(np.float32),
            "marks": np.array(event_classes, dtype=np.int64),
        })

    return sequences
