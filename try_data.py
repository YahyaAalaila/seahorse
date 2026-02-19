#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def smoothstep(u: float) -> float:
    """C^1 smooth interpolation on [0,1]."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class CornerSwitchHotspot:
    """
    Intensity:
      lambda(t, x) = base_rate + hotspot_weight * A(t) * exp(-||x - c(t)||^2/(2*sigma^2))

    c(t): corner A -> (move) -> corner B, with local jitter.
    A(t): positive amplitude modulation, with smooth periodic + mild random wiggle.
    """

    def __init__(
        self,
        T=5.0,
        a=5.0,
        b=5.0,
        t1=1.8,
        t2=2.4,
        sigma=0.75,
        base_rate=0.05,
        hotspot_weight=6.0,
        jitter_radius=0.45,
        jitter_f1=0.9,
        jitter_f2=1.3,
        amp0=0.0,
        amp1=0.45,
        amp_noise=0.12,
        seed=42,
    ):
        self.T = float(T)
        self.a = float(a)
        self.b = float(b)
        self.t1 = float(t1)
        self.t2 = float(t2)
        assert 0.0 <= self.t1 <= self.t2 <= self.T

        self.sigma = float(sigma)
        self.base_rate = float(base_rate)
        self.hotspot_weight = float(hotspot_weight)

        self.jitter_radius = float(jitter_radius)
        self.jitter_f1 = float(jitter_f1)
        self.jitter_f2 = float(jitter_f2)

        self.amp0 = float(amp0)
        self.amp1 = float(amp1)
        self.amp_noise = float(amp_noise)

        self.rng = np.random.RandomState(seed)
        self.seed = seed

        # corners
        self.corner_A = np.array([-self.a, -self.b], dtype=np.float64)
        self.corner_B = np.array([+self.a, +self.b], dtype=np.float64)

        # fixed phases for jitter and amplitude
        self.phi1 = self.rng.uniform(0, 2 * np.pi)
        self.phi2 = self.rng.uniform(0, 2 * np.pi)
        self.phiA = self.rng.uniform(0, 2 * np.pi)

        # amplitude noise as a smooth random process: pre-sample knots then interpolate
        self._noise_knots_t = np.linspace(0.0, self.T, 16)
        self._noise_knots_v = self.rng.normal(0.0, 1.0, size=self._noise_knots_t.shape[0])

        # conservative global upper bound for thinning (safe > true max)
        # A(t) = exp(amp0 + amp1*sin + amp_noise*noise_interp)
        # Use |sin|<=1 and noise_interp roughly in [-3,3] w.h.p.; pad further.
        A_max = np.exp(self.amp0 + abs(self.amp1) + 4.0 * abs(self.amp_noise))
        self.lambda_bar = self.base_rate + self.hotspot_weight * A_max

    def _noise_interp(self, t: float) -> float:
        # linear interpolation of knot noise
        return np.interp(t, self._noise_knots_t, self._noise_knots_v)

    def amplitude(self, t: float) -> float:
        # smooth periodic + smooth random
        # keep positivity by exponentiating
        val = self.amp0 + self.amp1 * np.sin(2 * np.pi * 0.35 * t + self.phiA) + self.amp_noise * self._noise_interp(t)
        return float(np.exp(val))

    def center(self, t: float) -> np.ndarray:
        # piecewise corner switching + smooth move + jitter
        if t <= self.t1:
            base = self.corner_A
        elif t <= self.t2:
            u = (t - self.t1) / max(self.t2 - self.t1, 1e-9)
            s = smoothstep(u)
            base = (1.0 - s) * self.corner_A + s * self.corner_B
        else:
            base = self.corner_B

        # local jitter
        jx = np.sin(2 * np.pi * self.jitter_f1 * t + self.phi1)
        jy = np.cos(2 * np.pi * self.jitter_f2 * t + self.phi2)
        jitter = self.jitter_radius * np.array([jx, jy], dtype=np.float64)

        c = base + jitter

        # clip to remain inside bounds (avoid drifting outside domain)
        c[0] = np.clip(c[0], -self.a, self.a)
        c[1] = np.clip(c[1], -self.b, self.b)
        return c

    def intensity(self, t: float, xy: np.ndarray) -> float:
        # xy shape (2,)
        c = self.center(t)
        A = self.amplitude(t)
        d2 = np.sum((xy - c) ** 2)
        hotspot = np.exp(-d2 / (2.0 * self.sigma ** 2))
        lam = self.base_rate + self.hotspot_weight * A * hotspot
        return float(lam)


def simulate_stpp_thinning(model: CornerSwitchHotspot, n_sequences=1):
    """
    Simulate an inhomogeneous Poisson STPP on [0,T] x [-a,a]x[-b,b]
    using standard space-time thinning with global bound lambda_bar.

    Returns list of dicts: {times: (n,), locations: (n,2)}
    """
    rng = model.rng
    T = model.T
    area = (2 * model.a) * (2 * model.b)
    lam_bar = model.lambda_bar

    sequences = []
    for _ in range(n_sequences):
        t = 0.0
        times = []
        locs = []

        while True:
            # propose next time using homogeneous rate lam_bar * area
            dt = rng.exponential(1.0 / max(lam_bar * area, 1e-12))
            t += dt
            if t >= T:
                break

            # propose location uniformly
            x = rng.uniform(-model.a, model.a)
            y = rng.uniform(-model.b, model.b)
            xy = np.array([x, y], dtype=np.float64)

            lam = model.intensity(t, xy)
            if rng.uniform() < lam / lam_bar:
                times.append(t)
                locs.append(xy.astype(np.float32))

        sequences.append(
            {
                "times": np.array(times, dtype=np.float32),
                "locations": np.array(locs, dtype=np.float32).reshape(-1, 2),
            }
        )
    return sequences


def make_gif(model: CornerSwitchHotspot, seq, out_path="corner_switch_hotspot.gif",
             grid_n=120, n_frames=80, dpi=110):
    """
    Create a GIF showing intensity heatmap and event points up to time t.
    """
    T = model.T
    xs = np.linspace(-model.a, model.a, grid_n)
    ys = np.linspace(-model.b, model.b, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    grid_xy = np.stack([X, Y], axis=-1)  # (grid_n, grid_n, 2)

    times = seq["times"]
    locs = seq["locations"]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.set_title("Corner-switching hotspot intensity + events")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-model.a, model.a)
    ax.set_ylim(-model.b, model.b)

    # initial frame at t=0
    t0 = 0.0
    lam0 = np.zeros((grid_n, grid_n), dtype=np.float64)
    for i in range(grid_n):
        for j in range(grid_n):
            lam0[j, i] = model.intensity(t0, grid_xy[j, i])

    im = ax.imshow(
        lam0,
        extent=[-model.a, model.a, -model.b, model.b],
        origin="lower",
        cmap="viridis",
        vmin=model.base_rate,
        vmax=min(model.lambda_bar, np.max(lam0) * 1.2 + 1e-6),
        aspect="equal",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("lambda(t,x)")

    (sc,) = ax.plot([], [], "r.", markersize=3, alpha=0.8)
    (center_pt,) = ax.plot([], [], "wo", markersize=6, markeredgecolor="k", markeredgewidth=0.8)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    frame_ts = np.linspace(0.0, T, n_frames)

    def update(frame_idx):
        t = float(frame_ts[frame_idx])

        # compute intensity grid (vectorize lightly)
        lam = np.zeros((grid_n, grid_n), dtype=np.float64)
        for i in range(grid_n):
            for j in range(grid_n):
                lam[j, i] = model.intensity(t, grid_xy[j, i])

        im.set_data(lam)
        im.set_clim(vmin=model.base_rate, vmax=max(model.base_rate + 1e-6, min(model.lambda_bar, np.max(lam) * 1.15)))

        # events up to t
        if times.size > 0:
            mask = times <= t
            pts = locs[mask]
            if pts.shape[0] > 0:
                sc.set_data(pts[:, 0], pts[:, 1])
            else:
                sc.set_data([], [])
        else:
            sc.set_data([], [])

        # show current center
        c = model.center(t)
        center_pt.set_data([c[0]], [c[1]])

        time_text.set_text(f"t = {t:.3f}   | events shown: {int(np.sum(times <= t))}")
        return im, sc, center_pt, time_text

    ani = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=12), dpi=dpi)
    plt.close(fig)


def main():
    # ---- user-facing knobs ----
    T = 5.0
    a = 5.0
    b = 5.0
    t1_frac = 0.32
    t2_frac = 0.46
    t1 = t1_frac * T
    t2 = t2_frac * T

    model = CornerSwitchHotspot(
        T=T, a=a, b=b,
        t1=t1, t2=t2,
        sigma=0.7,
        base_rate=0.3,
        hotspot_weight=7.0,
        jitter_radius=0.55,   # local motion around corner
        jitter_f1=0.8,
        jitter_f2=1.25,
        amp0=0.0,
        amp1=0.55,            # smooth magnitude variation
        amp_noise=0.10,       # mild random variation
        seed=7,
    )

    seqs = simulate_stpp_thinning(model, n_sequences=1)
    seq = seqs[0]
    print(f"Generated {seq['times'].shape[0]} events. lambda_bar={model.lambda_bar:.3f}")

    out_gif = "corner_switch_hotspot.gif"
    make_gif(model, seq, out_path=out_gif, grid_n=130, n_frames=90, dpi=120)
    print(f"Saved GIF to: {out_gif}")

    # quick static sanity plots at 3 times
    ts = [0.2, 0.5 * (t1 + t2), 0.9 * T]
    for t in ts:
        c = model.center(t)
        A = model.amplitude(t)
        print(f"t={t:.3f} center={c} amplitude={A:.3f}")


if __name__ == "__main__":
    main()
