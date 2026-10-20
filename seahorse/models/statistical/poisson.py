"""Analytic continuous-space Poisson controls for the statistical runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .core import Support, area, as_support, calendar_offset, forecast_origin, history_arrays, json_array, sequences


def _cell_indices(locations: np.ndarray, support: Support, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = support
    nx, ny = grid
    ix = np.minimum(((locations[:, 0] - xmin) / (xmax - xmin) * nx).astype(int), nx - 1)
    iy = np.minimum(((locations[:, 1] - ymin) / (ymax - ymin) * ny).astype(int), ny - 1)
    return ix, iy


@dataclass
class PoissonSTPP:
    """Piecewise-uniform spatial Poisson process with optional periodic rate.

    ``rates`` is a total rate per unit time, one value per calendar phase.
    Spatial probabilities are independent of phase.  This is the explicit
    spatial-times-seasonal product model, rather than a claim that it learns a
    time-varying hotspot map.
    """

    support: Support
    grid: tuple[int, int] = (1, 1)
    period: float | None = None
    n_phases: int = 1
    phase_origin: float = 0.0
    alpha: float = 0.5
    rates: np.ndarray | None = None
    probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.support = as_support(self.support)
        if len(self.grid) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in self.grid):
            raise ValueError("grid must contain two positive integers.")
        if not isinstance(self.n_phases, (int, np.integer)) or self.n_phases < 1:
            raise ValueError("n_phases must be a positive integer.")
        if self.period is not None and (not np.isfinite(self.period) or self.period <= 0):
            raise ValueError("period must be finite and positive.")
        if not np.isfinite(self.phase_origin) or not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("phase_origin must be finite and alpha finite and non-negative.")

    @property
    def n_params(self) -> int:
        return self._n_phase + self.grid[0] * self.grid[1] - 1

    @property
    def _n_phase(self) -> int:
        return 1 if self.period is None else int(self.n_phases)

    @property
    def _phase_width(self) -> float:
        return 1.0 if self.period is None else float(self.period) / self._n_phase

    def _phase(self, time: np.ndarray | float) -> np.ndarray:
        if self.period is None:
            return np.zeros(np.asarray(time).shape, dtype=int)
        value = np.asarray(time, dtype=float)
        return np.floor(np.mod(value - self.phase_origin, self.period) / self._phase_width).astype(int) % self._n_phase

    def _cell_area(self) -> float:
        return area(self.support) / (self.grid[0] * self.grid[1])

    def fit(self, train_seqs, *, seed: int, inference: Mapping | None = None) -> dict:
        del seed
        opts = dict(inference or {})
        if opts.pop("method", "mle").lower() not in {"mle", "analytic"}:
            raise ValueError("Poisson presets currently support analytic MLE with optional spatial smoothing.")
        if "alpha" in opts:
            self.alpha = float(opts.pop("alpha"))
        if opts:
            raise ValueError(f"Unknown Poisson inference options: {sorted(opts)}")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        data = sequences(train_seqs, self.support)
        phases = self._n_phase
        counts = np.zeros(phases)
        exposure = np.zeros(phases)
        cells = np.zeros(self.grid)
        for seq in data:
            counts += np.bincount(self._phase(seq.times + seq.calendar_offset), minlength=phases)
            exposure += self._phase_exposure(seq.t0 + seq.calendar_offset, seq.t1 + seq.calendar_offset)
            ix, iy = _cell_indices(seq.locations, self.support, self.grid)
            np.add.at(cells, (ix, iy), 1)
        if np.any(exposure <= 0):
            raise ValueError("Every declared seasonal phase needs positive training exposure.")
        self.rates = counts / exposure
        denominator = cells.sum() + self.alpha * cells.size
        self.probabilities = ((cells.reshape(-1) + self.alpha) / denominator
                              if denominator > 0 else np.full(cells.size, 1.0 / cells.size))
        method = "analytic_rate_mle_dirichlet_spatial_mean" if self.alpha > 0 and cells.size > 1 else "analytic_mle"
        return {"status": "converged", "method": method, "n_sequences": len(data),
                "n_events": int(counts.sum()), "exposure_by_phase": json_array(exposure),
                "spatial_distribution_identified": bool(cells.sum() > 0),
                "calendar_phase_policy": "window_start_plus_relative_days_if_present",
                "rates": json_array(self.rates), "n_params": self.n_params}

    def _require_fit(self) -> None:
        if self.rates is None or self.probabilities is None:
            raise RuntimeError("Model must be fit or restored before scoring.")

    def _phase_exposure(self, start: float, end: float) -> np.ndarray:
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError("Rate exposure requires finite ordered endpoints.")
        if self.period is None:
            return np.asarray([end - start])
        width = self._phase_width
        edges = np.arange(self._n_phase) * width
        def primitive(t):
            cycles, remainder = divmod(t - self.phase_origin, self.period)
            return cycles * width + np.clip(remainder - edges, 0.0, width)
        return np.maximum(primitive(end) - primitive(start), 0.0)

    def _rate_integral(self, start: float, end: float) -> float:
        return float(self._phase_exposure(start, end) @ self.rates)

    def _intensity(self, time: float, location: np.ndarray) -> float:
        self._require_fit()
        ix, iy = _cell_indices(np.asarray(location, dtype=float).reshape(1, 2), self.support, self.grid)
        cell = int(ix[0]) * self.grid[1] + int(iy[0])
        return float(self.rates[int(self._phase(time))] * self.probabilities[cell] / self._cell_area())

    def next_event_nll(self, history: Mapping, next_time: float, next_location: np.ndarray) -> float:
        times, _ = history_arrays(history, self.support)
        if not len(times):
            raise ValueError("Walk-forward scoring needs at least one context event.")
        next_time = float(next_time)
        origin = forecast_origin(history, times)
        point = np.asarray(next_location, dtype=float).reshape(1, 2)
        if not np.isfinite(next_time) or next_time <= origin or not bool(np.isfinite(point).all()):
            raise ValueError("Target must be finite and strictly after the history prefix.")
        from .core import contains
        if not bool(contains(self.support, point)[0]):
            return float("inf")
        offset = calendar_offset(history)
        intensity = self._intensity(next_time + offset, point[0])
        if intensity <= 0:
            return float("inf")
        return float(-math.log(intensity) + self._rate_integral(origin + offset, next_time + offset))

    def full_window_nll(self, seq: Mapping) -> float:
        data = sequences([seq], self.support)[0]
        self._require_fit()
        value = self._rate_integral(data.t0 + data.calendar_offset, data.t1 + data.calendar_offset)
        for t, x in zip(data.times, data.locations):
            intensity = self._intensity(float(t) + data.calendar_offset, x)
            if intensity <= 0:
                return float("inf")
            value -= math.log(intensity)
        return float(value)

    def _sample_one(self, history: Mapping, rng: np.random.Generator) -> tuple[float, np.ndarray]:
        times, _ = history_arrays(history, self.support)
        self._require_fit()
        current = forecast_origin(history, times)
        offset = calendar_offset(history)
        max_rate = float(np.max(self.rates))
        if max_rate <= 0:
            raise RuntimeError("No future event exists under zero Poisson intensity.")
        while True:
            current += float(rng.exponential(1.0 / max_rate))
            if rng.random() <= self.rates[int(self._phase(current + offset))] / max_rate:
                probs = self.probabilities
                cell = int(rng.choice(len(probs), p=probs))
                ix, iy = divmod(cell, self.grid[1])
                xmin, xmax, ymin, ymax = self.support
                x = xmin + (ix + rng.random()) * (xmax - xmin) / self.grid[0]
                y = ymin + (iy + rng.random()) * (ymax - ymin) / self.grid[1]
                return current, np.array([x, y])

    def sample_next_event(self, history: Mapping, *, n_samples: int, rng: np.random.Generator):
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        draws = [self._sample_one(history, rng) for _ in range(n_samples)]
        return np.asarray([x[0] for x in draws]), np.asarray([x[1] for x in draws])

    def sample_future(self, history: Mapping, *, horizon: float, rng: np.random.Generator, max_events: int = 10000) -> dict:
        times, locations = history_arrays(history, self.support)
        if not np.isfinite(horizon) or horizon <= 0 or max_events < 1:
            raise ValueError("horizon must be positive.")
        self._require_fit()
        start = forecast_origin(history, times)
        if np.max(self.rates) == 0:
            return {"times": [], "locations": []}
        out_t: list[float] = []
        out_x: list[np.ndarray] = []
        state = {**history, "times": times.copy(), "locations": locations.copy(), "t1": start}
        while len(out_t) < max_events:
            t, x = self._sample_one(state, rng)
            if t >= start + horizon:
                break
            out_t.append(t)
            out_x.append(x)
            state = {**state, "times": np.append(state["times"], t), "locations": np.vstack([state["locations"], x]), "t1": t}
        if len(out_t) == max_events:
            raise RuntimeError("max_events reached; forecast is not silently censored.")
        return {"times": json_array(np.asarray(out_t)), "locations": json_array(np.asarray(out_x).reshape(-1, 2))}

    def to_state(self) -> dict:
        self._require_fit()
        return {"kind": "poisson", "support": list(self.support), "grid": list(self.grid), "period": self.period, "n_phases": self.n_phases,
                "phase_origin": self.phase_origin, "alpha": self.alpha, "rates": json_array(self.rates),
                "probabilities": json_array(self.probabilities),
                "metadata": dict(getattr(self, "metadata", {}))}

    @classmethod
    def from_state(cls, state: Mapping) -> "PoissonSTPP":
        model = cls(support=as_support(state["support"]), grid=tuple(state["grid"]), period=state.get("period"), n_phases=int(state.get("n_phases", len(state["rates"]))),
                    phase_origin=float(state.get("phase_origin", 0.0)), alpha=float(state.get("alpha", 0.5)))
        model.rates = np.asarray(state["rates"], dtype=float)
        model.probabilities = np.asarray(state["probabilities"], dtype=float)
        model.metadata = dict(state.get("metadata", {}))
        if (model.rates.shape != (model._n_phase,) or not np.isfinite(model.rates).all()
                or np.any(model.rates < 0)):
            raise ValueError("Stored rates must be finite, non-negative and match the declared phases.")
        if (model.probabilities.shape != (model.grid[0] * model.grid[1],)
                or not np.isfinite(model.probabilities).all()
                or np.any(model.probabilities < 0)
                or not np.isclose(model.probabilities.sum(), 1.0, rtol=1e-12, atol=1e-12)):
            raise ValueError("Stored spatial probabilities must be a normalized distribution on the declared grid.")
        return model
