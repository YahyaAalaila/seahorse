"""Adapter around the vendored Hawkes source snapshot.

The mathematical process, parameter transforms, likelihood, source MLE and
simulator live in the vendored Hawkes source directory. This module only maps
Seahorse's independent-window and walk-forward protocol onto those components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
from scipy.special import ndtr, ndtri

from ._hawkes_source.hawkes_package.inference.likelihood import History, SpatioTemporalLogLikelihood, _bind_history
from ._hawkes_source.hawkes_package.inference.models import ProcessModel, spatio_temporal_model
from ._hawkes_source.hawkes_package.spatio_temporal.domains import Rectangle
from ._hawkes_source.hawkes_package.spatio_temporal.process import SpatioTemporalHawkesProcess
from ._hawkes_source.hawkes_package.base import _stalled_message
from .components import BACKGROUND_REGISTRY, INFERENCE_REGISTRY, SPATIAL_KERNEL_REGISTRY, TEMPORAL_KERNEL_REGISTRY, resolve_component
from .core import Support, as_support, contains, forecast_origin, history_arrays, json_array, sequences


def _history(times: np.ndarray, locations: np.ndarray, *, start: float, end: float) -> History:
    """Source History uses points shaped (ndim, n), unlike Seahorse rows."""
    return History.from_events(np.vstack([times, locations.T]), start=float(start), end=float(end))


def _rectangular_gaussian_mass(points: np.ndarray, sigma: float, support: Support) -> np.ndarray:
    """Exact mass of a planar isotropic Gaussian over Seahorse's rectangle.

    The copied source uses deterministic quadrature for arbitrary spatial
    kernels and domains.  The initial Gaussian port is a special case: both
    the domain and kernel admit this product-normal integral.  Keeping it in
    the adapter removes the source rule's *visibility* lower bound on sigma;
    it does not change the vendored source or approximate another kernel.
    """
    locations = np.asarray(points, dtype=float).reshape(-1, 2)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gaussian sigma must be finite and positive.")
    xmin, xmax, ymin, ymax = support
    x_mass = ndtr((xmax - locations[:, 0]) / sigma) - ndtr((xmin - locations[:, 0]) / sigma)
    y_mass = ndtr((ymax - locations[:, 1]) / sigma) - ndtr((ymin - locations[:, 1]) / sigma)
    return np.asarray(x_mass * y_mass, dtype=float)


class _AnalyticGaussianProcess(SpatioTemporalHawkesProcess):
    """Source simulator with the Gaussian rectangle integral substituted exactly.

    The source's thinning loop and MCMC location sampler are retained.  Only
    its spatial quadrature hooks are replaced for the one integral with a
    closed form, so their acceptance/bound identity remains exact in continuous
    space instead of only on quadrature nodes.
    """

    def __init__(self, *args, gaussian_sigma: float, **kwargs) -> None:
        self._gaussian_sigma = float(gaussian_sigma)
        super().__init__(*args, **kwargs)

    def _build_quadrature(self, n_quad: int | None) -> None:
        # No source quadrature is needed: _integrated_intensity and
        # _edge_scales below use exact Gaussian masses.  Preserve the requested
        # value for state/diagnostic compatibility without constructing a rule.
        self.n_quad = 0 if n_quad is None else int(n_quad)
        self._quadrature = None

    def _gaussian_masses(self, locations: np.ndarray) -> np.ndarray:
        bounds = np.asarray(self.domain.bounds, dtype=float)
        return _rectangular_gaussian_mass(
            locations,
            self._gaussian_sigma,
            (float(bounds[0, 0]), float(bounds[0, 1]), float(bounds[1, 0]), float(bounds[1, 1])),
        )

    def _edge_scales(self, count: int) -> np.ndarray:
        if not self._renormalise or count == 0:
            return np.ones(count, dtype=float)
        masses = self._gaussian_masses(self.events[1:, :count].T)
        if not np.all(np.isfinite(masses) & (masses > 0.0)):
            raise RuntimeError("Exact Gaussian edge mass is non-positive or non-finite.")
        return masses

    def _integrated_intensity(self, t: float, bound: bool = False) -> float:
        # ConstantBase is the only background registered for this first port.
        background = self._base_at(np.asarray(self.domain.interior_point), t, bound=bound) * self.domain.volume
        factors = self._temporal_factors(t, bound=bound)
        if factors.size == 0:
            return float(background)
        if self._renormalise:
            return float(background + np.sum(factors))
        masses = self._gaussian_masses(self._past_events(t, inclusive=bound)[1:].T)
        return float(background + np.dot(factors, masses))

    def _draw_location(self, event_time: float) -> np.ndarray:
        """Draw the exact rectangle-truncated Gaussian/background mixture.

        Source Ogata thinning still generates time candidates.  Conditional on
        an accepted time, constant background is uniform and every parent is a
        Gaussian restricted to the declared rectangle; their integrated
        intensity contributions are the mixture weights.
        """
        parents = self._past_events(event_time)[1:].T
        factors = self._temporal_factors(event_time)
        bounds = np.asarray(self.domain.bounds, dtype=float)
        background_weight = float(self._base_at(np.asarray(self.domain.interior_point), event_time) * self.domain.volume)
        if parents.size:
            masses = self._gaussian_masses(parents)
            parent_weights = factors if self._renormalise else factors * masses
        else:
            parent_weights = np.empty(0, dtype=float)
        weights = np.concatenate([[background_weight], np.asarray(parent_weights, dtype=float)])
        total = float(np.sum(weights))
        if not total > 0.0 or not np.isfinite(total):
            raise RuntimeError("Non-positive or non-finite conditional spatial mixture mass.")
        chosen = int(self.rng.choice(weights.size, p=weights / total))
        if chosen == 0:
            return np.asarray(self.rng.uniform(bounds[:, 0], bounds[:, 1]), dtype=float)
        centre = parents[chosen - 1]
        lo = ndtr((bounds[:, 0] - centre) / self._gaussian_sigma)
        hi = ndtr((bounds[:, 1] - centre) / self._gaussian_sigma)
        u = lo + (hi - lo) * self.rng.uniform(size=2)
        # The endpoints have probability zero, but clipping protects inverse
        # CDF evaluation when a floating-point draw rounds to an endpoint.
        return np.asarray(centre + self._gaussian_sigma * ndtri(np.clip(u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)), dtype=float)


class _AnalyticGaussianLikelihood(SpatioTemporalLogLikelihood):
    """Cached source likelihood with exact rectangle masses for Gaussian kernels."""

    def __init__(self, *args, support: Support, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._seahorse_support = support

    def _spatial_masses(self, cache, theta: np.ndarray) -> np.ndarray:
        points = cache.points.T
        masses = _rectangular_gaussian_mass(points, float(theta[self._spatial_slice][0]), self._seahorse_support)
        self.homogeneous_used = False
        scale = max(abs(float(np.mean(masses))) if masses.size else 0.0, np.finfo(float).tiny)
        self.spatial_spread = float(np.ptp(masses)) / scale if masses.size else 0.0
        return masses

    def _edge_scales(self, cache, theta: np.ndarray, n: int) -> np.ndarray | None:
        if not self.components.renormalises or n == 0:
            return None
        masses = _rectangular_gaussian_mass(
            cache.points[:, :n].T, float(theta[self._spatial_slice][0]), self._seahorse_support
        )
        if not np.all(np.isfinite(masses) & (masses > 0.0)):
            raise ValueError("Exact Gaussian edge mass is non-positive or non-finite.")
        return masses


def _source_simulate_one_after(process, start: float, *, horizon: float | None = None) -> bool:
    """One-event extraction from the copied source thinning loop.

    This is the source process's _propagate_until contract, stopped immediately
    after its first accepted event.  It deliberately uses the copied process
    bound, intensity, location sampler, event buffer and RNG rather than a
    second Seahorse simulator. A false return means its next candidate is past
    the optional horizon and is not appended, matching source simulate_until.
    """
    current = float(start)
    while True:
        bound = process._bound_at(current)
        advanced = current + process.rng.exponential() / bound
        if not advanced > current:
            raise RuntimeError(_stalled_message(current, bound, "while drawing one source continuation event"))
        current = advanced
        if horizon is not None and current > horizon:
            return False
        if process.rng.uniform() * bound <= process._integrated_intensity(current):
            process._record(current, process._draw_location(current))
            return True


@dataclass
class _IndependentLikelihood:
    """Explicit adaptation: sum source likelihoods over independent windows."""
    model: object
    likelihoods: list[SpatioTemporalLogLikelihood]
    histories: list[History]
    empty_exposure: float = 0.0

    def total(self, theta, _unused_history) -> float:
        observed = sum(likelihood.total(theta, history) for likelihood, history in zip(self.likelihoods, self.histories, strict=True))
        # ConstantBase is the only registered background in this initial scope.
        # An empty independent window has no event term or excitation, hence its
        # source log likelihood is exactly -mu times area-time exposure.
        return float(observed - float(np.asarray(theta, dtype=float)[0]) * self.empty_exposure)


@dataclass
class _ConditionalExpGaussianLikelihood:
    """H-event conditional composite likelihood for the common benchmark task.

    This caches bounded H-event sufficient arrays before source MLE.  It is not a
    full-window Hawkes likelihood: each target deliberately receives exactly H
    observed predecessors and no earlier excitation.  For the registered
    exponential/Gaussian pair, the source kernel and compensator have closed
    forms, so this avoids constructing an N-by-N geometry cache for a merged
    catalogue while retaining the source parameterization and optimizer.
    """

    model: ProcessModel
    support: Support
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    history_events: int

    def total(self, theta, _unused_history) -> float:
        values = np.asarray(theta, dtype=float).reshape(-1)
        scored = _conditional_exp_gaussian_nll(values, self.chunks, self.support, self.model.components.renormalises)
        return -math.inf if not np.all(np.isfinite(scored)) else float(-np.sum(scored))


def _conditional_exp_gaussian_nll(theta: np.ndarray, chunks, support: Support, renormalise: bool) -> np.ndarray:
    """Vectorized H-event conditional NLL, preserving source kernel formulas."""
    values = np.asarray(theta, dtype=float).reshape(-1)
    mu, alpha, beta, sigma = values
    if not (mu > 0.0 and alpha > 0.0 and beta > 0.0 and sigma > 0.0):
        return np.full(sum(chunk[0].shape[0] for chunk in chunks), np.inf)
    area = (support[1] - support[0]) * (support[3] - support[2])
    output = []
    for times, locations, origins, target_times, target_locations in chunks:
        parent_masses = _rectangular_gaussian_mass(locations.reshape(-1, 2), sigma, support).reshape(times.shape)
        if not np.all(parent_masses > 0.0):
            return np.full(sum(chunk[0].shape[0] for chunk in chunks), np.inf)
        lag_target = target_times[:, None] - times
        temporal_target = alpha * np.exp(-beta * lag_target)
        distances = np.linalg.norm(locations - target_locations[:, None, :], axis=2)
        spatial_target = np.exp(-0.5 * (distances / sigma) ** 2) / (2.0 * math.pi * sigma**2)
        if renormalise:
            spatial_target = spatial_target / parent_masses
            compensator_mass = np.ones_like(parent_masses)
        else:
            compensator_mass = parent_masses
        intensity = mu + np.sum(temporal_target * spatial_target, axis=1)
        if not np.all(np.isfinite(intensity) & (intensity > 0.0)):
            return np.full(sum(chunk[0].shape[0] for chunk in chunks), np.inf)
        # Integral from the declared forecast origin to the target.  Every
        # predecessor is at or before origin, and the exact Gaussian mass
        # makes this the continuous-space compensator, not a grid score.
        integral = mu * area * (target_times - origins)
        integral += np.sum(
            compensator_mass * (alpha / beta) *
            (np.exp(-beta * (origins[:, None] - times)) - np.exp(-beta * lag_target)), axis=1,
        )
        output.append(-np.log(intensity) + integral)
    if not output:
        raise ValueError("Conditional Hawkes fit received no contexts.")
    return np.concatenate(output)


def _cache_conditional_contexts(contexts, support: Support, history_events: int, *, chunk_size: int,
                                max_cache_bytes: int) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], list[float], int, int]:
    """Materialize H-event sufficient arrays once, in bounded vectorized chunks."""
    if chunk_size < 1 or max_cache_bytes < 1:
        raise ValueError("Conditional Hawkes chunk_size and cache capacity must be positive.")
    iterator = contexts() if callable(contexts) else iter(contexts)
    chunks, delays = [], []
    buffered: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]] = []
    used = 0

    def flush() -> None:
        nonlocal used
        if not buffered:
            return
        arrays = (
            np.stack([row[0] for row in buffered]), np.stack([row[1] for row in buffered]),
            np.asarray([row[2] for row in buffered]), np.asarray([row[3] for row in buffered]),
            np.stack([row[4] for row in buffered]),
        )
        required = sum(array.nbytes for array in arrays)
        if used + required > max_cache_bytes:
            raise MemoryError(
                f"Conditional Hawkes context cache needs at least {used + required} bytes, above "
                f"conditional_cache_bytes={max_cache_bytes}. Increase that declared capacity or reduce the fit corpus."
            )
        chunks.append(arrays)
        used += required
        buffered.clear()

    for row in iterator:
        try:
            history = row["history"]
            times, locations = history_arrays(history, support)
            target_time = float(row["next_time"])
            target_location = np.asarray(row["next_location"], dtype=float).reshape(2)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Conditional Hawkes contexts need history, next_time, and next_location.") from error
        if times.size != history_events:
            raise ValueError(f"Conditional Hawkes expected exactly {history_events} history events, got {times.size}.")
        origin = forecast_origin(history, times)
        if not np.isfinite(target_time) or target_time <= origin or not np.isfinite(target_location).all():
            raise ValueError("Conditional Hawkes target must be finite and strictly after its history origin.")
        if not bool(contains(support, target_location.reshape(1, 2))[0]):
            raise ValueError("Conditional Hawkes target lies outside declared support.")
        buffered.append((times, locations, origin, target_time, target_location))
        delays.append(target_time - origin)
        if len(buffered) >= chunk_size:
            flush()
    flush()
    return chunks, delays, sum(chunk[0].shape[0] for chunk in chunks), used


@dataclass
class HawkesSTPP:
    support: Support
    temporal: str = "exponential"
    spatial: str = "gaussian"
    quadrature_order: int = 32
    edge_correction: str = "auto"
    sampler_iterations: int = 2000
    parameters: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.support = as_support(self.support)
        supported = {("exponential", "gaussian"), ("omori", "pareto")}
        if (self.temporal, self.spatial) not in supported:
            raise ValueError(
                "Initial statistical presets support only exponential+gaussian "
                "and omori+pareto Hawkes component pairs."
            )
        if self.edge_correction not in {"auto", "renormalise", "none"}:
            raise ValueError("edge_correction must be auto, renormalise, or none.")
        if int(self.quadrature_order) != self.quadrature_order or self.quadrature_order < 4:
            raise ValueError("quadrature_order must be an integer of at least four.")
        self.quadrature_order = int(self.quadrature_order)
        if int(self.sampler_iterations) != self.sampler_iterations or self.sampler_iterations <= 500:
            raise ValueError("sampler_iterations must be an integer above the source 500-step burn-in.")
        self.sampler_iterations = int(self.sampler_iterations)
        if self.parameters is not None:
            self.parameters = self._validate_parameters(self.parameters)

    @property
    def n_params(self) -> int:
        return 4 if self.temporal == "exponential" else 6

    @property
    def nll_kind(self) -> str:
        """Numerical quality of the continuous next-event density."""
        return "exact" if (self.temporal, self.spatial) == ("exponential", "gaussian") else "approx"

    @property
    def nll_method(self) -> str:
        if self.nll_kind == "exact":
            return "analytic_continuous_next_event_density"
        return "source_quadrature_continuous_next_event_density"

    @property
    def nll_footnote(self) -> str:
        if self.nll_kind == "exact":
            return ""
        return "Omori/Pareto temporal and spatial integrals use declared source quadrature."

    @property
    def sampling_method(self) -> str:
        if (self.temporal, self.spatial) == ("exponential", "gaussian"):
            return "source_ogata_with_exact_truncated_gaussian_mixture"
        return "source_ogata_thinning_plus_finite_mcmc_location_sampling"

    def _domain(self) -> Rectangle:
        xmin, xmax, ymin, ymax = self.support
        return Rectangle(xmax - xmin, ymax - ymin, origin=(xmin, ymin))

    def _source_model(self):
        temporal = resolve_component(TEMPORAL_KERNEL_REGISTRY, self.temporal)
        spatial = resolve_component(SPATIAL_KERNEL_REGISTRY, self.spatial)
        source = spatio_temporal_model(self._domain(), base=resolve_component(BACKGROUND_REGISTRY, "constant"), temporal=temporal, spatial=spatial,
                                        n_quad=self.quadrature_order, edge_correction=self.edge_correction)
        if self.spatial != "gaussian":
            return source
        components = source.components
        n_base = len(components.base.spec)
        n_temporal = len(components.temporal.spec)
        spatial_slice = slice(n_base + n_temporal, len(source.spec))

        def build(theta: np.ndarray, rng) -> _AnalyticGaussianProcess:
            values = np.asarray(theta, dtype=float)
            return _AnalyticGaussianProcess(
                base=components.base.build(values[:n_base]),
                spatial=components.spatial.build(values[spatial_slice]),
                temporal=components.temporal.build(values[n_base:n_base + n_temporal]),
                domain=components.domain, monotone_temporal_kernel=components.temporal.monotone,
                rng=rng, gaussian_sigma=float(values[spatial_slice][0]), n_quad=self.quadrature_order,
                edge_correction=self.edge_correction,
            )

        def analytically_resolvable(theta: np.ndarray) -> np.ndarray:
            return np.ones(np.atleast_2d(np.asarray(theta, dtype=float)).shape[0], dtype=bool)

        return replace(source, builder=build, extra_support=analytically_resolvable)

    def _likelihood(self, model: ProcessModel):
        if self.spatial == "gaussian":
            return _AnalyticGaussianLikelihood(model, support=self.support, backend="cached")
        return SpatioTemporalLogLikelihood(model, backend="cached")

    def _validate_parameters(self, parameters) -> np.ndarray:
        theta = np.asarray(parameters, dtype=float).reshape(-1)
        if theta.shape != (self.n_params,) or not np.isfinite(theta).all():
            raise ValueError(f"Expected {self.n_params} finite Hawkes parameters.")
        if not bool(np.asarray(self._source_model().support(theta)).ravel()[0]):
            raise ValueError("Hawkes parameters are outside the copied source model support.")
        return theta

    def _source_history(self, seq) -> History:
        if len(seq.times) and seq.times[0] <= seq.t0:
            raise ValueError("Native source Hawkes requires events strictly after t0; declare a prior context window instead.")
        return _history(seq.times, seq.locations, start=seq.t0, end=seq.t1)

    @staticmethod
    def _valid_start(scale_t: float, scale_x: float, temporal: str, background: float) -> np.ndarray:
        """Dimensionally valid and strictly subcritical in the source convention."""
        if temporal == "exponential":
            beta = 1.0 / scale_t
            return np.array([background, 0.25 * beta, beta, scale_x])
        c, p = scale_t, 2.0
        return np.array([background, 0.25 * c * (p - 1.0), c, p, scale_x, 2.0])

    def fit(self, train_seqs, *, seed: int, inference: Mapping | None = None) -> dict:
        del seed
        options = dict(inference or {})
        unknown = set(options) - {"method", "optimizer", "max_iterations", "tol"}
        if unknown:
            raise ValueError(f"Unknown Hawkes inference options: {sorted(unknown)}")
        if options.get("method", "mle").lower() != "mle":
            raise ValueError("Initial native Hawkes ports support source MLE only.")
        data = sequences(train_seqs, self.support)
        histories = [self._source_history(seq) for seq in data]
        n_events = sum(history.n_events for history in histories)
        if n_events == 0:
            raise ValueError("Hawkes MLE is undefined for all-empty training sequences; use a Poisson control.")
        domain = self._domain()
        exposure = sum(history.duration * domain.volume for history in histories)
        background = max(n_events / exposure, np.finfo(float).tiny)
        gaps = np.concatenate([np.diff(history.times) for history in histories if history.n_events > 1]) if any(history.n_events > 1 for history in histories) else np.array([])
        scale_t = max(float(np.median(gaps)) if gaps.size else min(history.duration for history in histories), 1e-8)
        widest = float(np.max(domain.bounds[:, 1] - domain.bounds[:, 0]))
        scale_x = max(widest / 4.0, 2.0 * widest / self.quadrature_order)
        start = self._valid_start(scale_t, scale_x, self.temporal, background)
        model = self._source_model()
        if not bool(np.asarray(model.support(start)).ravel()[0]):
            raise RuntimeError("Constructed source-MLE start is outside source model support.")
        observed_histories = [history for history in histories if history.n_events]
        empty_exposure = sum(history.duration * domain.volume for history in histories if not history.n_events)
        likelihoods = [self._likelihood(model) for _ in observed_histories]
        combined = _IndependentLikelihood(model, likelihoods, observed_histories, empty_exposure)
        fit = INFERENCE_REGISTRY["mle"](combined, histories[0], start, method=str(options.get("optimizer", "Nelder-Mead")),
                      max_iterations=int(options.get("max_iterations", 300)), tol=options.get("tol"), recheck_resolution=False)
        theta = np.asarray(fit.theta, dtype=float)
        if not fit.converged:
            raise RuntimeError(f"Source Hawkes MLE did not converge to valid support: {fit.message}")
        theta = self._validate_parameters(theta)
        self.parameters = theta
        return {"status": "converged", "method": "source_fit_mle", "source_commit": "37e83bc30b3bd5781551ea1fc1e72ba657608ebb",
                "optimizer": str(options.get("optimizer", "Nelder-Mead")), "message": fit.message,
                "n_evaluations": fit.n_evaluations, "objective_nll": -fit.log_likelihood,
                "n_sequences": len(histories), "n_events": n_events, "observation_exposure": exposure, "parameters": json_array(theta),
                "branching_mass_upper_bound": float(np.asarray(model.branching(theta[None, :])).ravel()[0]),
                "n_params": self.n_params, "boundary": self.edge_correction, "quadrature_nodes_per_axis": self.quadrature_order,
                "spatial_integral": "analytic_rectangular_gaussian" if self.spatial == "gaussian" else "source_quadrature",
                "quadrature_resolution_checked": self.spatial != "gaussian"}

    def fit_conditional(self, contexts, *, seed: int, inference: Mapping | None = None) -> dict:
        """Fit an explicitly finite-history conditional Hawkes composite objective.

        This is the campaign route when the benchmark itself fixes a common
        number of observed predecessors.  It intentionally differs from
        :meth:`fit`: there is no independent-window terminal survival or
        background exposure for unscored intervals.
        """
        del seed
        options = dict(inference or {})
        unknown = set(options) - {"method", "optimizer", "max_iterations", "tol", "history_events", "conditional_chunk_size", "conditional_cache_bytes"}
        if unknown:
            raise ValueError(f"Unknown conditional Hawkes inference options: {sorted(unknown)}")
        if options.get("method", "mle").lower() != "mle":
            raise ValueError("Conditional native Hawkes ports support source MLE only.")
        history_events = options.get("history_events")
        if history_events != 20:
            raise ValueError("This benchmark conditional Hawkes port requires history_events=20.")
        if (self.temporal, self.spatial) != ("exponential", "gaussian"):
            raise ValueError("The scalable conditional composite fit currently supports hawkes_exp_gaussian only.")
        model = self._source_model()
        # Materialise only fixed-H sufficient arrays, not windows or a merged
        # catalogue.  Subsequent simplex evaluations are vectorized over these
        # chunks, so a full training split is read once rather than per step.
        chunks, delays, n_contexts, cache_bytes = _cache_conditional_contexts(
            contexts, self.support, int(history_events),
            chunk_size=int(options.get("conditional_chunk_size", 8192)),
            max_cache_bytes=int(options.get("conditional_cache_bytes", 512 * 1024 * 1024)),
        )
        if n_contexts == 0:
            raise ValueError("Conditional Hawkes fit received no contexts.")
        scale_t = max(float(np.median(delays)) if delays else 1.0, 1e-8)
        widest = float(np.max(self._domain().bounds[:, 1] - self._domain().bounds[:, 0]))
        start = self._valid_start(scale_t, widest / 4.0, self.temporal, 1.0 / (self._domain().volume * scale_t))
        if not bool(np.asarray(model.support(start)).ravel()[0]):
            raise RuntimeError("Constructed conditional source-MLE start is outside source support.")
        composite = _ConditionalExpGaussianLikelihood(model, self.support, chunks, int(history_events))
        placeholder = _history(np.empty(0), np.empty((0, 2)), start=0.0, end=0.0)
        fit = INFERENCE_REGISTRY["mle"](
            composite, placeholder, start, method=str(options.get("optimizer", "Nelder-Mead")),
            max_iterations=int(options.get("max_iterations", 300)), tol=options.get("tol"), recheck_resolution=False,
        )
        theta = np.asarray(fit.theta, dtype=float)
        if not fit.converged:
            raise RuntimeError(f"Conditional source Hawkes MLE did not converge to valid support: {fit.message}")
        self.parameters = self._validate_parameters(theta)
        return {
            "status": "converged", "method": "source_hawkes_h20_conditional_composite_mle",
            "objective_kind": "finite_history_conditional_composite_likelihood",
            "source_commit": "37e83bc30b3bd5781551ea1fc1e72ba657608ebb",
            "optimizer": str(options.get("optimizer", "Nelder-Mead")), "message": fit.message,
            "n_evaluations": fit.n_evaluations, "objective_nll": -fit.log_likelihood,
            "n_contexts": n_contexts, "history_events": int(history_events), "parameters": json_array(self.parameters),
            "conditional_cache_bytes": cache_bytes, "conditional_chunk_size": int(options.get("conditional_chunk_size", 8192)),
            "branching_mass_upper_bound": float(np.asarray(model.branching(self.parameters[None, :])).ravel()[0]),
            "n_params": self.n_params, "boundary": self.edge_correction,
            "spatial_integral": "analytic_rectangular_gaussian", "quadrature_resolution_checked": False,
        }

    def score_conditional(self, contexts, *, history_events: int = 20,
                          chunk_size: int = 8192, cache_bytes: int = 512 * 1024 * 1024) -> np.ndarray:
        """Score fixed-H contexts in vectorized chunks without source geometry caches."""
        if (self.temporal, self.spatial) != ("exponential", "gaussian"):
            raise ValueError("Vectorized conditional scoring currently supports hawkes_exp_gaussian only.")
        if history_events != 20:
            raise ValueError("This benchmark conditional scorer requires history_events=20.")
        chunks, _, count, _ = _cache_conditional_contexts(
            contexts, self.support, history_events, chunk_size=chunk_size, max_cache_bytes=cache_bytes
        )
        if count == 0:
            return np.empty(0, dtype=float)
        return _conditional_exp_gaussian_nll(
            self._require_fit(), chunks, self.support, self._source_model().components.renormalises
        )

    def _require_fit(self) -> np.ndarray:
        if self.parameters is None:
            raise RuntimeError("Model must be fit or restored before scoring.")
        return self.parameters

    def _history_start(self, history: Mapping, times: np.ndarray, origin: float) -> float:
        start = float(history.get("t0", origin))
        if len(times) and times[0] <= start:
            # A first event at t0 is valid as walk-forward context: the scoring
            # increment starts at the *last* context event, so this endpoint is
            # never integrated or scored. The source History object still needs
            # an open left endpoint to represent it; use its adjacent float,
            # rather than moving timestamps or pretending a full-window NLL is
            # defined at that boundary.
            start = float(np.nextafter(times[0], -np.inf))
        return start

    def next_event_nll(self, history: Mapping, next_time: float, next_location: np.ndarray) -> float:
        theta = self._require_fit()
        times, locations = history_arrays(history, self.support)
        point = np.asarray(next_location, dtype=float).reshape(1, 2)
        next_time = float(next_time)
        origin = forecast_origin(history, times)
        if not np.isfinite(next_time) or next_time <= origin or not bool(np.isfinite(point).all()):
            raise ValueError("Target must be finite and strictly after the forecast origin.")
        if not bool(contains(self.support, point)[0]):
            return float("inf")
        full = _history(np.append(times, next_time), np.vstack([locations, point]),
                        start=self._history_start(history, times, origin), end=next_time)
        likelihood = self._likelihood(self._source_model())
        state = likelihood.initial_state(origin)
        _, log_density = likelihood.extend(state, theta, full, next_time)
        return float("inf") if not math.isfinite(log_density) else float(-log_density)

    def full_window_nll(self, seq: Mapping) -> float:
        data = sequences([seq], self.support)[0]
        history = self._source_history(data)
        theta = self._require_fit()
        if history.n_events == 0:
            return float(theta[0] * self._domain().volume * history.duration)
        return float(-self._likelihood(self._source_model()).total(theta, history))

    def _process(self, history: Mapping, rng: np.random.Generator):
        times, locations = history_arrays(history, self.support)
        origin = forecast_origin(history, times)
        process = self._source_model()(self._require_fit(), rng=rng)
        process.n_iter = int(self.sampler_iterations)
        bound = _bind_history(process, _history(times, locations, start=self._history_start(history, times, origin), end=origin))
        return bound, origin

    def sample_next_event(self, history: Mapping, *, n_samples: int, rng: np.random.Generator):
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        output = []
        for _ in range(n_samples):
            process, origin = self._process(history, rng)
            _source_simulate_one_after(process, origin)
            output.append(process.events[:, -1])
        events = np.asarray(output, dtype=float)
        return events[:, 0], events[:, 1:].reshape(n_samples, 2)

    def sample_future(self, history: Mapping, *, horizon: float, rng: np.random.Generator, max_events: int = 10000) -> dict:
        if not np.isfinite(horizon) or horizon <= 0 or max_events < 1:
            raise ValueError("horizon and max_events must be positive.")
        process, origin = self._process(history, rng)
        initial_count = process.events.shape[1]
        current = origin
        while True:
            accepted = _source_simulate_one_after(process, current, horizon=origin + horizon)
            if not accepted:
                break
            current = float(process.events[0, -1])
            if process.events.shape[1] - initial_count >= max_events:
                raise RuntimeError("max_events reached in source Hawkes simulation; forecast was not silently censored.")
        count = process.events.shape[1] - initial_count
        generated = process.events[:, initial_count:] if count else np.empty((3, 0))
        return {"times": json_array(generated[0]), "locations": json_array(generated[1:].T.reshape(-1, 2))}

    def to_state(self) -> dict:
        return {"kind": "hawkes", "support": list(self.support), "temporal": self.temporal, "spatial": self.spatial,
                "quadrature_order": self.quadrature_order, "edge_correction": self.edge_correction,
                "sampler_iterations": self.sampler_iterations, "parameters": json_array(self._require_fit()),
                "source_commit": "37e83bc30b3bd5781551ea1fc1e72ba657608ebb",
                "metadata": dict(getattr(self, "metadata", {}))}

    @classmethod
    def from_state(cls, state: Mapping) -> "HawkesSTPP":
        model = cls(support=as_support(state["support"]), temporal=str(state["temporal"]), spatial=str(state["spatial"]),
                    quadrature_order=int(state.get("quadrature_order", 32)), edge_correction=str(state.get("edge_correction", "auto")),
                    sampler_iterations=int(state.get("sampler_iterations", 2000)))
        model.parameters = np.asarray(state["parameters"], dtype=float)
        model.parameters = model._validate_parameters(model.parameters)
        model.metadata = dict(state.get("metadata", {}))
        return model
