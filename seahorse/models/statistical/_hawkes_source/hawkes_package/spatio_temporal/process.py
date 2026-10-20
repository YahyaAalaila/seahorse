#!/usr/bin/env python3
r"""Domain-aware spatio-temporal Hawkes process simulation.

The intensity is separable,

.. math::

    \lambda(t, x \mid H_t) = \mu(x)
        + \sum_{t_i < t} \kappa_t(t - t_i)\, \kappa_s\!\left( d(x, x_i) \right),

where :math:`d(\cdot, \cdot)` is the geodesic distance on the domain.

Event times come from Ogata's thinning applied to the space-integrated
intensity; the location of each accepted event is then drawn by
Metropolis-Hastings from the conditional spatial density at that time.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from .._numerics import as_float, as_point, locate_peak
from ..base import HawkesProcess, SeedLike, _stalled_message
from ..mcmc import mcmc_sampler
from . import _integration
from .domains import Circle, SpatialDomain

__all__ = ["SpatioTemporalHawkesProcess"]


class SpatioTemporalHawkesProcess(HawkesProcess):
    r"""Spatio-temporal Hawkes process on an arbitrary :class:`SpatialDomain`.

    Parameters
    ----------
    base : callable
        Background intensity ``mu(x)`` as a function of spatial coordinate. Since
        1.0.0 it may instead carry ``time_varying = True`` and be a function of
        ``(t, x)`` -- see :class:`~hawkes_package.periodic.PeriodicBackground` --
        in which case it must also supply ``supremum(x)``.
    spatial : callable
        Isotropic spatial kernel evaluated at a non-negative distance.
    temporal : callable
        Temporal kernel evaluated at a non-negative time lag.
    domain : SpatialDomain, optional
        Spatial domain. Defaults to the unit :class:`Circle`.

        .. versionchanged:: 0.3.0
           May be a proper subset of its bounding box: the quadrature rule is
           masked by :meth:`~hawkes_package.SpatialDomain.contains` and weighted
           by :meth:`~hawkes_package.SpatialDomain.volume_element`, and the
           location sampler targets the intensity restricted to the domain
           rather than to the box. Before 0.3.0 a domain whose ``volume``
           differed from ``prod(bounds widths)`` raised :class:`ValueError`;
           the summed quadrature weights are now checked against ``volume``
           instead, warning above 1% apart and raising above 10%.
    monotone_temporal_kernel : bool
        Set ``True`` when `temporal` is monotone decreasing. This permits a
        tighter thinning bound and skips the numerical search for the kernel's
        maximum.
    rng : None, int or numpy.random.Generator
        Source of randomness. See :class:`~hawkes_package.base.HawkesProcess`.

        .. versionadded:: 0.2.0

    edge_correction : {'auto', 'renormalise', 'none'}
        What to do about excitation that falls off the edge of a bounded domain.

        On a closed surface the spatial kernel's mass over the domain is the
        same wherever an event lands, so an event's realised productivity does
        not depend on its position. On a domain with a boundary it does: an
        event near the edge spreads offspring into a region that is partly
        outside, so it produces fewer of them, and a fit that ignores this
        attributes the missing offspring to a weaker kernel.

        ``'renormalise'`` divides each event's spatial kernel by its own
        in-domain mass :math:`S_i = \int_D \kappa_s(d(y, x_i))\,dy`, so every
        event excites the same total amount wherever it sits. It also makes the
        stationarity condition *exact* rather than merely conservative: the
        branching ratio in :mod:`hawkes_package.inference` is
        ``mass(kappa_t) * mass(kappa_s)`` with the spatial mass taken over the
        whole model space, and renormalisation is what makes the realised mass
        equal to it.

        ``'none'`` leaves the intensity as written, with a position-dependent
        realised productivity. Correct, and what you want if the boundary is
        physical rather than an observation window.

        ``'auto'`` -- the default -- renormalises exactly when the domain sets
        :attr:`~hawkes_package.SpatialDomain.has_boundary`. Every domain that
        predates 1.0.0 does not, so nothing existing changes.

        :math:`S_i` is computed on the **same quadrature the bound and the
        acceptance test share**, which is what keeps ``M >= lambda`` exact: it is
        a per-event constant, so it divides both sides by the same positive
        number.

        .. versionadded:: 1.0.0

    Attributes
    ----------
    events : numpy.ndarray
        Shape ``(ndim + 1, n)``. Row 0 holds event times, rows 1.. hold
        coordinates.

        .. versionchanged:: 0.4.0
           Renamed from ``Events``, which still works and warns until 0.5.0.

    Examples
    --------
    >>> process = SpatioTemporalHawkesProcess(
    ...     base=lambda x: 0.5,
    ...     spatial=lambda d: max(0.0, 1 - d / np.pi),
    ...     temporal=lambda dt: 0.9 * np.exp(-5 * dt),
    ...     domain=Circle(),
    ...     monotone_temporal_kernel=True,
    ...     rng=0,
    ... )
    >>> process.simulate(5)
    >>> process.events.shape
    (2, 5)
    """

    def __init__(
        self,
        # `Callable[..., float]` for the reason `spatial` is: the signature is
        # decided by an opt-in attribute on the callable itself, `time_varying`
        # here and `pairwise` there, so no single arity describes both.
        base: Callable[..., float],
        spatial: Callable[..., Any],
        temporal: Callable[[float], float],
        domain: SpatialDomain | None = None,
        monotone_temporal_kernel: bool = False,
        rng: SeedLike = None,
        *,
        peak_lag: float | None = None,
        peak_value: float | None = None,
        n_quad: int | None = None,
        proposal_std: Any = None,
        n_iter: int = 2000,
        edge_correction: str = "auto",
    ) -> None:
        self.base = base
        self.spatial = spatial
        self.temporal = temporal
        # Opt-in dispatch, in the `PairwiseKernel` style: a background carrying
        # `time_varying = True` is handed `(t, x)`, everything else `(x)` exactly
        # as before. Widening the signature for everyone would break every user
        # callable and move every existing number; this way a configuration that
        # predates 1.0.0 is bit-identical, which the tests assert.
        self._time_varying_base = getattr(base, "time_varying", False) is True
        if self._time_varying_base and not hasattr(base, "supremum"):
            raise ValueError(
                "a time-varying background must supply `supremum(x)`, the largest "
                "value it takes at `x` over any later interval. The thinning bound "
                "is computed before the candidate is drawn, so bounding by the "
                "current value is smaller than the intensity the candidate meets: "
                "every candidate is accepted and the output is a Poisson process "
                "wearing a Hawkes costume, with nothing raising. See "
                "hawkes_package.periodic.PeriodicSchedule for a background whose "
                "supremum is closed form."
            )
        self.domain = domain if domain is not None else Circle()
        self.monotone_temporal_kernel = monotone_temporal_kernel

        if edge_correction not in {"auto", "renormalise", "none"}:
            raise ValueError(
                f"edge_correction must be 'auto', 'renormalise' or 'none', got {edge_correction!r}"
            )
        self.edge_correction = edge_correction
        # "auto" means "correct exactly where there is an edge to correct for",
        # which is what `has_boundary` declares. Every domain that predates
        # 1.0.0 reports False, so this resolves to `False` for all of them and
        # no previously produced number moves.
        self._renormalise = edge_correction == "renormalise" or (
            edge_correction == "auto" and bool(self.domain.has_boundary)
        )
        self._edge_times: np.ndarray = np.empty(0, dtype=float)
        self._edge_values: np.ndarray = np.empty(0, dtype=float)

        ndim = self.domain.bounds.shape[0]
        self._ndim = ndim
        # The record is one row of times above `ndim` rows of coordinates, and
        # the domain has to be known before the buffer can be shaped -- which is
        # why `super().__init__` comes after it rather than first.
        super().__init__(rng=rng, rows=ndim + 1)
        self.proposal_std = proposal_std
        self.n_iter = int(n_iter)
        self._build_quadrature(n_quad)

        if not monotone_temporal_kernel:
            if peak_lag is None:
                located = locate_peak(temporal, name="temporal kernel")
                self.temporal_extremum, self.temporal_peak = located.lag, located.value
            else:
                self.temporal_extremum = float(peak_lag)
                self.temporal_peak = as_float(
                    peak_value if peak_value is not None else temporal(self.temporal_extremum)
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _past_events(self, t: float, inclusive: bool = False) -> np.ndarray:
        """Return the events before time `t`, as a ``(ndim + 1, m)`` slice of the record.

        `inclusive` also admits an event at exactly `t`, which the thinning
        bound needs: at the start of a step `t` *is* the most recent event time.

        There is no ``0 < ...`` guard: it used to hide the bootstrap column, but
        it also silently discarded any user-supplied event at a non-positive
        time for the whole life of the object.

        .. versionchanged:: 0.4.0
           Returns the events rather than a list of their indices, and selects
           them with a mask rather than a Python loop over every column. It is
           called four times per thinning step and once per quadrature node
           within each, so the loop was a per-event constant on the hot path.
        """
        record = self.events
        times = record[0]
        keep = times <= t if inclusive else times < t
        return np.asarray(record[:, keep])

    def _temporal_factors(self, t: float, bound: bool = False) -> np.ndarray:
        """Temporal kernel factors at time `t`, or their future suprema.

        With ``bound=True`` each event contributes the largest value its kernel
        can still reach at any later time -- the peak if it has not yet peaked,
        its current value if it has. For a monotone-decreasing kernel the two
        coincide, so only the inclusive event set differs.
        """
        lags = np.asarray(t - self._past_events(t, inclusive=bound)[0])
        if lags.size == 0:
            return lags
        values = np.array([float(self.temporal(lag)) for lag in lags])
        if bound and not self.monotone_temporal_kernel:
            values = np.where(lags < self.temporal_extremum, self.temporal_peak, values)
        return values

    def _dist_temporal(self, t: float) -> np.ndarray:
        return self._temporal_factors(t)

    def _dist_spatial(self, x: Any, t: float, bound: bool = False) -> np.ndarray:
        past = self._past_events(t, inclusive=bound)
        # as_float on every element: a user `spatial` returning a 0-d or
        # shape-(1,) array would otherwise build a (n, 1) column here, which
        # broadcasts against the (n,) temporal factors to an (n, n) outer
        # product -- silently computing (sum kappa_t)(sum kappa_s).
        values = np.array([self._spatial_at(x, event) for event in past[1:].T], dtype=float)
        if bound:
            # sup_s kappa_t(s - t_i) * kappa_s(d) equals sup(kappa_t) * kappa_s(d)
            # only where kappa_s(d) >= 0; where it is negative the supremum is 0,
            # since kappa_t is non-negative and decays. Clipping *is* the
            # supremum, so an inhibitory spatial kernel stays correctly bounded.
            values = np.maximum(values, 0.0)
        if self._renormalise and values.size:
            # A per-event *constant*, which is the whole reason this is safe:
            # it does not depend on `x` or on `t`, so it divides the intensity
            # and its supremum by the same positive number and the domination
            # argument carries unchanged. Applied after the clip above, so the
            # bound stays a bound.
            values = values / self._edge_scales(values.size)
        return values

    def _in_domain_mass(self, y: Any) -> float:
        """Integral of the spatial kernel centred on `y`, over the domain.

        Computed on **the same quadrature the bound and the acceptance test
        share**. That is not an efficiency: using a finer rule here and the
        shared one there would divide the two sides of ``M >= lambda`` by
        different numbers, and the invariant would fail by whatever the two
        rules disagreed about.
        """
        mass = self._quadrature.integrate(lambda x: self._spatial_at(x, y))
        if not mass > 0.0:
            raise RuntimeError(
                f"the spatial kernel integrates to {mass!r} over the domain around "
                f"{np.asarray(y).tolist()}, so it cannot be renormalised. Either the "
                f"kernel's support is narrower than a quadrature panel -- raise n_quad "
                f"above {self.n_quad} -- or it is signed and integrates to nothing "
                "there, in which case pass edge_correction='none' and interpret the "
                "excitation as position-dependent."
            )
        return float(mass)

    def _edge_scales(self, count: int) -> np.ndarray:
        """In-domain kernel mass for each of the first `count` recorded events.

        Cached, and that matters: :meth:`_full_intensity` runs once per
        quadrature node, so computing this inside it would make a single space
        integral ``O(m**2 n)`` -- at 1024 nodes, a million kernel evaluations per
        node. The mass depends only on the event's location and the fixed rule,
        so it is computed once per event and reused.

        Keyed on the event *times* rather than on a count, because
        ``process.events = history`` replaces the whole record: a cache keyed on
        length would silently survive being handed a different realisation of
        the same size and scale every event by another one's mass.
        """
        times = self.events[0]
        cached = self._edge_values.size
        if cached > times.size or not np.array_equal(self._edge_times[:cached], times[:cached]):
            self._edge_values = np.empty(0, dtype=float)
            self._edge_times = np.empty(0, dtype=float)
            cached = 0

        if cached < count:
            fresh = self.events[1:, cached:count].T
            self._edge_values = np.concatenate(
                [self._edge_values, [self._in_domain_mass(point) for point in fresh]]
            )
            self._edge_times = np.asarray(times[: self._edge_values.size], dtype=float).copy()
        return np.asarray(self._edge_values[:count], dtype=float)

    def _base_at(self, x: Any, t: float, bound: bool = False) -> float:
        """Evaluate the background at `x`, or bound it over every later time.

        The `bound` branch is the one that matters. A candidate is drawn ahead
        of `t`, so the bound has to dominate the background the candidate will
        *meet*, not the one at the moment the bound is computed. For a periodic
        background those differ by the whole amplitude of the cycle, and the
        error is in the direction that accepts everything.
        """
        if not self._time_varying_base:
            return as_float(self.base(x))
        if bound:
            # The constructor has already refused a time-varying background
            # without this, which is the check mypy cannot see from a Callable.
            return as_float(self.base.supremum(x))  # type: ignore[attr-defined]
        return as_float(self.base(t, x))

    def _full_intensity(self, x: Any, t: float, bound: bool = False) -> float:
        # The single place a coordinate is normalised: `base` and `spatial` are
        # always handed a shape-(ndim,) point, whichever path called us.
        x = as_point(x, self._ndim)
        contrib = np.multiply(
            self._temporal_factors(t, bound=bound), self._dist_spatial(x, t, bound=bound)
        ).sum()
        return max(0.0, self._base_at(x, t, bound=bound) + float(contrib))

    def _spatial_at(self, x: Any, y: Any) -> float:
        """Evaluate the spatial kernel between two points.

        A kernel marked ``pairwise`` consumes both endpoints; an ordinary one
        consumes the geodesic distance between them. See
        :class:`~hawkes_package.spatio_temporal.kernels.PairwiseKernel`.
        """
        if getattr(self.spatial, "pairwise", False) is True:
            return as_float(self.spatial(x, y))
        return as_float(self.spatial(self.domain.distance(x, y)))

    def _build_quadrature(self, n_quad: int | None) -> None:
        """Build the fixed spatial integration rule, once, at construction."""
        bounds = self.domain.bounds
        # The domain's own recommendation, not a flat dimension-based default:
        # a hyperbolic polygon needs four times the nodes a flat one does, and
        # the only symptom of too few is a warning that the area -- and so the
        # event rate -- is out by several percent.
        self.n_quad = self.domain.nodes_per_axis if n_quad is None else int(n_quad)

        def make_rule(nodes_per_axis: int) -> _integration.TensorQuadrature:
            # Nodes outside the domain are dropped and the survivors weighted by
            # the measure density. Both are identities when the domain fills its
            # box with the flat measure, which is the default and covers every
            # domain that predates 0.3.0.
            return _integration.restrict(
                _integration.build(bounds, nodes_per_axis),
                self.domain.contains,
                self.domain.volume_element,
            )

        self._quadrature = make_rule(self.n_quad)
        self._check_quadrature_volume()

        # The kernel is what the rule has to resolve; check it once, here,
        # rather than silently returning a wrong integral for the whole run.
        # Probed from a point the domain vouches for: the centre of the bounding
        # box need not lie inside a domain that is a proper subset of it.
        centre = np.asarray(self.domain.interior_point, dtype=float)
        _integration.check_resolution(
            self._quadrature, make_rule, self.n_quad, lambda x: self._spatial_at(x, centre)
        )

        # The kernel is not the only thing the rule has to resolve. A peaked
        # *background* loses its mass between nodes in exactly the places the
        # events are, and the only symptom is an event rate wrong by that
        # fraction -- the same silent failure, from the other term.
        #
        # Guarded by one pass over the coarse nodes rather than run
        # unconditionally: a constant background is resolved exactly, and this
        # constructor runs once per particle per rejuvenation move, where a
        # doubled rule would be paid for nothing on every fit that predates a
        # varying background.
        # At the bound, which is the time-supremum for a periodic background:
        # that is the profile the quadrature has to resolve, and it is also the
        # only one available before the first event time exists.
        background = np.array(
            [self._base_at(node, 0.0, bound=True) for node in self._quadrature.nodes]
        )
        if background.size and float(np.ptp(background)) > 0.0:
            _integration.check_resolution(
                self._quadrature,
                make_rule,
                self.n_quad,
                lambda x: self._base_at(x, 0.0, bound=True),
                name="background",
            )

    def _check_quadrature_volume(self) -> None:
        """Check the rule reproduces the domain's own measure.

        The weights sum to whatever the rule thinks the domain measures, so
        comparing that against :attr:`SpatialDomain.volume` catches two distinct
        faults with one number: a domain whose declared volume is simply wrong,
        and a rule too coarse to resolve the domain's boundary. Either one
        scales the simulated event rate by the same factor it gets the measure
        wrong by -- and, as ever here, does so silently.

        For a domain that fills its bounding box nothing is masked and the
        weights sum to the box volume exactly, so this subsumes the flat
        ``volume == prod(bounds widths)`` check it replaces.
        """
        measured = float(self._quadrature.weights.sum())
        declared = float(self.domain.volume)
        scale = max(abs(declared), np.finfo(float).tiny)
        error = abs(measured - declared) / scale

        if error > 0.1:
            raise ValueError(
                f"{type(self.domain).__name__} declares volume {declared:.6g} but its "
                f"quadrature rule measures {measured:.6g} ({100 * error:.1f}% apart). The "
                "domain's `volume`, `bounds` and `contains` disagree with each other."
            )
        if error > 1e-2:
            warnings.warn(
                f"the quadrature rule measures {type(self.domain).__name__} as {measured:.6g} "
                f"against its declared volume {declared:.6g} ({100 * error:.1f}% apart): the "
                f"rule does not resolve the domain boundary and the simulated event rate will "
                f"be wrong by about as much. Raise n_quad above {self.n_quad}.",
                UserWarning,
                stacklevel=4,
            )

    def _integrated_intensity(self, t: float, bound: bool = False) -> float:
        """Intensity integrated over the spatial domain at time `t`.

        Deterministic: the bound and the acceptance test use the same nodes and
        strictly positive weights, so a pointwise-dominating integrand
        integrates to a dominating value and ``M >= lambda`` holds exactly.
        """
        return self._quadrature.integrate(lambda x: self._full_intensity(x, t, bound=bound))

    def _upper_bound(self, t: float) -> float:
        # Integrating the per-event suprema dominates the integrated intensity
        # at every later time, for monotone and bell-shaped kernels alike.
        return self._integrated_intensity(t, bound=True)

    def _confined_density(self, x: Any, t: float) -> float:
        r"""Evaluate the density the location sampler targets: zero outside the domain.

        Two corrections to the raw intensity, for two different reasons.

        **Confinement.** `bounds` is a *box*, and a domain that is a proper
        subset of it needs the chain confined to the domain itself. Folding the
        sample back afterwards is not the same thing and is not correct:
        `_full_intensity` off the domain is the periodic extension, so the box
        covers some parts of the domain twice and others once, and folding a
        box-distributed draw inherits that unevenness. Returning zero is the
        confinement -- :func:`~hawkes_package.mcmc.mcmc_sampler` already treats
        a zero density as a rejection, and rejection (rather than folding) is
        what keeps the chain reversible for a general pairing group.

        **The measure.** The event location is distributed as
        :math:`\lambda\,\mathrm{d}A` on the surface, but the sampler walks in
        *chart* coordinates with a symmetric Gaussian proposal and accepts on
        the raw ratio, so the density it must be handed is
        :math:`\lambda \cdot \sqrt{\det g}` -- the same
        :meth:`~hawkes_package.SpatialDomain.volume_element` factor
        :func:`~hawkes_package.spatio_temporal._integration.restrict` applies to
        the quadrature weights. Omitting it biases the sampled locations towards
        wherever the chart compresses area, by exactly the factor the chart
        compresses by: on the sphere it would pile events at the poles.

        For a domain that fills its box with the flat measure -- every domain
        that predates 0.4.0 -- ``contains`` is constantly ``True``,
        ``volume_element`` is constantly ``1.0``, and this is exactly
        `_full_intensity`.

        .. versionchanged:: 0.4.0
           Multiplies by ``volume_element``. No shipped result moves: every
           domain that existed before had the flat chart measure.
        """
        if not self.domain.contains(x):
            return 0.0
        return self._full_intensity(x, t) * self.domain.volume_element(x)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _bound_at(self, t: float) -> float:
        """Return the thinning bound at `t`, refusing a non-positive one."""
        bound = self._upper_bound(t)
        if not bound > 0:
            raise RuntimeError(
                f"Non-positive thinning bound M={bound!r} at t={t!r}; the "
                "background intensity must be positive somewhere."
            )
        return bound

    def _draw_location(self, event_time: float) -> np.ndarray:
        """Draw the location of an event at `event_time` from its conditional density.

        Costs :attr:`n_iter` Metropolis-Hastings steps, each one a full spatial
        intensity evaluation, so it is by far the most expensive thing the loop
        does per event. That is why
        :meth:`~hawkes_package.base.HawkesProcess.simulate_until` refuses an
        over-horizon candidate *before* getting here.
        """
        coord = mcmc_sampler(
            lambda x: self._confined_density(x, event_time),
            self.domain.bounds,
            n_iter=self.n_iter,
            proposal_std=self.proposal_std,
            seed=self.rng,
            # Folding is reversible only where `wrap` is a translation
            # symmetry; elsewhere out-of-domain proposals are rejected.
            transform=self.domain.wrap if self.domain.periodic else None,
        )
        # Idempotent: the sampler already returns an in-domain point. Kept only
        # as a guard against a third-party `wrap` returning a boundary-equal
        # representative.
        return np.asarray(self.domain.wrap(coord), dtype=float).reshape(-1)

    def _record(self, event_time: float, coord: np.ndarray) -> None:
        """Append one located event to the record."""
        self._events.append(np.concatenate([[event_time], coord]))
        self.n_simulated += 1

    def _propagate(self, k: int) -> None:
        for done in range(k):
            t = self._events.last_time

            # --- Temporal thinning on the space-integrated intensity ---
            while True:
                bound = self._bound_at(t)
                advanced = t + self.rng.exponential() / bound
                if not advanced > t:
                    raise RuntimeError(
                        _stalled_message(t, bound, f"after {done} of {k} requested events")
                    )
                t = advanced
                if self.rng.uniform() * bound <= self._integrated_intensity(t):
                    break

            # --- Spatial coordinate from the conditional density at that time ---
            self._record(t, self._draw_location(t))

    def _propagate_until(self, t_end: float, start: float) -> None:
        t = start
        accepted = 0

        while True:
            bound = self._bound_at(t)
            advanced = t + self.rng.exponential() / bound
            if not advanced > t:
                raise RuntimeError(
                    _stalled_message(t, bound, f"after {accepted} events past t={start!r}")
                )
            t = advanced

            # The horizon is checked before the acceptance test, and so before
            # `_draw_location`: the candidate that ends the run would otherwise
            # burn `n_iter` Metropolis-Hastings steps on an event that is thrown
            # away, which is most of the cost of one event.
            if t > t_end:
                return

            if self.rng.uniform() * bound <= self._integrated_intensity(t):
                self._record(t, self._draw_location(t))
                accepted += 1

    # ------------------------------------------------------------------
    # Intensity accessors
    # ------------------------------------------------------------------

    def intensity(self, t: float, x: Any) -> float:
        r"""Conditional intensity :math:`\lambda(t, x \mid H_t)` at one point.

        Parameters
        ----------
        t : float
            Time.
        x : array_like
            Spatial coordinate, of length ``ndim``.

        Returns
        -------
        float
            The intensity, including the background term and floored at zero.

        .. versionadded:: 0.2.0
        """
        return self._full_intensity(np.asarray(x, dtype=float), float(t))

    def intensity_over_interval(
        self,
        times: Any,
        points: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Evaluate the intensity on the tensor grid ``times`` x ``points``.

        Parameters
        ----------
        times : array_like, shape (n_t,)
            Times to evaluate at. The realised event times are merged in and
            the result is sorted and de-duplicated.
        points : array_like, shape (n_x, ndim), optional
            Spatial evaluation points. Defaults to 200 equispaced points across
            the domain when ``ndim == 1``. **Required** when ``ndim >= 2``,
            because no canonical ordering of a multi-dimensional grid exists.

        Returns
        -------
        times : numpy.ndarray, shape (n_t',)
            The sorted evaluation times, events merged in.
        points : numpy.ndarray, shape (n_x, ndim)
            The spatial points actually used.
        intensity : numpy.ndarray, shape (n_x, n_t')
            Rows index space, columns index time. This is the orientation
            :func:`matplotlib.pyplot.contourf` expects, so a field plot is
            ``plt.contourf(times, points[:, 0], intensity)``.

        Raises
        ------
        ValueError
            If `points` is omitted on a domain of two or more dimensions.

        Notes
        -----
        `times` and `points` are both returned because `times` is modified
        (events merged in) and `points` may have been defaulted; without them
        the caller cannot label the axes of `intensity`.

        .. versionadded:: 0.2.0
           Before 0.2.0 there was no way to evaluate the field intensity
           without re-implementing it by hand.
        """
        ndim = self.domain.bounds.shape[0]

        if points is None:
            if ndim != 1:
                raise ValueError(
                    f"points is required for a {ndim}-dimensional domain; there is no "
                    "canonical default grid above one dimension."
                )
            lo, hi = self.domain.bounds[0]
            points_arr = np.linspace(lo, hi, 200).reshape(-1, 1)
        else:
            points_arr = np.asarray(points, dtype=float)
            if points_arr.ndim == 1:
                points_arr = points_arr.reshape(-1, 1)
            if points_arr.shape[1] != ndim:
                raise ValueError(
                    f"points must have shape (n_x, {ndim}) for this domain, got {points_arr.shape}"
                )

        event_times = self.events[0, :]
        times_arr = np.unique(np.append(np.asarray(times, dtype=float).ravel(), event_times))

        intensity = np.array(
            [
                [self._full_intensity(points_arr[i], float(t)) for t in times_arr]
                for i in range(points_arr.shape[0])
            ]
        )
        return times_arr, points_arr, intensity
