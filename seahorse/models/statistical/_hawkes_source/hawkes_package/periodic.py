r"""A background that repeats: diurnal, weekly and seasonal structure.

Crime, emergency calls and social data all have a daily cycle, and a model
without one attributes that cycle to self-excitation -- the same confusion a
constant background makes with a spatially clustered one, moved from space into
time.

The shape is a truncated Fourier series on a **declared** period,

.. math::

    s(t) = 1 + \sum_{k=1}^{K} \big[a_k \cos(2\pi k t / P) + b_k \sin(2\pi k t / P)\big],

which is chosen over splines for three properties this package needs and knots
do not give: it is periodic by construction, its integral over any interval is
closed form, and **so is its supremum** -- and that last one is not a
convenience, it is the thinning bound.

Why the supremum is the whole story
-----------------------------------

Ogata's algorithm draws a candidate *ahead* in time and accepts it against a
bound computed *before* the draw. With a constant background that distinction
is invisible, because the background contributes the same value either way. With
a rising background it is the difference between a correct simulation and a
silent one: bounding by the value at the current time is smaller than the
intensity the candidate meets, every candidate is accepted, and the output is a
Poisson process wearing a Hawkes costume.

Since the schedule is periodic, the supremum over ``[t, ∞)`` is the supremum
over one period for any `t` -- a candidate can land arbitrarily far ahead, and
every window of a full period contains the maximum. So the bound is a constant
this class computes once, and computes by a dense scan of a closed form rather
than by a solver: an unvalidated numerical peak search is one of the two
recurring root causes of a bound that is too small in this package's history.

.. versionadded:: 1.0.0
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["PeriodicBackground", "PeriodicSchedule"]

#: Points per period in the supremum scan. A truncated Fourier series with `K`
#: harmonics has at most `2K` interior extrema, so a grid this fine brackets
#: every one of them for any `K` a user is plausibly fitting -- and the value
#: taken is the grid maximum inflated by the series' own Lipschitz bound over
#: half a grid step, which turns "very probably the maximum" into "at least the
#: maximum". A bound that is merely probably right is the failure this package
#: has shipped seven times.
_SCAN = 4096


class PeriodicSchedule:
    r"""A non-negative periodic multiplier on a background rate.

    ``schedule(t)`` is :math:`\max(0, 1 + \sum_k a_k\cos + b_k\sin)`, so it
    averages to one over a period when the floor does not bite, and multiplying
    a background by it leaves the *mean* rate where it was. That convention is
    what makes an amplitude readable: ``a_1 = 0.5`` is a cycle that doubles the
    rate at its peak, whatever the background it multiplies.

    Parameters
    ----------
    cosine, sine : array_like
        The coefficients :math:`a_k` and :math:`b_k`, `K` of each. Both may be
        empty, which is the constant schedule and is worth allowing: it makes
        "no cycle" a point in the parameter space rather than a different model.
    period : float
        The cycle length, **declared rather than fitted**. A fitted period is a
        multimodal likelihood -- every integer fraction of the truth is a local
        maximum -- and 24 hours is what applied users have.

    Attributes
    ----------
    supremum : float
        The largest value the schedule takes, over any interval. Certified
        rather than searched: see the module docstring.

    Raises
    ------
    ValueError
        If `period` is not positive, or the two coefficient arrays differ in
        length.

    Examples
    --------
    >>> import numpy as np
    >>> schedule = PeriodicSchedule([0.5], [0.0], period=24.0)
    >>> float(schedule(0.0))
    1.5
    >>> round(float(schedule.integral(0.0, 24.0)), 10)
    24.0
    >>> round(schedule.supremum, 4)
    1.5

    .. versionadded:: 1.0.0
    """

    def __init__(self, cosine: Any, sine: Any, *, period: float) -> None:
        self.cosine = np.asarray(cosine, dtype=float).reshape(-1)
        self.sine = np.asarray(sine, dtype=float).reshape(-1)
        if self.cosine.size != self.sine.size:
            raise ValueError(
                f"cosine has {self.cosine.size} coefficient(s) and sine has "
                f"{self.sine.size}; a truncated Fourier series has both per harmonic, "
                "with zeros where a term is absent"
            )
        self.period = float(period)
        if not self.period > 0:
            raise ValueError(f"period must be positive, got {period!r}")
        if not (np.all(np.isfinite(self.cosine)) and np.all(np.isfinite(self.sine))):
            raise ValueError("the Fourier coefficients must all be finite")

        self.harmonics = np.arange(1, self.cosine.size + 1, dtype=float)
        self.supremum = self._certified_supremum()

    def __repr__(self) -> str:
        return (
            f"PeriodicSchedule(cosine={self.cosine.tolist()}, "
            f"sine={self.sine.tolist()}, period={self.period})"
        )

    def _raw(self, t: Any) -> np.ndarray:
        """Evaluate the series before the floor, vectorized over `t`."""
        times = np.asarray(t, dtype=float)
        if self.cosine.size == 0:
            return np.ones_like(times, dtype=float)
        angle = 2.0 * math.pi * self.harmonics[None, :] * times.reshape(-1, 1) / self.period
        terms = np.cos(angle) @ self.cosine + np.sin(angle) @ self.sine
        return np.asarray(1.0 + terms.reshape(times.shape), dtype=float)

    def __call__(self, t: Any) -> np.ndarray:
        """Evaluate the schedule, floored at zero.

        The floor is what keeps a background non-negative for coefficients that
        would otherwise take the series below zero. It is applied here rather
        than left to the caller because a negative background is not a small
        modelling error -- it makes the intensity's floor the thing that decides
        the answer, and the cached likelihood backend refuses it outright.
        """
        # Wrapped rather than returned straight from the ufunc: numpy 2's stubs
        # type `np.maximum` as `Any`, and a function promising an ndarray must
        # not quietly return one. numpy 1's stubs do not, which is exactly why
        # this reached CI -- the two see different errors, so a clean local mypy
        # is not evidence.
        return np.asarray(np.maximum(0.0, self._raw(t)), dtype=float)

    def _certified_supremum(self) -> float:
        r"""Return a value the schedule never exceeds.

        A dense grid maximum plus the series' own Lipschitz bound over half a
        grid step. The derivative is bounded by
        :math:`\sum_k (2\pi k/P)\sqrt{a_k^2 + b_k^2}`, so the true maximum
        cannot exceed the best grid value by more than that times half the
        spacing -- which turns a scan into a certificate.

        The cheap alternative, :math:`1 + \sum_k \sqrt{a_k^2+b_k^2}`, is also
        valid and is what a reader expects. It is not used because it is loose
        by a factor that grows with the number of harmonics -- every harmonic
        peaking at once -- and a loose bound is accepted candidates thrown away,
        which costs simulation time on every step of every run.
        """
        if self.cosine.size == 0:
            return 1.0
        grid = np.linspace(0.0, self.period, _SCAN, endpoint=False)
        peak = float(np.max(self._raw(grid)))
        amplitudes = np.hypot(self.cosine, self.sine)
        lipschitz = float(np.sum(2.0 * math.pi * self.harmonics / self.period * amplitudes))
        certified = peak + lipschitz * 0.5 * self.period / _SCAN
        # Never above the every-harmonic-aligned bound, which is exact when the
        # scan happens to be loose.
        return float(min(certified, 1.0 + float(np.sum(amplitudes))))

    def integral(self, start: float, end: float) -> float:
        r"""Return :math:`\int_{start}^{end} s(u)\,du`, in closed form.

        Exact for the series; the floor is **not** applied, so a schedule whose
        coefficients push it below zero integrates to less than this over the
        clipped region. That case is refused where it matters -- see
        :meth:`is_non_negative` -- rather than silently mis-integrated, because a
        background integral that comes out too small is a penalty on a high
        intensity that never gets applied, and the excitation absorbs it.
        """
        a, b = float(start), float(end)
        if self.cosine.size == 0:
            return b - a
        omega = 2.0 * math.pi * self.harmonics / self.period
        cos_part = self.cosine * (np.sin(omega * b) - np.sin(omega * a)) / omega
        sin_part = self.sine * (np.cos(omega * a) - np.cos(omega * b)) / omega
        return float((b - a) + np.sum(cos_part) + np.sum(sin_part))

    def is_non_negative(self) -> bool:
        """Whether the series stays at or above zero, so the floor never bites.

        Checked on the same dense grid the supremum uses, less the same
        Lipschitz slack -- so a ``True`` here is a certificate and a ``False``
        may be conservative, which is the right way round for a predicate that
        gates an exact integral.
        """
        if self.cosine.size == 0:
            return True
        grid = np.linspace(0.0, self.period, _SCAN, endpoint=False)
        trough = float(np.min(self._raw(grid)))
        amplitudes = np.hypot(self.cosine, self.sine)
        lipschitz = float(np.sum(2.0 * math.pi * self.harmonics / self.period * amplitudes))
        return bool(trough - lipschitz * 0.5 * self.period / _SCAN >= 0.0)

    def mean(self) -> float:
        """Return the average over one period, which is ``1`` unless the floor bites."""
        return self.integral(0.0, self.period) / self.period


class PeriodicBackground:
    r"""A spatial background multiplied by a periodic schedule.

    :math:`\mu(t, x) = m(x)\, s(t)`, and the separability is a design decision
    rather than an approximation of one: it keeps the compensator a **product**
    of two integrals -- the spatial one the package already computes and the
    schedule's closed form -- instead of a two-dimensional quadrature. A
    background whose *shape* changes with the hour is a joint model and is out
    of scope.

    Carries ``time_varying = True``, which is how
    :class:`~hawkes_package.spatio_temporal.process.SpatioTemporalHawkesProcess`
    knows to hand it ``(t, x)``. Anything without that attribute is called as
    ``base(x)`` exactly as before, so every configuration that predates 1.0.0
    produces the same numbers.

    Parameters
    ----------
    spatial : callable
        The background's shape, ``m(x)``, per unit measure -- the same object
        that would have been passed as `base`.
    schedule : PeriodicSchedule
        The cycle. Its mean is one, so ``spatial`` keeps its meaning as the
        *average* background rather than acquiring a hidden factor.

    Examples
    --------
    >>> schedule = PeriodicSchedule([0.5], [0.0], period=24.0)
    >>> background = PeriodicBackground(lambda x: 0.5, schedule)
    >>> background.time_varying
    True
    >>> float(background(0.0, [0.1]))
    0.75
    >>> float(background.supremum([0.1]))
    0.75

    .. versionadded:: 1.0.0
    """

    #: The opt-in flag the simulator dispatches on.
    time_varying = True

    def __init__(self, spatial: Any, schedule: PeriodicSchedule) -> None:
        self.spatial = spatial
        self.schedule = schedule

    def __repr__(self) -> str:
        return f"PeriodicBackground({self.spatial!r}, {self.schedule!r})"

    def __call__(self, t: Any, x: Any) -> float:
        """Evaluate the background at time `t` and position `x`."""
        return float(self.spatial(x)) * float(self.schedule(float(t)))

    def supremum(self, x: Any) -> float:
        """Return the largest value at `x` over any later interval, the bound.

        ``m(x)`` times the schedule's own supremum, and that product is exact
        rather than conservative because the two factors are independent: the
        schedule reaches its maximum at some time in every period, and `x` does
        not move while a candidate is drawn.

        Requires ``m(x) >= 0``. A negative shape would turn this product into a
        *minimum*, and the bound would sit below the intensity everywhere the
        schedule is small -- which is the failure that accepts every candidate.
        """
        shape = float(self.spatial(x))
        if shape < 0.0:
            raise ValueError(
                f"the background's spatial shape is {shape} at {np.asarray(x).tolist()}. "
                "A negative shape makes `shape * schedule.supremum` a minimum rather "
                "than a maximum, so the thinning bound would sit below the intensity "
                "with nothing raising."
            )
        return shape * self.schedule.supremum
