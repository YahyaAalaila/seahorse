r"""Checking a fitted model against the data it was fitted to.

The sharpest available check on a point-process fit is the **time-rescaling
theorem**, which this package already uses to verify its simulator
(``docs/theory.md``). If the events really came from intensity
:math:`\lambda_\theta`, then mapping them through their own compensator

.. math::

    \Lambda(t) = \int_{start}^{t} \lambda_\theta(s)\,\mathrm{d}s

turns them into a unit-rate Poisson process, so the transformed gaps
:math:`\Lambda(t_{i+1}) - \Lambda(t_i)` are independent
:math:`\mathrm{Exp}(1)`. That single test catches a wrong kernel, a missing
background and a compensator computed too small -- none of which a plot of the
posterior would notice, because a posterior can be tight, stable and wrong.

The Kolmogorov-Smirnov machinery is written out here rather than imported from
:mod:`scipy.stats`. This package holds SciPy to exactly one call site, and the
Kolmogorov distribution is a four-line series. It is checked against
:func:`scipy.stats.kstest` in the test suite, where SciPy is a development
dependency and may be used freely.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .likelihood import History, LogLikelihood
from .smc import ParticleCloud

__all__ = ["KSResult", "ks_exponential", "posterior_report", "residuals"]

#: Terms of the Kolmogorov series. It converges like exp(-2 k^2 x^2), so a
#: hundred is far past the point where the terms stop changing a double -- the
#: count exists to bound the loop, not to reach the accuracy.
_KOLMOGOROV_TERMS = 100


def residuals(
    likelihood: LogLikelihood,
    theta: Any,
    history: History,
    *,
    upto: float | None = None,
) -> np.ndarray:
    r"""Time-rescaled inter-event gaps: :math:`\Lambda(t_{i+1}) - \Lambda(t_i)`.

    Under the true parameter these are independent :math:`\mathrm{Exp}(1)`, so
    :func:`ks_exponential` on them is a goodness-of-fit test for the whole
    model at once.

    The first gap is measured from ``history.start``, not from the first event.
    Dropping it would discard the one residual that carries information about
    the background rate alone, and it is the residual a wrong ``mu`` distorts
    most.

    Parameters
    ----------
    likelihood : LogLikelihood
        Supplies the compensator -- the same one the log-likelihood is built
        from, so a bug in it cannot pass this test and fail the fit or the
        reverse.
    theta : array_like
        A single parameter vector, usually
        :meth:`~hawkes_package.inference.smc.ParticleCloud.mean`.
    history : History
        The observed events.
    upto : float, optional
        Rescale only the events up to this time. Defaults to the whole window.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_events,)``. Empty for a history with no events.

    Raises
    ------
    ValueError
        If the compensator comes back decreasing. That is not a numerical
        nuisance but a broken intensity: a compensator is an integral of a
        non-negative function, so a decrease means the intensity went negative
        or the quadrature ran backwards.
    """
    end = history.end if upto is None else float(upto)
    times = history.times[history.times <= end]
    if times.size == 0:
        return np.empty(0, dtype=float)

    compensated = np.asarray(likelihood.compensator(theta, history, times), dtype=float)
    gaps = np.diff(np.concatenate([[0.0], compensated]))
    if np.any(gaps < 0.0):
        worst = float(np.min(gaps))
        raise ValueError(
            f"the compensator decreased by {-worst:.6g} between two events. It is the "
            "integral of a non-negative intensity, so this is a broken intensity or a "
            "quadrature evaluated out of order, not a rounding artefact."
        )
    return gaps


@dataclass(frozen=True)
class KSResult:
    """A Kolmogorov-Smirnov statistic and its p-value."""

    statistic: float
    pvalue: float

    def __repr__(self) -> str:
        return f"KSResult(statistic={self.statistic:.6g}, pvalue={self.pvalue:.6g})"


def kolmogorov_sf(x: float) -> float:
    r"""Evaluate the Kolmogorov distribution's survival function.

    :math:`Q(x) = 2\sum_{k \ge 1} (-1)^{k-1} e^{-2k^2x^2}`, the limiting
    distribution of :math:`\sqrt{n} D_n`. Alternating and rapidly convergent,
    so summing it directly is both accurate and stable -- the terms shrink
    faster than any cancellation can matter.
    """
    if x <= 0.0:
        return 1.0
    if x > 8.0:
        # Past here the first term is below 1e-55 and the sum underflows to
        # zero anyway; returning it directly avoids a loop that does nothing.
        return 0.0
    total = 0.0
    for k in range(1, _KOLMOGOROV_TERMS + 1):
        term = math.exp(-2.0 * (k * x) ** 2)
        total += term if k % 2 else -term
        if term < 1e-18:
            break
    return float(min(1.0, max(0.0, 2.0 * total)))


def ks_exponential(values: Any, *, rate: float = 1.0) -> KSResult:
    """One-sample Kolmogorov-Smirnov test against ``Exp(rate)``.

    Parameters
    ----------
    values : array_like
        The sample. Non-negative; a negative entry is refused rather than
        contributing zero to the empirical CDF, because it means the caller has
        handed this something that is not a set of gaps.
    rate : float
        The exponential rate to test against. ``1.0`` is what
        :func:`residuals` produces under a correct model.

    Returns
    -------
    KSResult
        The statistic and the asymptotic p-value, the latter with the standard
        small-sample correction ``(sqrt(n) + 0.12 + 0.11/sqrt(n)) * D``, which
        holds to a few per cent from about twenty observations upward.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> result = ks_exponential(rng.exponential(size=500))
    >>> result.pvalue > 1e-3
    True

    .. versionadded:: 0.5.0
    """
    sample = np.sort(np.asarray(values, dtype=float).ravel())
    n = sample.size
    if n == 0:
        raise ValueError("the sample is empty")
    if np.any(sample < 0.0):
        raise ValueError(
            f"an exponential sample cannot be negative, and this one reaches "
            f"{float(np.min(sample)):.6g}"
        )
    if not rate > 0:
        raise ValueError(f"rate must be positive, got {rate}")

    theoretical = 1.0 - np.exp(-rate * sample)
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    # Both sides of every step: the empirical CDF jumps at each observation, and
    # taking only the value after the jump misses the larger discrepancy exactly
    # where the sample is sparse -- the tail, which is what a wrong compensator
    # distorts first.
    statistic = float(max(np.max(upper - theoretical), np.max(theoretical - lower)))

    root = math.sqrt(n)
    return KSResult(
        statistic=statistic,
        pvalue=kolmogorov_sf((root + 0.12 + 0.11 / root) * statistic),
    )


def posterior_report(
    cloud: ParticleCloud,
    likelihood: LogLikelihood,
    history: History,
    *,
    truth: Any = None,
) -> str:
    """Summarise a fit: the marginals, then the residual test.

    Reported together on purpose. A tight posterior and a rejected
    goodness-of-fit test is the combination worth noticing -- it says the model
    is confidently wrong -- and reading the two from separate calls is how that
    combination goes unnoticed.

    Parameters
    ----------
    cloud : ParticleCloud
        The fitted posterior.
    likelihood : LogLikelihood
        Used for the residuals, at the posterior mean.
    history : History
        The data.
    truth : array_like, optional
        Known parameters, for a simulation study. When given, each marginal is
        marked with whether its 90% interval covers.

    .. versionadded:: 0.5.0
    """
    lines = [cloud.summary()]
    if truth is not None:
        values = np.asarray(truth, dtype=float).ravel()
        low, high = cloud.credible_interval(0.9)
        covered = (values > low) & (values < high)
        lines.append(
            "coverage: "
            + ", ".join(
                f"{name}={'in' if ok else 'OUT'}"
                for name, ok in zip(cloud.spec.names, covered, strict=True)
            )
        )

    gaps = residuals(likelihood, cloud.mean(), history)
    if gaps.size:
        result = ks_exponential(gaps)
        verdict = "consistent" if result.pvalue > 1e-3 else "REJECTED"
        lines.append(
            f"time-rescaling: D={result.statistic:.4g}, p={result.pvalue:.4g} "
            f"over {gaps.size} gaps -- {verdict} with a unit-rate Poisson process"
        )
    return "\n".join(lines)
