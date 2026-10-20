"""Priors over model parameters, as independent marginals plus a support.

Every prior here answers two questions and no others: draw ``n`` parameter
vectors, and evaluate ``log p(theta)`` on a batch of them. That is all an SMC
sampler needs, and keeping the interface that narrow is what lets a user pass
their own object without inheriting anything.

Densities are returned **unnormalised only where a constraint truncates them**.
:class:`ConstrainedPrior` divides by nothing: the normalising constant of the
truncated prior is a constant in ``theta``, so it cancels in every
Metropolis-Hastings ratio and in every importance weight, and estimating it
would cost a Monte Carlo integral to buy nothing. It does change the evidence
:attr:`~hawkes_package.inference.smc.SMCDiagnostics.log_evidence` by an unknown
additive constant, which is stated there rather than papered over.

The marginals are hand-written rather than taken from :mod:`scipy.stats`: this
package keeps SciPy to exactly one call site, and a log-pdf is four lines.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .parameters import ParameterSpec

__all__ = [
    "ConstrainedPrior",
    "Gamma",
    "IndependentPrior",
    "LogNormal",
    "Marginal",
    "Normal",
    "Prior",
    "Uniform",
    "stationarity",
]

_LOG_2PI = math.log(2.0 * math.pi)


@runtime_checkable
class Marginal(Protocol):
    """A scalar distribution: a log-density and a sampler."""

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Log density at each entry of `x`, ``-inf`` off the support."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` independent values."""
        ...


@runtime_checkable
class Prior(Protocol):
    """A joint prior over a parameter vector."""

    def log_pdf(self, theta: np.ndarray) -> np.ndarray:
        """Log density of each row of `theta`, shape ``(n_particles,)``."""
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` parameter vectors, shape ``(n, n_dim)``."""
        ...


# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------


class LogNormal:
    r"""Log-normal marginal on :math:`(0, \infty)`.

    The default choice for a rate: it has the right support, its tail is heavy
    enough that a prior guessed an order of magnitude wrong still puts mass
    where the data is, and its logarithm is where the sampler moves anyway.

    Parameters
    ----------
    mean_log, sd_log : float
        Mean and standard deviation of :math:`\log X`. Note that
        :math:`E[X] = \exp(\mathrm{mean\_log} + \mathrm{sd\_log}^2 / 2)`, which
        is above the median :math:`\exp(\mathrm{mean\_log})` -- the parameter is
        the median's logarithm, not the mean's.

    .. versionadded:: 0.5.0
    """

    def __init__(self, mean_log: float, sd_log: float) -> None:
        if not sd_log > 0:
            raise ValueError(f"sd_log must be positive, got {sd_log}")
        self.mean_log = float(mean_log)
        self.sd_log = float(sd_log)

    def __repr__(self) -> str:
        return f"LogNormal({self.mean_log!r}, {self.sd_log!r})"

    def log_pdf(self, x: Any) -> np.ndarray:
        """Log density, ``-inf`` at or below zero."""
        values = np.asarray(x, dtype=float)
        out = np.full(values.shape, -np.inf)
        positive = values > 0
        if np.any(positive):
            logs = np.log(values[positive])
            out[positive] = (
                -logs
                - math.log(self.sd_log)
                - 0.5 * _LOG_2PI
                - 0.5 * ((logs - self.mean_log) / self.sd_log) ** 2
            )
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` log-normal values."""
        return np.asarray(rng.lognormal(self.mean_log, self.sd_log, size=n), dtype=float)


class Gamma:
    r"""Gamma marginal on :math:`(0, \infty)`, in the shape-rate parameterisation.

    Density :math:`b^a x^{a-1} e^{-b x} / \Gamma(a)`, so the mean is
    :math:`a / b`. The rate convention, not the scale one -- NumPy's
    :meth:`~numpy.random.Generator.gamma` takes a scale, and the reciprocal is
    applied here so the two cannot be confused at the call site.

    Parameters
    ----------
    shape, rate : float
        Both strictly positive.

    .. versionadded:: 0.5.0
    """

    def __init__(self, shape: float, rate: float) -> None:
        if not shape > 0 or not rate > 0:
            raise ValueError(f"shape and rate must be positive, got {shape}, {rate}")
        self.shape = float(shape)
        self.rate = float(rate)
        self._log_norm = self.shape * math.log(self.rate) - math.lgamma(self.shape)

    def __repr__(self) -> str:
        return f"Gamma({self.shape!r}, {self.rate!r})"

    def log_pdf(self, x: Any) -> np.ndarray:
        """Log density, ``-inf`` at or below zero."""
        values = np.asarray(x, dtype=float)
        out = np.full(values.shape, -np.inf)
        positive = values > 0
        if np.any(positive):
            good = values[positive]
            out[positive] = self._log_norm + (self.shape - 1.0) * np.log(good) - self.rate * good
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` gamma values."""
        return np.asarray(rng.gamma(self.shape, 1.0 / self.rate, size=n), dtype=float)


class Normal:
    """Normal marginal on the whole line.

    Parameters
    ----------
    mean, sd : float
        Location and scale; `sd` strictly positive.

    .. versionadded:: 0.5.0
    """

    def __init__(self, mean: float, sd: float) -> None:
        if not sd > 0:
            raise ValueError(f"sd must be positive, got {sd}")
        self.mean = float(mean)
        self.sd = float(sd)

    def __repr__(self) -> str:
        return f"Normal({self.mean!r}, {self.sd!r})"

    def log_pdf(self, x: Any) -> np.ndarray:
        """Log density."""
        values = np.asarray(x, dtype=float)
        return np.asarray(
            -math.log(self.sd) - 0.5 * _LOG_2PI - 0.5 * ((values - self.mean) / self.sd) ** 2,
            dtype=float,
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` normal values."""
        return np.asarray(rng.normal(self.mean, self.sd, size=n), dtype=float)


class Uniform:
    """Uniform marginal on ``[low, high]``.

    Parameters
    ----------
    low, high : float
        Finite, with ``low < high``.

    .. versionadded:: 0.5.0
    """

    def __init__(self, low: float, high: float) -> None:
        if not math.isfinite(low) or not math.isfinite(high) or not low < high:
            raise ValueError(f"need finite low < high, got {low}, {high}")
        self.low = float(low)
        self.high = float(high)
        self._log_density = -math.log(self.high - self.low)

    def __repr__(self) -> str:
        return f"Uniform({self.low!r}, {self.high!r})"

    def log_pdf(self, x: Any) -> np.ndarray:
        """Log density, ``-inf`` outside the interval."""
        values = np.asarray(x, dtype=float)
        inside = (values >= self.low) & (values <= self.high)
        return np.where(inside, self._log_density, -np.inf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` uniform values."""
        return np.asarray(rng.uniform(self.low, self.high, size=n), dtype=float)


# ---------------------------------------------------------------------------
# Joint priors
# ---------------------------------------------------------------------------


class IndependentPrior:
    """A product of independent marginals, one per coordinate.

    Parameters
    ----------
    marginals : sequence of Marginal
        One per parameter, in the model's coordinate order.

    Examples
    --------
    >>> prior = IndependentPrior((LogNormal(0.5, 1.0), LogNormal(-1.0, 1.0)))
    >>> prior.sample(4, np.random.default_rng(0)).shape
    (4, 2)

    .. versionadded:: 0.5.0
    """

    def __init__(self, marginals: Sequence[Marginal]) -> None:
        self.marginals = tuple(marginals)
        if not self.marginals:
            raise ValueError("a prior needs at least one marginal")

    def __len__(self) -> int:
        return len(self.marginals)

    def __repr__(self) -> str:
        return f"IndependentPrior({list(self.marginals)!r})"

    def log_pdf(self, theta: Any) -> np.ndarray:
        """Sum of the marginal log-densities over each row of `theta`."""
        batch = np.atleast_2d(np.asarray(theta, dtype=float))
        if batch.shape[1] != len(self.marginals):
            raise ValueError(
                f"theta has {batch.shape[1]} coordinate(s) but the prior has "
                f"{len(self.marginals)} marginal(s)"
            )
        total = np.zeros(batch.shape[0], dtype=float)
        for j, marginal in enumerate(self.marginals):
            total = total + marginal.log_pdf(batch[:, j])
        return total

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` parameter vectors, one column per marginal."""
        return np.column_stack([m.sample(n, rng) for m in self.marginals])


class ConstrainedPrior:
    """A prior restricted to the region a predicate admits.

    Sampling is by rejection, which is the only method that does not need the
    truncated normalising constant. That makes the acceptance rate the thing
    that can go wrong, so it is measured rather than assumed: a support the base
    prior almost never lands in raises, naming the measured rate, instead of
    looping until the caller gives up.

    Parameters
    ----------
    base : Prior
        The unconstrained prior.
    support : callable
        Maps ``(n_particles, n_dim)`` to a boolean ``(n_particles,)``. The
        natural argument is
        :meth:`~hawkes_package.inference.models.ProcessModel.support`, which
        already knows every condition the process constructor would raise on.
    max_draws : int
        Total base draws allowed while filling one request.

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        base: Prior,
        support: Callable[[np.ndarray], np.ndarray],
        *,
        max_draws: int = 1_000_000,
    ) -> None:
        self.base = base
        self.support = support
        self.max_draws = int(max_draws)

    def __repr__(self) -> str:
        return f"ConstrainedPrior({self.base!r}, {self.support!r})"

    def log_pdf(self, theta: Any) -> np.ndarray:
        """Return the base log-density where the support admits, ``-inf`` elsewhere.

        Unnormalised: the truncation constant is a constant in ``theta``, so it
        cancels everywhere it is used except in the evidence.
        """
        batch = np.atleast_2d(np.asarray(theta, dtype=float))
        allowed = np.asarray(self.support(batch), dtype=bool)
        return np.where(allowed, self.base.log_pdf(batch), -np.inf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` vectors from the base prior, keeping those in the support."""
        kept: list[np.ndarray] = []
        collected = 0
        drawn = 0
        accepted = 0
        while collected < n:
            # Oversample by the observed rate, so a support holding a tenth of
            # the prior costs a handful of rounds rather than one per particle.
            rate = max(accepted / drawn, 1e-3) if drawn else 1.0
            block = min(max(int((n - collected) / rate) + 8, 64), self.max_draws - drawn)
            if block <= 0:
                break
            candidates = np.atleast_2d(self.base.sample(block, rng))
            good = candidates[np.asarray(self.support(candidates), dtype=bool)]
            drawn += block
            accepted += good.shape[0]
            if good.shape[0]:
                kept.append(good)
                collected += good.shape[0]

        if collected < n:
            raise RuntimeError(
                f"the constrained prior accepted {accepted} of {drawn} base draws "
                f"({100 * accepted / max(drawn, 1):.3f}%), not enough to fill {n} "
                "particles. The support and the base prior barely overlap: widen the "
                "support, or move the base prior into it."
            )
        return np.asarray(np.concatenate(kept, axis=0)[:n], dtype=float)


def stationarity(
    branching_ratio: Callable[[np.ndarray], np.ndarray],
    *,
    limit: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the support predicate ``branching_ratio(theta) < limit``.

    The branching ratio is the expected number of direct offspring per event. At
    or above one the process is supercritical: it produces infinitely many
    events in finite time, so there is no stationary law to fit and
    :meth:`~hawkes_package.base.HawkesProcess.simulate` on such a parameter
    either raises or takes unbounded time. A prior that puts mass there does not
    fail visibly -- the particles land in it, the likelihood is finite on a
    finite observed history, and the posterior simply carries an excitation term
    biased upward.

    Parameters
    ----------
    branching_ratio : callable
        Maps ``(n_particles, n_dim)`` to ``(n_particles,)``.
        :meth:`~hawkes_package.inference.models.ProcessModel.branching_ratio` is
        the one to pass.
    limit : float
        The threshold. Below 1 to keep the cloud away from the boundary, where
        simulation for a forecast becomes expensive long before it becomes
        impossible.

    Examples
    --------
    >>> from hawkes_package.inference import exponential_model
    >>> model = exponential_model()
    >>> keep = stationarity(model.branching_ratio, limit=0.9)
    >>> keep(np.array([[1.0, 0.5, 2.0], [1.0, 1.9, 2.0]])).tolist()
    [True, False]

    .. versionadded:: 0.5.0
    """
    if not limit > 0:
        raise ValueError(f"limit must be positive, got {limit}")

    def keep(theta: np.ndarray) -> np.ndarray:
        ratio = np.asarray(branching_ratio(theta), dtype=float)
        return np.asarray(np.isfinite(ratio) & (ratio < limit), dtype=bool)

    return keep


def spec_for(prior: Prior, spec: ParameterSpec) -> None:
    """Check that `prior` and `spec` describe the same number of coordinates.

    Called once when a sampler is built. A prior one coordinate short does not
    raise on its own -- NumPy broadcasts the mismatch into a plausible-looking
    array -- so the check has to be explicit.

    Raises
    ------
    ValueError
        If the two disagree.
    """
    sample = np.atleast_2d(prior.sample(2, np.random.default_rng(0)))
    if sample.shape[1] != len(spec):
        raise ValueError(
            f"the prior draws {sample.shape[1]} coordinate(s) but the model has "
            f"{len(spec)}: {list(spec.names)}"
        )
