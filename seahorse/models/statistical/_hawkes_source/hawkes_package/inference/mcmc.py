r"""An adaptive Metropolis chain, for reference posteriors.

This module exists to **check** the sequential sampler, not to replace it. A
Monte Carlo method that verifies itself verifies nothing, so
:func:`batch_posterior` computes the same posterior by a completely different
route -- one long chain over the whole history at once -- and
``tests/statistical/test_inference_recovery.py`` requires the two to agree. Two
independent samplers reaching the same distribution is evidence; one sampler
reaching a plausible-looking distribution is not.

**This is not** :func:`hawkes_package.mcmc.mcmc_sampler`, and the two must not
be confused. That one is on the Ogata correctness path: it draws the location of
every simulated event, works on the natural scale of a density, and returns only
the final state of its chain. Handing it a log-posterior would underflow it, and
handing it a target whose evaluation costs a full likelihood would make every
simulation unusable. So inference gets its own chain, named so it cannot read as
a drop-in. What is reused is the *idea*: Roberts and Rosenthal's 0.234 target
acceptance rate, and a multiplicative scale adaptation confined to the burn-in.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base import SeedLike
from .evolution import _cholesky_with_floor, default_scale
from .likelihood import History, LogLikelihood
from .priors import Prior

__all__ = ["ChainResult", "batch_posterior", "metropolis_chain"]

#: Target acceptance rate for the burn-in adaptation (Roberts and Rosenthal).
_TARGET_ACCEPTANCE = 0.234
_ADAPT_EVERY = 50

#: How far the adaptation may move the step from where it started, each way.
#: Without a floor the adaptation has a trap it cannot climb out of, and the
#: trap is not hypothetical: on one seed of the recovery sweep the step shrank
#: to 2e-159, every proposal became the point it started from, the acceptance
#: rate read 1.000 -- proposing the current point is always accepted -- and the
#: chain returned 40 000 copies of one sample while reporting a healthy-looking
#: run. The shape estimate closes the loop: a frozen chain has no scatter, a
#: zero scatter gives a zero proposal covariance, and a zero proposal keeps it
#: frozen.
_STEP_RANGE = (1e-4, 1e4)

#: Floor on the proposal's standard deviation, as a fraction of the initial
#: step. Absolute rather than a fraction of the estimated covariance, which is
#: the same trap from the other side: a ridge proportional to a covariance that
#: has collapsed has collapsed with it.
_SHAPE_FLOOR = 1e-3


@dataclass(frozen=True)
class ChainResult:
    """The output of one Metropolis run.

    Attributes
    ----------
    samples : numpy.ndarray
        Shape ``(n_samples, n_dim)``, post-burn-in, on whatever scale the target
        was written in.
    log_density : numpy.ndarray
        The target's value at each kept sample, shape ``(n_samples,)``.
    acceptance : float
        Post-burn-in acceptance rate. Outside roughly ``[0.1, 0.6]`` the chain
        is either crawling or bouncing, and in both cases the samples are far
        more correlated than their count suggests.
    scale : numpy.ndarray
        The proposal factor the adaptation settled on.
    """

    samples: np.ndarray
    log_density: np.ndarray
    acceptance: float
    scale: np.ndarray


def metropolis_chain(
    log_density: Callable[[np.ndarray], float],
    initial: Any,
    *,
    n_samples: int = 5000,
    burn_in: int | None = None,
    thin: int = 1,
    scale: float | None = None,
    rng: SeedLike = None,
    warn_acceptance: bool = True,
) -> ChainResult:
    r"""Run a random-walk Metropolis chain against `log_density`.

    Parameters
    ----------
    log_density : callable
        The unnormalised log target, taking one shape-``(n_dim,)`` point and
        returning a float. ``-inf`` is a rejection; ``nan`` is refused, because
        a comparison against ``nan`` is false and the proposal would be rejected
        forever with nothing said.
    initial : array_like
        Starting point, shape ``(n_dim,)``. Must have finite log density.
    n_samples : int
        Samples kept after burn-in and thinning.
    burn_in : int, optional
        Iterations discarded, during which the proposal adapts. Defaults to half
        of `n_samples`. Adaptation stops at the end of burn-in: an
        indefinitely-adapting chain is not Markov and does not have the target
        as its stationary distribution.
    thin : int
        Keep one sample in `thin`.
    scale : float, optional
        Initial step size. Defaults to Roberts and Rosenthal's
        :math:`2.38/\sqrt{d}`.
    rng : None, int or numpy.random.Generator
        Source of randomness.
    warn_acceptance : bool
        Warn when the post-burn-in acceptance rate leaves ``[0.1, 0.6]``.

    Returns
    -------
    ChainResult

    Raises
    ------
    ValueError
        If `initial` has non-finite log density -- the chain would have nothing
        to compare a proposal against and would accept the first one
        unconditionally, silently starting from a point the target never
        proposed.

    .. versionadded:: 0.5.0
    """
    generator = np.random.default_rng(rng)
    current = np.asarray(initial, dtype=float).ravel()
    n_dim = current.size
    kept = int(n_samples)
    if kept < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if thin < 1:
        raise ValueError(f"thin must be at least 1, got {thin}")
    warm = kept // 2 if burn_in is None else int(burn_in)
    if warm < 0:
        raise ValueError(f"burn_in must be non-negative, got {burn_in}")

    initial_step = default_scale(n_dim) if scale is None else float(scale)
    if not initial_step > 0:
        raise ValueError(f"scale must be positive, got {scale}")
    step = initial_step
    factor = step * np.eye(n_dim)

    current_density = float(log_density(current))
    if not np.isfinite(current_density):
        raise ValueError(
            f"the log density at the starting point {current.tolist()} is "
            f"{current_density}. A chain started outside the support accepts its first "
            "proposal whatever it is, so the run would look healthy while sampling from "
            "somewhere the target never sent it."
        )

    total = warm + kept * thin
    samples = np.empty((kept, n_dim), dtype=float)
    densities = np.empty(kept, dtype=float)

    # The running mean and scatter of the burn-in, used to shape the proposal.
    # A scalar step cannot fit a posterior whose coordinates differ in scale by
    # orders of magnitude, which a Hawkes posterior over (mu, alpha, beta)
    # routinely does.
    mean = current.copy()
    scatter = np.zeros((n_dim, n_dim), dtype=float)
    seen = 0

    window = 0
    accepted_total = 0
    index = 0

    for step_index in range(total):
        proposal = current + generator.normal(0.0, 1.0, size=n_dim) @ factor.T
        density = float(log_density(proposal))
        if math.isnan(density):
            raise ValueError(
                f"the log density returned nan at {proposal.tolist()}. nan compares "
                "false against everything, so it would be rejected forever and the chain "
                "would stall at its current point without the acceptance rate showing "
                "anything unusual."
            )
        if math.log(generator.uniform()) < density - current_density:
            current, current_density = proposal, density
            window += 1
            if step_index >= warm:
                accepted_total += 1

        if step_index < warm:
            seen += 1
            delta = current - mean
            mean += delta / seen
            scatter += np.outer(delta, current - mean)
            if (step_index + 1) % _ADAPT_EVERY == 0:
                step, window = _adapt(step, window, initial_step)
                factor = _proposal_factor(scatter, seen, step, n_dim, initial_step)
        elif (step_index - warm) % thin == 0 and index < kept:
            samples[index] = current
            densities[index] = current_density
            index += 1

    acceptance = accepted_total / max(kept * thin, 1)
    if warn_acceptance and not 0.1 <= acceptance <= 0.6:
        warnings.warn(
            f"the Metropolis acceptance rate is {acceptance:.3f}, outside [0.1, 0.6]. "
            "The chain is either crawling or bouncing, so its samples are far more "
            "correlated than their count suggests and any interval read off them is "
            "narrower than it should be. Lengthen the burn-in, or pass an explicit "
            "scale=.",
            UserWarning,
            stacklevel=2,
        )
    return ChainResult(
        samples=samples[:index], log_density=densities[:index], acceptance=acceptance, scale=factor
    )


def _adapt(step: float, accepted_window: int, initial_step: float) -> tuple[float, int]:
    """Rescale the step towards the target acceptance rate, within bounds.

    The same multiplicative schedule
    :func:`hawkes_package.mcmc.mcmc_sampler` uses, and for the same reason: a
    0.9-per-step schedule cannot cross several orders of magnitude inside a
    burn-in, and a badly scaled start needs to. The bounds are the difference,
    and they are not decoration -- see :data:`_STEP_RANGE`.
    """
    rate = accepted_window / _ADAPT_EVERY
    scaled = step * float(np.clip(np.exp(2.0 * (rate - _TARGET_ACCEPTANCE)), 0.5, 2.0))
    low, high = _STEP_RANGE
    return float(np.clip(scaled, low * initial_step, high * initial_step)), 0


def _proposal_factor(
    scatter: np.ndarray, seen: int, step: float, n_dim: int, initial_step: float
) -> np.ndarray:
    """Cholesky factor of the adapted proposal covariance.

    The ridge is a fraction of the *initial* step rather than of the estimated
    covariance, which is what stops a collapsed chain staying collapsed: a
    relative ridge on a covariance that has gone to zero is zero itself.
    """
    floor = (_SHAPE_FLOOR * initial_step) ** 2
    if seen < 2 * n_dim:
        # Too few points for a covariance that is not mostly noise; a scaled
        # identity is a worse proposal than the truth and a better one than a
        # rank-deficient estimate of it.
        return step * np.eye(n_dim)
    covariance = scatter / (seen - 1)
    covariance = 0.5 * (covariance + covariance.T)
    return _cholesky_with_floor(step**2 * covariance + floor * np.eye(n_dim))


def batch_posterior(
    likelihood: LogLikelihood,
    prior: Prior,
    history: History,
    *,
    n_samples: int = 5000,
    burn_in: int | None = None,
    thin: int = 1,
    initial: Any = None,
    scale: float | None = None,
    rng: SeedLike = None,
    warn_acceptance: bool = True,
) -> ChainResult:
    """Sample the posterior over the whole history with one Metropolis chain.

    The independent second opinion on
    :class:`~hawkes_package.inference.smc.SMCSampler`: no blocks, no weights, no
    resampling, no rejuvenation -- just the full log-likelihood evaluated at
    every proposal. Slow by construction, and the point.

    The chain moves on the **unconstrained** scale and the samples come back
    constrained, so the returned array can be compared with a
    :class:`~hawkes_package.inference.smc.ParticleCloud` directly. The Jacobian
    of that transform is part of the target; without it the two samplers would
    disagree by a factor of ``theta`` per positive coordinate, which is exactly
    the kind of discrepancy that gets blamed on Monte Carlo error.

    Parameters
    ----------
    likelihood, prior, history :
        As for :class:`~hawkes_package.inference.smc.SMCSampler`.
    n_samples, burn_in, thin, scale, rng, warn_acceptance :
        Passed to :func:`metropolis_chain`.
    initial : array_like, optional
        Starting parameter vector, on the **constrained** scale. Defaults to a
        prior draw inside the support.

    Returns
    -------
    ChainResult
        With `samples` on the constrained scale.

    .. versionadded:: 0.5.0
    """
    model = likelihood.model
    spec = model.spec
    generator = np.random.default_rng(rng)

    def log_target(z: np.ndarray) -> float:
        theta = spec.to_constrained(z)
        if not bool(model.support(theta)):
            return -math.inf
        log_prior = float(np.atleast_1d(prior.log_pdf(theta))[0])
        if not np.isfinite(log_prior):
            return -math.inf
        value = likelihood.total(theta, history)
        if not np.isfinite(value):
            return -math.inf
        return log_prior + value + float(spec.log_abs_det_jacobian(z))

    if initial is None:
        start = _first_supported_draw(prior, model, generator)
    else:
        start = np.asarray(initial, dtype=float).ravel()

    result = metropolis_chain(
        log_target,
        spec.to_unconstrained(start),
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        scale=scale,
        rng=generator,
        warn_acceptance=warn_acceptance,
    )
    return ChainResult(
        samples=spec.to_constrained(result.samples),
        log_density=result.log_density,
        acceptance=result.acceptance,
        scale=result.scale,
    )


def _first_supported_draw(
    prior: Prior, model: Any, rng: np.random.Generator, tries: int = 1000
) -> np.ndarray:
    """Draw from `prior` until a parameter inside the model's support appears."""
    for _ in range(tries):
        candidates = np.atleast_2d(prior.sample(32, rng))
        allowed = np.asarray(model.support(candidates), dtype=bool)
        if np.any(allowed):
            return np.asarray(candidates[np.argmax(allowed)], dtype=float)
    raise RuntimeError(
        f"no prior draw landed inside the model's support in {tries} rounds of 32. The "
        "prior and the model disagree about which parameters exist; wrap the prior in "
        "ConstrainedPrior(prior, model.support), or pass initial= explicitly."
    )
