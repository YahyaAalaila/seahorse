r"""How the parameter is allowed to move between blocks.

Two different statistical models hide behind one loop, and the difference is
worth stating plainly because the same code produces both.

**Static.** The parameter does not change. The sequence of targets is then the
data-tempered posteriors :math:`\pi_k(\theta) \propto p(\theta) e^{\ell_k}`, and
the block increments telescope: adding block ``k``'s increment to the log weight
is *exactly* reweighting from :math:`\pi_{k-1}` to :math:`\pi_k`. This is
Chopin's IBIS, and it is exact.

**Drifting.** The parameter is a latent state with its own dynamics, and the
sampler is a genuine particle filter over it. This costs an approximation that
does not announce itself: the block increment is computed with :math:`\theta_k`
for the *whole* intensity over that block, including the excitation contributed
by events generated earlier under different parameters. That is a
locally-stationary approximation, not an exact factorisation of the likelihood.
It is a good one when the drift is slow compared with the kernel's memory and a
poor one when it is not, and in neither case does anything raise -- the
posterior is simply more confident than it has earned. Under
:class:`Static` the same expression is exact, which is why the distinction lives
here rather than in a footnote.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = ["Evolution", "LiuWest", "RandomWalkDrift", "Static"]


@runtime_checkable
class Evolution(Protocol):
    """How particles move between blocks, on the unconstrained scale.

    Every implementation works in ``z``, not in ``theta``. A jitter that
    respected the bounds in ``theta`` would have to be truncated at them, and a
    truncated random walk pushes mass away from the boundary -- so a parameter
    genuinely near zero would drift upward at a rate set by the jitter, which
    looks exactly like a real trend.

    Read :mod:`~hawkes_package.inference.evolution`'s own docstring before
    choosing anything but :class:`Static`: a drifting parameter makes the block
    likelihood an approximation, and the sampler cannot tell you that it has
    become a bad one.
    """

    @property
    def static(self) -> bool:
        """Whether the target is a fixed posterior, so an invariant move means something."""
        ...

    def propagate(
        self,
        z: np.ndarray,
        log_weights: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the moved particles, shape ``(n_particles, n_dim)``."""
        ...


class Static:
    """The parameter does not change: the identity map.

    The default, and the only setting under which the sequence of targets is
    the exact data-tempered posterior and rejuvenation by an invariant MCMC
    kernel means anything.

    .. versionadded:: 0.5.0
    """

    static = True

    def __repr__(self) -> str:
        return "Static()"

    def propagate(
        self,
        z: np.ndarray,
        log_weights: np.ndarray,  # noqa: ARG002 - part of the protocol
        rng: np.random.Generator,  # noqa: ARG002 - part of the protocol
    ) -> np.ndarray:
        """Return `z` unchanged."""
        return z


class RandomWalkDrift:
    r"""A Gaussian random walk on the unconstrained scale.

    :math:`z' = z + \varepsilon`, :math:`\varepsilon \sim N(0, \mathrm{scale}^2)`.
    The simplest way to let a parameter move, and the one whose cost is easiest
    to see: the transition adds variance at every block whether or not the data
    supports a change, so the posterior can never contract past what one block's
    worth of data supports. Choose `scale` from how fast the parameter is
    believed to move, not from how well the filter tracks -- tuning it upward
    until the fit follows the data is fitting the noise.

    Parameters
    ----------
    scale : float or array_like
        Standard deviation per coordinate on the unconstrained scale, so for a
        positive parameter it is a *relative* step: 0.05 is about five per cent
        per block.

    .. versionadded:: 0.5.0
    """

    static = False

    def __init__(self, scale: Any = 0.05) -> None:
        values = np.asarray(scale, dtype=float)
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError(f"scale must be finite and positive, got {scale!r}")
        self.scale = values

    def __repr__(self) -> str:
        return f"RandomWalkDrift({self.scale.tolist()!r})"

    def propagate(
        self,
        z: np.ndarray,
        log_weights: np.ndarray,  # noqa: ARG002 - part of the protocol
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add independent Gaussian noise to every particle."""
        return np.asarray(z + rng.normal(0.0, 1.0, size=z.shape) * self.scale, dtype=float)


class LiuWest:
    r"""Liu and West's shrinkage jitter, on the unconstrained scale.

    A plain random walk inflates the cloud's variance at every block, so a
    parameter that is genuinely constant looks increasingly uncertain the longer
    it is watched. Liu and West's kernel adds the same jitter and pulls each
    particle back towards the weighted mean by exactly enough to cancel it:

    .. math::

        a = \frac{3\delta - 1}{2\delta}, \quad h^2 = 1 - a^2, \quad
        z' \sim N\!\left(a z_i + (1 - a)\bar{z},\ h^2 V\right),

    which leaves :math:`E[z'] = \bar{z}` and :math:`\mathrm{Var}[z'] = V`
    exactly. At the default :math:`\delta = 0.95` that is ``a = 0.9737`` and
    ``h^2 = 0.0519``.

    This is the simplified, non-auxiliary variant: the jitter is applied without
    the auxiliary-particle-filter lookahead that the original paper pairs it
    with. The lookahead needs a one-step-ahead predictive likelihood, which for
    a self-exciting process would mean simulating forward from every particle.

    The approximation described in this module's docstring applies: with a
    drifting parameter the block likelihood is locally stationary rather than
    exact, and the resulting tracking posterior is more confident than it should
    be.

    Parameters
    ----------
    delta : float
        Discount factor in ``(0.5, 1]``. Nearer one means less movement per
        block; at exactly one this is :class:`Static` with a resample. Below 0.5
        the shrinkage coefficient turns negative, which reflects the cloud
        through its mean rather than contracting it.

    .. versionadded:: 0.5.0
    """

    static = False

    def __init__(self, delta: float = 0.95) -> None:
        value = float(delta)
        if not 0.5 < value <= 1.0:
            raise ValueError(
                f"delta must lie in (0.5, 1], got {value}; at or below 0.5 the shrinkage "
                "coefficient (3*delta - 1) / (2*delta) is not positive and the kernel "
                "reflects the cloud through its mean instead of contracting it"
            )
        self.delta = value
        self.shrinkage = (3.0 * value - 1.0) / (2.0 * value)
        self.jitter_variance = 1.0 - self.shrinkage**2

    def __repr__(self) -> str:
        return f"LiuWest({self.delta!r})"

    def propagate(
        self,
        z: np.ndarray,
        log_weights: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Shrink towards the weighted mean, then add the matching jitter."""
        weights = np.exp(np.asarray(log_weights, dtype=float).ravel())
        mean = np.average(z, axis=0, weights=weights)
        centred = z - mean
        covariance = (centred * weights[:, None]).T @ centred
        # Symmetrise before factorising: the outer product above is symmetric in
        # exact arithmetic and is not quite symmetric in floating point, and
        # `cholesky` reads only the lower triangle -- so an unsymmetrised matrix
        # silently uses half of a matrix that is not the one intended.
        covariance = 0.5 * (covariance + covariance.T)
        scale = _cholesky_with_floor(self.jitter_variance * covariance)
        shifted = self.shrinkage * z + (1.0 - self.shrinkage) * mean
        return np.asarray(shifted + rng.normal(0.0, 1.0, size=z.shape) @ scale.T, dtype=float)


def _cholesky_with_floor(covariance: np.ndarray) -> np.ndarray:
    """Cholesky factor of `covariance`, nudged until it succeeds.

    The cloud that most needs a jitter is the one that has collapsed, and a
    collapsed cloud has a singular covariance -- so the factorisation fails
    exactly when it is most needed. Adding a multiple of the trace to the
    diagonal is scale-free and vanishes as the cloud recovers.
    """
    n_dim = covariance.shape[0]
    floor = 1e-10 * max(float(np.trace(covariance)) / n_dim, np.finfo(float).tiny)
    for attempt in range(8):
        try:
            return np.asarray(
                np.linalg.cholesky(covariance + floor * (10.0**attempt) * np.eye(n_dim)),
                dtype=float,
            )
        except np.linalg.LinAlgError:  # pragma: no cover - the loop escalates fast
            continue
    # Every particle identical in every coordinate: no direction left to move in.
    return np.zeros((n_dim, n_dim), dtype=float)


def default_scale(n_dim: int) -> float:
    r"""Return the optimal random-walk scaling :math:`2.38 / \sqrt{d}`.

    Roberts and Rosenthal's result for a Gaussian target: a proposal covariance
    of :math:`(2.38^2 / d)\,\hat{C}` maximises the asymptotic efficiency of a
    random-walk Metropolis chain, and lands the acceptance rate near 0.234. The
    same constant the spatial location sampler in
    :mod:`hawkes_package.mcmc` adapts towards, reused as an idea rather than as
    code.
    """
    return 2.38 / math.sqrt(max(n_dim, 1))
