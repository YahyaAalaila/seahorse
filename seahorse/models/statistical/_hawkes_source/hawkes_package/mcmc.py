"""Random-walk Metropolis-Hastings sampling over a rectangular domain."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._numerics import as_float, as_point

__all__ = ["mcmc_sampler"]

#: Target acceptance rate for the burn-in adaptation (Roberts & Rosenthal).
_TARGET_ACCEPTANCE = 0.234
_ADAPT_EVERY = 50


def mcmc_sampler(
    density: Callable[[Any], float],
    space: Any,
    n_iter: int = 2000,
    burn_in: int = 500,
    proposal_std: Any = None,
    seed: int | np.random.Generator | np.random.SeedSequence | None = None,
    return_diagnostics: bool = False,
    *,
    x0: Any = None,
    transform: Callable[[Any], Any] | None = None,
    max_init_tries: int = 1000,
) -> Any:
    """Draw a single sample from an unnormalised density restricted to `space`.

    Parameters
    ----------
    density : callable
        Unnormalised target density. Receives a shape-``(ndim,)`` array and
        returns a non-negative scalar.
    space : array_like of shape (ndim, 2)
        Bounds for each dimension, e.g. ``[[-pi, pi]]`` in 1-D. The chain is
        confined to this box: the target is `density` restricted to it.
    n_iter : int
        Total number of iterations, burn-in included.
    burn_in : int
        Number of leading iterations discarded. Must be less than `n_iter`.
    proposal_std : float or array_like, optional
        Standard deviation of the Gaussian proposal, per axis. Defaults to a
        tenth of each axis's width, which keeps the chain scaled to the domain;
        a fixed scalar is wrong on a large domain and wrong again on an
        anisotropic one.
    seed : None, int or numpy.random.Generator
        Source of randomness. Passing a :class:`~numpy.random.Generator` shares
        one stream with the caller and advances it.
    return_diagnostics : bool
        If True, also return the post-burn-in acceptance rate.
    x0 : array_like, optional
        Explicit starting point. Use it when the target is supported on a tiny
        fraction of `space`, where a uniform search may fail to find it.
    transform : callable, optional
        Maps a proposal back into the domain, instead of rejecting it. Only
        valid for a genuinely periodic domain, where folding is a symmetric
        move on the quotient and therefore still reversible; on a domain whose
        ``wrap`` clips, folding would pile every draw onto the boundary.
    max_init_tries : int
        Uniform draws allowed when searching for a starting point.

    Returns
    -------
    sample : numpy.ndarray of shape (ndim,)
        The final state of the chain, always inside `space`.
    acceptance_rate : float
        Only when `return_diagnostics` is True.

    Raises
    ------
    ValueError
        If `burn_in` is not smaller than `n_iter`.
    RuntimeError
        If no starting point with positive density can be found.

    Notes
    -----
    .. versionchanged:: 0.2.0
       The chain is confined to `space`. Previously the proposal was an
       unbounded random walk that never rejected an out-of-domain move, and
       `space` bounded only the initial draw. That is correct for a target
       periodic with the domain, but for a non-periodic background the sampled
       marginal was indistinguishable from uniform noise, and on a domain whose
       ``wrap`` clips, every draw landed exactly on a boundary.
    """
    space = np.asarray(space, dtype=float)
    if space.ndim != 2 or space.shape[1] != 2:
        raise ValueError(f"space must have shape (ndim, 2), got {space.shape}")
    if burn_in >= n_iter:
        raise ValueError(f"burn_in ({burn_in}) must be smaller than n_iter ({n_iter})")

    rng = np.random.default_rng(seed)
    dimension = space.shape[0]
    widths = space[:, 1] - space[:, 0]

    std = (
        0.1 * widths
        if proposal_std is None
        else np.broadcast_to(np.asarray(proposal_std, dtype=float), (dimension,)).astype(float)
    )

    # -- Starting point ----------------------------------------------------
    if x0 is not None:
        x = as_point(x0, dimension)
        current = as_float(density(x))
        if not (np.isfinite(current) and current > 0):
            raise RuntimeError(f"density at x0={x.tolist()} is {current}, not positive and finite")
    else:
        x = rng.uniform(space[:, 0], space[:, 1])
        current = as_float(density(x))
        for _ in range(max_init_tries - 1):
            if np.isfinite(current) and current > 0:
                break
            x = rng.uniform(space[:, 0], space[:, 1])
            current = as_float(density(x))
        else:
            if not (np.isfinite(current) and current > 0):
                raise RuntimeError(
                    f"could not find a starting point with positive density in "
                    f"{max_init_tries} uniform draws over {space.tolist()}. The target "
                    "appears to be supported on a tiny fraction of `space`; pass an "
                    "explicit x0= inside the support, or narrow `space`."
                )

    accepted_window = 0
    accepted_sampling = 0

    for i in range(n_iter):
        proposal = x + rng.normal(0.0, 1.0, dimension) * std

        if transform is not None:
            proposal = as_point(transform(proposal), dimension)
        elif np.any(proposal < space[:, 0]) or np.any(proposal > space[:, 1]):
            # Out of the domain: the target is zero there, so this is a
            # rejection. Counted as one, so the adaptation still sees it.
            if i < burn_in and (i + 1) % _ADAPT_EVERY == 0:
                std, accepted_window = _adapt(std, accepted_window)
            continue

        proposed = as_float(density(proposal))
        # No division: a zero or non-finite value is a rejection. Dividing gave
        # 0/0 -> nan, and min(1.0, nan) is 1.0 in Python, so a failed chain
        # accepted every proposal and returned a point outside the support.
        if np.isfinite(proposed) and proposed > 0 and rng.uniform() * current < proposed:
            x, current = proposal, proposed
            accepted_window += 1
            if i >= burn_in:
                accepted_sampling += 1

        if i < burn_in and (i + 1) % _ADAPT_EVERY == 0:
            std, accepted_window = _adapt(std, accepted_window)

    acceptance_rate = accepted_sampling / (n_iter - burn_in)
    if return_diagnostics:
        return x, acceptance_rate
    return x


def _adapt(std: np.ndarray, accepted_window: int) -> tuple[np.ndarray, int]:
    """Rescale the proposal towards the target acceptance rate.

    Multiplicative on the per-axis vector, so anisotropy is preserved. The
    schedule is aggressive enough to rescue a badly scaled start inside the
    burn-in: a target 0.2 wide in a box 100 wide needs several orders of
    magnitude of contraction, which a 0.9-per-step schedule cannot deliver.
    """
    rate = accepted_window / _ADAPT_EVERY
    factor = float(np.clip(np.exp(2.0 * (rate - _TARGET_ACCEPTANCE)), 0.5, 2.0))
    return std * factor, 0
