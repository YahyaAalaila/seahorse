"""Named, bounded parameters and the transforms that unconstrain them.

Every sampler in this subpackage moves in an **unconstrained** space and reports
in the constrained one. The reason is not convenience. A Gaussian random-walk
proposal on a rate that must stay positive spends its time being rejected at the
boundary, and the closer the posterior sits to zero the worse it gets; the
acceptance rate collapses without anything failing, so the cloud stops moving
and the posterior contracts around wherever it happened to be. Moving in ``z``
and mapping back through ``theta = a + exp(z)`` removes the boundary instead of
guarding it.

That map is not measure-preserving, so the log-density of the target on the
``z`` scale carries the Jacobian term :meth:`ParameterSpec.log_abs_det_jacobian`.
Dropping it does not raise either: it tilts the posterior towards small values
of every positive parameter by a factor of ``theta`` per coordinate, which for a
three-parameter Hawkes model is a factor of ``mu * alpha * beta``.

Arrays are **batch-first** throughout: parameters are ``(n_particles, n_dim)``,
so the whole SMC loop stays vectorized. A single point of shape ``(n_dim,)`` is
accepted everywhere and comes back with the same rank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = ["Parameter", "ParameterSpec"]


@dataclass(frozen=True)
class Parameter:
    """One scalar model parameter and the open interval it lives on.

    Parameters
    ----------
    name : str
        Identifier, used in diagnostics and to look the coordinate up by name.
    lower, upper : float
        Open bounds. The defaults describe a positive rate, which is what nearly
        every parameter in a Hawkes model is. Both may be infinite; ``lower``
        must be strictly below ``upper``.

    .. versionadded:: 0.5.0
    """

    name: str
    lower: float = 0.0
    upper: float = math.inf

    def __post_init__(self) -> None:
        """Refuse an empty or inverted interval at construction."""
        if not self.name:
            raise ValueError("a parameter needs a name")
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError(f"bounds of {self.name!r} must not be nan")
        if not self.lower < self.upper:
            raise ValueError(
                f"parameter {self.name!r} has lower={self.lower} >= upper={self.upper}; "
                "the interval is open and must be non-empty"
            )

    @property
    def kind(self) -> str:
        """Which of the four transforms this parameter's bounds select."""
        below = math.isfinite(self.lower)
        above = math.isfinite(self.upper)
        if below and above:
            return "interval"
        if below:
            return "positive"
        if above:
            return "bounded_above"
        return "real"


def _as_batch(values: Any, n_dim: int, what: str) -> tuple[np.ndarray, bool]:
    """Coerce `values` to ``(n, n_dim)``, reporting whether it arrived flat."""
    array = np.asarray(values, dtype=float)
    flat = array.ndim == 1
    if flat:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != n_dim:
        raise ValueError(
            f"{what} must have shape (n_dim,) or (n_particles, n_dim) with n_dim="
            f"{n_dim}, got shape {np.shape(values)}"
        )
    return array, flat


@dataclass(frozen=True)
class ParameterSpec:
    """An ordered tuple of :class:`Parameter`, with the transforms it induces.

    The transforms act coordinatewise, one of four depending on which bounds are
    finite:

    ==================  ==========================  ==========================
    bounds              ``theta`` from ``z``        ``log |d theta / d z|``
    ==================  ==========================  ==========================
    ``(a, inf)``        ``a + exp(z)``              ``z``
    ``(-inf, b)``       ``b - exp(-z)``             ``-z``
    ``(-inf, inf)``     ``z``                       ``0``
    ``(a, b)``          ``a + (b - a) sigmoid(z)``  ``log(b-a) - z - 2 s(-z)``
    ==================  ==========================  ==========================

    with ``s(x) = log(1 + exp(x))``, written as :func:`numpy.logaddexp` so the
    interval case does not overflow for ``|z|`` past about 700 -- reachable in a
    rejuvenation move on a cloud that has collapsed onto a boundary, which is
    exactly the state a rejuvenation move exists to escape.

    Parameters
    ----------
    parameters : tuple of Parameter
        The coordinates, in the order the model consumes them.

    .. versionadded:: 0.5.0
    """

    parameters: tuple[Parameter, ...]

    def __post_init__(self) -> None:
        """Refuse duplicate names, which would make :meth:`index` ambiguous."""
        if not self.parameters:
            raise ValueError("a parameter spec needs at least one parameter")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate parameter name(s) {duplicates}")

    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        """The parameter names, in order."""
        return tuple(p.name for p in self.parameters)

    @property
    def lower(self) -> np.ndarray:
        """Lower bounds, shape ``(n_dim,)``."""
        return np.array([p.lower for p in self.parameters], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        """Upper bounds, shape ``(n_dim,)``."""
        return np.array([p.upper for p in self.parameters], dtype=float)

    def index(self, name: str) -> int:
        """Position of the coordinate called `name`.

        Raises
        ------
        KeyError
            If no parameter has that name.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"no parameter named {name!r}; this spec has {list(self.names)}"
            ) from None

    def contains(self, theta: Any) -> np.ndarray:
        """Whether each row of `theta` lies strictly inside every bound.

        Strictly, because the transforms are onto the *open* interval: a value
        sitting exactly on a bound has no finite image, and admitting one would
        put a ``-inf`` or a ``nan`` into the cloud rather than rejecting the
        particle.

        Parameters
        ----------
        theta : array_like
            Shape ``(n_dim,)`` or ``(n_particles, n_dim)``.

        Returns
        -------
        numpy.ndarray
            Boolean, shape ``()`` or ``(n_particles,)``.
        """
        batch, flat = _as_batch(theta, len(self), "theta")
        inside = (
            np.isfinite(batch).all(axis=1)
            & (batch > self.lower).all(axis=1)
            & (batch < self.upper).all(axis=1)
        )
        return np.asarray(inside[0] if flat else inside, dtype=bool)

    def to_unconstrained(self, theta: Any) -> np.ndarray:
        """Map constrained parameters to the real line.

        Raises
        ------
        ValueError
            If any row lies outside the open bounds. Returning ``nan`` instead
            would travel: the cloud would carry it through a covariance, a
            Cholesky factor and every subsequent proposal, and the first thing
            to fail would be a linear-algebra call several steps away from the
            particle that caused it.
        """
        batch, flat = _as_batch(theta, len(self), "theta")
        if not np.all(self.contains(batch)):
            bad = int(np.argmin(self.contains(batch)))
            raise ValueError(
                f"theta[{bad}] = {batch[bad].tolist()} lies outside the open bounds "
                f"{list(zip(self.names, self.lower.tolist(), self.upper.tolist(), strict=True))}"
            )

        out = np.empty_like(batch)
        for j, parameter in enumerate(self.parameters):
            column = batch[:, j]
            kind = parameter.kind
            if kind == "positive":
                out[:, j] = np.log(column - parameter.lower)
            elif kind == "bounded_above":
                out[:, j] = -np.log(parameter.upper - column)
            elif kind == "real":
                out[:, j] = column
            else:
                width = parameter.upper - parameter.lower
                unit = (column - parameter.lower) / width
                out[:, j] = np.log(unit) - np.log1p(-unit)
        return out[0] if flat else out

    def to_constrained(self, z: Any) -> np.ndarray:
        """Map unconstrained coordinates back into the bounds.

        A coordinate large enough to overflow ``exp`` comes back as ``inf``
        rather than raising: :meth:`contains` then reports it outside the bounds
        and the sampler rejects the particle, which is the same treatment any
        other unusable proposal gets. Raising here instead would turn one wild
        proposal into a failed run.
        """
        batch, flat = _as_batch(z, len(self), "z")
        out = np.empty_like(batch)
        # `over` only: an underflow to zero is a parameter pinned at its bound,
        # which `contains` rejects, and a proposal that far out is a legitimate
        # thing for a random walk to try.
        with np.errstate(over="ignore"):
            for j, parameter in enumerate(self.parameters):
                column = batch[:, j]
                kind = parameter.kind
                if kind == "positive":
                    out[:, j] = parameter.lower + np.exp(column)
                elif kind == "bounded_above":
                    out[:, j] = parameter.upper - np.exp(-column)
                elif kind == "real":
                    out[:, j] = column
                else:
                    width = parameter.upper - parameter.lower
                    # expit, written so neither tail overflows.
                    out[:, j] = parameter.lower + width / (1.0 + np.exp(-column))
        return out[0] if flat else out

    def log_abs_det_jacobian(self, z: Any) -> np.ndarray:
        """Log absolute determinant of ``d theta / d z`` at `z`.

        The transform is coordinatewise, so the Jacobian is diagonal and this is
        a sum of scalar terms. It is added to the log prior wherever the target
        is evaluated on the ``z`` scale -- see the module docstring for what
        omitting it does.

        Returns
        -------
        numpy.ndarray
            Shape ``()`` or ``(n_particles,)``.
        """
        batch, flat = _as_batch(z, len(self), "z")
        total = np.zeros(batch.shape[0], dtype=float)
        for j, parameter in enumerate(self.parameters):
            column = batch[:, j]
            kind = parameter.kind
            if kind == "positive":
                total += column
            elif kind == "bounded_above":
                total -= column
            elif kind == "interval":
                width = parameter.upper - parameter.lower
                # log(width) + log sigma(z) + log(1 - sigma(z)), with both
                # sigmoid terms through logaddexp so a z of -1000 gives -1000
                # rather than an overflow warning and a nan.
                total += math.log(width) - column - 2.0 * np.logaddexp(0.0, -column)
        return total[0] if flat else total

    def prefixed(self, prefix: str) -> ParameterSpec:
        """Return this spec with `prefix` prepended to every name.

        What lets a composed model carry two kernels without their ``alpha``
        coordinates colliding.
        """
        return ParameterSpec(tuple(replace(p, name=f"{prefix}{p.name}") for p in self.parameters))

    def concat(self, other: ParameterSpec) -> ParameterSpec:
        """Return the spec holding this one's parameters followed by `other`'s."""
        return ParameterSpec(self.parameters + other.parameters)
