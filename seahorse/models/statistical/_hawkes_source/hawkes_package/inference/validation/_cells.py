r"""Residuals on a partition of the observation window.

Time rescaling answers "did the events arrive at the right *rate*". It cannot
answer "did they arrive in the right *places*", because it collapses space
before it starts -- so a model that puts the right number of events on the wrong
side of the domain passes it cleanly.

Cell residuals answer both at once. Partition the window into cells :math:`C`
and compare what was observed against what was predicted:

.. math::

    R(C) = N(C) - \int_C \lambda, \qquad
    R^{*}(C) = \frac{R(C)}{\sqrt{\int_C \lambda}}.

Under the true model :math:`N(C)` is Poisson with mean :math:`\int_C \lambda`, so
:math:`R^{*}` is approximately standard normal wherever that mean is not tiny --
which is the whole caveat, and why cells below a floor are excluded rather than
reported. Standardising by a near-zero expectation turns one stray event into a
residual of forty and buries every real signal beside it.

The integrals come from :func:`independent_compensator`, not from the
likelihood's own quadrature. That is the point of the subpackage: see its module
docstring for the cancellation it exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..likelihood import History, LogLikelihood
from ._compensator import independent_compensator

__all__ = ["CellResiduals", "cell_residuals"]

#: Cells expected to hold fewer events than this are reported but excluded from
#: the standardised summary. The normal approximation to a Poisson count is
#: conventionally taken as usable from about five; below one, dividing by
#: ``sqrt(mean)`` amplifies a single event into a residual large enough to
#: dominate every real one.
_MIN_EXPECTED = 1.0


@dataclass(frozen=True)
class CellResiduals:
    r"""Observed against expected counts on a partition of the window.

    Attributes
    ----------
    edges : numpy.ndarray
        Cell boundaries in time, shape ``(n_cells + 1,)``.
    counts : numpy.ndarray
        Events observed in each cell, shape ``(n_cells,)``.
    expected : numpy.ndarray
        :math:`\int_C \lambda` for each cell.
    usable : numpy.ndarray
        Boolean, shape ``(n_cells,)``: whether the expected count is large
        enough for the standardised residual to mean anything.

    .. versionadded:: 1.0.0
    """

    edges: np.ndarray
    counts: np.ndarray
    expected: np.ndarray
    usable: np.ndarray

    @property
    def raw(self) -> np.ndarray:
        """Return ``N(C) - int_C lambda``, which has mean zero under the true model."""
        return np.asarray(self.counts - self.expected, dtype=float)

    @property
    def standardised(self) -> np.ndarray:
        """The raw residual over ``sqrt(expected)``, approximately standard normal.

        ``nan`` in cells the expectation is too small to standardise by; use
        :attr:`usable` to select, rather than filtering on ``isfinite`` and
        hoping.
        """
        out = np.full(self.expected.shape, np.nan, dtype=float)
        safe = self.expected > 0.0
        out[safe] = self.raw[safe] / np.sqrt(self.expected[safe])
        out[~self.usable] = np.nan
        return out

    @property
    def total_expected(self) -> float:
        """Expected events over the whole window: the compensator at the end."""
        return float(np.sum(self.expected))

    def summary(self) -> str:
        """One line of counts and one of the standardised spread."""
        values = self.standardised[self.usable]
        head = (
            f"{int(self.counts.sum())} events observed against "
            f"{self.total_expected:.1f} expected over {self.counts.size} cells "
            f"({int(np.sum(self.usable))} usable)"
        )
        if values.size == 0:
            return head + "\nno cell holds enough expected events to standardise"
        worst = int(np.argmax(np.abs(values)))
        return (
            f"{head}\nstandardised residuals: mean {float(np.mean(values)):+.3f}, "
            f"sd {float(np.std(values)):.3f}, worst {float(values[worst]):+.2f}"
        )


def cell_residuals(
    likelihood: LogLikelihood,
    theta: Any,
    history: History,
    *,
    n_cells: int = 10,
    compensator: Any = None,
) -> CellResiduals:
    r"""Compare observed and expected counts on equal time cells.

    Parameters
    ----------
    likelihood : LogLikelihood
        Supplies the model. Its own compensator is **not** used unless
        `compensator` says so.
    theta : array_like
        A single parameter vector.
    history : History
        The observed events.
    n_cells : int
        Number of equal-width cells spanning ``[start, end]``.
    compensator : callable, optional
        Override the integrator, taking ``(theta, history, times)``. Pass
        ``likelihood.compensator`` to reproduce the shared-arithmetic behaviour
        deliberately -- which is what the guard test does, to show the two
        differ.

    Returns
    -------
    CellResiduals

    Raises
    ------
    ValueError
        If `n_cells` is not positive.

    Notes
    -----
    Equal-width cells rather than equal-count ones. Equal counts would put the
    boundaries where the events are, which makes every cell's observed count the
    same by construction and hides exactly the clustering a Hawkes model is for.

    .. versionadded:: 1.0.0
    """
    cells = int(n_cells)
    if cells < 1:
        raise ValueError(f"n_cells must be positive, got {n_cells!r}")

    edges = np.linspace(float(history.start), float(history.end), cells + 1)
    counts = np.histogram(history.times, bins=edges)[0].astype(float)

    integrate = (
        (lambda t: independent_compensator(likelihood, theta, history, t))
        if compensator is None
        else (lambda t: np.asarray(compensator(theta, history, t), dtype=float))
    )
    cumulative = np.concatenate([[0.0], integrate(edges[1:])])
    expected = np.diff(cumulative)

    return CellResiduals(
        edges=edges,
        counts=counts,
        expected=expected,
        usable=expected >= _MIN_EXPECTED,
    )
