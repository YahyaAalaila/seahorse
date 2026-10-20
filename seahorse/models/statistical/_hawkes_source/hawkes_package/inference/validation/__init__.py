"""Checking a fitted model against data, independently of how it was fitted.

:mod:`hawkes_package.inference.diagnostics` answers "do the two implementations
of this model agree". This subpackage answers a different and harder question:
**is the model right at all**, judged without reusing the arithmetic that
produced it.

That distinction is the whole reason this is a separate module rather than more
functions in the old one. ``residuals`` takes its compensator from the
``LogLikelihood`` it is handed, and its docstring calls that a feature -- "a bug
in it cannot pass this test and fail the fit or the reverse". For checking two
implementations against each other that is exactly right. For *validation* it is
exactly backwards, because of one specific cancellation:

**A fit made with a compensator 20% too small inflates the intensity by 25%, and
the two errors cancel exactly.** Rescale the events through the same broken
integral that produced the fit and the gaps come out unit-rate. The diagnostic
reports a clean bill of health, and the more badly the compensator is wrong the
more precisely the fit compensates for it.

So validation gets a second source: :func:`independent_compensator` integrates
the simulator's own intensity hook on a dense uniform grid -- a different
quadrature family, a different code path, and the intensity read from the hook
the simulator thins against rather than from the likelihood's cached
rearrangement. Two routes to one number. Where they disagree, one is wrong, and
saying so is more useful than averaging them.

.. versionadded:: 1.0.0
"""

from __future__ import annotations

from ._backtest import Backtest, OriginScore, rolling_origin
from ._baselines import (
    BaselineComparison,
    compare_with_baseline,
    homogeneous_log_likelihood,
)
from ._cells import CellResiduals, cell_residuals
from ._compensator import compensator_agreement, independent_compensator

__all__ = [
    "Backtest",
    "BaselineComparison",
    "CellResiduals",
    "OriginScore",
    "cell_residuals",
    "compare_with_baseline",
    "compensator_agreement",
    "homogeneous_log_likelihood",
    "independent_compensator",
    "rolling_origin",
]
