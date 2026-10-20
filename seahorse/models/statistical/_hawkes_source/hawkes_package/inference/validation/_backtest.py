r"""Scoring a model on data it was not fitted to.

Every other check here is in-sample: the model is compared against the events
that chose its parameters. That is worth doing and it is not what a user wants
to know. The question is whether the fit would have been any use *prospectively*,
which is answered by refitting at a series of origins and scoring only what came
next.

The score is the out-of-sample log-likelihood of each block, conditional on
everything before it -- so the intensity still sees the whole past, and only the
*parameters* are restricted to the prefix. That distinction is the one worth
getting right: a Hawkes intensity depends on history by construction, so
withholding the past would not be an honest forecast, it would be a different
model.

**The window expands rather than slides.** That matches the state
:meth:`~hawkes_package.inference.estimator.HawkesEstimator.partial_fit` already
keeps, and a sliding window would need new state to answer a question nobody
asked. It also means later origins are fitted on strictly more data, which is
what makes an improving score meaningful.

Fitting is the caller's job. This module supplies the orchestration, the scoring
and the one guarantee that matters -- that the fitter is handed the prefix and
nothing else.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from ..likelihood import History, LogLikelihood
from ._baselines import homogeneous_log_likelihood

__all__ = ["Backtest", "OriginScore", "rolling_origin"]


@dataclass(frozen=True)
class OriginScore:
    """One origin's worth of out-of-sample performance.

    .. versionadded:: 1.0.0
    """

    origin: float
    horizon: float
    n_train: int
    n_test: int
    log_score: float
    baseline: float

    @property
    def skill(self) -> float:
        """Log-likelihood gained over a constant rate, per event of the test block.

        Per event because blocks differ in how many events they happen to hold,
        and a total would rank a busy block above a well-predicted one.
        """
        return (self.log_score - self.baseline) / self.n_test if self.n_test else 0.0


@dataclass(frozen=True)
class Backtest:
    """The scores from every origin, in order.

    .. versionadded:: 1.0.0
    """

    scores: tuple[OriginScore, ...]

    @property
    def mean_skill(self) -> float:
        """Mean skill over the origins that held any events."""
        usable = [s.skill for s in self.scores if s.n_test]
        return float(np.mean(usable)) if usable else 0.0

    @property
    def beat_baseline(self) -> int:
        """How many origins the model out-predicted a constant rate at."""
        return sum(1 for s in self.scores if s.n_test and s.skill > 0.0)

    def summary(self) -> str:
        """One line per origin, then the aggregate."""
        lines = [
            f"origin {s.origin:8.3f}: trained on {s.n_train:4d}, scored {s.n_test:3d}, "
            f"skill {s.skill:+.4f}"
            for s in self.scores
        ]
        lines.append(
            f"mean skill {self.mean_skill:+.4f} over {len(self.scores)} origins; "
            f"beat the constant rate at {self.beat_baseline}"
        )
        return "\n".join(lines)


def rolling_origin(
    likelihood: LogLikelihood,
    history: History,
    fit: Callable[[History], Any],
    *,
    origins: Sequence[float],
) -> Backtest:
    """Refit at each origin and score only what came after it.

    Parameters
    ----------
    likelihood : LogLikelihood
        Scores the blocks. Its parameters come from `fit`.
    history : History
        The whole observed record.
    fit : callable
        Given a :class:`History` truncated at an origin, return a parameter
        vector. **It is handed the prefix and nothing else**, which is the one
        guarantee this function exists to provide.
    origins : sequence of float
        Increasing times inside the window. Each block runs from one origin to
        the next, and the last runs to ``history.end``.

    Returns
    -------
    Backtest

    Raises
    ------
    ValueError
        If `origins` is empty, not increasing, or leaves the window.

    Notes
    -----
    The block score is ``total(upto=next) - total(upto=this)``: the whole past
    still drives the intensity, and only the parameters are restricted. Scoring
    a block on a likelihood that had been denied the earlier events would not be
    a stricter test, it would be a different model.

    .. versionadded:: 1.0.0
    """
    cuts = np.asarray(origins, dtype=float).ravel()
    if cuts.size == 0:
        raise ValueError("origins must name at least one cut")
    if np.any(np.diff(cuts) <= 0.0):
        raise ValueError("origins must be strictly increasing")
    if cuts[0] <= history.start or cuts[-1] >= history.end:
        raise ValueError(
            f"origins must lie strictly inside (start, end) = ({history.start}, "
            f"{history.end}), but they span [{cuts[0]}, {cuts[-1]}]"
        )

    boundaries = np.concatenate([cuts, [history.end]])
    scores = []
    for origin, horizon in pairwise(boundaries):
        prefix = history.upto(float(origin))
        theta = fit(prefix)

        before = float(likelihood.total(theta, history, float(origin)))
        after = float(likelihood.total(theta, history, float(horizon)))
        block = history.times[(history.times > origin) & (history.times <= horizon)]

        scores.append(
            OriginScore(
                origin=float(origin),
                horizon=float(horizon),
                n_train=prefix.n_events,
                n_test=int(block.size),
                log_score=after - before,
                baseline=homogeneous_log_likelihood(
                    likelihood,
                    History(block, None, float(origin), float(horizon)),
                ),
            )
        )
    return Backtest(scores=tuple(scores))
