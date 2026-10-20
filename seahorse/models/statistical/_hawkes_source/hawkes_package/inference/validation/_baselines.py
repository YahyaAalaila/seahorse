r"""What a fitted model has to beat before it has shown anything.

A goodness-of-fit test says the model is *not obviously wrong*. It does not say
the model is worth having. A Hawkes fit that passes every residual check and
predicts no better than a constant rate has found nothing, and nothing in the
existing diagnostics would say so.

The bar is the **homogeneous Poisson process at its own maximum likelihood**,
:math:`\hat\nu = n / T`. Not a Poisson process at some convenient rate: the best
constant-rate model there is, so that beating it cannot be an artefact of having
handed the baseline a bad parameter. Its log-likelihood has a closed form,

.. math::

    \ell_0 = n \log \hat\nu - \hat\nu T = n \log(n / T) - n,

with :math:`T` the window length -- or the window length times the domain
measure for a spatio-temporal model, since there the intensity is a density per
unit area as well as per unit time and the two log-likelihoods are otherwise not
comparable.

**A baseline that is too weak is worse than none**, because beating it reads as
evidence. That is why the rate is the MLE and not the fitted background: a
constant rate fixed at the fitted ``mu`` is *guaranteed* to lose, since ``mu``
sits below the observed rate precisely because the excitation accounts for the
rest. Reporting a win against that would be reporting arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..likelihood import History, LogLikelihood
from ..models import SpatialComponents

__all__ = ["BaselineComparison", "compare_with_baseline", "homogeneous_log_likelihood"]


def _window_measure(likelihood: LogLikelihood, history: History) -> float:
    """Length of the observation window, times the domain measure where there is one.

    A spatio-temporal intensity is a density per unit area as well as per unit
    time, so its log-likelihood carries a ``-n log|D|`` that a temporal one does
    not. Comparing the two without it would credit or penalise the model for the
    size of the domain it happens to live on.
    """
    duration = float(history.end - history.start)
    model = getattr(likelihood, "model", None)
    components = getattr(model, "components", None)
    if isinstance(components, SpatialComponents):
        return duration * float(components.domain.volume)
    return duration


def homogeneous_log_likelihood(likelihood: LogLikelihood, history: History) -> float:
    r"""Log-likelihood of the best constant-rate process on this window.

    The maximum over :math:`\nu` of :math:`n\log\nu - \nu V`, attained at
    :math:`\hat\nu = n / V` and equal to :math:`n\log(n/V) - n`.

    Returns
    -------
    float
        ``-inf`` for a window with no events, which is what a Poisson process
        with rate zero assigns to anything.

    .. versionadded:: 1.0.0
    """
    n = history.n_events
    measure = _window_measure(likelihood, history)
    if n == 0 or measure <= 0.0:
        return -math.inf
    return float(n * math.log(n / measure) - n)


@dataclass(frozen=True)
class BaselineComparison:
    """A fitted model beside the constant-rate process it has to beat.

    .. versionadded:: 1.0.0
    """

    fitted: float
    homogeneous: float
    n_events: int

    @property
    def improvement(self) -> float:
        """Log-likelihood gained over the baseline, in nats."""
        return self.fitted - self.homogeneous

    @property
    def per_event(self) -> float:
        """The same, per event -- which is what makes two windows comparable."""
        return self.improvement / self.n_events if self.n_events else 0.0

    @property
    def beats_baseline(self) -> bool:
        """Whether the fit is worth having at all."""
        return self.improvement > 0.0

    def summary(self) -> str:
        """One line, saying plainly whether the model earned its complexity."""
        verdict = "beats" if self.beats_baseline else "LOSES TO"
        return (
            f"log-likelihood {self.fitted:.2f} against {self.homogeneous:.2f} for the "
            f"best constant rate: {verdict} the baseline by {self.improvement:+.2f} nats "
            f"({self.per_event:+.4f} per event over {self.n_events})"
        )


def compare_with_baseline(
    likelihood: LogLikelihood, theta: Any, history: History
) -> BaselineComparison:
    """Score a fitted model against the best constant-rate process.

    Parameters
    ----------
    likelihood : LogLikelihood
        The fitted model's likelihood.
    theta : array_like
        A single parameter vector, usually the posterior mean.
    history : History
        The data both are scored on.

    Returns
    -------
    BaselineComparison

    Notes
    -----
    Both log-likelihoods are on the same window and the same events, so the
    difference is a likelihood ratio and needs no further normalisation. It is
    *not* a hypothesis test: the models are not nested in a way that makes the
    usual asymptotics apply, and the number is reported as a margin rather than
    dressed as a p-value.

    .. versionadded:: 1.0.0
    """
    return BaselineComparison(
        fitted=float(likelihood.total(theta, history)),
        homogeneous=homogeneous_log_likelihood(likelihood, history),
        n_events=history.n_events,
    )
