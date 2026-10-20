r"""Maximum likelihood, beside the sequential machinery rather than instead of it.

The package's position is that the sequential Monte Carlo path is the recommended
one: it reports a posterior, it absorbs data in blocks, and its diagnostics say
when it has failed. This module exists for two things that position cannot
supply. A reviewer comparing point-process libraries runs a maximum-likelihood
fit, and "we do SMC instead" is not an answer they can check. And a mode is a
good place to centre an initial cloud, which
:func:`~hawkes_package.inference.mle.warm_start_proposal` turns into a real saving.

**SciPy is used here, and that widens a stated rule.** Runtime dependencies are
still numpy and scipy only, but until 1.0.0 scipy was held to a single call
site -- ``minimize_scalar`` in :mod:`hawkes_package._numerics` -- which is why
:func:`~hawkes_package.inference.diagnostics.ks_exponential` hand-rolls the
Kolmogorov series and :mod:`hawkes_package.inference.priors` hand-writes its
marginals. The rule is now that scipy is reached for where a hand-written version
would be *worse*, not wherever it is convenient. An optimiser is squarely on that
side; a twenty-line series is not, and those stay hand-written.

Three things here are not obvious, and each is a way an MLE goes quietly wrong.

**The Jacobian is deliberately absent.** The optimisation runs on the
unconstrained scale, through
:meth:`~hawkes_package.inference.parameters.ParameterSpec.to_constrained`, and
the log-determinant of that transform is *not* added. Adding it would be natural
-- the sampler does it -- and would silently return the mode of a density under a
flat prior on the unconstrained scale, which is a different estimator wearing the
same name.

**An unconstrained maximiser has no prior to protect it from a bad compensator.**
Every unit of :math:`\int\lambda` that goes missing is a penalty on a high
intensity that never gets applied, so the optimiser takes it: the background and
the excitation both come back large and the fit looks excellent. So the
compensator's resolution check is re-run **at the optimum**, where the integrand
is sharpest, and not only at the starting value.

**An inverse Hessian is not an interval.** At an optimum near the stationarity
boundary a differenced Hessian can come back indefinite, and the standard error
it reports is a plausible number with no content -- the same failure as a
collapsed particle cloud reading as confidence. :func:`profile_interval` walks
the likelihood instead.

.. versionadded:: 1.0.0
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .likelihood import History, LogLikelihood, _bind_history
from .models import ProcessModel
from .parameters import ParameterSpec
from .priors import IndependentPrior, Normal, Prior

__all__ = [
    "HawkesMLE",
    "MaximumLikelihoodFit",
    "fit_mle",
    "profile_interval",
    "warm_start_proposal",
]

#: Half of the chi-squared(1) quantile at a few standard levels, which is the
#: drop in log-likelihood that bounds a profile interval. Tabulated rather than
#: taken from `scipy.stats`: three numbers do not justify the import, and the
#: widened rule above is about optimisers, not about everything.
_CHI2_HALF = {0.90: 1.352772, 0.95: 1.920729, 0.99: 3.317448}

#: What the objective returns outside the support. **Finite on purpose**, though
#: `inf` is the honest value: Nelder-Mead's own convergence test computes
#: ``max|f[0] - f[1:]|``, which is ``inf - inf`` and therefore `nan` as soon as
#: two simplex vertices sit outside, and numpy warns. A suite running
#: ``filterwarnings = ["error"]`` -- this one, and any careful downstream one --
#: then fails inside SciPy, on a fit that was going perfectly well.
#:
#: A minimiser treats a large value as a wall rather than a direction, and 1e12
#: is roughly eight orders above the largest log-likelihood any realistic history
#: produces, so nothing outside the support can ever win. The returned optimum is
#: checked against the support anyway, because "can never win" is an argument and
#: the check is a fact.
_OUTSIDE = 1e12


@dataclass(frozen=True)
class MaximumLikelihoodFit:
    """The result of :func:`fit_mle`.

    Attributes
    ----------
    theta : numpy.ndarray
        The maximiser, on the model's own (constrained) scale.
    log_likelihood : float
        Its log-likelihood, in the same units
        :meth:`~hawkes_package.inference.likelihood.LogLikelihood.total` returns.
    n_evaluations : int
        Objective evaluations the optimiser used. Worth reading: a fit that took
        thousands is one whose objective is rough, usually a compensator that is
        not resolved.
    converged : bool
        What the optimiser reported. ``False`` is accompanied by a warning at fit
        time -- see :func:`fit_mle`.
    message : str
        The optimiser's own account of why it stopped.
    spec : ParameterSpec
        The coordinates, so a result can name its own numbers.
    """

    theta: np.ndarray
    log_likelihood: float
    n_evaluations: int
    converged: bool
    message: str
    spec: ParameterSpec
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def named(self) -> dict[str, float]:
        """Return the estimate as ``{name: value}``."""
        return {name: float(v) for name, v in zip(self.spec.names, self.theta, strict=True)}

    def __repr__(self) -> str:
        """Show the estimate by name, which is what a reader wants."""
        body = ", ".join(f"{k}={v:.4g}" for k, v in self.named().items())
        state = "converged" if self.converged else "NOT CONVERGED"
        return f"MaximumLikelihoodFit({body}, log_lik={self.log_likelihood:.4f}, {state})"


def _model_of(likelihood: LogLikelihood) -> ProcessModel:
    """Return the model a likelihood was built for."""
    model = getattr(likelihood, "model", None)
    if model is None:
        raise ValueError(
            f"{type(likelihood).__name__} does not carry a `model`, so its "
            "parameters have no names, bounds or support to optimise inside"
        )
    if not isinstance(model, ProcessModel):  # pragma: no cover - defensive
        raise TypeError(f"{type(likelihood).__name__}.model is a {type(model).__name__}")
    return model


def _objective(likelihood: LogLikelihood, history: History, model: ProcessModel) -> Any:
    """Return the negative log-likelihood as a function of the unconstrained vector."""
    spec = model.spec

    def negative(z: np.ndarray) -> float:
        theta = spec.to_constrained(np.asarray(z, dtype=float))
        if not bool(np.asarray(model.support(theta)).ravel()[0]):
            return _OUTSIDE
        value = float(likelihood.total(theta, history))
        # A `-inf` log-likelihood -- a mark below its threshold, an intensity
        # that reached zero at an event -- is the same situation as being outside
        # the support, and gets the same wall for the same reason.
        return -value if math.isfinite(value) else _OUTSIDE

    return negative


def fit_mle(
    likelihood: LogLikelihood,
    history: History,
    start: Any,
    *,
    method: str = "Nelder-Mead",
    max_iterations: int | None = None,
    tol: float | None = None,
    recheck_resolution: bool = True,
) -> MaximumLikelihoodFit:
    r"""Maximise `likelihood` over the model's parameters.

    The objective is the same
    :meth:`~hawkes_package.inference.likelihood.LogLikelihood.total` the
    sequential sampler uses -- never a second expression of the intensity, which
    is the documented way this package has gone wrong before.

    Parameters
    ----------
    likelihood : LogLikelihood
        Any of the likelihoods, carrying the model whose parameters are fitted.
    history : History
        The data.
    start : array_like
        Starting parameters, on the model's own scale. Must lie inside the
        support: a start the process cannot be built at is refused here rather
        than becoming an ``inf`` the optimiser cannot escape from.
    method : str
        Passed to :func:`scipy.optimize.minimize`. The default is
        ``"Nelder-Mead"`` on purpose. The objective is ``inf`` outside the
        support, and a gradient-based method differencing across that boundary
        takes a step nobody can interpret, where a simplex reflects away from it.
    max_iterations : int, optional
        Iteration cap. ``None`` leaves the method's own default.
    tol : float, optional
        Passed through as the method's tolerance.
    recheck_resolution : bool
        Re-run the compensator's order-``P``-versus-``2P`` check at the optimum.
        Defaults to ``True``, and the reason is in the module docstring: an
        unconstrained maximiser has no prior to protect it from an
        under-integrated compensator, which rewards exactly the parameters it
        should penalise.

    Returns
    -------
    MaximumLikelihoodFit

    Raises
    ------
    ValueError
        If `start` lies outside the model's support.

    Warns
    -----
    UserWarning
        If the optimiser reports that it did not converge. Returning the last
        iterate silently is how a failed fit gets published, and the suite runs
        with ``filterwarnings = ["error"]`` so that a downstream test sees it.

    Examples
    --------
    >>> import numpy as np
    >>> import hawkes_package as hp
    >>> from hawkes_package.inference import (
    ...     ExponentialLogLikelihood,
    ...     History,
    ...     exponential_model,
    ...     fit_mle,
    ... )
    >>> process = hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=0)
    >>> process.simulate(400)
    >>> history = History.from_simulation(process)
    >>> model = exponential_model()
    >>> fit = fit_mle(ExponentialLogLikelihood(model), history, [1.0, 0.3, 1.0])
    >>> fit.converged
    True

    .. versionadded:: 1.0.0
    """
    model = _model_of(likelihood)
    spec = model.spec
    theta0 = np.asarray(start, dtype=float).reshape(-1)
    if theta0.size != len(spec):
        raise ValueError(
            f"start has {theta0.size} entries but this model has {len(spec)}: {list(spec.names)}"
        )
    if not bool(np.asarray(model.support(theta0)).ravel()[0]):
        raise ValueError(
            f"start={theta0.tolist()} is outside the model's support, so the "
            "objective is infinite there and the optimiser has no direction to "
            "move in. Start from a parameter the process can be built at."
        )

    negative = _objective(likelihood, history, model)
    options: dict[str, Any] = {}
    if max_iterations is not None:
        options["maxiter"] = int(max_iterations)

    result = minimize(
        negative,
        spec.to_unconstrained(theta0),
        method=method,
        tol=tol,
        options=options or None,
    )

    theta = spec.to_constrained(np.asarray(result.x, dtype=float))
    inside = bool(np.asarray(model.support(theta)).ravel()[0])
    if not inside:
        # Unreachable by the argument at `_OUTSIDE`, and checked anyway: the
        # alternative is returning parameters the process cannot be built at,
        # which every downstream call would then fail on with a message about
        # something else.
        warnings.warn(
            f"the optimiser finished outside the model's support at "
            f"{np.round(theta, 4).tolist()}. This is a fit to discard, not a "
            "parameter to use.",
            UserWarning,
            stacklevel=2,
        )
    if not result.success:
        warnings.warn(
            f"the optimiser did not converge: {result.message} It returned "
            f"{np.round(theta, 4).tolist()} after {int(result.nfev)} evaluations, "
            "which is the last iterate and not a maximum. Raise max_iterations, "
            "or start closer.",
            UserWarning,
            stacklevel=2,
        )

    if recheck_resolution:
        _recheck(likelihood, theta, history)

    return MaximumLikelihoodFit(
        theta=theta,
        log_likelihood=-float(result.fun),
        n_evaluations=int(result.nfev),
        converged=bool(result.success) and inside,
        message=str(result.message),
        spec=spec,
    )


def _recheck(likelihood: LogLikelihood, theta: np.ndarray, history: History) -> None:
    """Re-run the compensator's resolution check at the fitted parameters.

    The check is a once-per-likelihood affair by design -- it doubles the
    quadrature work -- so it normally fires at whatever parameters happened to be
    evaluated first. Those are the *starting* parameters, and the integrand there
    is not the one the answer was read off. Resetting the flag and evaluating
    once more puts the check where the optimiser left the fit.
    """
    checked = getattr(likelihood, "_checked", None)
    if checked is None or not getattr(likelihood, "check", False):
        return
    # The flag belongs to this check: it exists so the doubling runs once, and
    # running it once *at the optimum* is more useful than once at the start.
    # `LogLikelihood` is a Protocol and does not promise it -- the two lines
    # above are exactly the test for whether this one is a member.
    likelihood._checked = False  # type: ignore[attr-defined]
    likelihood.total(theta, history)


def profile_interval(
    likelihood: LogLikelihood,
    history: History,
    fit: MaximumLikelihoodFit,
    name: str,
    *,
    level: float = 0.95,
    span: float = 4.0,
    n_grid: int = 24,
) -> tuple[float, float]:
    r"""Return a profile-likelihood interval for one coordinate.

    Walks `name` away from the maximum in both directions, re-maximising every
    other coordinate at each step, and reports where the log-likelihood has
    dropped by :math:`\chi^2_1(\text{level}) / 2`.

    **Not an inverse Hessian**, and that is the point. A numerically differenced
    Hessian at an optimum near the stationarity boundary can come back
    indefinite, and the standard error it produces is a plausible-looking number
    with no content. A profile interval is asymmetric where the likelihood is
    asymmetric, which near a boundary it always is.

    Each step **warm-starts the inner maximisation from the previous step's
    solution**, which is not only faster. Restarting it from the maximum every
    time leaves the profile carrying the inner optimiser's own noise -- measured
    at 0.35 log-likelihood units on a 400-event exponential fit, against a
    threshold of 1.92 -- and a walk that stops at the first crossing would then
    stop wherever that noise dipped.

    Parameters
    ----------
    likelihood : LogLikelihood
        The same objective the fit used.
    history : History
        The same data.
    fit : MaximumLikelihoodFit
        The maximum to walk out from.
    name : str
        Which coordinate to profile.
    level : float
        One of 0.90, 0.95 or 0.99.
    span : float
        How far to search, as a multiple of the estimate. **A returned endpoint
        at the search boundary means the data did not bound this coordinate**,
        which is information rather than a failure -- for an exponential kernel
        it is the ordinary state of ``alpha``, whose lower profile at 400 events
        drops only 0.85 by the time ``alpha`` has fallen a hundredfold, because
        ``beta`` follows it down and the branching ratio is what the data
        actually pins.
    n_grid : int
        Steps per side.

    Returns
    -------
    tuple of float
        ``(lower, upper)`` on the model's own scale.

    .. versionadded:: 1.0.0
    """
    if level not in _CHI2_HALF:
        raise ValueError(f"level must be one of {sorted(_CHI2_HALF)}, got {level!r}")
    model = _model_of(likelihood)
    spec = model.spec
    index = spec.index(name)
    target = fit.log_likelihood - _CHI2_HALF[level]
    centre = float(fit.theta[index])
    if not centre > 0.0 and spec.lower[index] == 0.0:
        raise ValueError(f"the fitted {name} is {centre}, which has no interval to walk")

    others = [i for i in range(len(spec)) if i != index]
    at_optimum = spec.to_unconstrained(fit.theta)

    def maximise_others(value: float, warm: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the profile at `value`, and the free coordinates that gave it."""
        if not others:
            theta = np.array([value], dtype=float)
            inside = bool(np.asarray(model.support(theta)).ravel()[0])
            return (float(likelihood.total(theta, history)) if inside else -math.inf), warm

        def negative(free: np.ndarray) -> float:
            filled = at_optimum.copy()
            filled[others] = free
            theta = spec.to_constrained(filled)
            theta[index] = value
            if not bool(np.asarray(model.support(theta)).ravel()[0]):
                return _OUTSIDE
            got = float(likelihood.total(theta, history))
            return -got if math.isfinite(got) else _OUTSIDE

        result = minimize(negative, warm, method="Nelder-Mead")
        return -float(result.fun), np.asarray(result.x, dtype=float)

    return (
        _walk(maximise_others, centre, target, spec, index, -1, span, n_grid, at_optimum[others]),
        _walk(maximise_others, centre, target, spec, index, +1, span, n_grid, at_optimum[others]),
    )


def _walk(
    maximise_others: Any,
    centre: float,
    target: float,
    spec: ParameterSpec,
    index: int,
    direction: int,
    span: float,
    n_grid: int,
    free: np.ndarray,
) -> float:
    """Step out from `centre` until the profile drops below `target`.

    Linear interpolation on the bracket rather than a root solve. The profile is
    itself the output of an optimiser, so it carries that optimiser's tolerance
    as noise, and bisecting would chase it to a precision it does not have.
    """
    bound = spec.lower[index] if direction < 0 else spec.upper[index]
    if direction > 0:
        reach = centre * (1.0 + span)
        if math.isfinite(bound):
            reach = min(reach, centre + 0.999 * (bound - centre))
    else:
        reach = centre * (1.0 - min(span, 0.99))
        if math.isfinite(bound):
            reach = max(reach, bound + 0.001 * (centre - bound))

    previous_value = centre
    previous_profile, free = maximise_others(centre, free)
    for step in range(1, n_grid + 1):
        value = centre + (reach - centre) * (step / n_grid)
        current, free = maximise_others(value, free)
        if current < target:
            gap = previous_profile - current
            weight = 0.0 if gap <= 0 else (previous_profile - target) / gap
            return float(previous_value + weight * (value - previous_value))
        previous_value, previous_profile = value, current
    return float(reach)


def warm_start_proposal(
    fit: MaximumLikelihoodFit,
    *,
    width: float = 0.5,
) -> Prior:
    """Return a distribution centred on the maximum, to *start* a cloud from.

    The strongest practical reason to have an MLE at all: pass this to
    :meth:`~hawkes_package.inference.smc.SMCSampler.initialise` as ``proposal=``
    and the cloud begins where the likelihood is, instead of spending its early
    blocks travelling there. Normal on the **unconstrained** scale, which is
    where the sampler's own moves live, so `width` is in the units the
    rejuvenation scale is.

    **Give it to `proposal=`, never to `prior=`, and the distinction is not
    pedantry.** As a proposal it is corrected for by weight and the posterior is
    unchanged; as a prior it *is* part of the model, and a tight one centred on
    the maximum pulls the answer to the maximum. Measured on 300 events at
    ``width=0.5``: used as a proposal, the four-block posterior mean stays the
    vague prior's ``(0.95, 0.21, 1.52)`` and the log evidence agrees to within
    Monte Carlo error; used as a prior, the mean moves to ``(1.07, 0.30, 2.39)``
    and the log evidence rises by two nats, because that is a different model
    and it fits better.

    What it costs is effective sample size rather than correctness: the initial
    weights are ``log prior - log proposal``, so a proposal far from the prior
    starts with an uneven cloud, and :attr:`SMCDiagnostics.min_ess_fraction`
    reports it. Measured on the same data, one block: 0.123 warm against 0.050
    cold, and the warm cloud's mean is 0.284 from the eight-block answer against
    the cold cloud's 0.506.

    Parameters
    ----------
    fit : MaximumLikelihoodFit
        The maximum to centre on.
    width : float
        Standard deviation per coordinate, on the unconstrained scale. Too
        narrow is not a correctness problem here -- it is an effective sample
        size problem, which is visible.

    .. versionadded:: 1.0.0
    """
    if not fit.converged:
        warnings.warn(
            "warm-starting from a fit that did not converge centres the cloud on "
            "the optimiser's last iterate, which is not a maximum.",
            UserWarning,
            stacklevel=2,
        )
    centre = fit.spec.to_unconstrained(fit.theta)
    return _UnconstrainedNormalPrior(fit.spec, centre, float(width))


@dataclass(frozen=True)
class _UnconstrainedNormalPrior:
    """A normal cloud on the unconstrained scale, mapped back through the spec."""

    spec: ParameterSpec
    centre: np.ndarray
    width: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` parameter vectors, on the model's own scale."""
        generator = np.random.default_rng(rng)
        z = self.centre[None, :] + self.width * generator.standard_normal(
            (int(n), self.centre.size)
        )
        return np.asarray(self.spec.to_constrained(z), dtype=float)

    def log_pdf(self, theta: Any) -> np.ndarray:
        """Log density on the model's own scale, **with** the Jacobian.

        Included here and deliberately absent from the optimiser's objective,
        and the difference is the whole distinction this module rests on: a
        prior is a density over parameters and has to transform like one, while
        a likelihood is not and must not, or the maximum becomes a MAP estimate
        under a flat prior on the unconstrained scale.
        """
        values = np.atleast_2d(np.asarray(theta, dtype=float))
        z = self.spec.to_unconstrained(values)
        gaussian = IndependentPrior(tuple(Normal(float(m), self.width) for m in self.centre))
        density = np.asarray(gaussian.log_pdf(z), dtype=float)
        return np.asarray(density - self.spec.log_abs_det_jacobian(z), dtype=float)


class HawkesMLE:
    """The maximum-likelihood fit behind the same surface :class:`HawkesEstimator` has.

    A **sibling** of that class rather than a ``method="mle"`` mode of it, which
    is the recommendation the programme plan made and the reason is worth
    keeping: one class would leave `diagnostics_` half populated -- an
    `SMCDiagnostics` with no cloud behind it -- and those diagnostics exist to be
    read. A user who wants a posterior should reach for a class that has one.

    What this deliberately does **not** have: `partial_fit`, because a maximum
    is not something you extend by a block; `sample_posterior`, because there is
    no posterior; and `predict_intensity_band`, because a point estimate has no
    band. `profile_interval_` is the honest replacement for the last of those,
    one coordinate at a time.

    Parameters
    ----------
    model : ProcessModel or str
        As for :class:`~hawkes_package.inference.estimator.HawkesEstimator`.
    start : array_like
        Starting parameters, on the model's own scale. Required: an optimiser
        needs somewhere to begin, and a default here would be a modelling
        assumption dressed as a convenience.
    likelihood : LogLikelihood, optional
        Defaults to the fastest one exact for the model.
    method : str
        Passed to :func:`scipy.optimize.minimize`; see :func:`fit_mle`.
    max_iterations : int, optional
        Iteration cap.
    recheck_resolution : bool
        Re-run the compensator's resolution check at the optimum.

    Attributes
    ----------
    theta_ : numpy.ndarray
        The estimate, once fitted.
    fit_ : MaximumLikelihoodFit
        The whole result, including whether the optimiser converged.
    model_, likelihood_, history_
        As for :class:`~hawkes_package.inference.estimator.HawkesEstimator`.

    Examples
    --------
    >>> import numpy as np
    >>> import hawkes_package as hp
    >>> from hawkes_package.inference import History, HawkesMLE
    >>> process = hp.ExponentialHawkes(np.array([1.0, 0.5, 2.0]), rng=0)
    >>> process.simulate(300)
    >>> estimator = HawkesMLE("exponential", start=[1.0, 0.3, 1.0])
    >>> estimator.fit(History.from_simulation(process)).theta_.shape
    (3,)

    .. versionadded:: 1.0.0
    """

    def __init__(
        self,
        model: Any,
        *,
        start: Any,
        likelihood: LogLikelihood | None = None,
        method: str = "Nelder-Mead",
        max_iterations: int | None = None,
        recheck_resolution: bool = True,
    ) -> None:
        self.model = model
        self.start = start
        self.likelihood = likelihood
        self.method = method
        self.max_iterations = max_iterations
        self.recheck_resolution = recheck_resolution

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor arguments, scikit-learn style."""
        return {
            "model": self.model,
            "start": self.start,
            "likelihood": self.likelihood,
            "method": self.method,
            "max_iterations": self.max_iterations,
            "recheck_resolution": self.recheck_resolution,
        }

    def set_params(self, **params: Any) -> HawkesMLE:
        """Set constructor arguments by name, refusing one this class does not have."""
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(
                    f"{type(self).__name__} has no parameter {key!r}; it has "
                    f"{sorted(self.get_params())}"
                )
            setattr(self, key, value)
        return self

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        end: float | None = None,
        start: float | None = None,
    ) -> HawkesMLE:
        """Maximise the likelihood over the whole history.

        Parameters are as for
        :meth:`~hawkes_package.inference.estimator.HawkesEstimator.fit`, down to
        `end` being required with a bare array and refused alongside a
        `History` -- reading ``end = X.max()`` drops the interval between the
        last event and the end of observation, which is the data saying nothing
        happened there, and biases ``mu`` upward.
        """
        # Imported here rather than at module scope: `estimator` imports this
        # module for `warm_start_proposal`, and a top-level import either way would
        # be a cycle.
        from .estimator import HawkesEstimator

        helper = HawkesEstimator(self.model, prior=None)  # type: ignore[arg-type]
        helper._check_target(y)
        model = helper._resolve_model()
        history = helper._as_history(X, model.ndim, end=end, start=start, caller="fit")
        likelihood = self.likelihood or helper._resolve_likelihood(model)

        fit = fit_mle(
            likelihood,
            history,
            self.start,
            method=self.method,
            max_iterations=self.max_iterations,
            recheck_resolution=self.recheck_resolution,
        )

        self.model_ = model
        self.likelihood_ = likelihood
        self.history_ = history
        self.fit_ = fit
        self.theta_ = fit.theta
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the conditional intensity at each requested time.

        At the estimate, and only at the estimate -- where
        :meth:`~hawkes_package.inference.estimator.HawkesEstimator.predict`
        averages the intensity over a posterior. The difference is not
        cosmetic: averaging is what makes that method's answer unbiased where
        the posterior has width, and this one is the plug-in it is warned about.
        """
        self._require_fit("predict")
        from .estimator import predictable_times

        times = predictable_times(self.model_, self.history_, X, caller="predict")
        process = self.model_(self.theta_)
        _bind_history(process, self.history_)
        # The hook the simulator thins against, which is the only definition of
        # the intensity in this package. `HawkesProcess` declares it on the
        # temporal subclass, and `predictable_times` has already refused a
        # spatio-temporal model, where it does not exist.
        intensity = process._conditional_intensity  # type: ignore[attr-defined]
        return np.array([float(intensity(float(t))) for t in times])

    def score(self, X: Any, y: Any = None, *, end: float) -> float:
        """Return the log-likelihood of a *later* block, at the fitted estimate.

        Later, because scoring the data a fit was made on measures the fit's
        appetite rather than its quality. The block must start after the fitted
        window, and is refused otherwise.
        """
        self._require_fit("score")
        events = np.asarray(X, dtype=float).reshape(-1)
        if events.size and float(events.min()) <= self.history_.end:
            raise ValueError(
                f"score expects events after the fitted window, which ends at "
                f"{self.history_.end}; the earliest given is {float(events.min())}"
            )
        if float(end) <= self.history_.end:
            raise ValueError(
                f"end={end} does not advance the window, which already reaches {self.history_.end}"
            )
        extended = History(
            np.concatenate([self.history_.times, events]),
            self.history_.points,
            self.history_.start,
            float(end),
        )
        whole = float(self.likelihood_.total(self.theta_, extended))
        fitted = float(self.likelihood_.total(self.theta_, self.history_))
        return whole - fitted

    def profile_interval_(
        self, name: str, *, level: float = 0.95, **kwargs: Any
    ) -> tuple[float, float]:
        """Return a profile-likelihood interval for one fitted coordinate."""
        self._require_fit("profile_interval_")
        return profile_interval(
            self.likelihood_, self.history_, self.fit_, name, level=level, **kwargs
        )

    def warm_start_proposal_(self, width: float = 0.5) -> Prior:
        """Return a proposal centred on the estimate, for starting a sequential fit.

        For ``fit_smc(..., proposal=)``, not for ``prior=``: see
        :func:`warm_start_proposal`.
        """
        self._require_fit("warm_start_proposal_")
        return warm_start_proposal(self.fit_, width=width)

    def _require_fit(self, caller: str) -> None:
        """Refuse a call that needs a fit, naming the one that is missing."""
        if not hasattr(self, "theta_"):
            raise ValueError(f"{caller} needs a fitted estimator; call fit first")

    def __repr__(self) -> str:
        """Show the estimate when there is one, and the configuration when not."""
        if hasattr(self, "fit_"):
            return f"HawkesMLE(fitted, {self.fit_!r})"
        return f"HawkesMLE(model={self.model!r}, start={list(np.asarray(self.start, dtype=float))})"
