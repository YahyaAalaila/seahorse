r"""A scikit-learn-shaped front door to the sequential Monte Carlo fit.

One object where :mod:`~hawkes_package.inference` otherwise asks for four -- a
model, a prior, a likelihood and a sampler -- behind the method names a user of
scikit-learn already knows. **Nothing new is inferred here.** :meth:`fit` builds
an :class:`~hawkes_package.inference.smc.SMCSampler` and runs it,
:meth:`partial_fit` calls
:meth:`~hawkes_package.inference.smc.SMCSampler.update`, and every forecast and
diagnostic is the existing function with the fitted objects already filled in.
Two exact equalities say so, and are tested: a fit is
:func:`~hawkes_package.inference.smc.fit_smc` bit-for-bit at the same seed, and a
`partial_fit` per block is ``fit(blocks=k)`` bit-for-bit.

scikit-learn is not a dependency and is not imported at module scope
-------------------------------------------------------------------

``clone``, ``Pipeline`` and ``GridSearchCV`` reach an estimator through
``get_params``/``set_params`` and never through ``isinstance`` -- ``clone``'s own
gate is ``hasattr(estimator, "get_params")`` -- so inheriting
:class:`sklearn.base.BaseEstimator` buys no behaviour. It costs some: a base
class chosen by whichever packages happen to be installed makes ``repr``,
parameter ordering and pickling differ between environments, which is the class
of difference this package exists not to ship silently. It is also impossible
under ``mypy --strict``, because scikit-learn ships no ``py.typed``: the base is
``Any`` and ``disallow_subclassing_any`` refuses it, installed or not.

So :class:`HawkesEstimator` reimplements the protocol in :class:`_ParamsMixin`,
and ``tests/inference/test_sklearn_interop.py`` pins that against
``BaseEstimator``'s own implementation -- which is what stops the two drifting.
A caller who genuinely needs the ``isinstance`` writes two lines::

    from sklearn.base import BaseEstimator


    class SklearnHawkes(BaseEstimator, HawkesEstimator): ...

Why nothing is validated in ``__init__``
----------------------------------------

``clone`` reconstructs an estimator as ``klass(**est.get_params())`` and then
asserts that every value is the *same object*. So ``int(n_particles)`` here would
break cloning for a NumPy integer rather than catch a mistake, and
``np.random.default_rng(rng)`` would break it always. The rule is therefore
absolute: **one assignment per parameter, in signature order, and nothing else.**

That is a smaller departure from this package's habits than it looks, because
almost none of the validation was ours to begin with:
:class:`~hawkes_package.inference.smc.SMCSampler` already refuses
``n_particles < 2``, an ``ess_threshold`` outside ``(0, 1]``, a negative
``n_move``, an MCMC move under a drifting evolution, a non-positive ``scale`` and
an unknown ``on_invalid``, with better messages than a second copy would carry;
:class:`~hawkes_package.inference.models.ProcessModel`,
:class:`~hawkes_package.inference.likelihood.History` and
``priors.spec_for`` cover the rest. :meth:`HawkesEstimator.fit` builds the
sampler **before it touches the data**, so those errors still arrive on the first
statement of the fit rather than after a minute of quadrature.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from ..base import SeedLike, TemporalHawkesProcess
from .diagnostics import posterior_report, residuals
from .evolution import Evolution
from .forecast import posterior_predictive, predictive_counts, predictive_interval
from .likelihood import (
    ExponentialLogLikelihood,
    History,
    LogLikelihood,
    MarkedLogLikelihood,
    MultivariateExponentialLogLikelihood,
    MultivariateLogLikelihood,
    SpatioTemporalLogLikelihood,
    TemporalLogLikelihood,
    _bind_history,
)
from .models import (
    MarkedComponents,
    MultivariateComponents,
    ProcessModel,
    bell_shape_model,
    exponential_model,
    monotone_model,
)
from .parameters import ParameterSpec
from .priors import Prior
from .resample import log_sum_exp, systematic
from .smc import ParticleCloud, SMCDiagnostics, SMCSampler

__all__ = ["HawkesEstimator"]

#: The families `model=` accepts as a string. Spatio-temporal is deliberately
#: absent: `spatio_temporal_model()` defaults its domain to a `Circle`, and
#: reaching that default through a string would make the geometry -- the most
#: consequential choice in the model -- invisible at the call site.
_TEMPORAL_FACTORIES: dict[str, Callable[[], ProcessModel]] = {
    "exponential": exponential_model,
    "monotone": monotone_model,
    "bell_shape": bell_shape_model,
}

_S = TypeVar("_S", bound="_ParamsMixin")


class _ParamsMixin:
    """scikit-learn's parameter protocol, reimplemented rather than inherited.

    See the module docstring for why this is not
    :class:`sklearn.base.BaseEstimator`. The implementation is deliberately
    faithful to that one, including the details that are easy to miss:
    keyword-only parameters *are* included (only ``**kwargs`` is filtered, and
    ``*args`` raises), the names come back **sorted**, and ``deep=True`` recurses
    only into values that themselves have ``get_params``.
    """

    @classmethod
    def _param_names(cls) -> list[str]:
        """Return the constructor's parameter names, sorted."""
        signature = inspect.signature(cls.__init__)
        names = []
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls.__name__} takes *args in __init__, so its parameters cannot be "
                    "introspected and it cannot be cloned. Use explicit keyword arguments."
                )
            names.append(name)
        return sorted(names)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters, exactly as they were stored.

        Parameters
        ----------
        deep : bool
            Also return the parameters of any value that is itself an estimator.
            None of this class's are, so ``prior__sd_log`` and the like do not
            resolve: the priors and models here are plain frozen objects, not
            estimators, and are replaced wholesale.
        """
        out: dict[str, Any] = {}
        for name in self._param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for key, sub in value.get_params().items():
                    out[f"{name}__{key}"] = sub
            out[name] = value
        return out

    def set_params(self: _S, **params: Any) -> _S:
        """Set constructor parameters in place and return ``self``."""
        if not params:
            return self
        valid = self.get_params(deep=True)
        nested: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        for key, value in params.items():
            name, delimiter, sub_key = key.partition("__")
            if name not in valid:
                raise ValueError(
                    f"invalid parameter {name!r} for {type(self).__name__}. Valid "
                    f"parameters are {self._param_names()}."
                )
            if delimiter:
                nested[name][sub_key] = value
            else:
                setattr(self, name, value)
                valid[name] = value
        for name, sub_params in nested.items():
            valid[name].set_params(**sub_params)
        return self

    def __repr__(self) -> str:
        """Name the class and the parameters that differ from their defaults."""
        defaults = inspect.signature(type(self).__init__).parameters
        shown = []
        for name in self._param_names():
            value = getattr(self, name)
            default = defaults[name].default
            if default is not inspect.Parameter.empty and value is default:
                continue
            shown.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(shown)})"


class HawkesEstimator(_ParamsMixin):
    """Fit a Hawkes process to observed events, and predict its intensity.

    A wrapper over :class:`~hawkes_package.inference.smc.SMCSampler` with the
    scikit-learn method names. The sampler is a particle filter over the
    data-tempered posterior, so :meth:`partial_fit` is not a convenience bolted
    on afterwards -- it is the algorithm's own entry point, and the reason this
    class is worth having.

    Parameters
    ----------
    model : ProcessModel or str
        The parameterised family to fit. A string names one of the temporal
        factories -- ``"exponential"``, ``"monotone"`` or ``"bell_shape"``.
        Spatio-temporal models must be passed as objects, from
        :func:`~hawkes_package.inference.models.spatio_temporal_model`, so the
        domain is visible where the fit is written.
    prior : Prior
        Where you think the truth might be. **No default**, deliberately: a
        default prior is a silent modelling choice of the same kind
        :class:`~hawkes_package.inference.likelihood.History` refuses for its
        window, and its dimension depends on the model. The usual form is
        ``ConstrainedPrior(IndependentPrior(marginals), model.support)``; an
        untruncated prior is not wrapped for you, because
        :meth:`~hawkes_package.inference.smc.SMCSampler.initialise` already
        refuses it with a message naming the offending draw, its branching ratio
        and the wrapper to apply.
    likelihood : LogLikelihood, optional
        Defaults to the fastest implementation that is exact for `model`:
        :class:`~hawkes_package.inference.likelihood.ExponentialLogLikelihood`
        for the exponential family,
        :class:`~hawkes_package.inference.likelihood.SpatioTemporalLogLikelihood`
        for a spatial one,
        :class:`~hawkes_package.inference.likelihood.TemporalLogLikelihood`
        otherwise. This is a speed choice and not a modelling one -- the two
        temporal paths compute the same number, at ``O(n)`` against ``O(n^2 P)``.
        The one thing it cannot guess is ``extra_lags`` for a kernel with an
        interior kink; pass the likelihood yourself for that, and note that
        ``TemporalLogLikelihood`` warns on the first evaluation when its
        quadrature is under-resolved.

        A likelihood passed here is **reused across fits**, and carries state --
        a geometry cache, a resolved backend. Fitting one instance to two
        different histories therefore raises, rather than answering from the
        wrong cache.
    n_particles : int
        Cloud size. What a small cloud gets wrong is the posterior's *width*,
        not its centre.
    blocks : int or sequence of float
        How :meth:`fit` splits the history. **Eight, not**
        :func:`~hawkes_package.inference.smc.fit_smc`'s **one**: a single block
        is IBIS with one tempering step, which is importance sampling from the
        prior, and it degenerates on any history long enough to be worth
        fitting.
    evolution : Evolution, optional
        Defaults to :class:`~hawkes_package.inference.evolution.Static`, the
        exact IBIS sampler. Anything else makes this a filter over a drifting
        parameter and requires ``n_move=0``.
    ess_threshold : float
        Resample when the effective sample size falls below this fraction.
    n_move : int
        Metropolis steps per particle after a resample.
    scale : float
        Multiplier on the optimal random-walk scaling.
    jitter : float
        Relative ridge on the proposal covariance before factorising.
    resampler : callable
        :func:`~hawkes_package.inference.resample.systematic` by default.
    on_invalid : {"raise", "reject"}
        What to do when a particle's likelihood cannot be evaluated.
    rng : None, int or numpy.random.Generator
        Source of randomness. Named `rng` rather than scikit-learn's
        ``random_state`` because ``check_random_state`` returns a legacy
        ``RandomState``, which this package never constructs -- accepting a
        ``Generator`` under that name would be a false promise. Nothing in
        ``clone`` or ``get_params`` keys on the name.

    Attributes
    ----------
    model_ : ProcessModel
        The resolved model. Present only after a fit.
    likelihood_ : LogLikelihood
        The resolved or auto-selected likelihood.
    sampler_ : SMCSampler
        The live sampler. This is what :meth:`partial_fit` continues.
    history_ : History
        The cumulative history fitted so far, window included.

    Examples
    --------
    >>> est = HawkesEstimator("exponential", prior, n_particles=128, rng=0)
    >>> est.fit(times, end=100.0).theta_.shape
    (3,)
    >>> est.diagnostics_.warnings()  # an empty list is the health check
    []

    Notes
    -----
    What this is not: it is not a :class:`sklearn.base.BaseEstimator` subclass
    (see the module docstring), it does not pass ``check_estimator``, and it
    cannot be used with ``GridSearchCV`` or ``cross_val_score`` -- not for a
    technical reason but a statistical one, since a point-process history cannot
    be sliced into folds when every fold's likelihood depends on the events
    before it. Sweep parameters with an explicit loop over ``clone``,
    :meth:`set_params`, :meth:`fit` and a chronological :meth:`score` instead.

    .. versionadded:: 0.5.0
    """

    model_: ProcessModel
    likelihood_: LogLikelihood
    sampler_: SMCSampler
    history_: History

    def __init__(
        self,
        model: ProcessModel | str,
        prior: Prior,
        *,
        likelihood: LogLikelihood | None = None,
        n_particles: int = 512,
        blocks: int | Sequence[float] = 8,
        evolution: Evolution | None = None,
        ess_threshold: float = 0.5,
        n_move: int = 3,
        scale: float = 2.38,
        jitter: float = 1e-10,
        resampler: Callable[[Any, np.random.Generator], np.ndarray] = systematic,
        on_invalid: str = "raise",
        rng: SeedLike = None,
    ) -> None:
        # One assignment per parameter and nothing else -- see the module
        # docstring. Coercing anything here would break `clone`, which asserts
        # that a reconstructed estimator holds the *same objects*.
        self.model = model
        self.prior = prior
        self.likelihood = likelihood
        self.n_particles = n_particles
        self.blocks = blocks
        self.evolution = evolution
        self.ess_threshold = ess_threshold
        self.n_move = n_move
        self.scale = scale
        self.jitter = jitter
        self.resampler = resampler
        self.on_invalid = on_invalid
        self.rng = rng

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        end: float | None = None,
        start: float | None = None,
    ) -> HawkesEstimator:
        """Fit the whole history, in `blocks` successive chunks, from a fresh sampler.

        Parameters
        ----------
        X : array_like or History
            The observed events. A
            :class:`~hawkes_package.inference.likelihood.History` carries its own
            window. An array is shape ``(n,)`` -- or ``(n, 1)``, scikit-learn's
            column -- for a temporal model, and ``(ndim + 1, n)`` with times in
            row 0 for a spatio-temporal one.
        y : None
            There is no target. A non-``None`` value raises rather than being
            ignored, because supplying one means expecting supervised semantics
            this class does not have.
        end : float
            End of the observation window. **Required with an array, and it has
            no default.** ``end = X.max()`` looks harmless and is not: it drops
            the interval between the last event and the end of observation, which
            is the data saying nothing happened there, and biases ``mu`` upward.
            Must not be given alongside a `History`, which already carries it.
        start : float, optional
            Beginning of the window; ``0.0`` with an array. Likewise refused
            alongside a `History`.

        Returns
        -------
        HawkesEstimator
            ``self``, fitted.

        Notes
        -----
        A second fit **discards** any online state: the sampler is rebuilt,
        because a likelihood cannot be extended backwards and so a sampler cannot
        be rewound. The fitted attributes are assigned only once the run
        succeeds, so a failed fit leaves the previous one intact.
        """
        self._check_target(y)
        model = self._resolve_model()
        history = self._as_history(X, model.ndim, end=end, start=start, caller="fit")

        likelihood = self._resolve_likelihood(model)
        # Built before the data is touched, so a bad `n_particles` or
        # `ess_threshold` raises here rather than after a minute of quadrature.
        sampler = self._build_sampler(likelihood)
        sampler.initialise(start=history.start)
        sampler.run(history, blocks=self.blocks)

        self.model_ = model
        self.likelihood_ = likelihood
        self.sampler_ = sampler
        self.history_ = history
        return self

    def partial_fit(self, X: Any, y: Any = None, *, end: float) -> HawkesEstimator:
        """Absorb the events in ``(history_.end, end]`` as one further block.

        The online entry point, and the one the sampler is actually built around.

        Parameters
        ----------
        X : array_like
            **Only the new events**, not the history so far -- the class
            accumulates that itself. Empty is a legitimate block: it says nothing
            happened, which is information. A `History` is refused, because the
            window it carries would contradict the accumulated one.
        y : None
            As for :meth:`fit`.
        end : float
            The time the process has now been *observed* to. Must advance
            strictly past ``history_.end``, and every event in `X` must lie in
            ``(history_.end, end]``.

        Returns
        -------
        HawkesEstimator
            ``self``.

        Notes
        -----
        One call is one block, and one block is one reweighting. Feeding a large
        batch through a single call is where a cloud collapses; split it.

        Called before any :meth:`fit`, this initialises from a window starting at
        ``0.0``. There is no ``start`` argument, because after the first call it
        is not the caller's to choose -- use :meth:`fit` for the first block if
        you need a different origin.
        """
        self._check_target(y)
        if isinstance(X, History):
            raise ValueError(
                "partial_fit takes the new events, not a History: the window a History "
                "carries would contradict the one accumulated so far. Pass the event "
                "times and the new end=, or use fit to start over."
            )
        if not hasattr(self, "sampler_"):
            self._initialise_online()

        horizon = float(end)
        previous = self.history_.end
        if not horizon > previous:
            raise ValueError(
                f"end={horizon} does not advance the observation window, which already "
                f"reaches {previous}. A block that observes no new time carries no "
                "information and would still cost a resample."
            )

        record = self._as_record(X, self.model_.ndim, caller="partial_fit")
        times = record if record.ndim == 1 else record[0]
        if times.size and (times.min() <= previous or times.max() > horizon):
            raise ValueError(
                f"partial_fit takes only the events in (history_.end, end] = "
                f"({previous}, {horizon}], but the block spans "
                f"[{times.min()}, {times.max()}]. Pass the new events alone -- the "
                "history so far is already held."
            )

        grown = History.from_events(
            np.concatenate([self.history_.as_process_events(), record], axis=-1),
            start=self.history_.start,
            end=horizon,
        )
        self.sampler_.update(grown, horizon)
        self.history_ = grown
        return self

    # -- prediction --------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        r"""Conditional intensity at each of `X`, averaged over the posterior.

        Returns :math:`\sum_p w_p\,\lambda_{\theta_p}(t \mid H)`, **not**
        :math:`\lambda(t \mid \bar\theta)`. The two differ, and not by noise:
        :math:`\lambda` is convex in the decay rate, so by Jensen the plug-in
        sits systematically *below* the marginal wherever the posterior has
        width. It is the same argument
        :mod:`~hawkes_package.inference.forecast` makes for drawing a fresh
        particle per path rather than simulating from the posterior mean.

        Parameters
        ----------
        X : array_like
            Times, shape ``(k,)`` or ``(k, 1)``. Every one must lie inside the
            observation window.

        Returns
        -------
        numpy.ndarray
            Shape ``(k,)``, in the order given.

        Raises
        ------
        ValueError
            If any time leaves ``[history_.start, history_.end]``. Past the
            window, the intensity computed from the observed record is the
            intensity conditional on *no events having happened since* -- it
            understates the truth by exactly the excitation of the events that
            would have occurred, and the gap grows with the horizon. Use
            :meth:`forecast`, :meth:`predict_counts` or :meth:`predict_interval`,
            which answer that question by simulating forward.

            Also if the fitted model is spatio-temporal, where a temporal
            intensity is not defined without a location.

        Notes
        -----
        At an observed event time this is the **left limit**
        :math:`\lambda(t^-)`: the intensity hook filters ``t_j < t`` strictly,
        which is the same convention the log-likelihood's sum uses. So at a fixed
        parameter, ``predict(history_.times)`` is exactly the vector the
        log-likelihood takes logs of.

        Cost is ``n_particles`` process builds and ``n_particles * k`` intensity
        evaluations, each linear in the number of events. Vectorising across
        particles would need a second expression for the intensity, which is the
        one thing this package does not do; the knob is ``n_particles``.
        """
        times = self._predictable_times(X)
        return np.asarray(self._intensity_matrix(times).T @ self.cloud_.weights, dtype=float)

    def predict_intensity_band(
        self, X: Any, *, level: float = 0.9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior quantiles of the intensity at each of `X`.

        The band :meth:`predict` returns the mean of, and the reason
        marginalising over the cloud is worth its cost.

        Parameters
        ----------
        X : array_like
            Times, as for :meth:`predict`.
        level : float
            Coverage of the equal-tailed band.

        Returns
        -------
        lower, median, upper : numpy.ndarray
            Each shape ``(k,)``.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        times = self._predictable_times(X)
        matrix = self._intensity_matrix(times)
        tail = 0.5 * (1.0 - level)
        bands = self._weighted_quantiles(matrix, [tail, 0.5, 1.0 - tail])
        return bands[0], bands[1], bands[2]

    def score(self, X: Any, y: Any = None, *, end: float) -> float:
        r"""Log posterior-predictive density of the window ``(history_.end, end]``.

        :math:`\log \sum_p w_p \exp \ell_{\theta_p}`, higher being better. It is
        a proper scoring rule, it marginalises over the posterior exactly as
        :meth:`predict` does, and it is the same quantity
        :attr:`~hawkes_package.inference.smc.SMCDiagnostics.log_evidence`
        accumulates -- so scoring a block and then absorbing it with
        :meth:`partial_fit` gives the same number twice.

        Parameters
        ----------
        X : array_like
            The events on the scored window only, as for :meth:`partial_fit`.
        y : None
            As for :meth:`fit`.
        end : float
            End of the scored window.

        Returns
        -------
        float
            A total, not a per-event average. Dividing by the event count would
            divide by a *random* quantity the model itself predicts, which
            destroys properness: a model forecasting fewer events would score
            better for nothing.

        Notes
        -----
        The scored window must **abut** the fitted one, because the process has
        memory: the intensity on ``(T, T']`` depends on every event before ``T``,
        so a segment scored as though it began fresh is not the same quantity.
        The conditioning history is always ``history_``, and a gap is refused --
        the compensator over unobserved time can be neither included nor left
        out.

        Nothing is mutated: no sampler is built, no particle state advances and
        ``history_`` is untouched. On a spatio-temporal fit the likelihood's
        geometry cache does grow to cover the scored window, which is what makes
        the intended score-then-absorb sequence cheap.
        """
        self._check_fitted()
        self._check_target(y)
        horizon = float(end)
        if not horizon > self.history_.end:
            raise ValueError(
                f"end={horizon} does not advance past the fitted window, which reaches "
                f"{self.history_.end}. There is nothing to score."
            )

        record = self._as_record(X, self.model_.ndim, caller="score")
        times = record if record.ndim == 1 else record[0]
        if times.size and (times.min() <= self.history_.end or times.max() > horizon):
            raise ValueError(
                f"score takes only the events in (history_.end, end] = "
                f"({self.history_.end}, {horizon}], but the window spans "
                f"[{times.min()}, {times.max()}]."
            )

        extended = History.from_events(
            np.concatenate([self.history_.as_process_events(), record], axis=-1),
            start=self.history_.start,
            end=horizon,
        )
        cloud = self.cloud_
        weights = cloud.weights
        increments = np.full(cloud.n_particles, -math.inf, dtype=float)
        for index in range(cloud.n_particles):
            if weights[index] == 0.0:
                # An abandoned particle may sit outside the support, where the
                # process cannot be built at all. It carries no weight, so its
                # increment never reaches the sum.
                continue
            theta = cloud.theta[index]
            state = self.likelihood_.initial_state(self.history_.start)
            state, _ = self.likelihood_.extend(state, theta, extended, self.history_.end)
            _, increment = self.likelihood_.extend(state, theta, extended, horizon)
            increments[index] = increment
        return float(log_sum_exp(cloud.log_weights + increments))

    # -- forecasting -------------------------------------------------------

    def forecast(
        self, *, horizon: float, n_paths: int = 200, rng: SeedLike = None
    ) -> list[np.ndarray]:
        """Simulate `n_paths` continuations past the window, one per posterior draw.

        A thin wrapper over
        :func:`~hawkes_package.inference.forecast.posterior_predictive`. Each
        path draws a fresh particle, so the result carries the uncertainty about
        *which* process as well as the process's own variability, and each starts
        at ``history_.end`` rather than at the last event.

        Parameters
        ----------
        horizon : float
            Length of the forecast window, measured from ``history_.end``.
        n_paths : int
            Number of paths.
        rng : None, int or numpy.random.Generator
            Deliberately independent of the fit's own stream: drawing from that
            one would make a later :meth:`partial_fit` depend on how many
            forecasts had been taken in between.

        Returns
        -------
        list of numpy.ndarray
            One entry per path, holding only the new events. An empty entry is a
            genuine outcome.
        """
        self._check_fitted()
        return posterior_predictive(
            self.model_,
            self.cloud_,
            self.history_,
            horizon=horizon,
            n_paths=n_paths,
            rng=rng,
        )

    def predict_counts(
        self, *, horizon: float, n_paths: int = 200, rng: SeedLike = None
    ) -> np.ndarray:
        """Count the new events on each forecast path.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_paths,)``. Its spread is the predictive distribution of
            the count, so read a quantile rather than the mean alone.
        """
        return predictive_counts(self.forecast(horizon=horizon, n_paths=n_paths, rng=rng))

    def predict_interval(
        self,
        X: Any,
        *,
        level: float = 0.9,
        n_paths: int = 200,
        rng: SeedLike = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predictive band for the cumulative new-event count at each of `X`.

        Parameters
        ----------
        X : array_like
            Times to evaluate the running count at. All must lie **past**
            ``history_.end`` -- this is the forecast counterpart of
            :meth:`predict`, and the horizon is taken from the furthest one.
        level : float
            Coverage of the equal-tailed band.
        n_paths : int
            Number of simulated paths.
        rng : None, int or numpy.random.Generator
            As for :meth:`forecast`.

        Returns
        -------
        lower, median, upper : numpy.ndarray
            Each shape ``(k,)``. Counts of events after ``history_.end``.
        """
        self._check_fitted()
        grid = np.asarray(X, dtype=float).ravel()
        if grid.size == 0:
            raise ValueError("predict_interval needs at least one time")
        if grid.min() <= self.history_.end:
            raise ValueError(
                f"predict_interval forecasts past the observation window, so every time "
                f"must exceed history_.end={self.history_.end}; the earliest given is "
                f"{grid.min()}. Use predict for times inside the window."
            )
        paths = self.forecast(
            horizon=float(grid.max()) - self.history_.end, n_paths=n_paths, rng=rng
        )
        return predictive_interval(paths, grid, level=level)

    def sample_posterior(self, n_draws: int = 1, *, rng: SeedLike = None) -> np.ndarray:
        """Draw parameter vectors from the fitted posterior, in proportion to weight.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_draws, n_dim)``. Drawing by weight means an unresampled
            cloud is used correctly without one.
        """
        self._check_fitted()
        draws = int(n_draws)
        if draws < 1:
            raise ValueError(f"n_draws must be at least 1, got {n_draws}")
        cloud = self.cloud_
        generator = np.random.default_rng(rng)
        indices = generator.choice(cloud.n_particles, size=draws, p=cloud.weights)
        return np.asarray(cloud.theta[indices], dtype=float)

    # -- checking ----------------------------------------------------------

    def residuals(self, *, upto: float | None = None) -> np.ndarray:
        """Time-rescaled inter-event gaps at the posterior mean.

        Independent ``Exp(1)`` under the true parameter, so
        :func:`~hawkes_package.inference.diagnostics.ks_exponential` on them
        tests the whole model at once -- which is the check a tight posterior
        cannot make, because a posterior can be tight, stable and wrong.
        """
        self._check_fitted()
        return residuals(self.likelihood_, self.theta_, self.history_, upto=upto)

    def report(self, *, truth: Any = None) -> str:
        """Return the posterior marginals and the time-rescaling verdict, together.

        Together on purpose: a confident posterior beside a rejected fit is the
        combination worth noticing, and the one that goes unnoticed when the two
        are read apart.
        """
        self._check_fitted()
        return posterior_report(self.cloud_, self.likelihood_, self.history_, truth=truth)

    # -- the fitted surface ------------------------------------------------

    @property
    def cloud_(self) -> ParticleCloud:
        """The posterior sample."""
        self._check_fitted()
        cloud = self.sampler_.cloud
        if cloud is None:  # pragma: no cover - a fitted sampler always holds one
            raise AttributeError("the sampler holds no cloud; call fit")
        return cloud

    @property
    def diagnostics_(self) -> SMCDiagnostics:
        """The per-block record. An empty ``diagnostics_.warnings()`` is the health check."""
        self._check_fitted()
        return self.sampler_.diagnostics

    @property
    def theta_(self) -> np.ndarray:
        """Posterior mean, in the constrained parameterisation.

        A mean, not a maximum -- and not what :meth:`predict` uses. See there.
        """
        return np.asarray(self.cloud_.mean(), dtype=float)

    @property
    def spec_(self) -> ParameterSpec:
        """The fitted model's coordinates and their bounds."""
        self._check_fitted()
        return self.model_.spec

    @property
    def parameter_names_(self) -> tuple[str, ...]:
        """Names of the fitted coordinates, in the order every array here uses.

        Not ``feature_names_in_``: that names the columns of ``X``, and ``X`` here
        is event times, which have no feature axis.
        """
        return self.spec_.names

    @property
    def n_events_(self) -> int:
        """Number of events fitted so far."""
        self._check_fitted()
        return self.history_.n_events

    @property
    def log_evidence_(self) -> float:
        """Accumulated log evidence, up to the prior's normalising constant.

        Differences between models sharing a prior are meaningful; the absolute
        value is not, and under a
        :class:`~hawkes_package.inference.priors.ConstrainedPrior` it is not even
        normalised.
        """
        return self.diagnostics_.log_evidence

    # -- scikit-learn hooks ------------------------------------------------

    def __sklearn_is_fitted__(self) -> bool:
        """Report whether a fit has run, so ``check_is_fitted`` need not guess."""
        return hasattr(self, "sampler_")

    def __sklearn_tags__(self) -> Any:
        """Answer scikit-learn's 1.6+ tag protocol without inheriting from it.

        The import is inside the method because scikit-learn is not a dependency:
        nothing reaches this unless scikit-learn itself does.
        ``BaseEstimator.__sklearn_tags__`` reads nothing off ``self``, so calling
        it unbound lets the installed scikit-learn build its own ``Tags``
        dataclass -- a field added in a later release then does not become a
        ``TypeError`` from this package.
        """
        from sklearn.base import BaseEstimator

        tags = BaseEstimator.__sklearn_tags__(self)
        # There is no `y` here at all: `fit` refuses a target rather than
        # ignoring one.
        tags.target_tags.required = False
        return tags

    # -- internals ---------------------------------------------------------

    def _resolve_model(self) -> ProcessModel:
        """Return the model to fit, applying a factory named by string."""
        if isinstance(self.model, ProcessModel):
            return self.model
        try:
            factory = _TEMPORAL_FACTORIES[self.model]
        except (KeyError, TypeError):
            raise ValueError(
                f"model must be a ProcessModel or one of {sorted(_TEMPORAL_FACTORIES)}, "
                f"got {self.model!r}. A spatio-temporal model has no string spelling on "
                "purpose: build it with spatio_temporal_model(domain), so the domain is "
                "visible here."
            ) from None
        return factory()

    def _resolve_likelihood(self, model: ProcessModel) -> LogLikelihood:
        """Return the given likelihood, or the fastest one exact for `model`.

        The two families with their own likelihood are dispatched on their
        `components` type rather than on `ndim`, which is 0 for both of them.
        Without that, the fall-through hands them to `TemporalLogLikelihood`,
        whose constructor refuses them -- correctly, since it would sum the
        total intensity where a multivariate log-sum needs the intensity of the
        type each event carries, and would drop the mark density entirely -- so
        the estimator raised on two of the five model families it is meant to be
        the convenient front door for.
        """
        if self.likelihood is not None:
            return self.likelihood
        if model.ndim > 0:
            return SpatioTemporalLogLikelihood(model)
        if isinstance(model.components, MultivariateComponents):
            try:
                return MultivariateExponentialLogLikelihood(model)
            except ValueError:
                # Not the exponential parameterisation; the hook-based form is
                # exact for any kernel shape, at O(n^2 P) instead of O(n d).
                return MultivariateLogLikelihood(model)
        if isinstance(model.components, MarkedComponents):
            return MarkedLogLikelihood(model)
        if model.family == "exponential" and model.spec.names == ("mu", "alpha", "beta"):
            return ExponentialLogLikelihood(model)
        return TemporalLogLikelihood(model)

    def _build_sampler(self, likelihood: LogLikelihood) -> SMCSampler:
        """Construct the sampler, which is where the hyperparameters are validated."""
        return SMCSampler(
            likelihood,
            self.prior,
            n_particles=self.n_particles,
            evolution=self.evolution,
            ess_threshold=self.ess_threshold,
            n_move=self.n_move,
            scale=self.scale,
            jitter=self.jitter,
            resampler=self.resampler,
            on_invalid=self.on_invalid,
            rng=self.rng,
        )

    def _initialise_online(self) -> None:
        """Set up an empty fit, for a `partial_fit` that comes before any `fit`."""
        model = self._resolve_model()
        likelihood = self._resolve_likelihood(model)
        sampler = self._build_sampler(likelihood)
        sampler.initialise(start=0.0)
        self.model_ = model
        self.likelihood_ = likelihood
        self.sampler_ = sampler
        self.history_ = History.from_events(
            np.empty(0) if model.ndim == 0 else np.empty((model.ndim + 1, 0)),
            start=0.0,
            end=0.0,
        )

    def _as_record(self, X: Any, ndim: int, *, caller: str) -> np.ndarray:
        """Return `X` in the layout an event record uses, refusing an ambiguous shape."""
        return as_record(X, ndim, caller=caller)

    def _as_history(
        self,
        X: Any,
        ndim: int,
        *,
        end: float | None,
        start: float | None,
        caller: str,
    ) -> History:
        """Build the history to fit, refusing a window that was guessed or doubled."""
        if isinstance(X, History):
            if end is not None or start is not None:
                raise ValueError(
                    f"{caller} was given a History and an explicit window. The History "
                    "already carries start and end; pass one or the other."
                )
            return X
        if end is None:
            raise ValueError(
                f"{caller} needs the observation window: pass end=. There is no default, "
                "and end=times[-1] is not a safe one -- it drops the interval between the "
                "last event and the end of observation, which is the data saying nothing "
                "happened there, and biases mu upward. Pass a History if the window "
                "travels with the data."
            )
        return History.from_events(
            self._as_record(X, ndim, caller=caller),
            start=0.0 if start is None else float(start),
            end=float(end),
        )

    def _predictable_times(self, X: Any) -> np.ndarray:
        """Return the times to evaluate the intensity at, refusing the ones that lie."""
        self._check_fitted()
        return predictable_times(self.model_, self.history_, X, caller="predict")

    def _bound_process(self, theta: np.ndarray) -> TemporalHawkesProcess:
        """Build the process at `theta` and condition it on the observed history."""
        process = self.model_(theta)
        _bind_history(process, self.history_)
        if not isinstance(process, TemporalHawkesProcess):  # pragma: no cover - guarded above
            raise TypeError(f"{self.model_.family} did not build a temporal process")
        return process

    def _intensity_matrix(self, times: np.ndarray) -> np.ndarray:
        """Return the per-particle intensities, shape ``(n_particles, len(times))``.

        Through ``_conditional_intensity`` -- the hook the Ogata loop thins
        against -- and not through ``intensity_over_interval``, which merges the
        event times into the grid and so returns more values than it was given.
        """
        cloud = self.cloud_
        weights = cloud.weights
        out = np.zeros((cloud.n_particles, times.size), dtype=float)
        for index in range(cloud.n_particles):
            if weights[index] == 0.0:
                # Carries no weight, and may sit outside the support where the
                # process cannot be built. Its row of zeros never reaches the sum.
                continue
            process = self._bound_process(cloud.theta[index])
            for column, time in enumerate(times):
                out[index, column] = process._conditional_intensity(float(time))
        return out

    def _weighted_quantiles(self, matrix: np.ndarray, probabilities: Any) -> np.ndarray:
        """Weighted quantiles down the particle axis of `matrix`.

        Interpolated on the *midpoints* of the weight steps, as
        :meth:`~hawkes_package.inference.smc.ParticleCloud.quantile` is: stepping
        through the raw cumulative sum biases every quantile upward by half a
        particle's weight, systematically and in one direction.
        """
        probs = np.atleast_1d(np.asarray(probabilities, dtype=float))
        weights = self.cloud_.weights
        out = np.empty((probs.size, matrix.shape[1]), dtype=float)
        for column in range(matrix.shape[1]):
            order = np.argsort(matrix[:, column])
            ordered_weights = weights[order]
            cumulative = np.cumsum(ordered_weights)
            midpoints = cumulative - 0.5 * ordered_weights
            out[:, column] = np.interp(probs, midpoints, matrix[order, column])
        return out

    def _check_fitted(self) -> None:
        """Refuse a call that needs a fit before there has been one."""
        if not hasattr(self, "sampler_"):
            raise RuntimeError(
                f"this {type(self).__name__} is not fitted; call fit or partial_fit first"
            )

    def _check_target(self, y: Any) -> None:
        """Refuse a target, rather than ignoring one as scikit-learn would."""
        if y is not None:
            raise ValueError(
                "a Hawkes fit has no target: the events are the data. Passing y means "
                "expecting supervised semantics this estimator does not have, so it is "
                "refused rather than ignored."
            )


def predictable_times(
    model: ProcessModel, history: History, X: Any, *, caller: str = "predict"
) -> np.ndarray:
    """Return the times to evaluate an intensity at, refusing the ones that lie.

    Module level since 1.0.0 so that
    :class:`~hawkes_package.inference.mle.HawkesMLE` refuses the same times with
    the same words. Two estimators that must reject the same input are two places
    for the explanation to drift, and the explanation is the useful part -- past
    the window the intensity computed from the observed record is the intensity
    *given that nothing has happened since*, which is a different quantity, not a
    less accurate one.

    .. versionadded:: 1.0.0
    """
    if model.ndim > 0:
        raise ValueError(
            f"{caller} is temporal: a spatio-temporal intensity is not defined without "
            "a location, and the space-integrated one is a different quantity under "
            "the same name. Use predict_counts or forecast."
        )
    times = as_record(X, 0, caller=caller)
    if times.size and (times.min() < history.start or times.max() > history.end):
        raise ValueError(
            f"{caller} evaluates inside the observation window "
            f"[{history.start}, {history.end}], but the times span "
            f"[{times.min()}, {times.max()}]. Past the window the intensity computed "
            "from the observed record is the intensity given that nothing has "
            "happened since -- biased low by exactly the excitation of the events "
            "that would have occurred. Use forecast, predict_counts or "
            "predict_interval, which simulate forward instead."
        )
    return times


def as_record(X: Any, ndim: int, *, caller: str) -> np.ndarray:
    """Return `X` in the layout an event record uses, refusing an ambiguous shape.

    Module level for the reason :func:`predictable_times` is.

    .. versionadded:: 1.0.0
    """
    if isinstance(X, History):
        raise ValueError(f"{caller} takes an array of events here, not a History")
    array = np.asarray(X, dtype=float)
    if ndim == 0:
        if array.size == 0:
            return np.empty(0, dtype=float)
        if array.ndim == 2 and array.shape[1] == 1:
            # scikit-learn's column vector. Ravelled here rather than left to
            # `History.from_events`, which reads any 2-D array as
            # `(ndim + 1, n)` -- so a column of k times would be taken for a
            # single event in k - 1 dimensions.
            array = array.ravel()
        if array.ndim != 1:
            raise ValueError(
                f"{caller} expects event times of shape (n,) or (n, 1) for a temporal "
                f"model, got shape {array.shape}"
            )
        return array
    if array.size == 0:
        return np.empty((ndim + 1, 0), dtype=float)
    if array.ndim != 2 or array.shape[0] != ndim + 1:
        raise ValueError(
            f"{caller} expects a spatio-temporal record of shape ({ndim + 1}, n) with "
            f"times in row 0, got shape {array.shape}"
        )
    return array
