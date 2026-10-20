r"""Sequential Monte Carlo over the parameters, with resample-move rejuvenation.

The algorithm is Chopin's **IBIS**: an SMC sampler over the data-tempered
sequence :math:`\pi_k(\theta) \propto p(\theta)\exp\ell_k(\theta)`, where
:math:`\ell_k` is the log-likelihood of the first ``k`` blocks. Particles carry
parameters, not states; a block of data arrives, every particle's log weight
gains that block's log-likelihood increment, and when the weights have
concentrated the cloud is resampled and moved.

**Why not a bootstrap filter.** For a fully observed Hawkes process the
parameter is *static*: there is no transition to propagate through. A bootstrap
filter over a static parameter degenerates, and it does so quietly. With no
transition noise the cloud can never regain diversity -- resampling only ever
deletes particles -- so after a few hundred observations the posterior is a
single point carrying weight one. It has zero variance, so every credible
interval it reports is empty, and its location is wherever the noise happened
to leave it. Nothing about the output says any of that happened, which is why
:class:`SMCDiagnostics` records the effective sample size and the number of
distinct particles at every step rather than only at the end.

**Rejuvenation is what fixes it** (Gilks and Berzuini's resample-move). After
resampling, each particle takes a few Metropolis-Hastings steps targeting the
current posterior. The move is invariant for :math:`\pi_k`, so it changes the
particles without changing what they represent -- but only while
:math:`\pi_k` *is* a fixed target. Under a drifting parameter there is no such
target, so :class:`SMCSampler` refuses that combination at construction rather
than producing plausible output aimed at nothing.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base import SeedLike
from . import resample as resampling
from .evolution import Evolution, Static, _cholesky_with_floor
from .likelihood import History, LikelihoodState, LogLikelihood
from .parameters import ParameterSpec
from .priors import Prior, spec_for

__all__ = [
    "ParticleCloud",
    "SMCDiagnostics",
    "SMCSampler",
    "StepRecord",
    "block_boundaries",
    "fit_smc",
]

#: Effective sample size below which a cloud is reported as degenerate -- but
#: only on a step where no resample followed. A low ESS *before* a resample is
#: not a fault: it says the block was informative, and over eight blocks of a
#: 400-event history the worst value across seeds 0-20 is 0.04, with the cloud
#: fully restored by the move that follows. What is a fault is a cloud left
#: concentrated going into the next block, which is what this catches.
_DEGENERATE_ESS = 0.1

#: Distinct-ancestor fraction below which resampling has taken most of the
#: cloud's diversity. Recorded separately because the effective sample size
#: cannot see it: N copies of one particle, all with equal weights, score a
#: perfect ESS.
_DEGENERATE_UNIQUE = 0.05

#: Rejuvenation displacement, relative to the cloud's own width, below which the
#: move has not moved anything. A high acceptance rate does not imply a move: a
#: proposal scaled to 1e-12 is accepted essentially always, because it proposes
#: the point it starts from -- so the acceptance rate reads 1.0 while the cloud
#: stays exactly where the resample left it. The displacement is what tells the
#: two apart, and it is why it is recorded rather than inferred.
_DEGENERATE_MOVE = 1e-3


# ---------------------------------------------------------------------------
# The cloud
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleCloud:
    """A weighted sample from a posterior over parameters.

    Attributes
    ----------
    theta : numpy.ndarray
        Shape ``(n_particles, n_dim)``, in the constrained parameterisation the
        model uses -- what a caller wants to read. The sampler converts to the
        unconstrained scale where it needs to move.
    log_weights : numpy.ndarray
        Shape ``(n_particles,)``, **normalised**: their exponentials sum to one.
        Checked at construction, because an unnormalised cloud reports a
        healthy effective sample size while being degenerate.
    spec : ParameterSpec
        Names and bounds of the coordinates.

    .. versionadded:: 0.5.0
    """

    theta: np.ndarray
    log_weights: np.ndarray
    spec: ParameterSpec

    def __post_init__(self) -> None:
        """Refuse a cloud whose weights are not a probability distribution."""
        theta = np.atleast_2d(np.asarray(self.theta, dtype=float))
        log_weights = np.asarray(self.log_weights, dtype=float).ravel()
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "log_weights", log_weights)

        if theta.shape[0] != log_weights.size:
            raise ValueError(f"{theta.shape[0]} particle(s) but {log_weights.size} weight(s)")
        if theta.shape[1] != len(self.spec):
            raise ValueError(
                f"particles have {theta.shape[1]} coordinate(s), the spec has "
                f"{len(self.spec)}: {list(self.spec.names)}"
            )
        total = resampling.log_sum_exp(log_weights)
        if not abs(total) < 1e-8:
            raise ValueError(
                f"log weights are not normalised: they sum in probability to "
                f"{math.exp(total) if total < 700 else float('inf'):.6g}. Everything "
                "downstream -- the effective sample size above all -- reads them as a "
                "distribution, and an unnormalised cloud reports a healthy ESS while "
                "carrying no mass."
            )

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return int(self.theta.shape[0])

    @property
    def weights(self) -> np.ndarray:
        """The normalised weights on the natural scale."""
        return np.asarray(np.exp(self.log_weights), dtype=float)

    @property
    def effective_sample_size(self) -> float:
        """Kish's effective sample size of the current weights."""
        return resampling.effective_sample_size(self.log_weights)

    def mean(self) -> np.ndarray:
        """Weighted posterior mean, shape ``(n_dim,)``."""
        return np.asarray(np.average(self.theta, axis=0, weights=self.weights), dtype=float)

    def covariance(self) -> np.ndarray:
        """Weighted posterior covariance, shape ``(n_dim, n_dim)``."""
        centred = self.theta - self.mean()
        covariance = (centred * self.weights[:, None]).T @ centred
        return np.asarray(0.5 * (covariance + covariance.T), dtype=float)

    def std(self) -> np.ndarray:
        """Weighted posterior standard deviation per coordinate."""
        return np.asarray(np.sqrt(np.diag(self.covariance())), dtype=float)

    def quantile(self, q: Any) -> np.ndarray:
        """Weighted quantiles per coordinate.

        Parameters
        ----------
        q : float or array_like
            Probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(q), n_dim)``, or ``(n_dim,)`` for a scalar `q`.

        Notes
        -----
        The weighted empirical CDF is stepped through with
        :func:`numpy.interp` on the *midpoints* of the weight steps. Using the
        cumulative sum directly biases every quantile towards the upper end by
        half a particle's weight, which for a 128-particle cloud is 0.4% of the
        distribution -- small, and systematically in one direction, so it does
        not average out over coordinates or over runs.
        """
        probabilities = np.atleast_1d(np.asarray(q, dtype=float))
        flat = np.isscalar(q) or np.ndim(q) == 0
        out = np.empty((probabilities.size, self.theta.shape[1]), dtype=float)
        for j in range(self.theta.shape[1]):
            order = np.argsort(self.theta[:, j])
            values = self.theta[order, j]
            weights = self.weights[order]
            cumulative = np.cumsum(weights)
            midpoints = cumulative - 0.5 * weights
            out[:, j] = np.interp(probabilities, midpoints, values)
        return out[0] if flat else out

    def credible_interval(self, level: float = 0.9) -> np.ndarray:
        """Equal-tailed credible interval per coordinate.

        Parameters
        ----------
        level : float
            Coverage, e.g. ``0.9`` for the interval between the 5th and 95th
            percentiles.

        Returns
        -------
        numpy.ndarray
            Shape ``(2, n_dim)``: lower bounds then upper bounds.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        tail = 0.5 * (1.0 - level)
        return self.quantile([tail, 1.0 - tail])

    def resample(
        self,
        rng: np.random.Generator,
        resampler: Callable[[Any, np.random.Generator], np.ndarray] = resampling.systematic,
    ) -> tuple[ParticleCloud, np.ndarray]:
        """Resample to equal weights, returning the new cloud and the ancestors."""
        indices = resampler(self.log_weights, rng)
        flat = np.full(self.n_particles, -math.log(self.n_particles), dtype=float)
        return ParticleCloud(self.theta[indices].copy(), flat, self.spec), indices

    def summary(self) -> str:
        """Tabulate the marginal mean, standard deviation and credible interval."""
        mean, sd = self.mean(), self.std()
        low, high = self.credible_interval(0.9)
        width = max(len(name) for name in self.spec.names)
        lines = [
            f"{'parameter':<{width}}  {'mean':>10}  {'sd':>10}  {'90% interval':>24}",
        ]
        for j, name in enumerate(self.spec.names):
            lines.append(
                f"{name:<{width}}  {mean[j]:>10.4g}  {sd[j]:>10.4g}  "
                f"{f'[{low[j]:.4g}, {high[j]:.4g}]':>24}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """What happened at one block of the fit.

    Attributes
    ----------
    upto : float
        The time the block ended at.
    n_events : int
        Events accounted for in total by the end of the block.
    ess : float
        Effective sample size after reweighting, before any resample.
    resampled : bool
        Whether the effective sample size fell below the threshold.
    move_acceptance : float
        Fraction of Metropolis proposals accepted during rejuvenation, or
        ``nan`` when no move was made. Zero means nothing moved -- but so, in
        effect, does an acceptance rate of one with a proposal scaled to
        nothing, which is why `move_size` is recorded beside it.
    move_size : float
        Root-mean-square distance a particle travelled during rejuvenation, as a
        fraction of the cloud's own width on the unconstrained scale, or ``nan``
        when no move was made. Around a half for a healthy move. **Near zero is the number
        to look for**: the cloud is then exactly where the resample left it, so
        the posterior is a resampled prior wearing a posterior's weights.
    unique_fraction : float
        Distinct ancestors as a fraction of the cloud, measured after resampling
        and before any move -- so it says how much diversity the resample took,
        which is the quantity rejuvenation exists to put back.
    log_evidence_increment : float
        ``log p(block | earlier blocks)``.
    n_invalid : int
        Particles whose likelihood could not be evaluated.
    """

    upto: float
    n_events: int
    ess: float
    resampled: bool
    move_acceptance: float
    move_size: float
    unique_fraction: float
    log_evidence_increment: float
    n_invalid: int


@dataclass
class SMCDiagnostics:
    """The record of a whole fit, and what it says about whether to trust it.

    Attributes
    ----------
    steps : list of StepRecord
        One per block.
    n_particles : int
        Size of the cloud.
    backend : str
        Which likelihood path ran, where the likelihood has more than one.

    .. versionadded:: 0.5.0
    """

    n_particles: int
    steps: list[StepRecord] = field(default_factory=list)
    backend: str = ""

    @property
    def log_evidence(self) -> float:
        r"""The accumulated :math:`\log p(\text{data})`.

        Exact only up to the normalising constant of the prior. A
        :class:`~hawkes_package.inference.priors.ConstrainedPrior` is
        unnormalised by design -- the truncation constant cancels everywhere
        else -- so an evidence computed under one is shifted by an unknown
        additive constant. Differences between models sharing a prior are still
        meaningful; absolute values are not.
        """
        return float(sum(step.log_evidence_increment for step in self.steps))

    @property
    def min_ess_fraction(self) -> float:
        """The lowest effective sample size seen, as a fraction of the cloud.

        Measured *before* each resample, so a low value on its own is a report
        about how informative a block was rather than about the cloud's health.
        Pair it with :attr:`min_unique_fraction`, and with
        :meth:`ess_recovered`, which is the property that actually matters.
        """
        if not self.steps:
            return math.nan
        return min(step.ess for step in self.steps) / self.n_particles

    def ess_recovered(self, threshold: float = _DEGENERATE_ESS) -> bool:
        """Whether every badly concentrated cloud was resampled rather than carried.

        The health question the raw minimum cannot answer. Weights concentrate
        whenever a block is informative; the sampler is working as long as it
        notices and rebuilds the cloud, and is failing as soon as it carries a
        concentrated one into the next block.
        """
        return all(step.resampled for step in self.steps if step.ess < threshold * self.n_particles)

    @property
    def min_unique_fraction(self) -> float:
        """The lowest distinct-ancestor fraction seen."""
        if not self.steps:
            return math.nan
        return min(step.unique_fraction for step in self.steps)

    @property
    def min_move_size(self) -> float:
        """The smallest rejuvenation displacement seen, relative to the cloud's width."""
        moved = [s.move_size for s in self.steps if not math.isnan(s.move_size)]
        return min(moved) if moved else math.nan

    def warnings(self) -> list[str]:
        """Everything about this fit that a reader should be told unprompted."""
        notes: list[str] = []
        if not self.steps:
            return notes
        if not self.ess_recovered():
            worst = min(
                (s for s in self.steps if not s.resampled), key=lambda s: s.ess, default=None
            )
            if worst is not None:
                notes.append(
                    f"the effective sample size fell to "
                    f"{100 * worst.ess / self.n_particles:.1f}% of the cloud at t="
                    f"{worst.upto:.4g} and no resample followed, so a posterior carried by "
                    "a handful of particles was taken into the next block. Its width is "
                    "not to be trusted. Raise ess_threshold, or use more particles."
                )
        if self.min_unique_fraction < _DEGENERATE_UNIQUE:
            notes.append(
                f"resampling left only {100 * self.min_unique_fraction:.1f}% of the cloud "
                "distinct: nearly every particle is a copy. Rejuvenation has to put that "
                "diversity back, so check move_size and move_acceptance, and use smaller "
                "blocks."
            )

        sizes = [s.move_size for s in self.steps if not math.isnan(s.move_size)]
        if sizes and max(sizes) < _DEGENERATE_MOVE:
            notes.append(
                f"rejuvenation moved the cloud by at most {max(sizes):.2g} of its own "
                "width, so it never moved: the posterior is the resampled prior wearing a "
                "posterior's weights. The proposal scale is far too small -- note that "
                "the acceptance rate cannot show this, because a proposal scaled to "
                "nothing is accepted almost always."
            )

        accepted = [s.move_acceptance for s in self.steps if not math.isnan(s.move_acceptance)]
        if accepted and max(accepted) <= 0.0:
            notes.append(
                "no rejuvenation proposal was ever accepted, so the cloud never moved "
                "after resampling. The proposal scale is far too large."
            )
        return notes

    def summary(self) -> str:
        """Report the fit: a per-block table, then anything wrong with it."""
        header = (
            f"{'upto':>10}  {'events':>7}  {'ESS':>8}  {'resamp':>6}  "
            f"{'accept':>7}  {'move':>8}  {'unique':>7}  {'dlogZ':>10}"
        )
        title = f"SMC over {self.n_particles} particles"
        if self.backend:
            title += f", backend={self.backend}"
        lines = [title, header]
        for step in self.steps:
            acceptance = "-" if math.isnan(step.move_acceptance) else f"{step.move_acceptance:.3f}"
            travelled = "-" if math.isnan(step.move_size) else f"{step.move_size:.2e}"
            lines.append(
                f"{step.upto:>10.4g}  {step.n_events:>7d}  {step.ess:>8.1f}  "
                f"{'yes' if step.resampled else 'no':>6}  {acceptance:>7}  "
                f"{travelled:>8}  {step.unique_fraction:>7.3f}  "
                f"{step.log_evidence_increment:>10.4g}"
            )
        lines.append(f"log evidence {self.log_evidence:.6g} (up to the prior's constant)")
        for note in self.warnings():
            lines.append(f"WARNING: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------


class SMCSampler:
    """Fit a Hawkes model by sequential Monte Carlo, in blocks as data arrives.

    Parameters
    ----------
    likelihood : LogLikelihood
        Any of the three implementations in
        :mod:`~hawkes_package.inference.likelihood`.
    prior : Prior
        Must draw as many coordinates as the model has; checked at construction,
        because a prior one coordinate short broadcasts into a plausible array
        rather than failing.
    n_particles : int
        Cloud size. The posterior's *width* is what a small cloud gets wrong,
        not its centre, so a run that looks converged at 64 particles may still
        be reporting intervals half the size they should be.
    evolution : Evolution, optional
        Defaults to :class:`~hawkes_package.inference.evolution.Static`, which
        is the exact IBIS sampler. Anything else makes this a filter over a
        drifting parameter, with the approximation
        :mod:`~hawkes_package.inference.evolution` describes.
    ess_threshold : float
        Resample when the effective sample size falls below this fraction of the
        cloud.
    n_move : int
        Metropolis steps per particle after a resample. **Must be zero for a
        non-static evolution**, and the constructor raises otherwise: an
        invariant kernel targeting a posterior the model says no longer exists
        is incoherent, and it is exactly the configuration reached by tuning
        until the output looks plausible.
    scale : float
        Multiplier on the optimal random-walk scaling. The proposal covariance
        is ``(scale**2 / n_dim) * C`` with ``C`` the weighted covariance of the
        cloud before resampling; 2.38 is Roberts and Rosenthal's constant.
    jitter : float
        Relative ridge added to that covariance before factorising. The cloud
        that most needs a move is the one that has collapsed, and a collapsed
        cloud has a singular covariance -- so this is what stops the
        factorisation failing exactly when it is needed.
    resampler : callable
        :func:`~hawkes_package.inference.resample.systematic` by default.
    on_invalid : {"raise", "reject"}
        What to do when a particle's likelihood cannot be evaluated. Raising is
        the default on purpose: reading an exception as "zero posterior mass" is
        how a broadcasting typo in a user kernel becomes a confident posterior.
    rng : None, int or numpy.random.Generator
        Source of randomness. Never the global stream.

    Attributes
    ----------
    cloud : ParticleCloud
        The current posterior sample.
    diagnostics : SMCDiagnostics

    Examples
    --------
    >>> import hawkes_package as hp
    >>> from hawkes_package.inference import (
    ...     ExponentialLogLikelihood,
    ...     History,
    ...     IndependentPrior,
    ...     LogNormal,
    ...     SMCSampler,
    ...     exponential_model,
    ... )
    >>> truth = hp.ExponentialHawkes(np.array([2.0, 0.5, 1.0]), rng=7)
    >>> truth.simulate(200)
    >>> history = History.from_simulation(truth)
    >>> model = exponential_model()
    >>> smc = SMCSampler(
    ...     ExponentialLogLikelihood(model),
    ...     IndependentPrior((LogNormal(0.5, 1.0), LogNormal(-1.0, 1.0), LogNormal(0.0, 1.0))),
    ...     n_particles=64,
    ...     rng=0,
    ... )
    >>> cloud = smc.run(history, blocks=2)
    >>> cloud.mean().shape
    (3,)

    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        likelihood: LogLikelihood,
        prior: Prior,
        *,
        n_particles: int = 512,
        evolution: Evolution | None = None,
        ess_threshold: float = 0.5,
        n_move: int = 3,
        scale: float = 2.38,
        jitter: float = 1e-10,
        resampler: Callable[[Any, np.random.Generator], np.ndarray] = resampling.systematic,
        on_invalid: str = "raise",
        rng: SeedLike = None,
    ) -> None:
        self.likelihood = likelihood
        self.model = likelihood.model
        self.spec: ParameterSpec = self.model.spec
        self.prior = prior
        spec_for(prior, self.spec)

        self.n_particles = int(n_particles)
        if self.n_particles < 2:
            raise ValueError(f"n_particles must be at least 2, got {n_particles}")
        self.evolution: Evolution = Static() if evolution is None else evolution
        if not 0.0 < ess_threshold <= 1.0:
            raise ValueError(f"ess_threshold must lie in (0, 1], got {ess_threshold}")
        self.ess_threshold = float(ess_threshold)
        self.n_move = int(n_move)
        if self.n_move < 0:
            raise ValueError(f"n_move must be non-negative, got {n_move}")
        if self.n_move > 0 and not self.evolution.static:
            raise ValueError(
                f"n_move={self.n_move} with {type(self.evolution).__name__} asks for an "
                "MCMC move that is invariant for a posterior this model says does not "
                "exist: under a drifting parameter the target changes at every block, so "
                "there is nothing for the kernel to leave invariant. Either set n_move=0 "
                "and let the evolution supply the diversity, or use Static()."
            )
        if not scale > 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = float(scale)
        self.jitter = float(jitter)
        self.resampler = resampler
        if on_invalid not in {"raise", "reject"}:
            raise ValueError(f"on_invalid must be raise or reject, got {on_invalid!r}")
        self.on_invalid = on_invalid
        self.rng = np.random.default_rng(rng)

        self.cloud: ParticleCloud | None = None
        self.diagnostics = SMCDiagnostics(n_particles=self.n_particles)
        self._states: list[LikelihoodState] = []
        self._log_prior: np.ndarray = np.empty(0, dtype=float)

    # -- setup -------------------------------------------------------------

    def initialise(self, *, start: float = 0.0, proposal: Prior | None = None) -> None:
        """Draw the cloud and set every particle's state to `start`.

        Parameters
        ----------
        start : float
            Where each particle's likelihood state begins.
        proposal : Prior, optional
            Draw from this instead of the prior, and correct for it by weight.
            **The target is unchanged**: the particles carry
            ``log prior - log proposal`` from the outset, so the posterior is the
            posterior of `prior` and the proposal only decides where the cloud
            starts looking.

            That distinction is the whole reason this argument exists rather
            than a suggestion to pass a tighter prior. Measured on 300 events
            with a proposal centred on the maximum likelihood estimate: used as
            a *prior* it moves the four-block posterior mean from
            ``(0.95, 0.21, 1.52)`` to ``(1.07, 0.30, 2.39)`` -- the prior was
            concentrated where the likelihood is and the answer moved with it --
            while used as a proposal here the answer is the vague prior's
            answer, reached from a cloud that started near it.

            A proposal far from the prior costs effective sample size instead of
            correctness, which :attr:`diagnostics` reports.

        Raises
        ------
        ValueError
            If the prior -- or the proposal, when given -- puts a particle
            outside the model's support. The prior is the caller's statement
            about where the truth might be, and a statement that includes
            parameters the process cannot be simulated at is worth correcting
            rather than quietly filtering -- wrap it in a
            :class:`~hawkes_package.inference.priors.ConstrainedPrior`.

        .. versionchanged:: 1.0.0
           Gained `proposal`.
        """
        source = self.prior if proposal is None else proposal
        drawn = source.sample(self.n_particles, self.rng)
        theta = np.atleast_2d(np.asarray(drawn, dtype=float))
        outside = ~np.asarray(self.model.support(theta), dtype=bool)
        if np.any(outside):
            first = int(np.argmax(outside))
            label = "prior" if proposal is None else "proposal"
            raise ValueError(
                f"{int(np.sum(outside))} of {self.n_particles} {label} draws lie outside the "
                f"model's support, the first being {theta[first].tolist()} with branching "
                f"ratio {float(self.model.branching_ratio(theta[first])):.4g}. Wrap the "
                f"{label} in ConstrainedPrior({label}, model.support) so the cloud starts "
                "where the process exists."
            )
        self._log_prior = np.asarray(self.prior.log_pdf(theta), dtype=float)
        if proposal is None:
            log_weights = np.full(self.n_particles, -math.log(self.n_particles), dtype=float)
        else:
            # The importance correction, and the reason the target does not move.
            log_weights = self._log_prior - np.asarray(proposal.log_pdf(theta), dtype=float)
            log_weights = log_weights - resampling.log_sum_exp(log_weights)
        self.cloud = ParticleCloud(theta, log_weights, self.spec)
        self._states = [
            self.likelihood.initial_state(float(start)) for _ in range(self.n_particles)
        ]
        self.diagnostics = SMCDiagnostics(n_particles=self.n_particles)

    # -- the online step ---------------------------------------------------

    def update(self, history: History, upto: float) -> None:
        """Absorb the block of data ending at `upto`.

        The online entry point: call it as each block of events arrives, with
        the history grown to include them and `upto` the time up to which the
        process has now been *observed* -- which is not the same as the last
        event time, and the difference is the information that nothing happened
        since.
        """
        if self.cloud is None:
            self.initialise(start=history.start)
        assert self.cloud is not None

        if not self.evolution.static:
            self._propagate(history)

        increments, invalid = self._reweight(self.cloud.theta, history, float(upto))

        # log Z increment, from the *normalised* pre-update weights.
        evidence = resampling.log_sum_exp(self.cloud.log_weights + increments)
        log_weights = resampling.log_normalise(self.cloud.log_weights + increments)
        self.cloud = ParticleCloud(self.cloud.theta, log_weights, self.spec)

        ess = self.cloud.effective_sample_size
        resampled = ess < self.ess_threshold * self.n_particles
        acceptance = math.nan
        move_size = math.nan
        unique = 1.0

        if resampled:
            # The covariance comes from the weighted cloud *before* resampling:
            # afterwards the duplicates have collapsed it towards whatever
            # survived, and a proposal scaled to that explores a region the
            # posterior has already been narrowed out of.
            proposal = self._proposal_factor(self.cloud)
            self.cloud, ancestors = self.cloud.resample(self.rng, self.resampler)
            self._states = [self._states[i] for i in ancestors]
            self._log_prior = self._log_prior[ancestors]
            unique = resampling.unique_fraction(ancestors, self.n_particles)
            if self.n_move:
                acceptance, move_size = self._rejuvenate(proposal, history, float(upto))

        self.diagnostics.steps.append(
            StepRecord(
                upto=float(upto),
                # Read off the history rather than off a particle. Every
                # particle sees the same events, so any of them would do --
                # except one that was abandoned, whose count stops advancing.
                n_events=int(np.count_nonzero(history.times <= float(upto))),
                ess=ess,
                resampled=resampled,
                move_acceptance=acceptance,
                move_size=move_size,
                unique_fraction=unique,
                log_evidence_increment=evidence,
                n_invalid=invalid,
            )
        )
        self.diagnostics.backend = str(getattr(self.likelihood, "backend_used", ""))

    def run(self, history: History, *, blocks: int | Any = 1) -> ParticleCloud:
        """Fit the whole history, in `blocks` successive chunks.

        Parameters
        ----------
        history : History
            The observed data and its window.
        blocks : int or array_like
            An integer splits the events into that many chunks of roughly equal
            count; an array is used as the block boundaries directly. The last
            boundary is always ``history.end``, so the empty tail after the last
            event is accounted for.

        Returns
        -------
        ParticleCloud
            The posterior sample after the whole history.
        """
        if self.cloud is None:
            self.initialise(start=history.start)
        for boundary in block_boundaries(history, blocks):
            self.update(history, boundary)
        assert self.cloud is not None
        return self.cloud

    # -- internals ---------------------------------------------------------

    def _reweight(self, theta: np.ndarray, history: History, upto: float) -> tuple[np.ndarray, int]:
        """Extend every particle's likelihood to `upto`, returning the increments."""
        increments = np.empty(self.n_particles, dtype=float)
        supported = np.asarray(self.model.support(theta), dtype=bool)
        invalid = 0

        for i in range(self.n_particles):
            if not supported[i]:
                increments[i] = self._abandon(i, upto)
                invalid += 1
                continue
            try:
                self._states[i], increments[i] = self.likelihood.extend(
                    self._states[i], theta[i], history, upto
                )
            except (ValueError, ArithmeticError, RuntimeError) as failure:
                if self.on_invalid == "raise":
                    raise RuntimeError(
                        f"the likelihood failed at particle {i}, theta="
                        f"{theta[i].tolist()}: {failure}. This parameter is inside the "
                        "model's declared support, so the failure is a bug rather than a "
                        'datum -- pass on_invalid="reject" only once you know which.'
                    ) from failure
                increments[i] = self._abandon(i, upto)
                invalid += 1

        if invalid and self.on_invalid == "reject":
            warnings.warn(
                f"{invalid} of {self.n_particles} particles could not be evaluated and "
                "were given zero weight. Each one narrows the posterior without "
                "contributing to it.",
                UserWarning,
                stacklevel=3,
            )
        return increments, invalid

    def _abandon(self, index: int, upto: float) -> float:
        """Give particle `index` zero weight, and move its state on regardless.

        Advancing the state matters even though the weight is zero. Rejuvenation
        compares a proposal's log-likelihood *at* the block boundary against the
        particle's own, so a state left behind at an earlier time would be
        compared against something it is not commensurable with. Systematic
        resampling essentially never selects a zero-weight particle, so this is
        a case that should not arise -- which is exactly the kind that is worth
        making unreachable rather than unlikely.
        """
        state = self._states[index]
        self._states[index] = LikelihoodState(
            upto=float(upto), n_events=state.n_events, log_lik=-math.inf, carry=state.carry
        )
        return -math.inf

    def _propagate(self, history: History) -> None:
        """Move the cloud through the evolution, then rebuild what it invalidated."""
        assert self.cloud is not None
        z = self.spec.to_unconstrained(self.cloud.theta)
        moved = self.evolution.propagate(z, self.cloud.log_weights, self.rng)
        theta = self.spec.to_constrained(moved)

        # A drifting particle can walk out of the support -- past the
        # stationarity boundary above all. Holding it where it was keeps the
        # cloud a valid sample of *something*, which discarding it would not.
        outside = ~np.asarray(self.model.support(theta), dtype=bool)
        if np.any(outside):
            theta[outside] = self.cloud.theta[outside]

        self.cloud = ParticleCloud(theta, self.cloud.log_weights, self.spec)
        self._log_prior = np.asarray(self.prior.log_pdf(theta), dtype=float)
        # The resumption summary was computed under the old parameters -- for the
        # exponential path it is a decayed sum whose decay rate has just changed
        # -- so it has to be rebuilt. This is the locally-stationary
        # approximation made concrete: the block ahead is scored entirely under
        # the new parameters, including the excitation left by events generated
        # under the old ones.
        for i in range(self.n_particles):
            state = self._states[i]
            rebuilt = self.likelihood.extend(
                self.likelihood.initial_state(history.start), theta[i], history, state.upto
            )[0]
            self._states[i] = LikelihoodState(
                upto=state.upto,
                n_events=state.n_events,
                log_lik=state.log_lik,
                carry=rebuilt.carry,
            )

    def _proposal_factor(self, cloud: ParticleCloud) -> np.ndarray:
        """Cholesky factor of the random-walk proposal covariance."""
        z = self.spec.to_unconstrained(cloud.theta)
        weights = cloud.weights
        mean = np.average(z, axis=0, weights=weights)
        centred = z - mean
        covariance = (centred * weights[:, None]).T @ centred
        covariance = 0.5 * (covariance + covariance.T)
        n_dim = len(self.spec)
        ridge = self.jitter * max(float(np.trace(covariance)) / n_dim, np.finfo(float).tiny)
        scaled = (self.scale**2 / n_dim) * covariance + ridge * np.eye(n_dim)
        return _cholesky_with_floor(scaled)

    def _rejuvenate(self, factor: np.ndarray, history: History, upto: float) -> tuple[float, float]:
        """Take `n_move` Metropolis steps per particle.

        Returns the acceptance rate and the root-mean-square displacement
        relative to the cloud's own width -- the second because the first cannot
        distinguish a move that worked from a proposal so small that every
        candidate is the point it started from.
        """
        assert self.cloud is not None
        z = self.spec.to_unconstrained(self.cloud.theta)
        started_at = z.copy()
        log_target = (
            self._log_prior
            + np.array([state.log_lik for state in self._states])
            + self.spec.log_abs_det_jacobian(z)
        )

        accepted = 0
        proposed = 0
        for _ in range(self.n_move):
            step = self.rng.normal(0.0, 1.0, size=z.shape) @ factor.T
            candidate = z + step
            theta = self.spec.to_constrained(candidate)
            # The support is gated first, so a proposal the process cannot be
            # built at costs nothing rather than an exception to interpret.
            allowed = np.asarray(self.model.support(theta), dtype=bool)
            log_prior = np.full(self.n_particles, -math.inf, dtype=float)
            if np.any(allowed):
                log_prior[allowed] = np.asarray(self.prior.log_pdf(theta[allowed]), dtype=float)
            allowed &= np.isfinite(log_prior)

            jacobian = self.spec.log_abs_det_jacobian(candidate)
            uniform = np.log(self.rng.uniform(size=self.n_particles))

            for i in range(self.n_particles):
                proposed += 1
                if not allowed[i]:
                    continue
                # An accepted move voids the carry, so the likelihood is
                # recomputed in full here. This is the dominant cost of the
                # whole fit: n_particles * n_move full evaluations per resample.
                try:
                    state = self.likelihood.extend(
                        self.likelihood.initial_state(history.start), theta[i], history, upto
                    )[0]
                except (ValueError, ArithmeticError, RuntimeError) as failure:
                    # `on_invalid` has to mean the same thing here as it does
                    # when a block is absorbed, or a parameter the caller has
                    # chosen to tolerate would still end the run -- just later,
                    # and only on the steps that happen to resample.
                    if self.on_invalid == "raise":
                        raise RuntimeError(
                            f"the likelihood failed on a rejuvenation proposal for "
                            f"particle {i}, theta={theta[i].tolist()}: {failure}. This "
                            "parameter is inside the model's declared support, so the "
                            "failure is a bug rather than a datum -- pass "
                            'on_invalid="reject" only once you know which.'
                        ) from failure
                    continue
                proposal_target = log_prior[i] + state.log_lik + jacobian[i]
                if not np.isfinite(proposal_target):
                    continue
                if uniform[i] < proposal_target - log_target[i]:
                    z[i] = candidate[i]
                    log_target[i] = proposal_target
                    self._log_prior[i] = log_prior[i]
                    self._states[i] = state
                    accepted += 1

        self.cloud = ParticleCloud(self.spec.to_constrained(z), self.cloud.log_weights, self.spec)
        # The width of the cloud the move started from, so the displacement is
        # reported in units of the thing it is meant to explore.
        width = float(np.sqrt(np.sum(np.var(started_at, axis=0))))
        # Root mean square, not the median. At the optimal acceptance rate of
        # 0.234 and three moves a particle has a 0.55 chance of never moving at
        # all, so on a perfectly healthy step the median displacement is exactly
        # zero -- which read as "the kernel is frozen" on five of twenty-one
        # seeds before this was measured properly.
        travelled = float(np.sqrt(np.mean(np.sum((z - started_at) ** 2, axis=1))))
        move_size = travelled / width if width > 0 else math.nan
        return (accepted / proposed if proposed else math.nan), move_size


def block_boundaries(history: History, blocks: int | Any = 1) -> list[float]:
    """Return the times a history is split at for a blocked fit.

    An integer gives that many chunks of roughly equal event count; an array is
    taken as the boundaries themselves. Either way the last boundary is
    ``history.end``, because the interval between the last event and the end of
    the observation window carries the information that nothing happened in it.
    """
    if isinstance(blocks, (int, np.integer)):
        count = int(blocks)
        if count < 1:
            raise ValueError(f"blocks must be at least 1, got {count}")
        if history.n_events == 0:
            return [history.end]
        cuts = np.unique(np.linspace(0, history.n_events, count + 1).astype(int))[1:]
        times = [float(history.times[c - 1]) for c in cuts]
        times[-1] = history.end
        return times

    values = [float(t) for t in np.asarray(blocks, dtype=float).ravel()]
    if not values:
        raise ValueError("blocks must not be empty")
    if any(b <= a for a, b in itertools.pairwise(values)):
        raise ValueError(f"block boundaries must be strictly increasing, got {values}")
    if values[0] <= history.start or values[-1] > history.end:
        raise ValueError(
            f"block boundaries must lie in ({history.start}, {history.end}], got "
            f"[{values[0]}, {values[-1]}]"
        )
    if values[-1] < history.end:
        values.append(history.end)
    return values


def fit_smc(
    likelihood: LogLikelihood,
    prior: Prior,
    history: History,
    *,
    blocks: int | Sequence[float] = 1,
    proposal: Prior | None = None,
    **kwargs: Any,
) -> SMCSampler:
    """Build an :class:`SMCSampler`, run it over `history`, and return it.

    The sampler rather than the cloud, so the diagnostics come back with it:
    a posterior read without its effective sample size is a posterior whose
    width means nothing in particular.

    Parameters
    ----------
    likelihood, prior, history :
        As for :class:`SMCSampler`.
    blocks : int or sequence of float
        Passed to :meth:`SMCSampler.run`.
    proposal : Prior, optional
        Passed to :meth:`SMCSampler.initialise`: where the cloud starts, with an
        importance correction so that the posterior remains the posterior under
        `prior`. See
        :func:`~hawkes_package.inference.mle.warm_start_proposal`.
    **kwargs :
        Passed to :class:`SMCSampler`.

    .. versionchanged:: 1.0.0
       Gained `proposal`.

    .. versionadded:: 0.5.0
    """
    sampler = SMCSampler(likelihood, prior, **kwargs)
    sampler.initialise(start=history.start, proposal=proposal)
    sampler.run(history, blocks=blocks)
    return sampler
