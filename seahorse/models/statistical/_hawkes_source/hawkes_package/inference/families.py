"""Parameterised kernels, backgrounds and nonlinearities.

The simulation side takes plain callables: a kernel is whatever function the
caller hands it. Inference needs more than a callable, because it has to build
one *per particle, per move* and then say things about it in closed form. A
family is that extra structure -- a :class:`ParameterSpec`, a builder, and two
summaries the sampler cannot afford to compute numerically:

* :meth:`~KernelFamily.peak`, because
  :class:`~hawkes_package.bell_shape.BellShapeHawkes` and the non-monotone
  spatio-temporal path locate the kernel's maximum at construction with
  :func:`~hawkes_package._numerics.locate_peak`, a 513-point scan with a bounded
  refinement. That is the right thing to do for a kernel nobody can
  differentiate, and the wrong thing to do 512 times per rejuvenation move for a
  kernel whose maximum is one line of algebra. Every family here returns the
  algebra, and the tests check it against the scan.
* :meth:`~KernelFamily.mass`, the integral over all lags, because the branching
  ratio decides whether a parameter is one the process can be *simulated* at.
  Guessing it wrong does not raise: the particle survives, the likelihood on a
  finite window is finite, and only the excitation estimate is wrong.

.. versionadded:: 0.5.0
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._numerics import PeakLocation
from ..periodic import PeriodicBackground, PeriodicSchedule
from .parameters import Parameter, ParameterSpec

__all__ = [
    "BaseFamily",
    "CompactSpatial",
    "ConstantBase",
    "ExcitationMatrix",
    "ExponentialKernel",
    "GammaKernel",
    "GaussianSpatial",
    "KernelFamily",
    "LinearNonlinearity",
    "LogLinearBase",
    "MultivariateBase",
    "NonlinearityFamily",
    "OmoriUtsuKernel",
    "ParetoSpatial",
    "PeriodicBase",
    "SoftPlusNonlinearity",
    "SpatialKernelFamily",
    "UnitExponentialKernel",
]


def _batch(theta: Any, n_dim: int) -> tuple[np.ndarray, bool]:
    """Coerce to ``(n, n_dim)``, reporting whether the input was a single vector."""
    array = np.asarray(theta, dtype=float)
    flat = array.ndim == 1
    if flat:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != n_dim:
        raise ValueError(f"expected {n_dim} parameter(s), got shape {np.shape(theta)}")
    return array, flat


def _unbatch(values: np.ndarray, flat: bool) -> np.ndarray:
    """Undo :func:`_batch`'s promotion."""
    return values[0] if flat else values


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class KernelFamily(Protocol):
    """A parameterised temporal excitation kernel."""

    @property
    def spec(self) -> ParameterSpec:
        """The kernel's own parameters, in the order it consumes them."""
        ...

    @property
    def monotone(self) -> bool:
        """Whether the kernel decreases from lag zero onwards."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel at one parameter vector, vectorized over lags."""
        ...

    def peak(self, theta: Any) -> PeakLocation:
        """Where the kernel attains its maximum, and the value there."""
        ...

    def mass(self, theta: Any) -> np.ndarray:
        """Integral of the kernel over ``[0, inf)``, batched over `theta`."""
        ...


@runtime_checkable
class SpatialKernelFamily(Protocol):
    """A parameterised isotropic spatial kernel, evaluated at a distance.

    **A family here must be normalised to unit mass on the model space**, and
    that is a requirement rather than a convention every shipped family happens
    to meet. It is what makes the mass over a *compact* domain at most one, and
    so what makes the temporal kernel's mass a sufficient bound on the branching
    ratio -- with no quadrature, and no dependence on which surface the process
    lives on. A family normalised some other way would leave
    :meth:`ProcessModel.support` admitting supercritical parameters, and the
    failure would appear as an explosion during simulation rather than as a
    rejected proposal.

    The kernel receives a scalar distance, so it is isotropic by construction.
    An anisotropic kernel is not excluded by anything here, but it needs the
    ``pairwise`` protocol the periodised images already use -- both endpoints
    rather than the distance between them -- rather than a wider version of this
    one.

    .. versionchanged:: 1.0.0
       Unit mass is stated as a requirement, and ``mass`` may report ``inf``
       where a family's own integral diverges.
    """

    @property
    def spec(self) -> ParameterSpec:
        """The kernel's own parameters."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel at one parameter vector, vectorized over distances."""
        ...

    def mass(self, theta: Any) -> np.ndarray:
        """Integral over the whole model space, batched over `theta`.

        ``1.0`` for a family that meets the normalisation above, and ``inf``
        wherever the integral does not converge -- never a plausible finite
        number there, because
        :meth:`~hawkes_package.inference.models.ProcessModel.support` evaluates
        the branching callable on every row of a batch before the parameter
        bounds filter it.
        """
        ...

    def min_scale(self, theta: Any) -> np.ndarray:
        """Return the smallest length the kernel varies on, batched over `theta`."""
        ...


@runtime_checkable
class BaseFamily(Protocol):
    """A parameterised background intensity."""

    @property
    def spec(self) -> ParameterSpec:
        """The background's own parameters."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the background at one parameter vector, as a function of position."""
        ...

    def at(self, theta: Any, points: Any) -> np.ndarray:
        """Evaluate the background at `points`, shape ``(m, ndim)`` -> ``(m,)``."""
        ...


@runtime_checkable
class NonlinearityFamily(Protocol):
    """A parameterised monotone-increasing map applied to the excitation sum."""

    @property
    def spec(self) -> ParameterSpec:
        """The nonlinearity's own parameters."""
        ...

    @property
    def lipschitz(self) -> float:
        """A Lipschitz constant, which bounds the stability condition."""
        ...

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the nonlinearity at one parameter vector."""
        ...


# ---------------------------------------------------------------------------
# Temporal kernels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExponentialKernel:
    r"""The kernel :math:`\kappa(s) = \alpha e^{-\beta s}`.

    Written in the same convention
    :class:`~hawkes_package.exponential.ExponentialHawkes` uses, so ``alpha`` is
    the jump in intensity an event causes and the branching ratio is
    ``alpha / beta`` rather than ``alpha``. Keeping the two conventions apart
    matters: reading ``alpha`` as a branching ratio understates the excitation
    by a factor of ``beta``, and nothing about the fitted numbers says so.

    Monotone decreasing, so its maximum is at lag zero.
    """

    monotone: bool = True

    @property
    def spec(self) -> ParameterSpec:
        """``(alpha, beta)``, both positive."""
        return ParameterSpec((Parameter("alpha"), Parameter("beta")))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative lags."""
        values, _ = _batch(theta, 2)
        alpha, beta = float(values[0, 0]), float(values[0, 1])
        return lambda s: alpha * np.exp(-beta * np.asarray(s, dtype=float))

    def peak(self, theta: Any) -> PeakLocation:
        """Return the maximum, at lag ``0`` with value ``alpha``."""
        values, _ = _batch(theta, 2)
        return PeakLocation(lag=0.0, value=float(values[0, 0]))

    def mass(self, theta: Any) -> np.ndarray:
        """``alpha / beta``, the branching ratio of the linear process."""
        values, flat = _batch(theta, 2)
        return _unbatch(values[:, 0] / values[:, 1], flat)


@dataclass(frozen=True)
class UnitExponentialKernel:
    r"""The kernel :math:`\kappa(s) = e^{-\beta s}`, with no amplitude of its own.

    :class:`ExponentialKernel` carries ``alpha``, which is right when the kernel
    is the whole excitation. In a multivariate model it is not: the excitation
    matrix already scales every ordered pair, so ``A[i, j] * alpha`` would be the
    quantity that matters and only the *product* would be identifiable.
    Multiplying every matrix entry by ``c`` and dividing ``alpha`` by ``c`` gives
    the same process, so the posterior would wander along that ridge forever
    while the effective sample size, the acceptance rate and the move size all
    looked healthy -- and the reported matrix would be arbitrary.

    So the amplitude lives in exactly one place. This family carries the shape
    only, and mass ``1 / beta`` means ``A / beta`` is the branching matrix
    directly. It is the family
    :class:`~hawkes_package.multivariate.MultivariateExponentialHawkes` is
    parameterised by.

    .. versionadded:: 1.0.0
    """

    monotone: bool = True

    @property
    def spec(self) -> ParameterSpec:
        """``(beta,)``, positive."""
        return ParameterSpec((Parameter("beta"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative lags."""
        values, _ = _batch(theta, 1)
        beta = float(values[0, 0])
        return lambda s: np.exp(-beta * np.asarray(s, dtype=float))

    def peak(self, theta: Any) -> PeakLocation:
        """Return the maximum, at lag ``0`` with value ``1``."""
        _batch(theta, 1)
        return PeakLocation(lag=0.0, value=1.0)

    def mass(self, theta: Any) -> np.ndarray:
        """``1 / beta``."""
        values, flat = _batch(theta, 1)
        return _unbatch(1.0 / values[:, 0], flat)


@dataclass(frozen=True)
class GammaKernel:
    r"""The kernel :math:`\kappa(s) = \alpha\, b^k s^{k-1} e^{-b s} / \Gamma(k)`.

    A gamma density scaled by ``alpha``, so ``alpha`` **is** the branching ratio
    -- the mass is ``alpha`` whatever the shape. With ``shape > 1`` it rises from
    zero, peaks at ``(shape - 1) / rate`` and decays: the bell shape
    :class:`~hawkes_package.bell_shape.BellShapeHawkes` exists for, and the one
    whose thinning bound is invalid if the peak is mislocated.

    The shape is bounded below by 1 rather than by 0. Below 1 the kernel
    diverges at lag zero, and an unbounded kernel has no thinning bound at all.
    """

    monotone: bool = False

    @property
    def spec(self) -> ParameterSpec:
        """``(alpha, shape, rate)``; ``shape`` above 1, the others above 0."""
        return ParameterSpec((Parameter("alpha"), Parameter("shape", lower=1.0), Parameter("rate")))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative lags."""
        values, _ = _batch(theta, 3)
        alpha, shape, rate = (float(v) for v in values[0])
        log_norm = math.log(alpha) + shape * math.log(rate) - math.lgamma(shape)

        def kernel(s: Any) -> np.ndarray:
            lags = np.asarray(s, dtype=float)
            positive = lags > 0.0
            # `log` is evaluated at 1 rather than at 0 off the support: masking
            # the *result* still evaluates the expression everywhere, so a bare
            # log(0) would emit a divide-by-zero warning -- which this suite
            # turns into a failure -- before `where` discarded it.
            safe = np.where(positive, lags, 1.0)
            return np.asarray(
                np.where(
                    positive,
                    np.exp(log_norm + (shape - 1.0) * np.log(safe) - rate * safe),
                    0.0,
                ),
                dtype=float,
            )

        return kernel

    def peak(self, theta: Any) -> PeakLocation:
        """Return the interior maximum, at ``(shape - 1) / rate``.

        Evaluated in logs. Written directly, ``rate ** shape`` overflows for a
        rate past about 300 at shape 4 while the kernel value itself is
        perfectly ordinary -- the intermediate blows up, not the answer.
        """
        values, _ = _batch(theta, 3)
        alpha, shape, rate = (float(v) for v in values[0])
        lag = (shape - 1.0) / rate
        if lag <= 0.0:
            # shape is bounded below by 1, so this is the boundary case only.
            return PeakLocation(lag=0.0, value=alpha * rate)
        log_value = (
            math.log(alpha)
            + shape * math.log(rate)
            + (shape - 1.0) * math.log(lag)
            - (shape - 1.0)
            - math.lgamma(shape)
        )
        return PeakLocation(lag=lag, value=math.exp(log_value))

    def mass(self, theta: Any) -> np.ndarray:
        """``alpha``: the kernel is a probability density times ``alpha``."""
        values, flat = _batch(theta, 3)
        return _unbatch(values[:, 0].copy(), flat)


@dataclass(frozen=True)
class OmoriUtsuKernel:
    r"""The power law :math:`\kappa(s) = \alpha\,(s + c)^{-p}`.

    The applied standard for aftershock decay, and the reason the package needed
    a second temporal shape: an exponential kernel says the excitation is gone
    after a few multiples of :math:`1/\beta`, and real catalogues decay far more
    slowly than that.

    ``alpha`` is the **amplitude**, in the convention
    :class:`ExponentialKernel` uses rather than :class:`GammaKernel`'s: the
    kernel starts at :math:`\alpha c^{-p}` and the branching ratio is the mass,
    :math:`\alpha c^{1-p} / (p - 1)`. Reading ``alpha`` as the branching ratio
    understates or overstates the excitation by whatever that factor is, and no
    fitted number says so.

    Monotone decreasing, so the maximum is at lag zero and the thinning bound is
    the value at the current time -- no peak search, and so none of the failure
    the bell-shaped path exists to guard.

    **The exponent is bounded above 1, and that bound is the stationarity
    condition rather than a convenience.** At :math:`p \le 1` the integral
    diverges: every event has infinitely many offspring in expectation while any
    finite window looks ordinary. `spec` refuses that region and `mass` returns
    ``inf`` there anyway, because
    :meth:`~hawkes_package.inference.models.ProcessModel.support` evaluates the
    branching callable on **every** row of a batch before the bounds filter it,
    so a plausible finite number returned here would admit a parameter the
    process cannot be simulated at.

    .. versionadded:: 1.0.0
    """

    monotone: bool = True

    #: Gauss-Legendre nodes per panel the compensator needs for this shape, read
    #: by the likelihoods when the caller does not name one. **Measured, not
    #: chosen.** The compensator of a power law is systematically *too small* at
    #: the package default of 8 -- and a compensator too small is a penalty on a
    #: high intensity that never gets applied, so the excitation comes back too
    #: large and the fit looks converged. Against the closed-form integral, worst
    #: relative error over a 300-event history at branching ratio 0.68:
    #:
    #: ===========  ========  ========  ========  ========
    #: ``c``        ``P=8``   ``P=12``  ``P=16``  ``P=20``
    #: ===========  ========  ========  ========  ========
    #: 0.5          1.5e-04   2.4e-06   3.4e-08   4.5e-10
    #: 0.2          2.6e-03   1.7e-04   1.0e-05   5.6e-07
    #: 0.05         3.0e-02   7.6e-03   1.7e-03   3.5e-04
    #: ===========  ========  ========  ========  ========
    #:
    #: (worst over the compensator evaluated *at intermediate times*, which is
    #: what time-rescaling residuals read; the whole-window integral is milder,
    #: because the exactly-integrated background dilutes it)
    #:
    #: 16 is where the middle row stops mattering, at twice the integrand
    #: evaluations. An exponential kernel is exact to 7e-13 at 8, which is why
    #: this is a per-family number rather than a new default for everyone.
    #:
    #: It does not rescue the bottom row: a core four times narrower than the
    #: median inter-event gap is a *panel* problem, not an order problem, and
    #: there the order-``P``-versus-``2P`` check fires and says so.
    quadrature_order: int = 16

    @property
    def spec(self) -> ParameterSpec:
        """``(alpha, c, p)``: amplitude and offset positive, exponent above 1."""
        return ParameterSpec((Parameter("alpha"), Parameter("c"), Parameter("p", lower=1.0)))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative lags."""
        values, _ = _batch(theta, 3)
        alpha, c, p = (float(v) for v in values[0])

        def kernel(s: Any) -> np.ndarray:
            lags = np.asarray(s, dtype=float)
            # `c > 0`, so the base is bounded away from zero for every
            # non-negative lag and no guard against `0 ** -p` is needed.
            return np.asarray(alpha * (lags + c) ** (-p), dtype=float)

        return kernel

    def peak(self, theta: Any) -> PeakLocation:
        """Return the maximum, at lag ``0`` with value ``alpha * c**-p``."""
        values, _ = _batch(theta, 3)
        alpha, c, p = (float(v) for v in values[0])
        return PeakLocation(lag=0.0, value=alpha * c**-p)

    def mass(self, theta: Any) -> np.ndarray:
        r""":math:`\alpha c^{1-p}/(p-1)`, and ``inf`` where the integral diverges."""
        values, flat = _batch(theta, 3)
        alpha, c, p = values[:, 0], values[:, 1], values[:, 2]
        out = np.full(values.shape[0], np.inf, dtype=float)
        # Computed only where it converges: `np.where` would evaluate the
        # division at `p == 1` as well and warn, which this suite turns into a
        # failure -- the same shape as the marked model's divergent expectation.
        usable = p > 1.0
        out[usable] = alpha[usable] * c[usable] ** (1.0 - p[usable]) / (p[usable] - 1.0)
        return _unbatch(out, flat)


# ---------------------------------------------------------------------------
# Spatial kernels
# ---------------------------------------------------------------------------


def _sphere_surface(ndim: int) -> float:
    """Surface measure of the unit sphere in `ndim` dimensions.

    ``2`` on the line, ``2*pi`` in the plane. The factor every isotropic
    normalisation on this page shares, written once.
    """
    return float(2.0 * math.pi ** (0.5 * ndim) / math.gamma(0.5 * ndim))


@dataclass(frozen=True)
class GaussianSpatial:
    r"""Isotropic Gaussian :math:`(2\pi\sigma^2)^{-n/2} e^{-d^2 / 2\sigma^2}`.

    Normalised to unit mass on :math:`\mathbb{R}^n`, so on a *compact* domain its
    mass is at most one, with equality in the limit of a kernel narrow compared
    to the domain. That inequality is the whole reason for the normalisation:
    it makes the temporal kernel's mass a **sufficient** bound on the branching
    ratio of the separable spatio-temporal process, with no quadrature and no
    dependence on which surface the process lives on.

    Strictly positive everywhere, which is what the cached likelihood backend
    needs: the separability identity it rests on holds only where the pre-floor
    integrand is non-negative at every node.

    Parameters
    ----------
    ndim : int
        Dimension of the domain, which fixes the normalising constant.
    """

    ndim: int

    def __post_init__(self) -> None:
        """Refuse a dimension the normalising constant is not written for."""
        if self.ndim < 1:
            raise ValueError(f"ndim must be at least 1, got {self.ndim}")

    @property
    def spec(self) -> ParameterSpec:
        """``(sigma,)``, positive."""
        return ParameterSpec((Parameter("sigma"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative distances."""
        values, _ = _batch(theta, 1)
        sigma = float(values[0, 0])
        norm = (2.0 * math.pi * sigma**2) ** (-0.5 * self.ndim)

        def kernel(d: Any) -> np.ndarray:
            distances = np.asarray(d, dtype=float)
            return np.asarray(norm * np.exp(-0.5 * (distances / sigma) ** 2), dtype=float)

        return kernel

    def mass(self, theta: Any) -> np.ndarray:
        """``1.0``: unit mass on the model space, and at most that on a domain."""
        values, flat = _batch(theta, 1)
        return _unbatch(np.ones(values.shape[0], dtype=float), flat)

    def min_scale(self, theta: Any) -> np.ndarray:
        """``sigma``, the length the kernel varies on."""
        values, flat = _batch(theta, 1)
        return _unbatch(values[:, 0].copy(), flat)


@dataclass(frozen=True)
class ParetoSpatial:
    r"""Isotropic power law :math:`f(r) \propto (r^2 + d^2)^{-q}`, unit mass.

    The Gaussian tail is too light for most real data: it puts almost nothing
    past three standard deviations, so a fit to a catalogue with distant
    offspring either widens `sigma` until the near field is wrong or attributes
    the far field to the background. This is the shape that does not have to
    choose.

    Normalised to unit mass on :math:`\mathbb{R}^n`, like
    :class:`GaussianSpatial` and for the same reason: it makes the *temporal*
    kernel's mass a sufficient bound on the branching ratio, with no quadrature
    and no dependence on which surface the process lives on. The normaliser is
    closed form,

    .. math::

        \int_{\mathbb{R}^n} (r^2 + d^2)^{-q}\,\mathrm{d}x
            = S_{n-1}\, d^{\,n-2q}\,
              \frac{\Gamma(n/2)\,\Gamma(q - n/2)}{2\,\Gamma(q)},

    which needs :math:`\Gamma` from the standard library and **not**
    ``scipy.special`` -- runtime SciPy is held to one call site on purpose.

    **The exponent is bounded by the dimension**, ``q > ndim / 2``, because that
    is exactly where the integral above converges. In the plane a `q` at or below
    1 is a kernel with infinite mass: the branching ratio it implies is infinite
    while every simulated catalogue looks ordinary, so `spec` refuses it rather
    than leaving it to be discovered as an explosion at simulation time.

    Parameters
    ----------
    ndim : int
        Dimension of the domain, which fixes both the normaliser and the lower
        bound on ``q``.

    .. versionadded:: 1.0.0
    """

    ndim: int

    def __post_init__(self) -> None:
        """Refuse a dimension the normalising constant is not written for."""
        if self.ndim < 1:
            raise ValueError(f"ndim must be at least 1, got {self.ndim}")

    @property
    def spec(self) -> ParameterSpec:
        """``(d, q)``: the core radius positive, the exponent above ``ndim/2``."""
        return ParameterSpec((Parameter("d"), Parameter("q", lower=0.5 * self.ndim)))

    def _normaliser(self, d: float, q: float) -> float:
        """Return the integral of the unnormalised kernel over the model space."""
        return float(
            _sphere_surface(self.ndim)
            * d ** (self.ndim - 2.0 * q)
            * math.gamma(0.5 * self.ndim)
            * math.gamma(q - 0.5 * self.ndim)
            / (2.0 * math.gamma(q))
        )

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative distances."""
        values, _ = _batch(theta, 2)
        d, q = float(values[0, 0]), float(values[0, 1])
        norm = 1.0 / self._normaliser(d, q)

        def kernel(r: Any) -> np.ndarray:
            distances = np.asarray(r, dtype=float)
            return np.asarray(norm * (distances**2 + d**2) ** (-q), dtype=float)

        return kernel

    def mass(self, theta: Any) -> np.ndarray:
        """``1.0`` where the integral converges, ``inf`` where it does not."""
        values, flat = _batch(theta, 2)
        out = np.full(values.shape[0], np.inf, dtype=float)
        out[values[:, 1] > 0.5 * self.ndim] = 1.0
        return _unbatch(out, flat)

    def min_scale(self, theta: Any) -> np.ndarray:
        """``d``: inside the core the kernel is flat, and outside it is a power law.

        The length a quadrature rule has to resolve, which is what this feeds --
        a heavy tail is easy to integrate and a narrow core is not.
        """
        values, flat = _batch(theta, 2)
        return _unbatch(values[:, 0].copy(), flat)


@dataclass(frozen=True)
class CompactSpatial:
    r"""Isotropic :math:`f(r) \propto 1 - (r/R)^2` on :math:`r \le R`, zero past it.

    The one family here whose support ends. That is worth having for its own
    sake -- an excitation that is genuinely local is a modelling statement, not
    an approximation of one -- and it is what makes neighbour skipping **exact**:
    dropping a pair separated by more than `R` is not truncating the sum, it is
    declining to add zero.

    Unit mass on :math:`\mathbb{R}^n`, with normaliser
    :math:`S_{n-1} R^n \cdot 2 / (n(n+2))`.

    Two things to know before using it.

    It is **not** strictly positive, so it cannot be used with the cached
    spatio-temporal backend's separability shortcut wherever the pre-floor
    integrand would go negative -- that backend raises rather than degrading, and
    a kernel that is zero over most of a large domain is the most likely thing to
    meet it.

    And it puts a floating-point comparison inside the intensity sum. On the
    spatio-temporal path that is already accepted and exact-value reproducibility
    is not asserted; a *temporal* compact kernel would be a different question,
    which is why there is not one here.

    Parameters
    ----------
    ndim : int
        Dimension of the domain, which fixes the normalising constant.

    .. versionadded:: 1.0.0
    """

    ndim: int

    def __post_init__(self) -> None:
        """Refuse a dimension the normalising constant is not written for."""
        if self.ndim < 1:
            raise ValueError(f"ndim must be at least 1, got {self.ndim}")

    @property
    def spec(self) -> ParameterSpec:
        """``(radius,)``, positive. Past it the kernel is exactly zero."""
        return ParameterSpec((Parameter("radius"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the kernel as a vectorized callable on non-negative distances."""
        values, _ = _batch(theta, 1)
        radius = float(values[0, 0])
        norm = 1.0 / (
            _sphere_surface(self.ndim) * radius**self.ndim * 2.0 / (self.ndim * (self.ndim + 2.0))
        )

        def kernel(r: Any) -> np.ndarray:
            distances = np.asarray(r, dtype=float)
            inside = distances < radius
            # `np.where` on the *result* rather than a masked expression: the
            # quadratic is finite everywhere, so both branches are safe to
            # evaluate and nothing warns.
            return np.asarray(
                np.where(inside, norm * (1.0 - (distances / radius) ** 2), 0.0), dtype=float
            )

        return kernel

    def cutoff(self, theta: Any) -> np.ndarray:
        """Return the distance past which the kernel is **exactly** zero.

        The presence of this method is the contract: a family that declares a
        cutoff may have pairs beyond it skipped, and one that does not may not.
        """
        values, flat = _batch(theta, 1)
        return _unbatch(values[:, 0].copy(), flat)

    def mass(self, theta: Any) -> np.ndarray:
        """``1.0``: unit mass on the model space, and at most that on a domain."""
        values, flat = _batch(theta, 1)
        return _unbatch(np.ones(values.shape[0], dtype=float), flat)

    def min_scale(self, theta: Any) -> np.ndarray:
        """``radius``, which is both where the kernel ends and the length it varies on."""
        values, flat = _batch(theta, 1)
        return _unbatch(values[:, 0].copy(), flat)


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantBase:
    """A background intensity that does not vary over the domain.

    ``mu`` is the intensity **per unit measure**, so the background event rate
    is ``mu * domain.volume``. Reading it as a rate instead scales every fitted
    background by the area of the surface, which on a unit circle is a factor of
    6.28 and on a genus-2 surface a factor of 25.
    """

    @property
    def spec(self) -> ParameterSpec:
        """``(mu,)``, positive."""
        return ParameterSpec((Parameter("mu"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the background as a callable on a domain point."""
        values, _ = _batch(theta, 1)
        mu = float(values[0, 0])
        return lambda x: mu  # noqa: ARG005 - constant in position, by definition

    def at(self, theta: Any, points: Any) -> np.ndarray:
        """Evaluate at every row of `points`, shape ``(m, ndim)`` -> ``(m,)``."""
        values, _ = _batch(theta, 1)
        nodes = np.asarray(points, dtype=float)
        return np.full(nodes.shape[0], float(values[0, 0]), dtype=float)


#: The largest exponent a double survives. `exp` of anything past this is `inf`,
#: and numpy warns -- which `filterwarnings = ["error"]` turns into a failure in
#: the middle of an otherwise healthy fit. Clipping here is not a modelling
#: choice: a background of 1e308 per unit measure gives a compensator no finite
#: log-likelihood can survive, so the particle is rejected either way. It is the
#: difference between rejecting it and raising.
_MAX_EXPONENT = 709.0


@dataclass(frozen=True)
class LogLinearBase:
    r"""A background that varies over the domain, log-linear in covariates.

    .. math::

        \mu(x) = \exp\!\left(\beta_0 + \sum_k \beta_k\, z_k(x)\right),

    **per unit measure**, the same convention :class:`ConstantBase` states: the
    background event rate is the integral of this over the domain, not the value
    times anything. Two backgrounds are comparable only if both mean that.

    The link is logarithmic rather than affine so the value is **positive
    everywhere by construction**. That is not cosmetic: the cached
    spatio-temporal backend's separability identity holds only where the
    pre-floor integrand is non-negative at every quadrature node, and it raises
    rather than degrading when it is not. An affine background with a covariate
    coefficient of the wrong sign trips that on a fit that is otherwise going
    fine; this one cannot.

    With no covariates at all it is a constant background in an unusual
    parameterisation, which is a useful thing to have: it is what the tests
    compare against :class:`ConstantBase` to show the plumbing carries a varying
    background without changing any number.

    Parameters
    ----------
    covariates : sequence of callable
        Each takes an ``(m, ndim)`` array of positions and returns ``(m,)``
        values. **Vectorized**, because these are evaluated at every quadrature
        node for every particle of every rejuvenation move. Non-finite values
        are refused where they appear, naming the covariate: a ``nan`` covariate
        otherwise turns the whole log-likelihood into ``nan``, which the sampler
        reads as an invalid particle and quietly resamples away.
    names : sequence of str, optional
        Names for the coefficients, used in the parameter names and so in every
        diagnostic. Defaults to positional ``b_0``, ``b_1``, ...

    Notes
    -----
    Nothing is centred or standardised for you. A covariate with a large mean
    trades off against ``log_mu0`` almost exactly, and a pair of parameters that
    trade off exactly is a ridge the posterior never leaves -- the same failure
    :class:`SoftPlusNonlinearity` avoids by not fitting its scale. Centre the
    covariate if the marginal for ``log_mu0`` comes back wide and correlated.

    Examples
    --------
    >>> import numpy as np
    >>> east = lambda points: np.asarray(points, dtype=float)[:, 0]
    >>> base = LogLinearBase((east,), names=("east",))
    >>> base.spec.names
    ('log_mu0', 'b_east')
    >>> base.at(np.array([0.0, 1.0]), np.array([[0.0], [1.0]]))
    array([1.        , 2.71828183])

    .. versionadded:: 1.0.0
    """

    covariates: tuple[Callable[[Any], Any], ...] = ()
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Normalise both sequences to tuples and check they describe each other."""
        fields = tuple(self.covariates)
        object.__setattr__(self, "covariates", fields)
        if self.names is None:
            return
        labels = tuple(str(name) for name in self.names)
        if len(labels) != len(fields):
            raise ValueError(
                f"got {len(labels)} names for {len(fields)} covariates; a name per "
                "covariate or none at all, since the names are what a diagnostic "
                "reports the coefficients under"
            )
        object.__setattr__(self, "names", labels)

    @property
    def spec(self) -> ParameterSpec:
        """``(log_mu0, b_...)``, every one of them on the **whole real line**.

        The first parameters in this package that are not positive rates. A
        coefficient's sign is the direction of the effect, and ``log_mu0`` is a
        log, so both are unbounded and
        :class:`~hawkes_package.inference.parameters.ParameterSpec` transforms
        them with the identity.
        """
        labels = (
            tuple(f"b_{i}" for i in range(len(self.covariates)))
            if self.names is None
            else tuple(f"b_{name}" for name in self.names)
        )
        return ParameterSpec(
            (
                Parameter("log_mu0", lower=-math.inf, upper=math.inf),
                *(Parameter(label, lower=-math.inf, upper=math.inf) for label in labels),
            )
        )

    def _design(self, points: Any) -> np.ndarray:
        """Evaluate every covariate at `points`, shape ``(m, ndim)`` -> ``(m, k)``."""
        nodes = np.atleast_2d(np.asarray(points, dtype=float))
        rows = nodes.shape[0]
        if not self.covariates:
            return np.empty((rows, 0), dtype=float)

        columns = []
        for index, covariate in enumerate(self.covariates):
            column = np.asarray(covariate(nodes), dtype=float).reshape(-1)
            if column.size != rows:
                label = index if self.names is None else self.names[index]
                raise ValueError(
                    f"covariate {label!r} returned {column.size} values for {rows} "
                    "positions; a covariate takes an (m, ndim) array and returns one "
                    "value per row, vectorized"
                )
            if not np.all(np.isfinite(column)):
                label = index if self.names is None else self.names[index]
                raise ValueError(
                    f"covariate {label!r} returned a non-finite value. It would turn "
                    "the whole log-likelihood into nan, which the sampler reads as an "
                    "invalid particle and resamples away without ever saying why."
                )
            columns.append(column)
        return np.stack(columns, axis=1)

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the background at one parameter vector, as a function of position."""
        values, _ = _batch(theta, len(self.covariates) + 1)
        vector = np.asarray(values[0], dtype=float)

        def background(x: Any) -> float:
            return float(self.at(vector, np.atleast_2d(np.asarray(x, dtype=float)))[0])

        return background

    def at(self, theta: Any, points: Any) -> np.ndarray:
        """Evaluate at every row of `points`, shape ``(m, ndim)`` -> ``(m,)``."""
        values, _ = _batch(theta, len(self.covariates) + 1)
        vector = np.asarray(values[0], dtype=float)
        design = self._design(points)
        linear = vector[0] + design @ vector[1:]
        # See `_MAX_EXPONENT`: clipped where a double stops holding the answer,
        # not where the model stops being sensible.
        return np.asarray(np.exp(np.minimum(linear, _MAX_EXPONENT)), dtype=float)


# ---------------------------------------------------------------------------
# Nonlinearities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearNonlinearity:
    r"""The identity plus a background: :math:`\varphi(x) = \mu + x`.

    Makes a :class:`~hawkes_package.monotone.MonotoneKernelHawkes` or
    :class:`~hawkes_package.bell_shape.BellShapeHawkes` a *linear* Hawkes
    process with background ``mu`` and a general kernel, which is what makes it
    comparable with :class:`~hawkes_package.exponential.ExponentialHawkes`.
    """

    lipschitz: float = 1.0

    @property
    def spec(self) -> ParameterSpec:
        """``(mu,)``, positive."""
        return ParameterSpec((Parameter("mu"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the nonlinearity at one parameter vector."""
        values, _ = _batch(theta, 1)
        mu = float(values[0, 0])
        return lambda x: mu + np.asarray(x, dtype=float)


@dataclass(frozen=True)
class SoftPlusNonlinearity:
    r"""A saturating nonlinearity, :math:`\varphi(x) = \mu + c\,\log(1 + e^{x/c})`.

    Increasing and 1-Lipschitz for every ``c > 0``, so the same
    ``mass(kernel) < 1`` condition bounds it -- by Bremaud and Massoulie's
    criterion, which asks for the Lipschitz constant times the kernel mass, not
    the branching ratio of a linear process that does not exist here.

    Bounded below by ``mu``, so the intensity stays positive and the thinning
    loop's non-positive-bound guard is unreachable.

    Parameters
    ----------
    scale : float
        ``c``, the excitation at which the response starts to saturate. Not
        fitted: it trades off against the kernel amplitude almost exactly, and
        a pair of parameters that trade off exactly is a ridge the posterior
        never leaves.
    """

    scale: float = 1.0
    lipschitz: float = 1.0

    def __post_init__(self) -> None:
        """Refuse a non-positive saturation scale."""
        if not self.scale > 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    @property
    def spec(self) -> ParameterSpec:
        """``(mu,)``, positive."""
        return ParameterSpec((Parameter("mu"),))

    def build(self, theta: Any) -> Callable[[Any], Any]:
        """Return the nonlinearity at one parameter vector."""
        values, _ = _batch(theta, 1)
        mu = float(values[0, 0])
        scale = self.scale

        def phi(x: Any) -> np.ndarray:
            # logaddexp, not log1p(exp(.)): the excitation sum reaches a few
            # hundred on a burst, where exp overflows and log1p sees inf.
            return np.asarray(
                mu + scale * np.logaddexp(0.0, np.asarray(x, dtype=float) / scale),
                dtype=float,
            )

        return phi


# ---------------------------------------------------------------------------
# Multivariate structure
# ---------------------------------------------------------------------------


def spectral_radii(matrices: Any) -> np.ndarray:
    """Largest absolute eigenvalue of each matrix in a batch.

    Parameters
    ----------
    matrices : array_like of shape (n, d, d)
        One branching matrix per particle.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``. A row carrying any non-finite entry returns ``inf``,
        which is what :meth:`~hawkes_package.inference.models.ProcessModel.support`
        already filters on.

    Notes
    -----
    The non-finite rows are masked out and written back rather than passed to
    the eigensolver, and that is not defensive tidiness. ``ProcessModel.support``
    evaluates ``branching`` on **every** row before ``spec.contains`` filters
    any, so rows carrying ``inf`` or ``nan`` do reach here -- and
    :func:`numpy.linalg.eigvals` *raises* ``LinAlgError`` on those rather than
    returning ``nan``. That would turn a particle the sampler was about to
    reject anyway into a crashed fit, on a proposal that is drawn hundreds of
    times per move.

    .. versionadded:: 1.0.0
    """
    batch = np.asarray(matrices, dtype=float)
    if batch.ndim != 3 or batch.shape[1] != batch.shape[2]:
        raise ValueError(f"expected a batch of square matrices, got shape {batch.shape}")

    radii = np.full(batch.shape[0], np.inf, dtype=float)
    usable = np.all(np.isfinite(batch), axis=(1, 2))
    if np.any(usable):
        eigenvalues = np.linalg.eigvals(batch[usable])
        radii[usable] = np.max(np.abs(eigenvalues), axis=1)
    return radii


@dataclass(frozen=True)
class ExcitationMatrix:
    """The ``(d, d)`` matrix of excitation scales in a multivariate model.

    ``A[i, j]`` scales the excitation type *j* exerts on type *i*, against one
    shared kernel shape. Every entry is positive, so the parameters are
    unconstrained-transformable coordinatewise like every other rate here, and
    the thinning bound's per-term supremum stays valid.

    Parameters are named ``a_i_j`` and laid out **row-major**, matching
    :func:`numpy.reshape`, so ``theta.reshape(d, d)`` is the matrix.

    .. versionadded:: 1.0.0
    """

    n_types: int

    def __post_init__(self) -> None:
        """Refuse a type count that cannot index a matrix."""
        count = int(self.n_types)
        if count != self.n_types or count < 1:
            raise ValueError(f"n_types must be a positive whole number, got {self.n_types!r}")
        object.__setattr__(self, "n_types", count)

    @property
    def size(self) -> int:
        """Number of coordinates the matrix consumes."""
        return self.n_types * self.n_types

    @property
    def spec(self) -> ParameterSpec:
        """``(a_0_0, a_0_1, ..., a_{d-1}_{d-1})``, all positive, row-major."""
        return ParameterSpec(
            tuple(Parameter(f"a_{i}_{j}") for i in range(self.n_types) for j in range(self.n_types))
        )

    def matrices(self, theta: Any) -> np.ndarray:
        """Reshape a batch of parameter rows into ``(n, d, d)`` matrices."""
        values, _ = _batch(theta, self.size)
        return values.reshape(-1, self.n_types, self.n_types)

    def matrix(self, theta: Any) -> np.ndarray:
        """Return the ``(d, d)`` matrix at one parameter vector."""
        return np.asarray(self.matrices(theta)[0], dtype=float)

    def spectral_radius(self, theta: Any, mass: Any) -> np.ndarray:
        """Spectral radius of the branching matrix ``A * mass``, batched.

        `mass` is the shared kernel's integral over all lags, one per row, so
        ``A[i, j] * mass`` is the expected number of type-*i* offspring a
        type-*j* event produces directly. Below one the process is stationary.
        At one type this is exactly ``alpha / beta``.
        """
        values, flat = _batch(theta, self.size)
        weights = np.atleast_1d(np.asarray(mass, dtype=float))
        branching = values.reshape(-1, self.n_types, self.n_types) * weights[:, None, None]
        return _unbatch(spectral_radii(branching), flat)


@dataclass(frozen=True)
class MultivariateBase:
    """A constant background rate per event type.

    The multivariate counterpart of :class:`ConstantBase`, and purely temporal:
    there is no domain to be per unit measure of, so ``mu_i`` is the background
    event rate of type *i* directly.

    .. versionadded:: 1.0.0
    """

    n_types: int

    def __post_init__(self) -> None:
        """Refuse a type count that cannot index a background vector."""
        count = int(self.n_types)
        if count != self.n_types or count < 1:
            raise ValueError(f"n_types must be a positive whole number, got {self.n_types!r}")
        object.__setattr__(self, "n_types", count)

    @property
    def spec(self) -> ParameterSpec:
        """``(mu_0, ..., mu_{d-1})``, all positive."""
        return ParameterSpec(tuple(Parameter(f"mu_{i}") for i in range(self.n_types)))

    def rates(self, theta: Any) -> np.ndarray:
        """Return the background vector at one parameter vector, shape ``(d,)``."""
        values, _ = _batch(theta, self.n_types)
        return np.asarray(values[0], dtype=float)


@dataclass(frozen=True)
class PeriodicBase:
    r"""A background whose rate repeats: :math:`\mu(t, x) = m(x)\,s(t)`.

    Wraps another :class:`BaseFamily` -- the *shape* -- and adds a truncated
    Fourier schedule on a declared period. Its coordinates are the shape's,
    followed by ``(a_1, b_1, ..., a_K, b_K)`` on the whole real line.

    **Separable on purpose.** The compensator then factorises into the spatial
    integral the package already computes and the schedule's own closed form,
    instead of a two-dimensional quadrature. A background whose spatial *shape*
    changes with the hour is a joint model and is out of scope.

    **Zero coefficients are the constant background**, exactly: the schedule is
    then 1 at every time and its integral is the interval length, so a fit with
    the amplitudes pinned at zero is the fit that predates this class. That is
    what makes "is there a cycle at all" a question about a parameter rather
    than a comparison of two models.

    The confounding this exists to resolve is worth naming: a daily cycle and
    self-excitation both produce clustering, and a fit that finds plausible
    amounts of each may have split them anywhere along a ridge. The evidence
    that they are separable at all is a recovery test with each effect switched
    off in turn, not the model's own report.

    Parameters
    ----------
    shape : BaseFamily
        The spatial background, per unit measure.
    n_harmonics : int
        How many ``(cos, sin)`` pairs. One is a single daily cycle; more buy
        shape at the cost of a rougher integrand, and the compensator's
        resolution check is what says when that has gone too far.
    period : float
        The cycle length, **declared and not fitted**. A fitted period is
        multimodal -- every integer fraction of the truth is a local maximum --
        and applied users have a period they know.

    .. versionadded:: 1.0.0
    """

    shape: BaseFamily
    n_harmonics: int
    period: float

    def __post_init__(self) -> None:
        """Refuse a period or harmonic count the schedule cannot be built from."""
        if int(self.n_harmonics) != self.n_harmonics or self.n_harmonics < 0:
            raise ValueError(
                f"n_harmonics must be a non-negative whole number, got {self.n_harmonics!r}"
            )
        if not float(self.period) > 0:
            raise ValueError(f"period must be positive, got {self.period!r}")
        object.__setattr__(self, "n_harmonics", int(self.n_harmonics))
        object.__setattr__(self, "period", float(self.period))

    @property
    def spec(self) -> ParameterSpec:
        """The shape's coordinates, then ``a_k`` and ``b_k`` on the whole line."""
        if self.n_harmonics == 0:
            # Returned before the harmonics are built rather than after: an
            # empty `ParameterSpec` is refused at construction, so building one
            # to discard it raises. Zero harmonics is a supported configuration
            # -- it is the constant background, reached through this class.
            return self.shape.spec
        return self.shape.spec.concat(
            ParameterSpec(
                tuple(
                    Parameter(name, lower=-math.inf, upper=math.inf)
                    for k in range(1, self.n_harmonics + 1)
                    for name in (f"cos_{k}", f"sin_{k}")
                )
            )
        )

    def _split(self, theta: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a parameter vector into the shape's part and the two coefficient sets."""
        values = np.asarray(theta, dtype=float).reshape(-1)
        cut = len(self.shape.spec)
        return values[:cut], values[cut::2], values[cut + 1 :: 2]

    def schedule(self, theta: Any) -> PeriodicSchedule:
        """Return the :class:`~hawkes_package.periodic.PeriodicSchedule` at `theta`."""
        _, cosine, sine = self._split(theta)
        return PeriodicSchedule(cosine, sine, period=self.period)

    def build(self, theta: Any) -> Callable[..., Any]:
        """Return the background as the simulator's time-varying callable."""
        head, _, _ = self._split(theta)
        return PeriodicBackground(self.shape.build(head), self.schedule(theta))

    def at(self, theta: Any, points: Any) -> np.ndarray:
        """Evaluate the **spatial shape** at `points`, without the schedule.

        Deliberately the shape alone, which is what lets the cached likelihood
        backend keep computing one spatial integral and multiply it by the
        schedule's own integral. :meth:`time_factor` is the other half.
        """
        head, _, _ = self._split(theta)
        return np.asarray(self.shape.at(head, points), dtype=float)

    def time_factor(self, theta: Any, times: Any) -> np.ndarray:
        """Evaluate the schedule at `times`."""
        return np.asarray(self.schedule(theta)(np.asarray(times, dtype=float)), dtype=float)

    def time_integral(self, theta: Any, start: float, end: float) -> float:
        """Integrate the schedule over ``[start, end]``, in closed form.

        Exact where the schedule stays non-negative, which
        :meth:`~hawkes_package.periodic.PeriodicSchedule.is_non_negative`
        certifies. Below zero the schedule is floored and this over-states the
        integral, so a compensator computed from it would be too *large* -- the
        safe direction, and the opposite of the failure this package guards
        against, but still worth knowing about rather than discovering.
        """
        return self.schedule(theta).integral(float(start), float(end))
