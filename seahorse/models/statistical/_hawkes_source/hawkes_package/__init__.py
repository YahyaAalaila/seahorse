"""Simulation of self-exciting (Hawkes) point processes.

Temporal processes
------------------
:class:`ExponentialHawkes`, :class:`MonotoneKernelHawkes`,
:class:`BellShapeHawkes`

Spatio-temporal processes
-------------------------
:class:`SpatioTemporalHawkesProcess`

Spatial domains
---------------
:class:`SpatialDomain`, :class:`Circle`, :class:`Torus2D`, :class:`Sphere` and
:class:`FundamentalDomain` — a convex geodesic polygon with side pairings, which
between them reach **every closed surface**: the sphere and the projective plane,
the torus and the Klein bottle, and every orientable or non-orientable surface of
higher genus. Plus :func:`make_periodic` for summing a kernel over a domain's
image points.

Inference
---------
:mod:`hawkes_package.inference` fits the parameters of any of these to observed
events, in blocks as the data arrives, by sequential Monte Carlo with MCMC
rejuvenation. Its likelihood is computed from the same intensity hooks the
simulator thins against.

.. versionadded:: 0.5.0

Notes
-----
Every process takes ``rng=``, accepting ``None``, an ``int`` seed or an existing
:class:`numpy.random.Generator`. As of 0.2.0 :func:`numpy.random.seed` no longer
influences simulations.
"""

from . import inference
from .base import (
    HawkesProcess,
    MarkedTemporalHawkesProcess,
    MultivariateTemporalHawkesProcess,
    TemporalHawkesProcess,
)
from .bell_shape import BellShapeHawkes
from .exponential import ExponentialHawkes
from .marked import ExponentialMarkedHawkes, MarkedHawkes
from .mcmc import mcmc_sampler
from .monotone import MonotoneKernelHawkes
from .multivariate import MultivariateExponentialHawkes, MultivariateHawkes
from .periodic import PeriodicBackground, PeriodicSchedule
from .spatio_temporal import (
    Circle,
    FundamentalDomain,
    Polygon,
    Rectangle,
    SpatialDomain,
    SpatioTemporalHawkesProcess,
    Sphere,
    Torus2D,
    make_periodic,
)

__version__ = "1.0.0"

__all__ = [
    "BellShapeHawkes",
    "Circle",
    "ExponentialHawkes",
    "ExponentialMarkedHawkes",
    "FundamentalDomain",
    "HawkesProcess",
    "MarkedHawkes",
    "MarkedTemporalHawkesProcess",
    "MonotoneKernelHawkes",
    "MultivariateExponentialHawkes",
    "MultivariateHawkes",
    "MultivariateTemporalHawkesProcess",
    "PeriodicBackground",
    "PeriodicSchedule",
    "Polygon",
    "Rectangle",
    "SpatialDomain",
    "SpatioTemporalHawkesProcess",
    "Sphere",
    "TemporalHawkesProcess",
    "Torus2D",
    "__version__",
    "inference",
    "make_periodic",
    "mcmc_sampler",
]
