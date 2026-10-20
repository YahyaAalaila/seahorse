"""Deterministic quadrature over a rectangular spatial domain.

Ogata's algorithm requires ``M >= lambda`` *pathwise*. Estimating the spatial
integral by Monte Carlo breaks that in two ways at once: the bound becomes an
unbiased estimate rather than an upper bound, so it sits below the truth about
half the time; and the loop drew a fresh point set for the bound and another for
the acceptance test, so the ratio of two independent estimates exceeded 1
whenever noise favoured the second. Measured on ``Torus2D``:
``P(lambda_hat > M_hat) = 0.437``.

A fixed rule removes both. Because the bound integrand dominates the true
integrand *pointwise* and both are evaluated on the same nodes with strictly
positive weights, ``sum(w_i f_bound(x_i)) >= sum(w_i f_true(x_i))`` holds with no
tolerance at all.

Composite Gauss-Legendre is also markedly cheaper than the adaptive ``quad`` it
replaces on this integrand -- roughly 8x on the legacy kernel, where ``quad``
exhausts its subdivision limit on the ``max(0, .)`` kinks.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["TensorQuadrature", "build", "default_nodes_per_axis", "restrict"]

#: Nodes per axis, by dimension. The cost is ``nodes ** ndim``, so the count
#: falls as the dimension rises.
_DEFAULT_NODES = {1: 256, 2: 32}
_FALLBACK_NODES = 12

#: Points per Gauss-Legendre panel. Low order with many panels handles the
#: kinks from ``max(0, .)`` and indicator kernels better than a single
#: high-order rule.
_PANEL_ORDER = 4


def default_nodes_per_axis(ndim: int) -> int:
    """Default node count per axis for a domain of dimension `ndim`."""
    return _DEFAULT_NODES.get(ndim, _FALLBACK_NODES)


@dataclass(frozen=True)
class TensorQuadrature:
    """A fixed tensor-product quadrature rule over a box.

    Attributes
    ----------
    nodes : numpy.ndarray
        Shape ``(m, ndim)``; each row is a point, so it can be handed straight
        to a callable expecting a shape-``(ndim,)`` coordinate.
    weights : numpy.ndarray
        Shape ``(m,)``, all strictly positive -- which is what makes a
        pointwise-dominating integrand integrate to a dominating value.
    """

    nodes: np.ndarray
    weights: np.ndarray

    def integrate(self, fn: Callable[[np.ndarray], float]) -> float:
        """Integrate `fn` over the box the rule was built for."""
        return float(np.dot(self.weights, [fn(node) for node in self.nodes]))


def _axis_rule(lo: float, hi: float, nodes_per_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes and weights on ``[lo, hi]``."""
    panels = max(1, int(nodes_per_axis) // _PANEL_ORDER)
    base_nodes, base_weights = np.polynomial.legendre.leggauss(_PANEL_ORDER)
    edges = np.linspace(lo, hi, panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * base_nodes[None, :]).ravel()
    weights = (half[:, None] * base_weights[None, :]).ravel()
    return nodes, weights


def build(bounds: Any, nodes_per_axis: int) -> TensorQuadrature:
    """Build a tensor-product rule over `bounds`.

    Parameters
    ----------
    bounds : array_like of shape (ndim, 2)
        The box to integrate over.
    nodes_per_axis : int
        Approximate node count per axis; rounded down to a whole number of
        4-point panels.
    """
    bounds = np.asarray(bounds, dtype=float)
    per_axis = [_axis_rule(lo, hi, nodes_per_axis) for lo, hi in bounds]

    grids = np.meshgrid(*[nodes for nodes, _ in per_axis], indexing="ij")
    weight_grids = np.meshgrid(*[weights for _, weights in per_axis], indexing="ij")

    nodes = np.stack([g.ravel() for g in grids], axis=-1)
    weights = np.prod([w.ravel() for w in weight_grids], axis=0)
    return TensorQuadrature(nodes=nodes, weights=weights)


def restrict(
    rule: TensorQuadrature,
    contains: Callable[[np.ndarray], bool],
    volume_element: Callable[[np.ndarray], float],
) -> TensorQuadrature:
    """Cut `rule` down to a domain that is a proper subset of its bounding box.

    Nodes outside the domain are dropped and the survivors' weights are scaled
    by the measure density there.

    Both operations preserve the property the whole thinning bound rests on. The
    argument in this module's docstring needs only that the bound and the
    acceptance test see *the same nodes* with *strictly positive weights*; it
    never needed those nodes to fill a box, or those weights to be the flat
    ones. So ``sum(w_i f_bound(x_i)) >= sum(w_i f_true(x_i))`` still holds with
    no tolerance, and ``M >= lambda`` still holds pathwise.

    What masking does cost is accuracy: the integrand is effectively multiplied
    by an indicator, and a Gauss rule resolves the resulting jump only to the
    width of a panel. That error shows up as a mismatch between the summed
    weights and the domain's declared volume, which is why the caller checks it.
    """
    keep = np.array([bool(contains(node)) for node in rule.nodes], dtype=bool)
    if not keep.any():
        raise ValueError(
            "no quadrature node lies inside the domain: it is smaller than one panel of "
            "the rule. Raise n_quad."
        )

    nodes = rule.nodes[keep]
    density = np.array([float(volume_element(node)) for node in nodes], dtype=float)
    if not np.all(np.isfinite(density) & (density > 0.0)):
        raise ValueError(
            "volume_element must be finite and strictly positive at every node; the "
            "thinning bound dominates only because every quadrature weight does."
        )
    return TensorQuadrature(nodes=nodes, weights=rule.weights[keep] * density)


def check_resolution(
    rule: TensorQuadrature,
    refine: Callable[[int], TensorQuadrature],
    nodes_per_axis: int,
    fn: Callable[[np.ndarray], float],
    *,
    rtol: float = 1e-2,
    name: str = "spatial kernel",
) -> None:
    """Warn if `fn` is too narrow to be resolved by `rule`.

    Replaces inspecting the error estimate that ``quad`` returned and the code
    discarded. Deterministic, and fires once at construction rather than on
    every one of the hundreds of thousands of integrations a simulation runs.

    The failure this catches is real: with a spatial kernel of width 0.005 on a
    unit circle, ``quad`` returned exactly the background-only integral, so the
    excitation was invisible to the temporal thinning while the spatial sampler
    still saw it -- the process silently degenerated towards Poisson in time.

    `refine` rebuilds *the same* rule at a given node count. It is a callable
    rather than a bounding box because the rule may be restricted to a domain
    inside that box: comparing a masked rule against an unmasked finer one would
    measure the mask, not the resolution.
    """
    coarse = rule.integrate(fn)
    fine = refine(nodes_per_axis * 2).integrate(fn)
    scale = max(abs(fine), np.finfo(float).tiny)
    if abs(coarse - fine) / scale > rtol:
        warnings.warn(
            f"the {name} is too narrow for the quadrature rule: doubling the node count "
            f"changes the integral by {100 * abs(coarse - fine) / scale:.1f}%. The "
            f"simulated event rate will be wrong. Raise n_quad above {nodes_per_axis}.",
            UserWarning,
            stacklevel=3,
        )
