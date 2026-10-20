"""Registrable source-backed components used by native statistical presets."""

from __future__ import annotations

from ._hawkes_source.hawkes_package.inference.families import (
    ConstantBase,
    ExponentialKernel,
    GaussianSpatial,
    OmoriUtsuKernel,
    ParetoSpatial,
)
from ._hawkes_source.hawkes_package.inference.mle import fit_mle


BACKGROUND_REGISTRY = {"constant": ConstantBase}
TEMPORAL_KERNEL_REGISTRY = {"exponential": ExponentialKernel, "omori": OmoriUtsuKernel}
SPATIAL_KERNEL_REGISTRY = {
    "gaussian": lambda: GaussianSpatial(2),
    "pareto": lambda: ParetoSpatial(2),
}
INFERENCE_REGISTRY = {"mle": fit_mle}


def resolve_component(registry: dict, name: str):
    try:
        return registry[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown statistical component {name!r}; available={sorted(registry)}") from exc
