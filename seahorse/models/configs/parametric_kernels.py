"""Construction configs for the parametric Hawkes/Poisson baselines."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from .base import BaseModelConfig, ConfigRegistry


@dataclasses.dataclass
class _ParametricKernelConfig(BaseModelConfig):
    """Shared options of the parametric exact-likelihood baselines."""

    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw"})
    _BLOCK: ClassVar[str] = ""

    eta_cap: float = 0.95
    min_history: int = 1
    regularization_weight: float = 1.0e-5
    mu_init: float = 2.0
    eta_init: float = 0.45
    killing_rate_init: float = 1.0
    background_std_init: float = 1.0

    @classmethod
    def _block(cls, d: dict[str, Any]) -> dict[str, Any]:
        return dict(d.get(cls._BLOCK, {}) or {})

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        hidden_dim: int = 128,
        spatial_dim: int = 2,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        n_marks: int = 0,
    ):
        del event_cov_dim, field_cov_dim, n_marks
        block = cls._block(d)
        return cls(
            hidden_dim=int(hidden_dim),
            spatial_dim=int(spatial_dim),
            eta_cap=float(block.get("eta_cap", 0.95)),
            min_history=int(block.get("min_history", 1)),
            regularization_weight=float(block.get("regularization_weight", 1.0e-5)),
            mu_init=float(block.get("mu_init", 2.0)),
            eta_init=float(block.get("eta_init", 0.45)),
            killing_rate_init=float(block.get("killing_rate_init", 1.0)),
            background_std_init=float(block.get("background_std_init", 1.0)),
        )

    def _event_kwargs(self) -> dict[str, Any]:
        return {
            "spatial_dim": self.spatial_dim,
            "eta_cap": self.eta_cap,
            "min_history": self.min_history,
            "regularization_weight": self.regularization_weight,
            "mu_init": self.mu_init,
            "eta_init": self.eta_init,
            "killing_rate_init": self.killing_rate_init,
            "background_std_init": self.background_std_init,
        }

    def _build(self, event_model):
        from seahorse.models.state_models.factorized import FactorizedStateModel
        from seahorse.models.unified_model import UnifiedSTPP

        state_model = FactorizedStateModel(input_transform={})
        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )


@ConfigRegistry.register("static_hawkes")
@dataclasses.dataclass
class StaticGaussianHawkesConfig(_ParametricKernelConfig):
    """Exact Hawkes baseline whose offspring displacement ignores event age."""

    _BLOCK: ClassVar[str] = "static_hawkes"

    static_std_init: float = 0.40

    @classmethod
    def from_dict(cls, d: dict[str, Any], **dims) -> "StaticGaussianHawkesConfig":
        obj = super().from_dict(d, **dims)
        obj.static_std_init = float(cls._block(d).get("static_std_init", 0.40))
        return obj

    def build_model(self):
        from seahorse.models.event_models.parametric_kernels import (
            StaticGaussianHawkesEventModel,
        )

        return self._build(
            StaticGaussianHawkesEventModel(
                static_std_init=self.static_std_init, **self._event_kwargs()
            )
        )


@ConfigRegistry.register("gaussian_poisson")
@dataclasses.dataclass
class GaussianPoissonConfig(_ParametricKernelConfig):
    """Exact homogeneous-Poisson null with the shared Gaussian background."""

    _BLOCK: ClassVar[str] = "gaussian_poisson"

    def build_model(self):
        from seahorse.models.event_models.parametric_kernels import GaussianPoissonEventModel

        return self._build(
            GaussianPoissonEventModel(
                spatial_dim=self.spatial_dim,
                min_history=self.min_history,
                regularization_weight=self.regularization_weight,
                mu_init=self.mu_init,
                background_std_init=self.background_std_init,
            )
        )
