"""Configuration adapters for non-Torch statistical STPP presets."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .base import ConfigRegistry


class _StatisticalConfig:
    """Retain declared options for direct ``ModelConfig`` construction."""

    _preset: str = ""

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.options = dict(options)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], **_dimensions: Any) -> "_StatisticalConfig":
        return cls(copy.deepcopy(dict(data)))

    def build_model(self):
        from seahorse.models.statistical import build_statistical_model

        return build_statistical_model(self._preset, self.options)

    @classmethod
    def resolve_accelerator(cls, _requested: str) -> str:
        """Native statistical fitting is CPU-only."""
        return "cpu"

    @classmethod
    def data_init_overrides(cls, _dm):
        return {}

    @classmethod
    def fit_transform_artifact(cls, _dm):
        return None

    @classmethod
    def transform_init_overrides(cls, _artifact):
        return {}


for _name in (
    "homogeneous_poisson", "spatial_poisson", "seasonal_poisson",
    "hawkes_exp_gaussian", "hawkes_omori_pareto",
):
    config_cls = type(f"{_name.title().replace('_', '')}Config", (_StatisticalConfig,), {"_preset": _name})
    ConfigRegistry.register(_name, runner_kind="statstpp")(config_cls)
