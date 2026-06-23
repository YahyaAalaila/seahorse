"""Config for the tutorial ``demo_gru_gaussian`` research model."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

from .base import BaseModelConfig, ConfigRegistry


@ConfigRegistry.register("demo_gru_gaussian")
@dataclasses.dataclass
class DemoGRUGaussianConfig(BaseModelConfig):
    """Small encoder-evolve-decoder model used in the extension tutorial."""

    _STATE_MODEL = "demo_gru_decay"
    _EVENT_MODEL = "demo_temporal_gaussian"

    decay_init: float = 0.25
    min_scale: float = 0.03
    max_log_scale: float = 1.5

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        *,
        hidden_dim: int = 32,
        spatial_dim: int = 2,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        n_marks: int = 0,
    ) -> "DemoGRUGaussianConfig":
        del event_cov_dim, field_cov_dim, n_marks
        state = dict(d.get("state", {}) or {})
        decoder = dict(d.get("decoder", {}) or {})
        return cls(
            hidden_dim=int(hidden_dim),
            spatial_dim=int(spatial_dim),
            decay_init=float(state.get("decay_init", d.get("decay_init", 0.25))),
            min_scale=float(decoder.get("min_scale", d.get("min_scale", 0.03))),
            max_log_scale=float(
                decoder.get("max_log_scale", d.get("max_log_scale", 1.5))
            ),
        )

    def _state_kwargs(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "spatial_dim": self.spatial_dim,
            "decay_init": self.decay_init,
        }

    def _event_kwargs(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "spatial_dim": self.spatial_dim,
            "min_scale": self.min_scale,
            "max_log_scale": self.max_log_scale,
        }
