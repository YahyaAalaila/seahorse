"""Construction configuration for the independent DNSK+Barrier implementation.

Available preset; not benchmarked in the Seahorse paper. The implementation
follows Dong et al. (ICLR 2023) from the paper, without author source code.
"""
from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from .base import BaseModelConfig, ConfigRegistry


@ConfigRegistry.register("dnsk")
@dataclasses.dataclass
class DNSKConfig(BaseModelConfig):
    _STATE_MODEL: ClassVar[str] = "nsmpp_deepbasis"  # shared history-only container
    _EVENT_MODEL: ClassVar[str] = "dnsk"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset] = frozenset({"raw", "standard", "unified"})
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    input_transform: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, d, *, hidden_dim=128, spatial_dim=2, event_cov_dim=0, field_cov_dim=0, n_marks=0):
        if spatial_dim != 2 or event_cov_dim or field_cov_dim or n_marks:
            raise ValueError("This DNSK implementation is an unmarked two-dimensional STPP.")
        decoder = dict(d.get("decoder", {}))
        kind = decoder.pop("type", "dnsk")
        if kind != "dnsk":
            raise ValueError(f"DNSK decoder.type must be dnsk, got {kind!r}.")
        for source, target in (("support_space_min", "space_min"), ("support_space_max", "space_max"),
                               ("support_t0", "t0")):
            if source in d:
                decoder[target] = d[source]
        return cls(hidden_dim=hidden_dim, spatial_dim=spatial_dim, parameters=decoder,
                   input_transform=dict(d.get("input_transform", {})))

    def _state_kwargs(self):
        return {"input_transform": self.input_transform}

    def _event_kwargs(self):
        return dict(self.parameters)
