"""Tutorial event decoder for the ``demo_gru_gaussian`` preset."""

from __future__ import annotations

from typing import Dict, Optional

import math
import torch
import torch.nn as nn
from torch import Tensor

from unified_stpp.models.abstractions import EventCapabilities, EventModel, StateContext
from unified_stpp.models.model_registry import register_event


@register_event("demo_temporal_gaussian")
class DemoTemporalGaussianEventModel(EventModel):
    """Factorized next-event likelihood with exponential time and Gaussian space."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        spatial_dim: int = 2,
        min_scale: float = 0.03,
        max_log_scale: float = 1.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.spatial_dim = int(spatial_dim)
        self.min_scale = float(min_scale)
        self.max_log_scale = float(max_log_scale)
        self.rate_head = nn.Linear(self.hidden_dim, 1)
        self.loc_head = nn.Linear(self.hidden_dim, self.spatial_dim)
        self.log_scale_head = nn.Linear(self.hidden_dim, self.spatial_dim)

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="exact exponential-time + Gaussian-space NLL",
            nll_kind="exact",
            nll_description="exact joint NLL/event for the demo GRU-Gaussian model",
            has_density=True,
            exposes_eventwise_terms=True,
        )

    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        del state_regularization_terms, x_field_at_events, marks
        if device is None:
            device = times.device
        times = times.to(device)
        locations = locations.to(device)
        lengths = lengths.to(device)
        event_state = state.payload["event_state"].to(device)

        B, T = times.shape
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.float()

        previous_times = torch.cat([times.new_zeros(B, 1), times[:, :-1]], dim=1)
        delta_t = (times - previous_times).clamp_min(1e-5)
        rate = torch.nn.functional.softplus(self.rate_head(event_state)).squeeze(-1) + 1e-5
        temporal_nll_matrix = -(torch.log(rate) - rate * delta_t)

        loc_mean = self.loc_head(event_state)
        log_scale = self.log_scale_head(event_state).clamp(
            min=math.log(self.min_scale),
            max=self.max_log_scale,
        )
        inv_scale_resid = (locations - loc_mean) / torch.exp(log_scale)
        spatial_nll_matrix = 0.5 * (
            inv_scale_resid.pow(2)
            + 2.0 * log_scale
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)

        nll_matrix = temporal_nll_matrix + spatial_nll_matrix
        total_events = mask_f.sum().clamp_min(1.0)
        nll = (nll_matrix * mask_f).sum() / total_events
        temporal_nll = (temporal_nll_matrix * mask_f).sum() / total_events
        spatial_nll = (spatial_nll_matrix * mask_f).sum() / total_events

        event_index = torch.arange(T, device=device).unsqueeze(0)
        next_event_mask = mask & (event_index > 0)
        return {
            "loss": nll,
            "nll": nll,
            "total_events": total_events,
            "mask": mask_f,
            "next_event_mask": next_event_mask.float(),
            "nll_matrix": nll_matrix,
            "temporal_nll_matrix": temporal_nll_matrix,
            "spatial_nll_matrix": spatial_nll_matrix,
            "temporal_nll": temporal_nll,
            "spatial_nll": spatial_nll,
        }
