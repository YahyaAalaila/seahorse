"""
GRU Jump Updater — used in NeuralSTPP.

At each event arrival, the pre-event state z(t⁻) is updated via a GRU cell
that takes the new event (t, s, X) as input.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ..base import Updater


class GRUJumpUpdater(Updater):
    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        input_dim = 1 + spatial_dim + event_cov_dim + field_cov_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(
        self,
        z_pre: Tensor,
        t: Tensor,
        s: Tensor,
        x_event: Optional[Tensor] = None,
        x_field_at_event: Optional[Tensor] = None,
        encoder_states: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z_pre: (B, h) — pre-event latent state
            t: (B, 1)     — event time
            s: (B, d)     — event location
            x_event: (B, p) optional
            x_field_at_event: (B, r) optional
        Returns:
            z_post: (B, h)
        """
        parts = [t, s]
        if x_event is not None:
            parts.append(x_event)
        if x_field_at_event is not None:
            parts.append(x_field_at_event)
        event_input = torch.cat(parts, dim=-1)  # (B, 1+d+p+r)
        z_post = self.gru_cell(event_input, z_pre)
        return z_post
