"""
GRU Encoder — used in NeuralSTPP.

Processes event sequence (t_i, s_i, X_i) sequentially via a GRU.
Returns final hidden state and all intermediate states.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from ..base import Encoder


class GRUEncoder(Encoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        event_cov_dim: int = 0,
        **kwargs,
    ):
        # input_dim = 1 (time) + d (space) [+ p (event covariates)]
        total_input = input_dim + event_cov_dim
        super().__init__(input_dim=total_input, hidden_dim=hidden_dim)
        self.embed = nn.Linear(total_input, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.num_layers = num_layers

    def forward(
        self,
        events: Tensor,
        lengths: Tensor,
        x_event: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            events: (B, N, 1+d)  — concatenated (time, space)
            lengths: (B,)
            x_event: (B, N, p) optional
        Returns:
            z_final: (B, h)
            all_states: (B, N, h)
        """
        if x_event is not None:
            events = torch.cat([events, x_event], dim=-1)

        x = self.embed(events)  # (B, N, h)

        # Pack for variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        output_packed, h_n = self.gru(packed)
        all_states, _ = nn.utils.rnn.pad_packed_sequence(
            output_packed, batch_first=True
        )  # (B, N, h)

        z_final = h_n[-1]  # (B, h) — last layer final state
        return z_final, all_states
