"""
Transformer Encoder — used in DSTPP, NMSTP.

Similar to AttentionEncoder but uses continuous-time positional encoding
based on actual event timestamps rather than discrete positions.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import math
from ..base import Encoder


class TransformerEncoder(Encoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        event_cov_dim: int = 0,
        **kwargs,
    ):
        total_input = input_dim + event_cov_dim
        super().__init__(input_dim=total_input, hidden_dim=hidden_dim)

        self.embed = nn.Linear(total_input, hidden_dim)
        self.time_enc = ContinuousTimeEncoding(hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # Disable nested-tensor fast path: MPS currently misses
        # aten::_nested_tensor_from_mask_left_aligned.
        try:
            self.layers = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False
            )
        except TypeError:
            # Older PyTorch versions may not expose enable_nested_tensor.
            self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        events: Tensor,
        lengths: Tensor,
        x_event: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, N, D = events.shape
        if x_event is not None:
            events = torch.cat([events, x_event], dim=-1)

        # Extract timestamps (first column) for continuous time encoding
        times = events[:, :, 0]  # (B, N)

        x = self.embed(events)  # (B, N, h)
        x = x + self.time_enc(times)

        causal_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        arange = torch.arange(N, device=x.device).unsqueeze(0)
        pad_mask = arange >= lengths.unsqueeze(1)

        all_states = self.layers(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        all_states = self.norm(all_states)

        idx = (lengths - 1).clamp(min=0).long()
        z_final = all_states[torch.arange(B, device=x.device), idx]

        return z_final, all_states


class ContinuousTimeEncoding(nn.Module):
    """Fourier-based continuous time encoding."""

    def __init__(self, d_model: int, max_period: float = 100.0):
        super().__init__()
        assert d_model % 2 == 0
        half = d_model // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * (-math.log(max_period) / half)
        )
        self.register_buffer("freqs", freqs)  # (d/2,)

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B, N) — timestamps
        Returns:
            encoding: (B, N, d_model)
        """
        args = t.unsqueeze(-1) * self.freqs  # (B, N, d/2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
