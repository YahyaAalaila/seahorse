"""
Self-Attention Encoder — used in DeepSTPP, AutoSTPP, NMSTPP.

Processes event sequence via multi-head self-attention with causal masking.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import math
from ..base import Encoder


class AttentionEncoder(Encoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 2,
        num_layers: int = 3,
        dropout: float = 0,
        event_cov_dim: int = 0,
        **kwargs,
    ):
        total_input = input_dim + event_cov_dim
        super().__init__(input_dim=total_input, hidden_dim=hidden_dim)

        self.embed = nn.Linear(total_input, hidden_dim)
        self.pos_enc = LearnedPositionalEncoding(hidden_dim, max_len=1024)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # enable_nested_tensor=False: disables the nested-tensor fast-path which
        # is not implemented on MPS (Apple Silicon GPU). Has no effect on CUDA/CPU.
        self.attn_layers = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        events: Tensor,
        lengths: Tensor,
        x_event: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, N, _ = events.shape
        if x_event is not None:
            events = torch.cat([events, x_event], dim=-1)

        x = self.embed(events)  # (B, N, h)
        x = self.pos_enc(x)

        # Causal mask (bool): True entries are masked (future positions).
        causal_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Padding mask
        arange = torch.arange(N, device=x.device).unsqueeze(0)  # (1, N)
        pad_mask = arange >= lengths.unsqueeze(1)  # (B, N), True = masked

        all_states = self.attn_layers(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        all_states = self.norm(all_states)  # (B, N, h)

        # Extract final state per sequence
        idx = (lengths - 1).clamp(min=0).long()  # (B,)
        z_final = all_states[torch.arange(B, device=x.device), idx]  # (B, h)

        return z_final, all_states


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def _grow_if_needed(self, n_positions: int, device: torch.device):
        current = self.pos_embed.num_embeddings
        if n_positions <= current:
            return
        new_size = current
        while new_size < n_positions:
            new_size *= 2
        new_embed = nn.Embedding(new_size, self.d_model).to(device=device)
        with torch.no_grad():
            new_embed.weight[:current].copy_(self.pos_embed.weight)
            nn.init.normal_(new_embed.weight[current:], mean=0.0, std=0.02)
        self.pos_embed = new_embed

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        self._grow_if_needed(N, x.device)
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pos_embed(positions)
