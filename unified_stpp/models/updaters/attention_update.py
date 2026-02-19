"""
Attention Updater — used by DeepSTPP, DSTPP, and attention-based methods.

For attention-based models, the "update" is a cross-attention step where
the new event queries all previous encoder states. This avoids re-running 
the full encoder at each event while maintaining the modular interface.

In practice, for these models the encoder already produces all per-event 
states via causal attention, so the updater simply selects the appropriate 
state from the pre-computed encoder outputs.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ..base import Updater


class AttentionUpdater(Updater):
    """
    Two modes:
    1. If encoder_states are available (pre-computed): select the n-th state.
       This is the efficient path during training when the full sequence is known.
    2. If not: use a cross-attention layer to update z_pre given the new event.
       This is used during sampling / online inference.
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        num_heads: int = 4,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)

        input_dim = 1 + spatial_dim + event_cov_dim + field_cov_dim
        self.event_proj = nn.Linear(input_dim, hidden_dim)

        # Cross-attention for online mode
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Internal buffer for accumulating states during online inference
        self._state_buffer = None

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
            z_pre: (B, h)
            t: (B, 1)
            s: (B, d)
            x_event: (B, p) optional
            x_field_at_event: (B, r) optional
            encoder_states: (B, N, h) optional — if available, use pre-computed states
        Returns:
            z_post: (B, h)
        """
        # Build event embedding
        parts = [t, s]
        if x_event is not None:
            parts.append(x_event)
        if x_field_at_event is not None:
            parts.append(x_field_at_event)
        event_embed = self.event_proj(torch.cat(parts, dim=-1))  # (B, h)

        if encoder_states is not None:
            # Training mode: use cross-attention over encoder states
            query = event_embed.unsqueeze(1)  # (B, 1, h)
            out, _ = self.cross_attn(query, encoder_states, encoder_states)
            out = self.norm(out + query)
            out = self.norm2(out + self.ffn(out))
            return out.squeeze(1)  # (B, h)
        else:
            # Online mode: simple GRU-like update
            # (Fallback for sampling when encoder_states not available)
            gate = torch.sigmoid(event_embed)
            return gate * z_pre + (1 - gate) * event_embed
