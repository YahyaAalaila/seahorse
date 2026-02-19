"""
Mark Decoders — model p*(k | t, s, z) for discrete event types.

The mark distribution is factored from the ground process:
    λ*(t, s, k) = λ*(t, s) · p*(k | t, s, z)
    NLL = -log f*(t, s) - log p*(k | t, s, z)

The compensator integral does NOT involve marks (only the ground process),
so existing temporal/spatial decoders require zero modification.

Two implementations:
1. MLPMarkDecoder     — concatenate (z, t, s, x_field), MLP → K logits
2. AttentionMarkDecoder — score each mark via dot product with mark embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from ..base import MarkDecoder


class MLPMarkDecoder(MarkDecoder):
    """
    MLP mark decoder: concatenate (z, t, s, x_field) and predict K logits.

    Parameters
    ----------
    hidden_dim : int
        Latent state dimension (also used for hidden layers).
    spatial_dim : int
        Dimension of event locations.
    n_marks : int
        Number of discrete mark types.
    n_layers : int
        Number of hidden layers (default 2).
    field_cov_dim : int
        Dimension of optional field covariates (default 0).
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        n_marks: int,
        n_layers: int = 2,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__(hidden_dim, spatial_dim, n_marks)
        input_dim = hidden_dim + 1 + spatial_dim + field_cov_dim
        layers = []
        in_d = input_dim
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.SiLU()])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, n_marks))
        self.net = nn.Sequential(*layers)

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z: (B, h)
            t: (B, 1)
            s: (B, d)
            x_field: (B, r) optional
        Returns:
            log_probs: (B, K)
        """
        parts = [z, t, s]
        if x_field is not None:
            parts.append(x_field)
        logits = self.net(torch.cat(parts, dim=-1))  # (B, K)
        return F.log_softmax(logits, dim=-1)


class AttentionMarkDecoder(MarkDecoder):
    """
    Attention-based mark decoder.

    Scores each mark type via dot-product attention between a query
    vector (derived from z, t, s) and learned mark embeddings.

    Parameters
    ----------
    hidden_dim : int
        Latent state dimension.
    spatial_dim : int
        Dimension of event locations.
    n_marks : int
        Number of discrete mark types.
    field_cov_dim : int
        Dimension of optional field covariates (default 0).
    """

    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int,
        n_marks: int,
        field_cov_dim: int = 0,
        **kwargs,
    ):
        super().__init__(hidden_dim, spatial_dim, n_marks)
        input_dim = hidden_dim + 1 + spatial_dim + field_cov_dim
        self.mark_embed = nn.Embedding(n_marks, hidden_dim)
        self.query_proj = nn.Linear(input_dim, hidden_dim)

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z: (B, h)
            t: (B, 1)
            s: (B, d)
            x_field: (B, r) optional
        Returns:
            log_probs: (B, K)
        """
        parts = [z, t, s]
        if x_field is not None:
            parts.append(x_field)
        query = self.query_proj(torch.cat(parts, dim=-1))  # (B, h)

        # Score each mark via dot product with mark embeddings
        mark_embeds = self.mark_embed.weight  # (K, h)
        logits = torch.matmul(query, mark_embeds.T)  # (B, K)
        return F.log_softmax(logits, dim=-1)
