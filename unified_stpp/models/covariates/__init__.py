"""
Covariate handling: field covariates, event covariates, and lifting maps.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
import math


class LiftingMap(nn.Module):
    """
    L_θ: {(t_i, s_i, X_i^event)} → X^field(t, s)
    
    Lifts discrete event-level covariates to a continuous field
    via kernel smoothing or a learned interpolator.
    
    Required when only X^event is available but injection points
    (II) Dynamics or (IV) Decoder need field covariates.
    """

    def __init__(
        self,
        event_cov_dim: int,
        output_dim: int,
        spatial_dim: int,
        method: str = "kernel",  # "kernel" or "learned"
        bandwidth: float = 1.0,
    ):
        super().__init__()
        self.event_cov_dim = event_cov_dim
        self.output_dim = output_dim
        self.spatial_dim = spatial_dim
        self.method = method
        self.bandwidth = bandwidth

        if method == "learned":
            input_dim = 1 + spatial_dim + event_cov_dim + 1 + spatial_dim
            # Input: (t_query, s_query, t_event, s_event, x_event) → contribution
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.SiLU(),
                nn.Linear(64, output_dim),
            )
        elif method == "kernel":
            self.proj = nn.Linear(event_cov_dim, output_dim)
        else:
            raise ValueError(f"Unknown lifting method: {method}")

    def forward(
        self,
        t_query: Tensor,
        s_query: Tensor,
        event_times: Tensor,
        event_locs: Tensor,
        event_covs: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """
        Args:
            t_query: (B, 1) — query time
            s_query: (B, d) — query location
            event_times: (B, N) — past event times
            event_locs: (B, N, d) — past event locations
            event_covs: (B, N, p) — past event covariates
            lengths: (B,) — actual sequence lengths
        Returns:
            x_field: (B, r) — interpolated field covariate at (t_query, s_query)
        """
        B, N = event_times.shape
        d = self.spatial_dim
        device = t_query.device

        # Mask for valid events
        arange = torch.arange(N, device=device).unsqueeze(0)
        mask = (arange < lengths.unsqueeze(1)).float()  # (B, N)

        # Only use events before t_query
        causal_mask = (event_times < t_query.squeeze(-1).unsqueeze(-1)).float()
        mask = mask * causal_mask  # (B, N)

        if self.method == "kernel":
            # Kernel smoothing with Gaussian kernel
            dt = t_query - event_times.unsqueeze(-1)  # using broadcast
            dt = (t_query.squeeze(-1).unsqueeze(-1) - event_times)  # (B, N)
            ds = s_query.unsqueeze(1) - event_locs  # (B, N, d)
            dist_sq = dt ** 2 + (ds ** 2).sum(dim=-1)  # (B, N)

            weights = torch.exp(-dist_sq / (2 * self.bandwidth ** 2))  # (B, N)
            weights = weights * mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Weighted sum of projected covariates
            x_proj = self.proj(event_covs)  # (B, N, r)
            x_field = (weights.unsqueeze(-1) * x_proj).sum(dim=1)  # (B, r)
            return x_field

        elif self.method == "learned":
            # Learned interpolation
            t_q = t_query.unsqueeze(1).expand(B, N, 1)  # (B, N, 1)
            s_q = s_query.unsqueeze(1).expand(B, N, d)  # (B, N, d)
            t_e = event_times.unsqueeze(-1)  # (B, N, 1)
            inp = torch.cat([t_q, s_q, t_e, event_locs, event_covs], dim=-1)
            contrib = self.net(inp)  # (B, N, r)
            contrib = contrib * mask.unsqueeze(-1)
            return contrib.sum(dim=1) / (mask.sum(dim=-1, keepdim=True) + 1e-8)


class MarkEmbedding(nn.Module):
    """
    Embeds discrete mark indices into dense vectors for injection as x_event.

    Use this to feed mark types into the encoder and updater via the
    existing event-level covariate path (x_event), without modifying
    any Encoder or Updater signatures.
    """

    def __init__(self, n_marks: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_marks, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, marks: Tensor) -> Tensor:
        """
        Args:
            marks: (B, N) LongTensor — mark indices
        Returns:
            (B, N, embed_dim) float embeddings
        """
        return self.embedding(marks)


class FieldCovariateEncoder(nn.Module):
    """
    Encodes field covariates X^field(t, s) ∈ R^r into a representation
    suitable for injection into a model component.
    
    If the field is given as a callable (e.g., from gridded data),
    this module evaluates and projects it.
    If the field is pre-evaluated, it just projects.
    """

    def __init__(self, field_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(field_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x_field: Tensor) -> Tensor:
        return self.net(x_field)
