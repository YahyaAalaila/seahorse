"""
Identity Dynamics — z(t) = z_n for all t in (t_n, t_{n+1}).

Used by DeepSTPP, DSTPP, AutoSTPP, NMSTPP, and any method where
temporal dependencies are handled entirely by the encoder/updater.
"""

import torch
from torch import Tensor
from typing import Optional
from ..base import Dynamics


class IdentityDynamics(Dynamics):
    def __init__(self, hidden_dim: int, **kwargs):
        super().__init__(hidden_dim=hidden_dim)

    def forward(
        self,
        z_n: Tensor,
        dt: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            z_n: (B, h)
            dt: (B, M) — ignored
            x_field: ignored
        Returns:
            z_t: (B, M, h) — z_n broadcast over M query times
        """
        B, h = z_n.shape
        M = dt.shape[1]
        return z_n.unsqueeze(1).expand(B, M, h)
