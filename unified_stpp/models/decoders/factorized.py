"""
Factorized Decoder — f*(t, s) = f*(t) · f*(s | t).

Composes a temporal and spatial sub-decoder.
Used by NeuralSTPP and DeepSTPP.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple
from ..base import Decoder


class FactorizedDecoder(Decoder):
    """
    Wraps a temporal decoder and a spatial decoder into a single Decoder.
    
    log f*(t, s) = log f*(t) + log f*(s | t)
    
    The temporal decoder receives z and produces log f*(t | H).
    The spatial decoder receives z (and possibly t) and produces log f*(s | t, H).
    """

    def __init__(
        self,
        temporal_decoder,
        spatial_decoder,
        hidden_dim: int,
        spatial_dim: int,
        **kwargs,
    ):
        super().__init__(hidden_dim=hidden_dim, spatial_dim=spatial_dim)
        self.temporal = temporal_decoder
        self.spatial = spatial_decoder

    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        x_field_spatial: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Returns log f*(t, s) = log f*(t) + log f*(s|t).

        Args:
            x_field: field covariates passed to both sub-decoders (default).
            x_field_spatial: if provided, passed to the spatial decoder instead
                of x_field.  Use this for data-centered decoders that receive
                history windows rather than field covariates.
        """
        log_ft = self.temporal.log_prob(z, t, t_prev, x_field=x_field)
        spat_field = x_field_spatial if x_field_spatial is not None else x_field
        log_fs = self.spatial.log_prob(z, t, s, t_prev, x_field=spat_field)
        return log_ft + log_fs

    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
        x_field_spatial: Optional[Tensor] = None,
    ) -> Tensor:
        """
        NLL = -log f*(t) - log f*(s|t) = -log f*(t,s).

        For density-based decoders, the compensator is implicit
        (the density integrates to 1 by construction).
        """
        return -self.log_prob(z, t, s, t_prev, x_field=x_field,
                              x_field_spatial=x_field_spatial)

    @property
    def SEQUENCE_COUPLED(self) -> bool:
        """True when the spatial sub-decoder requires full-sequence processing."""
        return getattr(self.spatial, "SEQUENCE_COUPLED", False)

    def sequence_nll(
        self,
        z_seq: Tensor,
        t_seq: Tensor,
        s_seq: Tensor,
        t_prev_seq: Tensor,
        lengths: Tensor,
        mask: Tensor,
        x_field_seq: Optional[Tensor] = None,
    ) -> Tensor:
        """Per-event NLL for sequences with a sequence-coupled spatial sub-decoder.

        Temporal NLL is computed per-event (independent across positions).
        Spatial NLL is delegated to ``self.spatial.sequence_nll``.
        """
        B, L, h = z_seq.shape
        z_flat = z_seq.reshape(B * L, h)
        t_flat = t_seq.reshape(B * L, 1)
        t_prev_flat = t_prev_seq.reshape(B * L, 1)
        x_field_flat = (
            x_field_seq.reshape(B * L, -1) if x_field_seq is not None else None
        )

        temporal_nll = -self.temporal.log_prob(
            z_flat, t_flat, t_prev_flat, x_field=x_field_flat
        ).reshape(B, L)  # (B, L)

        spatial_nll = self.spatial.sequence_nll(
            z_seq, t_seq, s_seq, t_prev_seq, lengths, mask,
            x_field_seq=x_field_seq,
        )  # (B, L)

        return temporal_nll + spatial_nll

    def sample(
        self,
        z: Tensor,
        t_prev: Tensor,
        x_field_fn=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample (t, s) from f*(t) · f*(s|t).
        
        Two-step ancestral sampling:
          1. t ~ f*(t | z, H)        via temporal decoder
          2. s ~ f*(s | t, z, H)     via spatial decoder, conditioned on sampled t
        """
        # 1. Sample time
        t_sampled = self.temporal.sample(z, t_prev)  # (B,)
        t_sampled_col = t_sampled.unsqueeze(-1)  # (B, 1)

        # 2. Optionally evaluate field covariate at sampled (t, s_dummy)
        x_field = None
        if x_field_fn is not None:
            # For spatial sampling we don't know s yet; pass t only
            # The spatial decoder may ignore x_field or use temporal part
            x_field = x_field_fn(t_sampled_col)

        # 3. Sample location conditioned on sampled time
        s_sampled = self.spatial.sample(z, t_sampled_col, t_prev, x_field)  # (B, d)

        return t_sampled_col, s_sampled
