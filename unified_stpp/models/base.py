"""
Base abstract classes for the unified Neural STPP framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch.nn as nn
from torch import Tensor


class Encoder(ABC, nn.Module):
    """
    E_θ: (events, X_event) → z_0

    Input:
        events: (B, N, 1+d+p)  — time, space, event covariates (padded)
        lengths: (B,)           — actual sequence lengths
        x_event: (B, N, p) optional additional event-level covariates
    Output:
        z: (B, h)               — initial latent state
        states: (B, N, h)       — per-event hidden states (for attention-based updaters)
    """

    def __init__(self, input_dim: int, hidden_dim: int, **kwargs):
        ABC.__init__(self)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        events: Tensor,
        lengths: Tensor,
        x_event: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Returns (final_state, all_states)."""
        ...


class Decoder(ABC, nn.Module):
    """
    G_θ: (z(t), t, s, X_field) → log f*(t, s) or log λ*(t, s)

    Base class for all decoders. Must also provide the negative log-likelihood
    contribution (including the compensator integral where applicable).
    """

    SEQUENCE_COUPLED: bool = False
    """When True, this decoder requires the full event sequence to compute NLL.
    The caller will invoke ``sequence_nll`` instead of per-event ``nll``.
    """

    def __init__(self, hidden_dim: int, spatial_dim: int, **kwargs):
        ABC.__init__(self)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim

    @abstractmethod
    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """Log-density or log-intensity at (t, s)."""
        ...

    @abstractmethod
    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        t_prev: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Negative log-likelihood contribution for one event.

        For density-based decoders: -log f*(t, s)
        For intensity-based decoders: -log λ*(t, s) + ∫∫ λ* ds dt
        """
        ...

    def sequence_nll(
        self,
        z_seq: Tensor,
        t_seq: Tensor,
        s_seq: Tensor,
        t_prev_seq: Tensor,
        lengths: Tensor,
        mask: Tensor,
        **kwargs,
    ) -> Tensor:
        """Per-event NLL for an entire sequence (batch).

        Required when ``SEQUENCE_COUPLED = True``; the default raises
        ``NotImplementedError``.  When implemented, returns ``(B, L)``
        unmasked NLL values — the caller applies ``mask``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has SEQUENCE_COUPLED=True but does not "
            "implement sequence_nll()."
        )

    def sample(
        self,
        z: Tensor,
        t_prev: Tensor,
        x_field_fn=None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample (t, s) from the conditional distribution. Optional."""
        raise NotImplementedError("Sampling not implemented for this decoder.")
