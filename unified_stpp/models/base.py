"""
Base abstract classes for the unified Neural STPP framework.

A Neural STPP is M = (E, D, U, G) where:
  E: Encoder     — maps event history to initial latent state
  D: Dynamics    — evolves latent state between events  
  U: Updater     — updates latent state at event arrival
  G: Decoder     — maps latent state to intensity/density

Covariates enter at each component via standardized interfaces.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
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


class Dynamics(ABC, nn.Module):
    """
    D_θ: (z_n, Δt, X_field) → z(t)
    
    Evolves latent state from z_n over time interval Δt.
    
    Input:
        z_n: (B, h)            — state after last event
        dt: (B, M)             — query time offsets from t_n
        x_field: (B, M, r) optional field covariates at query times
    Output:
        z_t: (B, M, h)         — states at query times
    """

    def __init__(self, hidden_dim: int, **kwargs):
        ABC.__init__(self)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        z_n: Tensor,
        dt: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        ...


class Updater(ABC, nn.Module):
    """
    U_θ: (z(t⁻), t, s, X_event, X_field) → z⁺
    
    Updates latent state when a new event arrives.
    
    Input:
        z_pre: (B, h)          — pre-event state
        t: (B, 1)              — event time
        s: (B, d)              — event location
        x_event: (B, p) optional event-level covariates
        x_field_at_event: (B, r) optional field covariate at (t, s)
        encoder_states: (B, N, h) optional — all past encoder states
    Output:
        z_post: (B, h)         — post-event state
    """

    def __init__(self, hidden_dim: int, spatial_dim: int, **kwargs):
        ABC.__init__(self)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim

    @abstractmethod
    def forward(
        self,
        z_pre: Tensor,
        t: Tensor,
        s: Tensor,
        x_event: Optional[Tensor] = None,
        x_field_at_event: Optional[Tensor] = None,
        encoder_states: Optional[Tensor] = None,
    ) -> Tensor:
        ...


class Decoder(ABC, nn.Module):
    """
    G_θ: (z(t), t, s, X_field) → log f*(t, s) or log λ*(t, s)
    
    Base class for all decoders. Subclasses implement either:
    - Joint density/intensity: log_prob(z, t, s, ...)
    - Factorized: temporal log_prob + spatial log_prob
    
    Must also provide the negative log-likelihood contribution
    (including the compensator integral where applicable).
    
    Input:
        z: (B, h)              — latent state
        t: (B, 1)              — query time
        s: (B, d)              — query location  
        t_prev: (B, 1)         — previous event time (for inter-event interval)
        x_field: (B, r) optional field covariates at (t, s)
    Output:
        log_prob: (B,)         — log f*(t, s | H_t) or log λ*(t, s | H_t)
    """

    SEQUENCE_COUPLED: bool = False
    """When True, this decoder requires the full event sequence to compute NLL.
    ``UnifiedSTPP`` will call ``sequence_nll`` instead of per-event ``nll``.
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

        Args:
            z_seq:      (B, L, h) — conditioning state at each prediction step.
            t_seq:      (B, L, 1) — target event times.
            s_seq:      (B, L, d) — target event locations.
            t_prev_seq: (B, L, 1) — previous event times (inter-event interval).
            lengths:    (B,)      — actual sequence lengths.
            mask:       (B, L)    — 1.0 for valid positions, 0.0 for padding.
        Returns:
            nll: (B, L) — per-event NLL, unmasked.
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


class MarkDecoder(ABC, nn.Module):
    """
    G^m_θ: (z(t), t, s, X_field) → log p*(k | t, s, z)

    Models the conditional mark distribution given the latent state,
    event time, and event location. The mark NLL decomposes additively
    from the ground process NLL: no changes to existing Decoder subclasses.

    Input:
        z: (B, h)          — latent state
        t: (B, 1)          — event time
        s: (B, d)          — event location
        x_field: (B, r)    — optional field covariates
    Output:
        log_probs: (B, K)  — log-probabilities for each mark
    """

    def __init__(self, hidden_dim: int, spatial_dim: int, n_marks: int, **kwargs):
        ABC.__init__(self)
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.n_marks = n_marks

    @abstractmethod
    def log_prob(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns (B, K) log-probabilities."""
        ...

    def nll(
        self,
        z: Tensor,
        t: Tensor,
        s: Tensor,
        k: Tensor,
        x_field: Optional[Tensor] = None,
    ) -> Tensor:
        """
        NLL for observed mark k.

        Args:
            z: (B, h)
            t: (B, 1)
            s: (B, d)
            k: (B,) LongTensor — observed mark indices
            x_field: (B, r) optional
        Returns:
            nll: (B,)
        """
        log_probs = self.log_prob(z, t, s, x_field)  # (B, K)
        return -log_probs[torch.arange(k.shape[0], device=k.device), k]  # (B,)


class CovariateProcessor(nn.Module):
    """
    Processes and projects covariates for injection into a component.
    Handles both field and event-level covariates.
    """

    def __init__(
        self,
        field_dim: int = 0,
        event_dim: int = 0,
        output_dim: int = 0,
    ):
        super().__init__()
        self.field_dim = field_dim
        self.event_dim = event_dim
        total_in = field_dim + event_dim
        if total_in > 0 and output_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(total_in, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.proj = None

    def forward(
        self,
        x_field: Optional[Tensor] = None,
        x_event: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        parts = []
        if x_field is not None and self.field_dim > 0:
            parts.append(x_field)
        if x_event is not None and self.event_dim > 0:
            parts.append(x_event)
        if not parts:
            return None
        x = torch.cat(parts, dim=-1)
        if self.proj is not None:
            x = self.proj(x)
        return x
