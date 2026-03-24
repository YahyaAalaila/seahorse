"""EventModel for NeuralSTPP — owns JumpCNFSpatial or SelfAttentiveCNFSpatial."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext

_SPATIAL_DECODER_REGISTRY: Dict[str, Any] = {}


def _get_spatial_registry() -> Dict[str, Any]:
    if not _SPATIAL_DECODER_REGISTRY:
        from ..spatial_models import JumpCNFSpatial, SelfAttentiveCNFSpatial

        _SPATIAL_DECODER_REGISTRY.update(
            {
                "jump_cnf": JumpCNFSpatial,
                "self_attentive_cnf": SelfAttentiveCNFSpatial,
            }
        )
    return _SPATIAL_DECODER_REGISTRY


class NeuralSTPPEventModel(EventModel):
    """NeuralSTPP event model.  Owns the spatial decoder."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        field_cov_dim: int = 0,
        spatial_type: str = "jump_cnf",
        **spatial_extra,
    ):
        super().__init__()
        registry = _get_spatial_registry()
        if spatial_type not in registry:
            raise ValueError(
                f"Unsupported spatial_type '{spatial_type}'. "
                f"Available: {sorted(registry)}"
            )
        self.spatial_decoder = registry[spatial_type](
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            field_cov_dim=field_cov_dim,
            **spatial_extra,
        )

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="exact_nll",
            has_eval_nll=True,
            has_intensity=True,
            has_density=False,
            has_score=False,
            has_native_sampler=False,
            exposes_eventwise_terms=True,
        )

    @staticmethod
    def _as_like(x: Any, ref: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            if x.device == ref.device and x.dtype == ref.dtype:
                return x
            return x.to(device=ref.device, dtype=ref.dtype)
        return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _get_state_term(state_ctx: StateContext, key: str) -> Tensor:
        val = state_ctx.payload.get(key)
        if val is None:
            raise ValueError(f"NeuralSTPPEventModel requires state['{key}'].")
        return val

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state_ctx: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]],
        device,
    ) -> Dict[str, Tensor]:
        bsz = times.shape[0]
        max_len = int(lengths.max().item())
        if max_len < 2:
            zeros_per_seq = torch.zeros(bsz, device=device)
            zero_scalar = torch.tensor(0.0, device=device)
            empty_l = 0
            return {
                "loss": zero_scalar,
                "nll": zero_scalar,
                "nll_per_event": zeros_per_seq,
                "total_events": zero_scalar,
                "base_mean_nll": zero_scalar,
                "nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "mask": torch.zeros(bsz, empty_l, device=device),
                "temporal_nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "spatial_nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "temporal_energy_reg": zero_scalar,
                "spatial_reg": zero_scalar,
                "regularization_total": zero_scalar,
            }

        l_steps = max_len - 1
        temporal_nll = self._get_state_term(state_ctx, "temporal_nll_matrix")[:, :l_steps]
        z_seq = self._get_state_term(state_ctx, "z_seq")[:, :l_steps, :]

        t_seq = times[:, 1 : 1 + l_steps].unsqueeze(-1)
        s_seq = locations[:, 1 : 1 + l_steps, :]
        t_prev_seq = times[:, :l_steps].unsqueeze(-1)
        n_idx = torch.arange(l_steps, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

        spatial_nll = self.spatial_decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )

        nll_matrix = temporal_nll + spatial_nll
        nll_masked = nll_matrix * mask
        total_nll = nll_masked.sum(dim=1)
        n_events = mask.sum(dim=1)
        base_mean_nll = total_nll.sum() / n_events.sum().clamp(min=1)

        temporal_energy_reg_val = None
        if state_regularization_terms is not None:
            temporal_energy_reg_val = state_regularization_terms.get("temporal_energy_reg")
        if temporal_energy_reg_val is None:
            temporal_energy_reg_val = state_ctx.payload.get("temporal_energy_reg", 0.0)

        temporal_energy_reg_t = self._as_like(temporal_energy_reg_val, base_mean_nll)
        spatial_reg_t = self._as_like(
            getattr(self.spatial_decoder, "_energy_reg", 0.0), base_mean_nll
        )
        regularization_total = temporal_energy_reg_t + spatial_reg_t
        mean_nll = base_mean_nll + regularization_total

        return {
            "loss": mean_nll,
            "nll": mean_nll,
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events.sum(),
            "base_mean_nll": base_mean_nll,
            "nll_matrix": nll_matrix,
            "mask": mask,
            "temporal_nll_matrix": temporal_nll,
            "spatial_nll_matrix": spatial_nll,
            "temporal_energy_reg": temporal_energy_reg_t,
            "spatial_reg": spatial_reg_t,
            "regularization_total": regularization_total,
        }

    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        del x_field_at_events, marks
        if device is None:
            device = times.device
        return self._compute(
            times=times,
            locations=locations,
            lengths=lengths,
            state_ctx=state,
            state_regularization_terms=state_regularization_terms,
            device=device,
        )

    def eval_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        return self.training_loss(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            state_regularization_terms=state_regularization_terms,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )

    def intensity(
        self,
        *,
        state: StateContext,
        query_times: Tensor,       # (M, 1) normalized query times
        query_locations: Tensor,   # (M, d) normalized query locations
        query_lengths=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Tensor:                   # (M,) joint intensity λ*(t, s | H)
        """Evaluate λ*(t, s | H) = λ(t | H) × p(s | t, H).

        Runs the ODE from h_post_final at t_last to t_query to get the exact
        temporal intensity, then appends the query as event T+1 to get the
        conditional spatial log-density. Faithful to the original NeuralSTPP.
        """
        if device is None:
            device = query_times.device

        # ── 1. h(t_query) and λ(t_query) via state closure ──────────────── #
        t_q_tensor = query_times[0:1, :].to(device)   # (1, 1)
        h_query, lambda_q = state.payload["_h_at_query"](t_q_tensor)
        # h_query: (1, h),  lambda_q: (1,) scalar

        # ── 2. Build z_aug for conditional_logprob_fn ────────────────────── #
        # Skip event 0 to match the training regime.  During training,
        # t_seq = times[:, 1:L] so the JumpCNF inner loop at j=0 always
        # conditions on event 1 (NOT event 0).  Including event 0 here would
        # apply an extra untrained radial flow centred at s_0, producing a
        # spurious spike in the surface.  The fix is to use events 1..T-1 as
        # the conditioning history (length T-1), making the augmented sequence
        # length T (= query appended to T-1 history events), exactly the
        # length seen at position T in training.
        times_h   = state.payload["times"].to(device)        # (1, T_pad)
        locs_h    = state.payload["locations"].to(device)    # (1, T_pad, d)
        lengths_h = state.payload["lengths"].to(device)      # (1,)
        z_seq_h   = state.payload["z_seq"].to(device)        # (1, T-1, h)

        T = int(lengths_h[0].item())
        # Events 1..T-1 (skip event 0)
        event_times_1d = times_h[0, 1:T]    # (T-1,)
        event_locs_2d  = locs_h[0, 1:T]     # (T-1, d)

        if T <= 1:
            # 0 or 1 history events: after skipping event 0, no history remains
            z_aug = h_query                                           # (1, h)
        else:
            z_aug = torch.cat([
                z_seq_h[0, :T - 1],      # (T-1, h): h_pre at events 1..T-1
                h_query,                 # (1, h): h(t_query)
            ], dim=0)                                                  # (T, h)

        # ── 3. Conditional spatial log-density ───────────────────────────── #
        t_q_val = float(query_times[0, 0].item())
        logprob_fn = self.spatial_decoder.conditional_logprob_fn(
            t_query=t_q_val,
            event_times=event_times_1d,
            event_locs=event_locs_2d,
            z_aug=z_aug,
        )
        log_p_s = logprob_fn(query_locations.to(device))   # (M,) log p(s | t, H)

        # ── 4. Joint intensity ───────────────────────────────────────────── #
        return lambda_q * torch.exp(log_p_s)                # (M,)

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: "torch.Tensor",
        grid_locs: "torch.Tensor",
        **kwargs,
    ) -> "torch.Tensor":
        """Surface query contract: routes to intensity() for NeuralSTPP."""
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            device=grid_times.device,
        )
