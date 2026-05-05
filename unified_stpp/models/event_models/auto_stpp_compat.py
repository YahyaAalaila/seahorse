"""EventModel for the compatibility AutoSTPP preset."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event


@register_event("auto_stpp_legacy")
class AutoSTPPCompatEventModel(EventModel):
    """Compatibility AutoSTPP event model with a MonotoneIntegralDecoder."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        field_cov_dim: int = 0,
        n_components: int = 8,
        n_layers: int = 2,
        internal_dim: int = 64,
        x_lo: float = -3.5,
        x_hi: float = 3.5,
        y_lo: float = -3.5,
        y_hi: float = 3.5,
        **dec_extra,
    ):
        super().__init__()
        from ..temporal_models.monotone_integral import MonotoneIntegralDecoder

        self.integral_decoder = MonotoneIntegralDecoder(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
            n_components=n_components,
            n_layers=n_layers,
            internal_dim=internal_dim,
            x_lo=x_lo,
            x_hi=x_hi,
            y_lo=y_lo,
            y_hi=y_hi,
            **dec_extra,
        )

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="exact NLL",
            nll_kind="exact",
            nll_description="exact joint NLL/event (AutoInt monotone integral)",
            supports_raw_reporting=True,
            raw_nll_description=(
                "exact joint NLL/event (raw/original data space; AutoInt transform corrected)"
            ),
            has_intensity=True,
            exposes_eventwise_terms=True,
        )

    @staticmethod
    def _get_state_term(state_ctx: StateContext, key: str) -> Tensor:
        val = state_ctx.payload.get(key)
        if val is None:
            raise ValueError(f"AutoSTPPCompatEventModel requires state['{key}'].")
        if not isinstance(val, Tensor):
            raise TypeError(f"AutoSTPPCompatEventModel expects tensor for state['{key}'].")
        return val

    @staticmethod
    def _broadcast_or_match(tensor: Tensor, batch_size: int, name: str) -> Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise ValueError(
            f"AutoSTPPCompatEventModel state['{name}'] has batch={tensor.shape[0]} "
            f"but query batch={batch_size}."
        )

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state_ctx: StateContext,
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
                "sll": zero_scalar,
                "tll": zero_scalar,
                "nll_matrix": torch.zeros(bsz, empty_l, device=device),
                "sll_matrix": torch.zeros(bsz, empty_l, device=device),
                "tll_matrix": torch.zeros(bsz, empty_l, device=device),
                "mask": torch.zeros(bsz, empty_l, device=device),
                "lambs_sum": torch.zeros(bsz, empty_l, device=device),
                "lamb_t": torch.zeros(bsz, empty_l, device=device),
                "lamb_ints": torch.zeros(bsz, empty_l, device=device),
            }

        l_steps = max_len - 1
        z_seq = state_ctx.payload.get("z_seq")
        if isinstance(z_seq, Tensor):
            all_states = z_seq[:, :l_steps, :]
        else:
            all_states = self._get_state_term(state_ctx, "all_states")[:, :l_steps, :]
        t_target = times[:, 1 : 1 + l_steps].unsqueeze(-1)
        s_target = locations[:, 1 : 1 + l_steps, :]
        t_prev = times[:, :l_steps].unsqueeze(-1)

        n_idx = torch.arange(l_steps, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

        h = all_states.shape[-1]
        d = s_target.shape[-1]
        z_flat = all_states.reshape(bsz * l_steps, h)
        t_flat = t_target.reshape(bsz * l_steps, 1)
        s_flat = s_target.reshape(bsz * l_steps, d)
        t_prev_flat = t_prev.reshape(bsz * l_steps, 1)
        tau_flat = (t_flat - t_prev_flat).clamp(min=1e-6)

        nll_flat = self.integral_decoder.nll(z_flat, t_flat, s_flat, t_prev_flat, None)
        nll_matrix = nll_flat.reshape(bsz, l_steps)

        log_lamb_flat = self.integral_decoder.log_prob(z_flat, t_flat, s_flat, t_prev_flat, None)
        lambs_sum = torch.exp(log_lamb_flat).reshape(bsz, l_steps)
        lamb_ints = self.integral_decoder.compensator(z_flat, tau_flat).reshape(bsz, l_steps)

        tll_matrix = -nll_matrix
        sll_matrix = torch.zeros_like(tll_matrix)

        nll_masked = nll_matrix * mask
        total_nll = nll_masked.sum(dim=1)
        n_events = mask.sum(dim=1)
        n_events_total = n_events.sum().clamp(min=1)

        mean_nll = total_nll.sum() / n_events_total
        tll = (tll_matrix * mask).sum() / n_events_total
        sll = torch.zeros_like(tll)

        return {
            "loss": mean_nll,
            "nll": mean_nll,
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events.sum(),
            "sll": sll,
            "tll": tll,
            "nll_matrix": nll_matrix,
            "sll_matrix": sll_matrix,
            "tll_matrix": tll_matrix,
            "mask": mask,
            "lambs_sum": lambs_sum,
            "lamb_t": lambs_sum,
            "lamb_ints": lamb_ints,
            "background_rate": self.integral_decoder.mu().to(device=device, dtype=mean_nll.dtype),
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
        del state_regularization_terms, x_field_at_events, marks
        if device is None:
            device = times.device
        return self._compute(
            times=times,
            locations=locations,
            lengths=lengths,
            state_ctx=state,
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
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths: Optional[Tensor] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Tensor:
        del query_lengths, x_field_at_events, marks
        if query_times.ndim == 1:
            query_times = query_times.unsqueeze(-1)
        if query_locations.ndim == 3 and query_locations.shape[1] == 1:
            query_locations = query_locations.squeeze(1)
        if query_locations.ndim != 2:
            raise ValueError(
                "AutoSTPPCompatEventModel.intensity expects query_locations with shape (B, d)."
            )
        if device is None:
            device = query_times.device

        batch_size = query_times.shape[0]
        history_times = self._broadcast_or_match(
            self._get_state_term(state, "times"), batch_size, "times"
        ).to(device=device, dtype=query_times.dtype)
        history_lengths = self._broadcast_or_match(
            self._get_state_term(state, "lengths").long(), batch_size, "lengths"
        ).to(device=device)

        z_final = state.payload.get("z_final")
        if isinstance(z_final, Tensor):
            z_hist = self._broadcast_or_match(z_final, batch_size, "z_final").to(device=device)
        else:
            z_seq = self._broadcast_or_match(
                self._get_state_term(state, "z_seq"), batch_size, "z_seq"
            ).to(device=device)
            idx = (history_lengths - 1).clamp(min=0)
            b_idx = torch.arange(batch_size, device=device)
            z_hist = z_seq[b_idx, idx, :]

        prev_idx = (history_lengths - 1).clamp(min=0)
        b_idx = torch.arange(batch_size, device=device)
        t_prev = history_times[b_idx, prev_idx].unsqueeze(-1)

        log_lamb = self.integral_decoder.log_prob(
            z_hist,
            query_times.to(device=device, dtype=t_prev.dtype),
            query_locations.to(device=device),
            t_prev,
            None,
        )
        return torch.exp(log_lamb)

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: "torch.Tensor",
        grid_locs: "torch.Tensor",
        **kwargs,
    ) -> "torch.Tensor":
        """Surface query contract: routes to intensity() for AutoSTPP."""
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            device=grid_times.device,
        )
