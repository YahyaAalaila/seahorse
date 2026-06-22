"""Event-side contract for the shared Neural STPP family."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from seahorse.data.transforms import transform_from_spec
from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event

@register_event("neural_stpp")
class NeuralSTPPEventModel(EventModel):
    """NeuralSTPP event model.  Owns the spatial decoder."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        field_cov_dim: int = 0,
        spatial_type: str = "jump_cnf",
        temporal_hdim: Optional[int] = None,
        **spatial_extra,
    ):
        super().__init__()
        from ..model_registry import get_spatial_cls
        self.temporal_hdim = temporal_hdim
        self.spatial_decoder = get_spatial_cls(spatial_type)(
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            field_cov_dim=field_cov_dim,
            **spatial_extra,
        )

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="exact NLL",
            objective_includes_regularization=True,
            nll_kind="exact",
            nll_description="exact joint NLL/event (raw time, standardized space)",
            supports_raw_reporting=True,
            raw_nll_description="exact joint NLL/event (raw/original data space; NeuralSTPP spatial transform corrected)",
            has_intensity=True,
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

    def _spatial_sequence_inputs(self, state_ctx: StateContext) -> Tensor:
        if bool(getattr(self.spatial_decoder, "USES_NEURAL_AUX_STATE", False)):
            return self._get_state_term(state_ctx, "spatial_aux_seq")
        return self._get_state_term(state_ctx, "temporal_hidden_seq")

    @staticmethod
    def _query_locations_native(state_ctx: StateContext, query_locations: Tensor) -> Tensor:
        spec = state_ctx.payload.get("input_transform")
        transform = transform_from_spec(spec if isinstance(spec, dict) else None)
        if transform is None:
            return query_locations
        lengths = query_locations.new_full((query_locations.shape[0],), 1, dtype=torch.long)
        return transform.forward_locations(query_locations, lengths)

    def fixed_time_query_terms(
        self,
        *,
        state: StateContext,
        query_time: Tensor | float,
        device=None,
    ) -> Dict[str, Any]:
        """Return the faithful Neural-STPP factors at one fixed query time.

        The helper stays in model space:
        - ``query_time`` follows the same convention as ``intensity()``, i.e.
          normalized time when the family was trained with normalized inputs.
        - the returned ``logprob_fn`` expects standardized-space query points.

        This is intentionally minimal and family-specific. Temporary
        visualization code can use it to recover:
        - ``lambda_t``           — raw-time conditional intensity
        - ``logprob_fn(s_norm)`` — log p(s_norm | t, H)
        """
        if device is None:
            if isinstance(query_time, Tensor):
                device = query_time.device
            else:
                times_ref = state.payload.get("times")
                device = times_ref.device if isinstance(times_ref, Tensor) else torch.device("cpu")

        if isinstance(query_time, Tensor):
            query_time_t = query_time.to(device=device, dtype=torch.float32).reshape(1, 1)
        else:
            query_time_t = torch.tensor([[float(query_time)]], dtype=torch.float32, device=device)

        normalize_time_inputs = bool(state.payload.get("normalize_time_inputs", False))
        if normalize_time_inputs:
            time_mean = float(state.payload.get("time_mean", 0.0))
            time_std = float(state.payload.get("time_std", 1.0))
            query_time_raw = query_time_t * time_std + time_mean
        else:
            query_time_raw = query_time_t

        h_query, lambda_q = state.payload["_h_at_query_raw"](query_time_raw)

        times_h = state.payload.get("times_raw", state.payload["times"]).to(device)
        locs_h = state.payload.get("locations_norm", state.payload["locations"]).to(device)
        lengths_h = state.payload["lengths"].to(device)
        z_seq_h = self._spatial_sequence_inputs(state).to(device)

        t_hist = int(lengths_h[0].item())
        event_times_1d = times_h[0, :t_hist]
        event_locs_2d = locs_h[0, :t_hist]

        if bool(getattr(self.spatial_decoder, "USES_NEURAL_AUX_STATE", False)):
            aux_dim = int(state.payload.get("spatial_aux_dim", 0))
            h_query_for_spatial = h_query[:, -aux_dim:] if aux_dim > 0 else h_query[:, :0]
        else:
            h_query_for_spatial = h_query

        z_aug = torch.cat([z_seq_h[0, :t_hist], h_query_for_spatial], dim=0)
        t_q_val = float(query_time_raw[0, 0].item())
        logprob_fn = self.spatial_decoder.conditional_logprob_fn(
            t_query=t_q_val,
            event_times=event_times_1d,
            event_locs=event_locs_2d,
            z_aug=z_aug,
        )
        return {
            "lambda_t": lambda_q.reshape(()),
            "query_time_raw": t_q_val,
            "logprob_fn": logprob_fn,
        }

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
        if max_len < 1:
            zeros_per_seq = torch.zeros(bsz, device=device)
            zero_scalar = torch.tensor(0.0, device=device)
            empty_l = 0
            return {
                "loss": zero_scalar,
                "nll": zero_scalar,
                "temporal_nll": 0.0,
                "spatial_nll": 0.0,
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
                "extra_metrics": self.raw_reporting_metrics(
                    state=state_ctx,
                    nll=zero_scalar,
                    temporal_nll=zero_scalar,
                    spatial_nll=zero_scalar,
                    total_events=zero_scalar,
                ),
            }

        l_steps = max_len
        temporal_nll_matrix = self._get_state_term(state_ctx, "temporal_nll_matrix")[:, :l_steps]
        z_seq = self._spatial_sequence_inputs(state_ctx)[:, :l_steps, :]
        times_raw = state_ctx.payload.get("times_raw", times)
        locations_norm = state_ctx.payload.get("locations_norm", locations)
        t0_raw = state_ctx.payload.get(
            "t0_raw",
            torch.zeros(bsz, 1, device=device, dtype=times.dtype),
        )

        t_seq = times_raw[:, :l_steps].unsqueeze(-1)
        s_seq = locations_norm[:, :l_steps, :]
        if l_steps > 1:
            t_prev_seq = torch.cat([t0_raw, times_raw[:, : l_steps - 1]], dim=1).unsqueeze(-1)
        else:
            t_prev_seq = t0_raw.unsqueeze(-1)
        n_idx = torch.arange(l_steps, device=device)
        mask = (n_idx.unsqueeze(0) < lengths.unsqueeze(1)).float()

        spatial_nll_matrix = self.spatial_decoder.sequence_nll(
            z_seq=z_seq,
            t_seq=t_seq,
            s_seq=s_seq,
            t_prev_seq=t_prev_seq,
            lengths=lengths,
            mask=mask,
        )

        nll_matrix = temporal_nll_matrix + spatial_nll_matrix
        nll_masked = nll_matrix * mask
        total_nll = nll_masked.sum(dim=1)
        n_events = mask.sum(dim=1)
        n_events_total = n_events.sum().clamp(min=1)
        base_mean_nll = total_nll.sum() / n_events_total

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

        # Scalar NLL breakdowns (mean per event, no regularization).
        temporal_nll_mean = (temporal_nll_matrix * mask).sum() / n_events_total
        spatial_nll_mean = (spatial_nll_matrix * mask).sum() / n_events_total

        return {
            # loss = pure NLL + regularization (backprop target).
            "loss": base_mean_nll + regularization_total,
            # nll = pure NLL only — the selection metric and benchmark metric.
            "nll": base_mean_nll,
            "temporal_nll": float(temporal_nll_mean.item()),   # per-event breakdown
            "spatial_nll": float(spatial_nll_mean.item()),     # per-event breakdown
            "nll_per_event": total_nll / n_events.clamp(min=1),
            "total_events": n_events_total,
            "base_mean_nll": base_mean_nll,
            "nll_matrix": nll_matrix,
            "mask": mask,
            "temporal_nll_matrix": temporal_nll_matrix,
            "spatial_nll_matrix": spatial_nll_matrix,
            "temporal_energy_reg": temporal_energy_reg_t,
            "spatial_reg": spatial_reg_t,
            "regularization_total": regularization_total,
            "extra_metrics": self.raw_reporting_metrics(
                state=state_ctx,
                nll=base_mean_nll,
                temporal_nll=temporal_nll_mean,
                spatial_nll=spatial_nll_mean,
            ),
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
        query_locations: Tensor,   # (M, d) standardized query locations
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
        del query_lengths, x_field_at_events, marks
        if device is None:
            device = query_times.device
        q = query_times.to(device=device, dtype=torch.float32)
        if q.ndim == 1:
            q = q.unsqueeze(-1)
        q_flat = q.reshape(-1)
        if q_flat.numel() == 0:
            return torch.zeros(0, device=device, dtype=torch.float32)
        if not torch.allclose(q_flat, q_flat[0].expand_as(q_flat), atol=1e-7, rtol=1e-6):
            raise ValueError(
                "NeuralSTPPEventModel.intensity() expects a fixed query time per call. "
                "Use repeated query_times for one time-slice evaluation."
            )

        terms = self.fixed_time_query_terms(
            state=state,
            query_time=q_flat[0],
            device=device,
        )
        q_locs = self._query_locations_native(state, query_locations.to(device))
        log_p_s = terms["logprob_fn"](q_locs)
        lambda_t = self._as_like(terms["lambda_t"], log_p_s)
        return lambda_t * torch.exp(log_p_s)

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
