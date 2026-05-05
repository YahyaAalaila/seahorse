"""EventModel for upstream-faithful DeepSTPP fixed-window likelihoods."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event


@register_event("deep_stpp")
class DeepSTPPEventModel(EventModel):
    """DeepSTPP event model over paper-style fixed windows."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        field_cov_dim: int = 0,
        seq_len: int = 20,
        lookahead: int = 1,
        num_points: int = 20,
        sigma_min: float = 1e-4,
        n_layers: int = 3,
        expose_decoded_params: bool = False,
        **dec_extra,
    ):
        super().__init__()
        from ..spatial_models.hawkes_gaussian import HawkesGaussianDecoder

        self.hawkes_decoder = HawkesGaussianDecoder(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            field_cov_dim=field_cov_dim,
            seq_len=seq_len,
            num_points=num_points,
            sigma_min=sigma_min,
            n_layers=n_layers,
            **dec_extra,
        )
        self.seq_len = int(seq_len)
        self.lookahead = int(lookahead)
        self.num_points = int(num_points)
        self.spatial_dim = int(spatial_dim)
        self.expose_decoded_params = bool(expose_decoded_params)

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="exact NLL",
            nll_kind="exact",
            nll_description="exact joint NLL/event (paper DeepSTPP window semantics)",
            supports_raw_reporting=True,
            raw_nll_description="exact joint NLL/event (raw/original data space; DeepSTPP paper transform corrected)",
            has_intensity=True,
            exposes_eventwise_terms=True,
        )

    @staticmethod
    def _get_state_term(state_ctx: StateContext, key: str) -> Tensor:
        val = state_ctx.payload.get(key)
        if val is None:
            raise ValueError(f"DeepSTPPEventModel requires state['{key}'].")
        if not isinstance(val, Tensor):
            raise TypeError(f"DeepSTPPEventModel expects tensor for state['{key}'].")
        return val

    @staticmethod
    def _broadcast_or_match(tensor: Tensor, batch_size: int, name: str) -> Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise ValueError(
            f"DeepSTPPEventModel state['{name}'] has batch={tensor.shape[0]} "
            f"but query batch={batch_size}."
        )

    @staticmethod
    def _pad_by_sequence(
        flat_values: Tensor,
        seq_ids: Tensor,
        counts: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
        out = flat_values.new_zeros((batch_size, max_count) + flat_values.shape[1:])
        mask = flat_values.new_zeros(batch_size, max_count)
        if flat_values.shape[0] == 0 or max_count == 0:
            return out, mask

        cursor = torch.zeros(batch_size, dtype=torch.long, device=seq_ids.device)
        for i in range(flat_values.shape[0]):
            seq_idx = int(seq_ids[i].item())
            pos = int(cursor[seq_idx].item())
            out[seq_idx, pos] = flat_values[i]
            mask[seq_idx, pos] = 1.0
            cursor[seq_idx] += 1
        return out, mask

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state_ctx: StateContext,
        device,
    ) -> Dict[str, Tensor]:
        del times, locations

        if self.lookahead != 1:
            raise ValueError(
                f"DeepSTPPEventModel currently supports lookahead=1, got {self.lookahead}."
            )

        bsz = int(lengths.shape[0])
        paper_st_x = self._get_state_term(state_ctx, "paper_st_x")
        paper_st_y = self._get_state_term(state_ctx, "paper_st_y")
        seq_ids = self._get_state_term(state_ctx, "paper_seq_ids").long()
        counts = self._get_state_term(state_ctx, "paper_window_counts").long()
        n_windows = int(paper_st_x.shape[0])

        if n_windows == 0:
            zeros_per_seq = torch.zeros(bsz, device=device)
            zero_scalar = torch.tensor(0.0, device=device, dtype=torch.float32)
            extra_metrics = self.raw_reporting_metrics(
                state=state_ctx,
                nll=zero_scalar,
                temporal_nll=zero_scalar,
                spatial_nll=zero_scalar,
                total_events=zero_scalar,
            )
            return {
                "loss": zero_scalar,
                "nll": zero_scalar,
                "nll_per_event": zeros_per_seq,
                "total_events": zero_scalar,
                "sll": zero_scalar,
                "tll": zero_scalar,
                "nll_matrix": torch.zeros(bsz, 0, device=device),
                "sll_matrix": torch.zeros(bsz, 0, device=device),
                "tll_matrix": torch.zeros(bsz, 0, device=device),
                "mask": torch.zeros(bsz, 0, device=device),
                "extra_metrics": extra_metrics,
            }

        z_flat = self._get_state_term(state_ctx, "z")
        target = paper_st_y[:, 0, :]
        t_flat = target[:, 2:3]
        s_flat = target[:, : self.spatial_dim]

        w_i, b_i, _sigma, inv_var = self.hawkes_decoder.decode(z_flat)
        t_hist = torch.cumsum(paper_st_x[:, :, 2], dim=-1)
        s_hist = paper_st_x[:, :, : self.spatial_dim]
        tn_ti_h = t_hist[:, -1:].sub(t_hist)
        tn_ti_bg = torch.zeros(
            n_windows,
            self.num_points,
            device=device,
            dtype=t_hist.dtype,
        )
        tn_ti = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        t_ti = (tn_ti + t_flat).clamp(min=1e-6)

        background = self.hawkes_decoder.background
        if background is not None:
            bg = background.to(device=device, dtype=s_hist.dtype).unsqueeze(0).expand(
                n_windows, -1, -1
            )
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist
        s_diff = s_flat.unsqueeze(1) - centers

        tll_flat = self.hawkes_decoder.log_ft(w_i, b_i, tn_ti, t_ti)
        sll_flat = self.hawkes_decoder.log_s_intensity(w_i, b_i, t_ti, s_diff, inv_var)
        nll_flat = -(tll_flat + sll_flat)

        nll_matrix, mask = self._pad_by_sequence(nll_flat, seq_ids, counts, bsz)
        sll_matrix, _ = self._pad_by_sequence(sll_flat, seq_ids, counts, bsz)
        tll_matrix, _ = self._pad_by_sequence(tll_flat, seq_ids, counts, bsz)

        counts_float = counts.to(device=device, dtype=nll_flat.dtype)
        per_seq_total = nll_flat.new_zeros(bsz)
        per_seq_total.scatter_add_(0, seq_ids, nll_flat)
        nll_per_event = per_seq_total / counts_float.clamp(min=1.0)

        mean_nll = nll_flat.mean()
        sll = sll_flat.mean()
        tll = tll_flat.mean()

        out = {
            "loss": mean_nll,
            "nll": mean_nll,
            "temporal_nll": float((-tll).item()),
            "spatial_nll": float((-sll).item()),
            "nll_per_event": nll_per_event,
            "total_events": counts.sum().to(device=device, dtype=mean_nll.dtype),
            "sll": sll,
            "tll": tll,
            "nll_matrix": nll_matrix,
            "sll_matrix": sll_matrix,
            "tll_matrix": tll_matrix,
            "mask": mask,
            "extra_metrics": self.raw_reporting_metrics(
                state=state_ctx,
                nll=mean_nll,
                temporal_nll=-tll,
                spatial_nll=-sll,
            ),
        }
        kl_loss = state_ctx.payload.get("kl_loss")
        if isinstance(kl_loss, Tensor):
            out["kl_loss"] = kl_loss
        if self.expose_decoded_params:
            out["w_i"], _ = self._pad_by_sequence(w_i, seq_ids, counts, bsz)
            out["b_i"], _ = self._pad_by_sequence(b_i, seq_ids, counts, bsz)
            out["inv_var"], _ = self._pad_by_sequence(inv_var, seq_ids, counts, bsz)
        return out

    def training_loss(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms=None,
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
        state_regularization_terms=None,
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
                "DeepSTPPEventModel.intensity expects query_locations with shape (B, d)."
            )
        if device is None:
            device = query_times.device

        batch_size = query_times.shape[0]
        query_z = self._broadcast_or_match(
            self._get_state_term(state, "query_z"),
            batch_size,
            "query_z",
        ).to(device=device)
        query_st_x = self._broadcast_or_match(
            self._get_state_term(state, "query_st_x"),
            batch_size,
            "query_st_x",
        ).to(device=device)
        query_last_time_raw = self._broadcast_or_match(
            self._get_state_term(state, "query_last_time_raw"),
            batch_size,
            "query_last_time_raw",
        ).to(device=device, dtype=query_times.dtype)

        input_normalized_flag = (
            float(self._get_state_term(state, "input_normalized_flag").reshape(-1)[0].item())
            > 0.5
        )
        input_time_mean = self._get_state_term(state, "input_time_mean").to(
            device=device,
            dtype=query_times.dtype,
        )
        input_time_std = self._get_state_term(state, "input_time_std").to(
            device=device,
            dtype=query_times.dtype,
        ).clamp(min=1e-8)
        input_loc_mean = self._get_state_term(state, "input_loc_mean").to(
            device=device,
            dtype=query_locations.dtype,
        )
        input_loc_std = self._get_state_term(state, "input_loc_std").to(
            device=device,
            dtype=query_locations.dtype,
        ).clamp(min=1e-8)
        paper_dt_min = self._get_state_term(state, "paper_dt_min").to(
            device=device,
            dtype=query_times.dtype,
        )
        paper_dt_range = self._get_state_term(state, "paper_dt_range").to(
            device=device,
            dtype=query_times.dtype,
        ).clamp(min=1e-8)
        paper_loc_min = self._get_state_term(state, "paper_loc_min").to(
            device=device,
            dtype=query_locations.dtype,
        )
        paper_loc_range = self._get_state_term(state, "paper_loc_range").to(
            device=device,
            dtype=query_locations.dtype,
        ).clamp(min=1e-8)

        if input_normalized_flag:
            query_times_raw = query_times * input_time_std + input_time_mean
            query_locations_raw = query_locations * input_loc_std + input_loc_mean
        else:
            query_times_raw = query_times
            query_locations_raw = query_locations

        query_dt = query_times_raw - query_last_time_raw
        query_dt_paper = (query_dt - paper_dt_min) / paper_dt_range
        query_locs_paper = (query_locations_raw - paper_loc_min) / paper_loc_range

        d = query_locations.shape[-1]
        t_hist = torch.cumsum(query_st_x[:, :, 2], dim=-1)
        s_hist = query_st_x[:, :, :d]
        tn_ti_h = t_hist[:, -1:].sub(t_hist)
        tn_ti_bg = torch.zeros(batch_size, self.num_points, device=device, dtype=tn_ti_h.dtype)
        tn_ti = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        t_ti = (tn_ti + query_dt_paper).clamp(min=1e-6)

        background = self.hawkes_decoder.background
        if background is not None:
            bg = background.to(device=device, dtype=s_hist.dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist
        s_diff = query_locs_paper.unsqueeze(1) - centers

        w_i, b_i, _sigma, inv_var = self.hawkes_decoder.decode(query_z)
        log_v_i = torch.log(w_i) - b_i * t_ti
        log_lamb_t = torch.logsumexp(log_v_i, dim=-1)
        log_v_norm = log_v_i - log_lamb_t.unsqueeze(-1)
        log_gauss = (
            0.5 * inv_var.prod(dim=-1).clamp(min=1e-12).log()
            - (d / 2.0) * math.log(2.0 * math.pi)
            - 0.5 * (s_diff.pow(2) * inv_var).sum(dim=-1)
        )
        log_spatial = torch.logsumexp(log_v_norm + log_gauss, dim=-1)
        return torch.exp(log_lamb_t + log_spatial)

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: torch.Tensor,
        grid_locs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            device=grid_times.device,
        )
