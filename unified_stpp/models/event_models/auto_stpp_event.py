"""Event model for the AutoSTPP preset."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event
from .auto_stpp_kernel import AutoSTPPCuboid


@register_event("auto_stpp")
class AutoSTPPEventModel(EventModel):
    """Exact upstream-style AutoSTPP over fixed paper windows."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        n_prodnet: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        activation: str = "tanh",
        bias: bool = True,
        lookback: int = 20,
        lookahead: int = 1,
        trunc: bool = False,
        max_history: int = 20,
        temporal_diag_mode: str = "exact",
        temporal_mc_samples: int = 10,
        report_orig_space_metrics: bool = True,
        **_,
    ):
        super().__init__()
        if int(spatial_dim) != 2:
            raise ValueError(
                f"AutoSTPP event model requires spatial_dim=2, got {spatial_dim}."
            )
        self.kernel = AutoSTPPCuboid(
            n_prodnet=int(n_prodnet),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            activation=str(activation),
            bias=bool(bias),
        )
        self.background = torch.nn.Parameter(torch.ones(1))
        self.lookback = int(lookback)
        self.lookahead = int(lookahead)
        self.trunc = bool(trunc)
        self.max_history = int(max_history)
        self.temporal_diag_mode = str(temporal_diag_mode).lower()
        self.temporal_mc_samples = int(temporal_mc_samples)
        self.report_orig_space_metrics = bool(report_orig_space_metrics)
        if self.temporal_diag_mode not in {"exact", "mc"}:
            raise ValueError(
                "temporal_diag_mode must be 'exact' or 'mc', "
                f"got {temporal_diag_mode!r}."
            )
        self.project_parameters()

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="exact NLL",
            nll_kind="exact",
            nll_description="exact joint NLL/event (upstream AutoSTPP Cuboid, normalized space)",
            supports_raw_reporting=True,
            raw_nll_description="exact joint NLL/event (raw/original data space; AutoSTPP paper transform corrected)",
            has_intensity=True,
            exposes_eventwise_terms=True,
        )

    @staticmethod
    def _get_state_term(state_ctx: StateContext, key: str) -> Tensor:
        val = state_ctx.payload.get(key)
        if val is None:
            raise ValueError(f"AutoSTPPEventModel requires state['{key}'].")
        if not isinstance(val, Tensor):
            raise TypeError(
                f"AutoSTPPEventModel expects tensor for state['{key}']."
            )
        return val

    @staticmethod
    def _broadcast_or_match(tensor: Tensor, batch_size: int, name: str) -> Tensor:
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise ValueError(
            f"AutoSTPPEventModel state['{name}'] has batch={tensor.shape[0]} "
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

    def project_parameters(self) -> None:
        self.kernel.project()

    def _background_rate(self, *, device, dtype) -> Tensor:
        return torch.exp(self.background).to(device=device, dtype=dtype)

    def _temporal_intensity_exact(
        self,
        s_x: Tensor,
        t_diff: Tensor,
        *,
        device,
        dtype,
    ) -> Tensor:
        n_windows, lookback, _ = s_x.shape
        lamb_t = self.kernel.lamb_t_stpp(
            s_x.reshape(-1, 2),
            t_diff.reshape(-1, 1),
        ).view(n_windows, lookback)
        return lamb_t.sum(-1) + self._background_rate(device=device, dtype=dtype)

    def _temporal_intensity_mc(
        self,
        s_x: Tensor,
        t_diff: Tensor,
        *,
        device,
        dtype,
    ) -> Tensor:
        n_samples = max(1, self.temporal_mc_samples)
        n_windows, lookback, _ = s_x.shape
        rand_locs = torch.rand(
            n_samples,
            n_windows,
            lookback,
            2,
            device=device,
            dtype=dtype,
        ) - s_x.unsqueeze(0)
        t_part = t_diff.unsqueeze(0).unsqueeze(-1).expand(n_samples, -1, -1, 1)
        st_diff = torch.cat([rand_locs, t_part], dim=-1)
        lamb_t = self.kernel.forward(st_diff.reshape(-1, 3)).view(
            n_samples,
            n_windows,
            lookback,
        )
        lamb_t = lamb_t.mean(0).sum(-1)
        return lamb_t + self._background_rate(device=device, dtype=dtype)

    def _orig_space_metrics(
        self,
        *,
        state_ctx: StateContext,
        mean_nll: Tensor,
        sll: Tensor,
        tll: Tensor,
        total_events: Tensor | None = None,
        device,
        dtype,
    ) -> dict[str, float]:
        if not self.report_orig_space_metrics:
            return {}
        raw_metrics = self.raw_reporting_metrics(
            state=state_ctx,
            nll=mean_nll,
            temporal_nll=-tll,
            spatial_nll=-sll,
            total_events=total_events,
        )
        if not raw_metrics:
            paper_loc_range = self._get_state_term(state_ctx, "paper_loc_range").to(
                device=device,
                dtype=dtype,
            ).clamp(min=1e-8)
            paper_dt_range = self._get_state_term(state_ctx, "paper_dt_range").to(
                device=device,
                dtype=dtype,
            ).clamp(min=1e-8)
            log_s = torch.log(torch.prod(paper_loc_range))
            log_t = torch.log(paper_dt_range.reshape(-1)[0])
            raw_metrics = {
                "raw_space_nll": float((mean_nll + log_s + log_t).item()),
                "raw_space_spatial_nll": float((-sll + log_s).item()),
                "raw_space_temporal_nll": float((-tll + log_t).item()),
            }
        raw_metrics.update(
            {
                "orig_space_nll": raw_metrics["raw_space_nll"],
                "orig_space_spatial_nll": raw_metrics["raw_space_spatial_nll"],
                "orig_space_temporal_nll": raw_metrics["raw_space_temporal_nll"],
            }
        )
        return raw_metrics

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
                f"AutoSTPPEventModel currently supports lookahead=1, got {self.lookahead}."
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
            extra_metrics = self._orig_space_metrics(
                state_ctx=state_ctx,
                mean_nll=zero_scalar,
                sll=zero_scalar,
                tll=zero_scalar,
                total_events=zero_scalar,
                device=device,
                dtype=zero_scalar.dtype,
            )
            return {
                "loss": zero_scalar,
                "nll": zero_scalar,
                "temporal_nll": 0.0,
                "spatial_nll": 0.0,
                "nll_per_event": zeros_per_seq,
                "total_events": zero_scalar,
                "sll": zero_scalar,
                "tll": zero_scalar,
                "nll_matrix": torch.zeros(bsz, 0, device=device),
                "sll_matrix": torch.zeros(bsz, 0, device=device),
                "tll_matrix": torch.zeros(bsz, 0, device=device),
                "mask": torch.zeros(bsz, 0, device=device),
                "lambs_sum": torch.zeros(bsz, 0, device=device),
                "lamb_t": torch.zeros(bsz, 0, device=device),
                "lamb_ints": torch.zeros(bsz, 0, device=device),
                "background_rate": self._background_rate(
                    device=device,
                    dtype=zero_scalar.dtype,
                ),
                "extra_metrics": extra_metrics,
            }

        t_x_cum = torch.cumsum(paper_st_x[..., -1], dim=-1)
        t_diff = t_x_cum[:, -1:] - t_x_cum + paper_st_y[..., -1:, -1]
        s_x = paper_st_x[..., :2]
        s_y = paper_st_y[:, 0, :2]
        s_diff = s_y.unsqueeze(1) - s_x
        st_diff = torch.cat([s_diff, t_diff.unsqueeze(-1)], dim=-1)

        lambs = self.kernel.forward(st_diff.reshape(-1, 3)).view(n_windows, self.lookback)
        lambs_sum = lambs.sum(-1) + self._background_rate(device=device, dtype=lambs.dtype)
        if not torch.all(lambs_sum > 0):
            raise ValueError("AutoSTPP intensity became non-positive.")

        if self.temporal_diag_mode == "exact":
            lamb_t = self._temporal_intensity_exact(
                s_x,
                t_diff,
                device=device,
                dtype=lambs.dtype,
            )
        else:
            lamb_t = self._temporal_intensity_mc(
                s_x,
                t_diff,
                device=device,
                dtype=lambs.dtype,
            )
        lamb_ints = self.kernel.int_lamb_stpp(
            s_x.reshape(-1, 2),
            (t_x_cum[:, -1:] - t_x_cum).reshape(-1, 1),
            t_diff.reshape(-1, 1),
        ).view(n_windows, self.lookback)
        lamb_ints = lamb_ints.sum(-1)
        background_int = paper_st_y[:, 0, -1] * self._background_rate(
            device=device,
            dtype=lambs.dtype,
        )
        lamb_ints = lamb_ints + background_int

        ll_flat = torch.log(lambs_sum) - lamb_ints
        tll_flat = torch.log(lamb_t) - lamb_ints
        sll_flat = ll_flat - tll_flat
        nll_flat = -ll_flat

        nll_matrix, mask = self._pad_by_sequence(nll_flat, seq_ids, counts, bsz)
        sll_matrix, _ = self._pad_by_sequence(sll_flat, seq_ids, counts, bsz)
        tll_matrix, _ = self._pad_by_sequence(tll_flat, seq_ids, counts, bsz)
        lambs_sum_matrix, _ = self._pad_by_sequence(lambs_sum, seq_ids, counts, bsz)
        lamb_t_matrix, _ = self._pad_by_sequence(lamb_t, seq_ids, counts, bsz)
        lamb_ints_matrix, _ = self._pad_by_sequence(lamb_ints, seq_ids, counts, bsz)

        counts_float = counts.to(device=device, dtype=nll_flat.dtype)
        per_seq_total = nll_flat.new_zeros(bsz)
        per_seq_total.scatter_add_(0, seq_ids, nll_flat)
        nll_per_event = per_seq_total / counts_float.clamp(min=1.0)

        mean_nll = nll_flat.mean()
        sll = sll_flat.mean()
        tll = tll_flat.mean()

        return {
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
            "lambs_sum": lambs_sum_matrix,
            "lamb_t": lamb_t_matrix,
            "lamb_ints": lamb_ints_matrix,
            "background_rate": self._background_rate(
                device=device,
                dtype=mean_nll.dtype,
            ),
            "extra_metrics": self._orig_space_metrics(
                state_ctx=state_ctx,
                mean_nll=mean_nll,
                sll=sll,
                tll=tll,
                total_events=counts.sum().to(device=device, dtype=mean_nll.dtype),
                device=device,
                dtype=mean_nll.dtype,
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
                "AutoSTPPEventModel.intensity expects query_locations with shape (B, d)."
            )
        if device is None:
            device = query_times.device

        batch_size = query_times.shape[0]
        base_history_times_raw = self._get_state_term(state, "times_raw")
        base_history_locs_scaled = self._get_state_term(state, "paper_history_scaled")
        base_history_lengths = self._get_state_term(state, "lengths").long()

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

        query_locations_scaled = (query_locations_raw - paper_loc_min) / paper_loc_range
        background = self._background_rate(device=device, dtype=query_times.dtype).reshape(1)

        # Notebook-style surface queries broadcast one fixed history across many
        # spatial grid points. Handle that case in one vectorized kernel call.
        if (
            not self.trunc
            and base_history_times_raw.shape[0] == 1
            and base_history_locs_scaled.shape[0] == 1
            and base_history_lengths.shape[0] == 1
        ):
            n_events = int(base_history_lengths[0].item())
            if n_events <= 0:
                return background.expand(batch_size).to(dtype=torch.float32)

            times_raw = base_history_times_raw[0, :n_events].to(
                device=device,
                dtype=query_times.dtype,
            )
            locs_scaled = base_history_locs_scaled[0, :n_events, :2].to(
                device=device,
                dtype=query_locations.dtype,
            )
            valid = times_raw.unsqueeze(0) < query_times_raw[:, 0:1]
            s_diff = query_locations_scaled.unsqueeze(1) - locs_scaled.unsqueeze(0)
            t_diff = (
                (query_times_raw[:, 0:1] - times_raw.unsqueeze(0)).clamp(min=0.0)
                / paper_dt_range
            )
            st_diff = torch.cat([s_diff, t_diff.unsqueeze(-1)], dim=-1)
            lamb = self.kernel.forward(st_diff.reshape(-1, 3)).view(batch_size, n_events)
            values = lamb.mul(valid.to(dtype=lamb.dtype)).sum(-1) + background
            return values.to(dtype=torch.float32)

        history_times_raw = self._broadcast_or_match(
            base_history_times_raw,
            batch_size,
            "times_raw",
        ).to(device=device, dtype=query_times.dtype)
        history_locs_scaled = self._broadcast_or_match(
            base_history_locs_scaled,
            batch_size,
            "paper_history_scaled",
        ).to(device=device, dtype=query_locations.dtype)
        history_lengths = self._broadcast_or_match(
            base_history_lengths,
            batch_size,
            "lengths",
        ).to(device=device)

        values = torch.zeros(batch_size, device=device, dtype=query_times.dtype)
        for b in range(batch_size):
            n_events = int(history_lengths[b].item())
            if n_events <= 0:
                values[b] = background
                continue

            times_raw = history_times_raw[b, :n_events]
            locs_scaled = history_locs_scaled[b, :n_events, :2]
            valid = times_raw < query_times_raw[b, 0]
            if self.trunc and self.max_history > 0 and int(valid.sum().item()) > self.max_history:
                keep = torch.nonzero(valid, as_tuple=False).reshape(-1)[-self.max_history :]
                valid = torch.zeros_like(valid, dtype=torch.bool)
                valid[keep] = True
            if not bool(valid.any()):
                values[b] = background
                continue

            valid_locs_scaled = locs_scaled[valid]
            valid_times_raw = times_raw[valid]
            s_diff = query_locations_scaled[b : b + 1] - valid_locs_scaled
            t_diff = (query_times_raw[b, 0] - valid_times_raw).clamp(min=0.0) / paper_dt_range
            st_diff = torch.cat([s_diff, t_diff.unsqueeze(-1)], dim=-1)
            values[b] = self.kernel.forward(st_diff).sum() + background

        return values.to(dtype=torch.float32)

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: Tensor,
        grid_locs: Tensor,
        **kwargs,
    ) -> Tensor:
        del kwargs
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            device=grid_times.device,
        )
