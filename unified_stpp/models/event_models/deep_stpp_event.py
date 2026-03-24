"""EventModel for DeepSTPP — owns HawkesGaussianDecoder."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext


def _sliding_history_windows_with_times(
    times: Tensor, locations: Tensor, l_steps: int, seq_len: int
) -> Tensor:
    """Build history windows in DeepSTPP layout [times | flattened locations]."""
    bsz, _, d = locations.shape
    first_t = times[:, :1].expand(-1, seq_len - 1)
    padded_t = torch.cat([first_t, times[:, : l_steps + 1]], dim=1)
    first_s = locations[:, :1, :].expand(-1, seq_len - 1, -1)
    padded_s = torch.cat([first_s, locations[:, : l_steps + 1, :]], dim=1)

    t_windows = torch.stack([padded_t[:, n : n + seq_len] for n in range(l_steps)], dim=1)
    s_windows = torch.stack([padded_s[:, n : n + seq_len, :] for n in range(l_steps)], dim=1)
    return torch.cat([t_windows, s_windows.reshape(bsz, l_steps, seq_len * d)], dim=-1)


class DeepSTPPEventModel(EventModel):
    """DeepSTPP event model.  Owns HawkesGaussianDecoder."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        spatial_dim: int,
        field_cov_dim: int = 0,
        seq_len: int = 20,
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
        self.num_points = int(num_points)
        self.spatial_dim = int(spatial_dim)
        self.expose_decoded_params = bool(expose_decoded_params)

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

    def _build_query_history_x_field(
        self,
        *,
        history_times: Tensor,
        history_locations: Tensor,
        history_lengths: Tensor,
    ) -> Tensor:
        """Build DeepSTPP history payload [times | flattened locations] per query."""
        bsz = history_times.shape[0]
        d = history_locations.shape[-1]
        out_times = torch.empty(
            bsz, self.seq_len, device=history_times.device, dtype=history_times.dtype
        )
        out_locs = torch.empty(
            bsz,
            self.seq_len,
            d,
            device=history_locations.device,
            dtype=history_locations.dtype,
        )
        for b in range(bsz):
            n = int(history_lengths[b].item())
            if n < 1:
                out_times[b] = torch.zeros(self.seq_len, device=history_times.device, dtype=history_times.dtype)
                out_locs[b] = torch.zeros(self.seq_len, d, device=history_locations.device, dtype=history_locations.dtype)
                continue
            t_hist = history_times[b, :n]
            s_hist = history_locations[b, :n, :]
            if n >= self.seq_len:
                t_win = t_hist[n - self.seq_len : n]
                s_win = s_hist[n - self.seq_len : n]
            else:
                t_pad = t_hist[:1].expand(self.seq_len - n)
                s_pad = s_hist[:1].expand(self.seq_len - n, d)
                t_win = torch.cat([t_pad, t_hist], dim=0)
                s_win = torch.cat([s_pad, s_hist], dim=0)
            out_times[b] = t_win
            out_locs[b] = s_win
        return torch.cat([out_times, out_locs.reshape(bsz, self.seq_len * d)], dim=-1)

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
            }

        l_steps = max_len - 1
        z_all = self._get_state_term(state_ctx, "z")
        z_cond = z_all[:, :l_steps, :]

        t_target = times[:, 1 : 1 + l_steps].unsqueeze(-1)
        s_target = locations[:, 1 : 1 + l_steps, :]
        t_prev = times[:, :l_steps].unsqueeze(-1)

        n_idx = torch.arange(l_steps, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

        hist_windows = _sliding_history_windows_with_times(
            times, locations, l_steps, self.seq_len
        )
        x_field_flat = hist_windows.reshape(bsz * l_steps, -1)

        h = z_cond.shape[-1]
        d = s_target.shape[-1]
        z_flat = z_cond.reshape(bsz * l_steps, h)
        t_flat = t_target.reshape(bsz * l_steps, 1)
        s_flat = s_target.reshape(bsz * l_steps, d)
        t_prev_flat = t_prev.reshape(bsz * l_steps, 1)

        w_i, b_i, _sigma, inv_var = self.hawkes_decoder.decode(z_flat)

        t_hist = x_field_flat[:, : self.seq_len]
        s_hist = x_field_flat[:, self.seq_len :].reshape(bsz * l_steps, self.seq_len, d)
        tn_ti_h = (t_prev_flat - t_hist).clamp(min=0.0)
        tn_ti_bg = torch.zeros(bsz * l_steps, self.num_points, device=device)
        tn_ti = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        dt = (t_flat - t_prev_flat).clamp(min=1e-6).reshape(bsz * l_steps)
        t_ti = (tn_ti + dt.unsqueeze(-1)).clamp(min=1e-6)

        background = self.hawkes_decoder.background
        if background is not None:
            bg = background.unsqueeze(0).expand(bsz * l_steps, -1, -1)
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist
        s_diff = s_flat.unsqueeze(1) - centers

        tll_flat = self.hawkes_decoder.log_ft(w_i, b_i, tn_ti, t_ti)
        sll_flat = self.hawkes_decoder.log_s_intensity(w_i, b_i, t_ti, s_diff, inv_var)
        tll_matrix = tll_flat.reshape(bsz, l_steps)
        sll_matrix = sll_flat.reshape(bsz, l_steps)
        nll_matrix = -(tll_matrix + sll_matrix)

        nll_masked = nll_matrix * mask
        total_nll = nll_masked.sum(dim=1)
        n_events = mask.sum(dim=1)
        n_events_total = n_events.sum().clamp(min=1)
        mean_nll = total_nll.sum() / n_events_total
        sll = (sll_matrix * mask).sum() / n_events_total
        tll = (tll_matrix * mask).sum() / n_events_total

        out = {
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
        }
        kl_loss = state_ctx.payload.get("kl_loss")
        if isinstance(kl_loss, Tensor):
            out["kl_loss"] = kl_loss
        if self.expose_decoded_params:
            out["w_i"] = w_i.reshape(bsz, l_steps, -1)
            out["b_i"] = b_i.reshape(bsz, l_steps, -1)
            out["inv_var"] = inv_var.reshape(
                bsz, l_steps, inv_var.shape[1], inv_var.shape[2]
            )
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
        history_times = self._broadcast_or_match(
            self._get_state_term(state, "times"), batch_size, "times"
        ).to(device=device, dtype=query_times.dtype)
        history_locations = self._broadcast_or_match(
            self._get_state_term(state, "locations"), batch_size, "locations"
        ).to(device=device, dtype=query_locations.dtype)
        history_lengths = self._broadcast_or_match(
            self._get_state_term(state, "lengths").long(), batch_size, "lengths"
        ).to(device=device)

        z_final = state.payload.get("z_final")
        if isinstance(z_final, Tensor):
            z_hist = self._broadcast_or_match(z_final, batch_size, "z_final").to(device=device)
        else:
            z_all = self._broadcast_or_match(
                self._get_state_term(state, "all_states"), batch_size, "all_states"
            ).to(device=device)
            idx = (history_lengths - 1).clamp(min=0)
            b_idx = torch.arange(batch_size, device=device)
            z_hist = z_all[b_idx, idx, :]

        b_idx = torch.arange(batch_size, device=device)
        prev_idx = (history_lengths - 1).clamp(min=0)
        if history_times.shape[1] > 0:
            t_prev = history_times[b_idx, prev_idx].unsqueeze(-1)
        else:
            t_prev = torch.zeros(batch_size, 1, device=device, dtype=query_times.dtype)

        x_field = self._build_query_history_x_field(
            history_times=history_times,
            history_locations=history_locations,
            history_lengths=history_lengths,
        )
        d = query_locations.shape[-1]
        t_hist = x_field[:, : self.seq_len]
        s_hist = x_field[:, self.seq_len :].reshape(batch_size, self.seq_len, d)

        tn_ti_h = (t_prev - t_hist).clamp(min=0.0)
        tn_ti_bg = torch.zeros(
            batch_size,
            self.num_points,
            device=device,
            dtype=tn_ti_h.dtype,
        )
        tn_ti = torch.cat([tn_ti_h, tn_ti_bg], dim=-1)
        dt = (query_times - t_prev).clamp(min=1e-6)
        t_ti = (tn_ti + dt).clamp(min=1e-6)

        background = self.hawkes_decoder.background
        if background is not None:
            bg = background.to(device=device, dtype=s_hist.dtype).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            centers = torch.cat([s_hist, bg], dim=1)
        else:
            centers = s_hist
        s_diff = query_locations.unsqueeze(1) - centers

        w_i, b_i, _sigma, inv_var = self.hawkes_decoder.decode(z_hist)
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
        grid_times: "torch.Tensor",
        grid_locs: "torch.Tensor",
        **kwargs,
    ) -> "torch.Tensor":
        """Surface query contract: routes to intensity() for DeepSTPP."""
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            device=grid_times.device,
        )
