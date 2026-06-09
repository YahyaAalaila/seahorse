"""Direct joint-intensity event model for the NSMPP DeepBasis preset."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event
from .nsmpp_deepbasis_kernel import DeepBasisKernel


@register_event("nsmpp_deepbasis")
class NSMPPDeepBasisEventModel(EventModel):
    """Direct conditional intensity over the full event vector."""

    def __init__(
        self,
        *,
        spatial_dim: int,
        mu: float = 1.0,
        n_basis: int = 5,
        basis_dim: int = 10,
        nn_width: int = 10,
        int_res: int = 20,
        numerical_int: bool = True,
        init_gain: float = 1.5,
        init_bias: float = 0.0,
        init_std: float = 1.0,
        init_weight_mean: float = -3.0,
        intensity_eps: float = 1e-5,
        support_t0: float = 0.0,
        support_t1: float = 1.0,
        support_space_min: tuple[float, float] = (0.0, 0.0),
        support_space_max: tuple[float, float] = (1.0, 1.0),
        compensator_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.spatial_dim = int(spatial_dim)
        if self.spatial_dim != 2:
            raise ValueError(
                f"NSMPPDeepBasisEventModel currently requires spatial_dim=2, got {self.spatial_dim}."
            )

        self.data_dim = 1 + self.spatial_dim
        # Learnable background rate stored as an unconstrained parameter.
        # Store in softplus-inverse space so softplus(raw_mu) == mu at init.
        _mu = max(float(mu), 1e-6)
        _raw_mu_init = math.log(math.expm1(_mu))  # softplus_inverse(mu)
        self.raw_mu = torch.nn.Parameter(torch.tensor(_raw_mu_init, dtype=torch.float32))
        self.int_res = int(int_res)
        self.numerical_int = bool(numerical_int)
        self.intensity_eps = float(intensity_eps)
        self.compensator_chunk_size = max(1, int(compensator_chunk_size))

        self.kernel = DeepBasisKernel(
            n_basis=int(n_basis),
            data_dim=self.data_dim,
            basis_dim=int(basis_dim),
            init_gain=float(init_gain),
            init_bias=float(init_bias),
            init_std=float(init_std),
            init_weight_mean=float(init_weight_mean),
            nn_width=int(nn_width),
        )

        t0 = float(support_t0)
        t1 = float(support_t1)
        if t1 <= t0:
            t1 = t0 + 1e-6
        s_min = torch.tensor(support_space_min, dtype=torch.float32)
        s_max = torch.tensor(support_space_max, dtype=torch.float32)
        spatial_range = torch.clamp(s_max - s_min, min=1e-6)
        s_max = s_min + spatial_range
        support_volume = float((t1 - t0) * spatial_range.prod().item())
        unit_vol = support_volume / float(self.int_res ** self.data_dim)

        time_grid = torch.linspace(t0, t1, self.int_res, dtype=torch.float32)
        x_grid = torch.linspace(float(s_min[0].item()), float(s_max[0].item()), self.int_res, dtype=torch.float32)
        y_grid = torch.linspace(float(s_min[1].item()), float(s_max[1].item()), self.int_res, dtype=torch.float32)
        mesh_x, mesh_y = torch.meshgrid(x_grid, y_grid, indexing="ij")
        spatial_grid = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=-1)

        self.register_buffer("support_time_grid", time_grid)
        self.register_buffer("support_spatial_grid", spatial_grid)
        self.register_buffer("support_space_min", s_min)
        self.register_buffer("support_space_max", s_max)
        self.register_buffer("unit_vol", torch.tensor(unit_vol, dtype=torch.float32))

    @property
    def mu(self) -> Tensor:
        """Background rate in intensity space."""
        return F.softplus(self.raw_mu)

    @property
    def capabilities(self) -> EventCapabilities:
        return EventCapabilities(
            training_objective="nll_seq_mean",
            metric_key="objective",
            objective_description=(
                "exact negative log-likelihood "
                "(optimized as sequence mean; test NLL/event reported separately)"
            ),
            objective_includes_regularization=False,
            nll_kind="exact",
            nll_description="exact joint NLL/event (direct DeepBasis conditional intensity)",
            supports_raw_reporting=True,
            raw_nll_description="exact joint NLL/event (raw/original data space; NSMPP direct intensity)",
            has_intensity=True,
            exposes_eventwise_terms=True,
        )

    @staticmethod
    def _get_payload_tensor(state_ctx: StateContext, key: str) -> Tensor:
        val = state_ctx.payload.get(key)
        if not isinstance(val, Tensor):
            raise ValueError(f"NSMPPDeepBasisEventModel requires tensor state['{key}'].")
        return val

    def _conditional_intensity(
        self,
        *,
        query_events: Tensor,
        history_events: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        kernel_sum = self.kernel.sum_over_history(query_events, history_events, history_mask)
        # Single-softplus parameterization: softplus(raw_mu) matches the configured
        # zero-history background rate at initialization.
        raw = kernel_sum + self.raw_mu
        return F.softplus(raw) + self.intensity_eps

    def _event_intensities(self, event_vectors: Tensor, valid_mask: Tensor) -> Tensor:
        _, seq_len, _ = event_vectors.shape
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=event_vectors.device),
            diagonal=-1,
        ).unsqueeze(0)
        history_mask = valid_mask.unsqueeze(1) & causal
        return self._conditional_intensity(
            query_events=event_vectors,
            history_events=event_vectors,
            history_mask=history_mask,
        )

    def _numerical_compensator(self, event_vectors: Tensor, valid_mask: Tensor) -> Tensor:
        batch_size, _, _ = event_vectors.shape
        if not self.numerical_int:
            raise NotImplementedError("NSMPP DeepBasis currently supports numerical_int=True.")

        total = event_vectors.new_zeros(batch_size)
        history_times = event_vectors[..., 0]
        spatial_grid = self.support_spatial_grid.to(device=event_vectors.device, dtype=event_vectors.dtype)

        for t_val in self.support_time_grid.to(device=event_vectors.device, dtype=event_vectors.dtype):
            history_mask_t = valid_mask.unsqueeze(1) & (history_times.unsqueeze(1) <= t_val)
            for start in range(0, spatial_grid.shape[0], self.compensator_chunk_size):
                chunk = spatial_grid[start : start + self.compensator_chunk_size]
                chunk_size = int(chunk.shape[0])
                t_chunk = torch.full(
                    (batch_size, chunk_size, 1),
                    float(t_val.item()),
                    device=event_vectors.device,
                    dtype=event_vectors.dtype,
                )
                s_chunk = chunk.unsqueeze(0).expand(batch_size, -1, -1)
                query_events = torch.cat([t_chunk, s_chunk], dim=-1)
                lam_chunk = self._conditional_intensity(
                    query_events=query_events,
                    history_events=event_vectors,
                    history_mask=history_mask_t.expand(-1, chunk_size, -1),
                )
                total = total + lam_chunk.sum(dim=1)
        return total * self.unit_vol.to(device=event_vectors.device, dtype=event_vectors.dtype)

    def _compute_terms(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        device,
    ) -> Dict[str, Tensor]:
        del times, locations
        event_vectors = self._get_payload_tensor(state, "event_vectors").to(device)
        valid_mask = self._get_payload_tensor(state, "event_mask").to(device=device, dtype=torch.bool)
        lengths = self._get_payload_tensor(state, "lengths").to(device)

        total_events = lengths.sum().to(device=device, dtype=event_vectors.dtype)
        if int(total_events.item()) == 0:
            zero = event_vectors.new_zeros(())
            return {
                "total_events": zero,
                "sequence_mean_nll": zero,
                "per_event_nll": zero,
                "nll_per_event": zero.reshape(1).expand(lengths.shape[0]).clone(),
                "mask": valid_mask.to(dtype=event_vectors.dtype),
                "nll_matrix": event_vectors.new_zeros(valid_mask.shape),
            }

        intensities = self._event_intensities(event_vectors, valid_mask)
        log_intensities = torch.log(intensities)
        event_nll_matrix = -log_intensities * valid_mask.to(dtype=event_vectors.dtype)
        sumlog_per_seq = (log_intensities * valid_mask.to(dtype=event_vectors.dtype)).sum(dim=1)
        compensator = self._numerical_compensator(event_vectors, valid_mask)
        seq_nll = -(sumlog_per_seq - compensator)
        nll_per_event = seq_nll / lengths.to(dtype=event_vectors.dtype).clamp(min=1.0)
        sequence_mean_nll = seq_nll.mean()
        per_event_nll = seq_nll.sum() / total_events.clamp(min=1.0)

        return {
            "total_events": total_events,
            "sequence_mean_nll": sequence_mean_nll,
            "per_event_nll": per_event_nll,
            "nll_per_event": nll_per_event,
            "mask": valid_mask.to(dtype=event_vectors.dtype),
            "nll_matrix": event_nll_matrix,
        }

    def _pack_output(
        self,
        terms: Dict[str, Tensor],
        *,
        nll_value: Tensor,
        state: StateContext,
    ) -> Dict[str, Tensor]:
        extra_metrics = {
            "sequence_mean_nll": float(terms["sequence_mean_nll"].item()),
            "per_event_nll": float(terms["per_event_nll"].item()),
            "background_rate": float(self.mu.detach().item()),
            "support_t0": float(self.support_time_grid[0].item()),
            "support_t1": float(self.support_time_grid[-1].item()),
        }
        extra_metrics.update(
            self.raw_reporting_metrics(
                state=state,
                nll=nll_value,
                total_events=terms["total_events"],
            )
        )
        return {
            "loss": nll_value,
            "nll": nll_value,
            "total_events": terms["total_events"],
            "nll_per_event": terms["nll_per_event"],
            "mask": terms["mask"],
            "nll_matrix": terms["nll_matrix"],
            "extra_metrics": extra_metrics,
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
        terms = self._compute_terms(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            device=device,
        )
        return self._pack_output(terms, nll_value=terms["sequence_mean_nll"], state=state)

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
        del state_regularization_terms, x_field_at_events, marks
        if device is None:
            device = times.device
        terms = self._compute_terms(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            device=device,
        )
        return self._pack_output(terms, nll_value=terms["per_event_nll"], state=state)

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
        if device is None:
            device = query_times.device

        event_vectors = self._get_payload_tensor(state, "event_vectors").to(device)
        valid_mask = self._get_payload_tensor(state, "event_mask").to(device=device, dtype=torch.bool)
        history_times = event_vectors[..., 0]

        q_t = query_times.to(device=device, dtype=event_vectors.dtype)
        if q_t.ndim == 2 and q_t.shape[-1] == 1:
            q_t = q_t[:, 0]
        elif q_t.ndim != 1:
            raise ValueError(
                f"NSMPPDeepBasisEventModel.intensity expects query_times with shape (M,) or (M,1), got {tuple(query_times.shape)}."
            )
        q_s = query_locations.to(device=device, dtype=event_vectors.dtype)
        if q_s.ndim != 2 or q_s.shape[-1] != self.spatial_dim:
            raise ValueError(
                f"NSMPPDeepBasisEventModel.intensity expects query_locations with shape (M,{self.spatial_dim}), got {tuple(query_locations.shape)}."
            )

        query_batch = q_t.shape[0]
        state_batch = event_vectors.shape[0]
        if state_batch == 1:
            history = event_vectors.expand(query_batch, -1, -1)
            history_valid = valid_mask.expand(query_batch, -1)
            history_times = history_times.expand(query_batch, -1)
        elif state_batch == query_batch:
            history = event_vectors
            history_valid = valid_mask
        else:
            raise ValueError(
                f"NSMPPDeepBasisEventModel.intensity requires state batch size 1 or query batch size; got state={state_batch}, query={query_batch}."
            )

        query_events = torch.cat([q_t.unsqueeze(-1), q_s], dim=-1).unsqueeze(1)
        history_mask = history_valid.unsqueeze(1) & (history_times.unsqueeze(1) <= q_t.view(-1, 1, 1))
        lam = self._conditional_intensity(
            query_events=query_events,
            history_events=history,
            history_mask=history_mask,
        )
        return lam[:, 0]

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
            device=grid_locs.device,
        )
