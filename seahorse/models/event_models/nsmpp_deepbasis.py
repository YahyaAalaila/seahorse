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
        interval_int_res: int = 2,
        interval_chunk_size: int = 4096,
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
        # Quadrature nodes used per inter-event interval by ``_interval_compensators``
        # (composite midpoint rule), and the query-chunk size used to bound peak memory.
        self.interval_int_res = max(1, int(interval_int_res))
        self.interval_chunk_size = max(1, int(interval_chunk_size))

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
        # Area of one cell of the spatial quadrature grid. ``unit_vol`` folds in the
        # global time step as well; the per-interval compensator supplies its own time
        # weights, so it needs the purely spatial cell measure.
        spatial_cell_area = float(spatial_range.prod().item()) / float(
            self.int_res ** self.spatial_dim
        )

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
        self.register_buffer(
            "spatial_cell_area", torch.tensor(spatial_cell_area, dtype=torch.float32)
        )
        self.register_buffer("support_t0_value", torch.tensor(t0, dtype=torch.float32))

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

    def _interval_compensators(self, event_vectors: Tensor, valid_mask: Tensor) -> Tensor:
        """Per-event compensator increments, shape ``(batch, seq_len)``.

        Entry ``[b, j]`` is

            Lambda_j = int_{t_{j-1}}^{t_j} int_S lambda(u, s | H_u) ds du,

        with ``t_{-1} = support_t0``, i.e. the mass the model assigns to "no event
        anywhere in space" over the waiting interval that ends at event ``j``.

        Why this attribution
        --------------------
        ``_numerical_compensator`` returns a per-sequence total over the whole support
        box, which cannot be split after the fact: there is no way to recover from a
        single scalar how much of the integrated mass belongs to each waiting interval.
        Any post-hoc split (uniform, proportional-to-gap, ...) would be an invented
        quantity, not a likelihood term. So the decomposition is done at the source
        instead, by integrating each interval on its own.

        This is exactly the quantity the other exact families put in ``nll_matrix``:
        the survival integral over ``(t_{j-1}, t_j]`` (compare the
        ``survival_integral = background_integral + offspring_integral`` term in
        ``parametric_kernels.py``).
        Summing the increments recovers the compensator over ``(support_t0, t_N]``,
        so ``-log lambda_j + Lambda_j`` summed over events is the sequence NLL under
        the framework-wide last-event convention, and every per-event term depends
        only on the strict history of event ``j``. That causality is what makes the
        eventwise terms agree with an independent prefix-difference recomputation.

        The integral is a composite midpoint rule with ``interval_int_res`` nodes per
        interval in time, over the same spatial grid the sequence-level compensator
        uses. Because the intervals are the model's own inter-event gaps, this is a
        strictly finer time discretization than the fixed ``int_res``-node global grid
        whenever a sequence has more than ``int_res`` events, which is the normal case.
        """
        if not self.numerical_int:
            raise NotImplementedError("NSMPP DeepBasis currently supports numerical_int=True.")

        batch_size, seq_len, _ = event_vectors.shape
        device = event_vectors.device
        dtype = event_vectors.dtype
        if seq_len == 0:
            return event_vectors.new_zeros((batch_size, 0))

        mask_f = valid_mask.to(dtype=dtype)
        times = event_vectors[..., 0]
        t0 = self.support_t0_value.to(device=device, dtype=dtype)
        previous_times = torch.cat(
            [times.new_full((batch_size, 1), 0.0) + t0, times[:, :-1]], dim=1
        )
        widths = (times - previous_times).clamp_min(0.0) * mask_f  # (B, T)

        # Node j of the query grid integrates over (t_{j-1}, t_j], where the
        # conditioning history is exactly the events with index < j. Using the index
        # prefix rather than a time comparison keeps this identical to the causal mask
        # used by ``_event_intensities`` (strictly-earlier events only).
        prefix_counts = torch.arange(seq_len, device=device, dtype=torch.long)
        prefix_counts = prefix_counts.unsqueeze(0).expand(batch_size, seq_len)

        spatial_grid = self.support_spatial_grid.to(device=device, dtype=dtype)
        n_grid = int(spatial_grid.shape[0])
        cell_area = self.spatial_cell_area.to(device=device, dtype=dtype)

        n_nodes = self.interval_int_res
        offsets = (
            torch.arange(n_nodes, device=device, dtype=dtype) + 0.5
        ) / float(n_nodes)

        spatial_totals = event_vectors.new_zeros((batch_size, seq_len))
        n_queries = seq_len * n_grid
        flat_index = torch.arange(n_queries, device=device, dtype=torch.long)
        for offset in offsets:
            node_times = previous_times + widths * offset  # (B, T)
            for start in range(0, n_queries, self.interval_chunk_size):
                chunk = flat_index[start : start + self.interval_chunk_size]
                event_idx = torch.div(chunk, n_grid, rounding_mode="floor")
                grid_idx = chunk - event_idx * n_grid
                t_chunk = node_times[:, event_idx].unsqueeze(-1)  # (B, C, 1)
                s_chunk = spatial_grid[grid_idx].unsqueeze(0).expand(batch_size, -1, -1)
                query_events = torch.cat([t_chunk, s_chunk], dim=-1)
                kernel_sum = self.kernel.sum_over_prefix_history(
                    query_events,
                    prefix_counts[:, event_idx],
                    event_vectors,
                    valid_mask,
                )
                lam_chunk = F.softplus(kernel_sum + self.raw_mu) + self.intensity_eps
                spatial_totals = spatial_totals.index_add(1, event_idx, lam_chunk)

        # Each node carries time weight ``width / n_nodes``; spatial sums carry the
        # spatial cell measure.
        return spatial_totals * cell_area * widths / float(n_nodes)

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
                "next_event_mask": event_vectors.new_zeros(valid_mask.shape),
            }

        mask_f = valid_mask.to(dtype=event_vectors.dtype)
        intensities = self._event_intensities(event_vectors, valid_mask)
        log_intensities = torch.log(intensities)
        # Eventwise exact NLL: -log lambda(t_j, s_j | H_j) + the compensator increment
        # over the waiting interval (t_{j-1}, t_j] that this event terminates. Both
        # terms must be here: the benchmark scores from ``nll_matrix``, and every other
        # exact family (the parametric Hawkes/Poisson kernels) reports the
        # same two-term quantity. Carrying only -log lambda understates the NLL by
        # roughly one nat per event, since at a likelihood optimum the compensator over
        # the observation window is approximately the event count.
        interval_compensator = self._interval_compensators(event_vectors, valid_mask)
        event_nll_matrix = (-log_intensities + interval_compensator) * mask_f
        seq_nll = event_nll_matrix.sum(dim=1)
        nll_per_event = seq_nll / lengths.to(dtype=event_vectors.dtype).clamp(min=1.0)
        sequence_mean_nll = seq_nll.mean()
        per_event_nll = seq_nll.sum() / total_events.clamp(min=1.0)

        # The first event terminates the interval starting at ``support_t0`` rather
        # than at a previous event, so it is not a held-out next-event target. This
        # matches the ``next_event_mask`` convention in ``factorized.py``.
        next_event_mask = mask_f.clone()
        next_event_mask[:, 0] = 0.0

        return {
            "total_events": total_events,
            "sequence_mean_nll": sequence_mean_nll,
            "per_event_nll": per_event_nll,
            "nll_per_event": nll_per_event,
            "mask": mask_f,
            "nll_matrix": event_nll_matrix,
            "next_event_mask": next_event_mask,
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
            "next_event_mask": terms["next_event_mask"],
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
