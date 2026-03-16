"""EventModel wrapper for AutoSTPP joint-intensity likelihood."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext


NLLFn = Callable[[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]], Tensor]
LogProbFn = Callable[[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]], Tensor]
CompensatorFn = Callable[[Tensor, Tensor], Tensor]
MuFn = Callable[[], Tensor]


class AutoSTPPEventModel(EventModel):
    """AutoSTPP event model over current AutoInt decoder semantics."""

    def __init__(
        self,
        *,
        nll_fn: NLLFn,
        log_prob_fn: LogProbFn,
        compensator_fn: CompensatorFn,
        mu_fn: MuFn,
    ):
        super().__init__()
        self._nll_fn = nll_fn
        self._log_prob_fn = log_prob_fn
        self._compensator_fn = compensator_fn
        self._mu_fn = mu_fn

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
            raise ValueError(f"AutoSTPPEventModel requires state['{key}'].")
        if not isinstance(val, Tensor):
            raise TypeError(f"AutoSTPPEventModel expects tensor for state['{key}'].")
        return val

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

        nll_flat = self._nll_fn(z_flat, t_flat, s_flat, t_prev_flat, None)
        nll_matrix = nll_flat.reshape(bsz, l_steps)

        log_lamb_flat = self._log_prob_fn(z_flat, t_flat, s_flat, t_prev_flat, None)
        lambs_sum = torch.exp(log_lamb_flat).reshape(bsz, l_steps)
        lamb_ints = self._compensator_fn(z_flat, tau_flat).reshape(bsz, l_steps)

        # AutoSTPP is represented as a unified joint-intensity model in this repo.
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
            # Under current repo semantics this equals the joint intensity.
            "lamb_t": lambs_sum,
            "lamb_ints": lamb_ints,
            "background_rate": self._mu_fn().to(device=device, dtype=mean_nll.dtype),
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
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        state_regularization_terms: Optional[Dict[str, Tensor]] = None,
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Tensor:
        out = self.eval_nll(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            state_regularization_terms=state_regularization_terms,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )
        return out["lambs_sum"]
