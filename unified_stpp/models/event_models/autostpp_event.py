"""
EventModel wrapper for AutoSTPP joint-intensity likelihood.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from ..abstractions import EventModel


NLLFn = Callable[[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]], Tensor]
LogProbFn = Callable[[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]], Tensor]
CompensatorFn = Callable[[Tensor, Tensor], Tensor]
MuFn = Callable[[], Tensor]


class AutoSTPPEventModel(EventModel):
    """
    AutoSTPP event model over current AutoInt decoder semantics.

    The model is treated as a unified joint-intensity model. For explicit
    diagnostics, we expose:
      - lambs_sum: joint intensity λ*(t, s | H)
      - lamb_t: equal to joint intensity under current repo semantics
      - lamb_ints: compensator term
    """

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

    @staticmethod
    def _get_state_term(state: Dict[str, Any], key: str) -> Tensor:
        val = state.get(key)
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
        state: Dict[str, Any],
        device,
    ) -> Dict[str, Tensor]:
        B = times.shape[0]
        max_len = int(lengths.max().item())
        if max_len < 2:
            zeros_per_seq = torch.zeros(B, device=device)
            zero_scalar = torch.tensor(0.0, device=device)
            empty_l = 0
            return {
                "nll": zero_scalar,
                "nll_per_event": zeros_per_seq,
                "total_events": zero_scalar,
                "sll": zero_scalar,
                "tll": zero_scalar,
                "nll_matrix": torch.zeros(B, empty_l, device=device),
                "sll_matrix": torch.zeros(B, empty_l, device=device),
                "tll_matrix": torch.zeros(B, empty_l, device=device),
                "mask": torch.zeros(B, empty_l, device=device),
                "lambs_sum": torch.zeros(B, empty_l, device=device),
                "lamb_t": torch.zeros(B, empty_l, device=device),
                "lamb_ints": torch.zeros(B, empty_l, device=device),
            }

        L = max_len - 1
        z_seq = state.get("z_seq")
        if isinstance(z_seq, Tensor):
            all_states = z_seq[:, :L, :]
        else:
            all_states = self._get_state_term(state, "all_states")[:, :L, :]
        t_target = times[:, 1 : 1 + L].unsqueeze(-1)
        s_target = locations[:, 1 : 1 + L, :]
        t_prev = times[:, :L].unsqueeze(-1)

        n_idx = torch.arange(L, device=device)
        mask = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()

        h = all_states.shape[-1]
        d = s_target.shape[-1]
        z_flat = all_states.reshape(B * L, h)
        t_flat = t_target.reshape(B * L, 1)
        s_flat = s_target.reshape(B * L, d)
        t_prev_flat = t_prev.reshape(B * L, 1)
        tau_flat = (t_flat - t_prev_flat).clamp(min=1e-6)

        nll_flat = self._nll_fn(z_flat, t_flat, s_flat, t_prev_flat, None)
        nll_matrix = nll_flat.reshape(B, L)

        log_lamb_flat = self._log_prob_fn(z_flat, t_flat, s_flat, t_prev_flat, None)
        lambs_sum = torch.exp(log_lamb_flat).reshape(B, L)
        lamb_ints = self._compensator_fn(z_flat, tau_flat).reshape(B, L)

        # AutoSTPP is represented as a unified joint-intensity model in this repo.
        # We expose a degenerate decomposition for compatibility:
        #   tll := joint ll, sll := 0.
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

    def nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: Dict[str, Any],
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
            state=state,
            device=device,
        )

    def sequence_nll(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: Dict[str, Any],
        x_field_at_events: Optional[Tensor] = None,
        marks: Optional[Tensor] = None,
        device=None,
    ) -> Dict[str, Tensor]:
        return self.nll(
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
            x_field_at_events=x_field_at_events,
            marks=marks,
            device=device,
        )
