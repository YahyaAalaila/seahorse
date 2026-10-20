"""EventModel for factorized STPP baselines.

Owns a temporal process model and a spatial model. Computes the factorized NLL:

    NLL = -(tll + sll) / n_events_total

where:
    tll = Σ_b temporal_logprob[b] / n_events_total   (per-event average)
    sll = Σ_{b,i} spatial_logprob[b,i] * mask[b,i] / n_events_total

Time convention — sequence-relative shifting:
    Parametric temporal models (Poisson, Hawkes, SelfCorrecting) require non-negative
    absolute times. The framework always passes z-score-normalized times which can be
    negative. To bridge this, FactorizedEventModel internally shifts times per-sequence:

        t_shifted[b, i] = times[b, i] - times[b, 0]   ≥ 0 always

    Temporal models receive t_shifted so that:
      - All times are non-negative
      - Time differences are unchanged (shift cancels)
      - The observation window is [t0, t1] in shifted coordinates

    Config parameters t0 and t1 are interpreted in this shifted frame:
      t0 = 0.0 (default) → window starts at first observed event
      t1 = None (default) → last-event convention: window ends at last observed event
      t1 = float          → fixed window end in shifted time (for right-censored data)

    The spatial model (GaussianMixtureSpatialModel) also receives shifted times.
    Its mixing weights use time differences (t_i - t_j), which are shift-invariant, so
    the shift has no semantic effect on the spatial log-probability.

    Under the raw-first data path, any reversible coordinate transform is
    applied by the state model first; this event model then works entirely in
    that family-owned native space.

Marks
-----
With ``n_marks > 0`` and a temporal model that implements ``marked_logprob``
(THP and RMTPP; see seahorse/models/temporal_models/neural_temporal.py) the model
emits a normalized categorical over mark types jointly with time and space:

    λ(t, y, k | H) = λ(t | H) · p(k | t, H) · f(y | t, k, H)

    NLL = -(tll + sll + mll) / n_events_total

where mll = Σ_{b,i} log p(k_{b,i} | t_{b,i}, H) · mask[b,i] / n_events_total.

Because p(k | ·) is a proper pmf, Σ_k λ(t, y, k) = λ(t) f(y | t), so the mark
factor leaves the ground-process compensator untouched and the factorisation is
exact. Whether the *joint* NLL is exact is then decided solely by the temporal
model's compensator, which is what ``capabilities.nll_kind`` reports.

Every mark branch is gated on ``n_marks > 0``: with ``n_marks=0`` — or with
unmarked data — this module is byte-for-byte its pre-mark self.

Keys added to the output dict when marked (nothing is added when unmarked), for
downstream accuracy / macro-F1 / ECE metrics:

    mark_logprob_matrix  (B, T, M)      log p(k | t_i, H_i)
    mark_logprob_events  (n_events, M)  padding-free view of the above
    mark_targets_events  (n_events,)    ground-truth mark per valid event
    mark_pred_events     (n_events,)    argmax prediction
    mark_nll_matrix      (B, T)         -log p(k_i), zero on padding
    mark_ll / mark_nll                  mean mark log-prob / NLL per event
    n_marks                             M
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from seahorse.data.transforms import transform_from_spec
from ..abstractions import EventCapabilities, EventModel, StateContext


class FactorizedEventModel(EventModel):
    """
    Factorized STPP event model.

    Owns:
        temporal_model : module with .logprob(times, locations, mask, t0, t1) → (B,)
        spatial_model  : module with .logprob(times, locations, mask) → (B, T)

    Args:
        temporal_model : parametric temporal process
        spatial_model  : parametric spatial density
        t0             : window start in sequence-relative (shifted) time (default 0.0 = first event)
        t1             : window end in shifted time; None → last-event convention (see module docstring)
        n_marks        : number of discrete mark types; 0 (default) = unmarked (see below)
    """

    def __init__(
        self,
        *,
        temporal_model: nn.Module,
        spatial_model: nn.Module,
        t0: float = 0.0,
        t1: Optional[float] = None,
        n_marks: int = 0,
    ):
        super().__init__()
        self.temporal_model = temporal_model
        self.spatial_model = spatial_model
        self.t0 = t0
        self.t1 = t1
        self.n_marks = int(n_marks)

    @property
    def _marked(self) -> bool:
        """True when this instance is configured for marks AND the temporal model
        can produce them. Every mark-related branch in this class is gated on this,
        so an ``n_marks=0`` model is byte-for-byte the pre-mark implementation."""
        return self.n_marks > 0 and callable(
            getattr(self.temporal_model, "marked_logprob", None)
        )

    @property
    def _temporal_event_logprob_fn(self):
        """The temporal model's per-event log-prob hook, or ``None`` if unsupported.

        A temporal process that only implements the sequence-level ``logprob`` cannot
        contribute a per-event term. In that case this model must not advertise
        eventwise terms — see ``capabilities`` and ``_compute``.
        """
        fn = getattr(self.temporal_model, "event_logprob_matrix", None)
        return fn if callable(fn) else None

    @property
    def capabilities(self) -> EventCapabilities:
        # Eventwise terms are only honest when the temporal process can supply a
        # per-event log-prob. Declaring them unconditionally is what let the
        # benchmark score a spatial-only quantity for the RMTPP/THP presets.
        exposes_eventwise = self._temporal_event_logprob_fn is not None
        # The compensator route is decided by the temporal process: analytic for
        # the parametric processes and RMTPP, numerical quadrature for THP (whose
        # evaluation uses deterministic midpoint nodes; training draws Monte Carlo).
        closed_form = getattr(self.temporal_model, "COMPENSATOR", "closed_form") == "closed_form"
        route = "closed-form" if closed_form else "numerical-quadrature"
        route_footnote = "" if closed_form else "§ numerical-quadrature compensator"

        if not self._marked:
            return EventCapabilities(
                training_objective="nll",
                metric_key="nll",
                objective_description="exact NLL" if closed_form else "NLL (quadrature compensator)",
                nll_kind="exact",
                nll_path="closed_form" if closed_form else "quadrature",
                nll_description=f"joint NLL/event (factorized temporal + spatial; {route} compensator)",
                nll_footnote=route_footnote,
                supports_raw_reporting=True,
                raw_nll_description=(
                    f"joint NLL/event (raw/original data space; factorized transform corrected; {route} compensator)"
                ),
                has_intensity=True,
                has_density=True,
                exposes_eventwise_terms=exposes_eventwise,
            )

        # Marked. The mark factor p(k|t,H) is an exactly normalised categorical in
        # both supported parametrisations, so the *factorisation* is exact; whether
        # the joint NLL is exact overall is decided by the temporal compensator.
        nll_kind = "exact" if closed_form else "approx"
        note = (
            ""
            if closed_form
            else " (compensator ∫λ estimated by Monte Carlo)"
        )
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description=f"{'exact' if closed_form else 'approximate'} marked NLL",
            nll_kind=nll_kind,
            nll_path="closed_form" if closed_form else "monte_carlo",
            nll_description=(
                f"{'exact' if closed_form else 'approximate'} joint marked NLL/event "
                f"(factorized temporal x mark x spatial){note}"
            ),
            nll_footnote="" if closed_form else "‡ MC compensator",
            supports_raw_reporting=closed_form,
            raw_nll_description=(
                "exact joint marked NLL/event (raw/original data space; factorized "
                "transform corrected; the mark factor is a pmf and needs no Jacobian)"
            ),
            has_intensity=True,
            has_density=True,
            exposes_eventwise_terms=exposes_eventwise,
        )

    def _extract_history(self, state: StateContext, device):
        """Extract history tensors from state and apply sequence-relative time shift.

        Returns (times_shifted, locations, mask) all on device, with B=1.
        The shift maps times[:, 0] → 0 so temporal models receive non-negative inputs.
        """
        times     = state.payload["times"].to(device)      # (B, T)
        locations = state.payload["locations"].to(device)  # (B, T, D)
        lengths   = state.payload["lengths"].to(device)    # (B,)

        B, T = times.shape
        idx  = torch.arange(T, device=device)
        mask = (idx.unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T)

        t_shift = times[:, 0:1] if T > 0 else torch.zeros(B, 1, device=device, dtype=times.dtype)
        times_shifted  = (times - t_shift).clamp(min=0.0)        # (B, T)

        return times_shifted, locations, mask, t_shift

    @staticmethod
    def _transform_queries(
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
    ) -> tuple[Tensor, Tensor]:
        spec = state.payload.get("input_transform")
        transform = transform_from_spec(spec if isinstance(spec, dict) else None)
        if transform is None:
            return query_times, query_locations
        lengths = query_times.new_full((query_times.shape[0],), 1, dtype=torch.long)
        q_t = transform.forward_times(query_times, lengths)
        q_s = transform.forward_locations(query_locations, lengths)
        return q_t, q_s

    def intensity(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Tensor:
        """Conditional intensity λ*(t_query, s_query | H) = f*(t|H) × p(s|t,H).

        The intensity is the product of:
          - f*(t|H): exact analytic temporal intensity from the parametric process
          - p(s|t,H): exact parametric spatial density from the spatial model

        Args:
            state          : StateContext with payload {times, locations, lengths}
            query_times    : (M, 1) normalized query times
            query_locations: (M, D) normalized query locations
        Returns:
            (M,) intensity values in normalized space
        """
        if device is None:
            device = query_times.device

        times_shifted, locations, mask, t_shift = self._extract_history(state, device)
        query_times, query_locations = self._transform_queries(
            state,
            query_times.to(device),
            query_locations.to(device),
        )

        # Shift query times by the same offset used for history
        t_q_shifted = (query_times.squeeze(-1) - t_shift.squeeze()).clamp(min=0.0)  # (M,)

        M = t_q_shifted.shape[0]

        # Expand history from (B=1, T) → (M, T) for vectorized evaluation
        history_times_exp = times_shifted.expand(M, -1)     # (M, T)
        history_locs_exp  = locations.expand(M, -1, -1)     # (M, T, D)
        history_mask_exp  = mask.expand(M, -1)              # (M, T)

        # Temporal intensity f*(t | H)
        lambda_t = self.temporal_model.intensity_at(
            t_q_shifted, history_times_exp, history_mask_exp
        )  # (M,)

        # Spatial log-density log p(s | t, H)
        log_p_s = self.spatial_model.log_spatial_density_at(
            t_q_shifted, query_locations, history_times_exp, history_locs_exp, history_mask_exp
        )  # (M,)

        return lambda_t * torch.exp(log_p_s)  # (M,)

    def density(
        self,
        *,
        state: StateContext,
        query_times: Tensor,
        query_locations: Tensor,
        query_lengths=None,
        x_field_at_events=None,
        marks=None,
        device=None,
    ) -> Tensor:
        """Conditional spatial density p(s_query | t_query, H).

        Args:
            state          : StateContext with payload {times, locations, lengths}
            query_times    : (M, 1) normalized query times
            query_locations: (M, D) normalized query locations
        Returns:
            (M,) density values in normalized space
        """
        if device is None:
            device = query_times.device

        times_shifted, locations, mask, t_shift = self._extract_history(state, device)
        query_times, query_locations = self._transform_queries(
            state,
            query_times.to(device),
            query_locations.to(device),
        )

        t_q_shifted = (query_times.squeeze(-1) - t_shift.squeeze()).clamp(min=0.0)  # (M,)
        M = t_q_shifted.shape[0]

        history_times_exp = times_shifted.expand(M, -1)
        history_locs_exp  = locations.expand(M, -1, -1)
        history_mask_exp  = mask.expand(M, -1)

        log_p_s = self.spatial_model.log_spatial_density_at(
            t_q_shifted, query_locations, history_times_exp, history_locs_exp, history_mask_exp
        )  # (M,)
        return torch.exp(log_p_s)  # (M,)

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: "Tensor",
        grid_locs: "Tensor",
        **kwargs,
    ) -> "Tensor":
        """Surface query contract: routes to intensity() for factorized models."""
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            device=grid_times.device,
        )

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        state: StateContext,
        device,
        marks: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        times = state.payload.get("times", times).to(device)
        locations = state.payload.get("locations", locations).to(device)
        lengths = state.payload.get("lengths", lengths).to(device)

        B, T = times.shape
        n_idx = torch.arange(T, device=device)
        mask = (n_idx.unsqueeze(0) < lengths.unsqueeze(1)).float()  # (B, T)

        # Sequence-relative time shift: ensure all times are non-negative.
        # Parametric temporal models require t ≥ 0; z-score normalized times can be
        # negative. Shifting by times[:,0] maps the first event to t=0 in each sequence.
        # Time differences (used by Hawkes and GMM) are unchanged by this shift.
        t_shift = times[:, 0:1]                            # (B, 1) — first event time
        times_shifted = (times - t_shift).clamp(min=0.0)  # (B, T) — sequence-relative

        # Observation window in shifted coordinates
        t0_tensor = torch.full((B,), self.t0, device=device, dtype=times.dtype)

        if self.t1 is not None:
            # Fixed window end from config (in shifted time)
            t1_tensor = torch.full((B,), self.t1, device=device, dtype=times.dtype)
        else:
            # Last-event convention: window ends at last observed event (shifted)
            last_idx = (lengths - 1).clamp(min=0)
            t1_tensor = times_shifted[torch.arange(B, device=device), last_idx]  # (B,)
        # Temporal log-prob: (B,)
        # Marked models return the *ground* process log-prob plus a per-event
        # categorical over marks; unmarked models take the original path untouched.
        mark_log_probs = None
        if self._marked:
            # Always take the marked path, even when the caller supplies no marks:
            # for a marked model the ground process IS the marked one, so falling
            # back to temporal_model.logprob() could report a likelihood from a head
            # the model never trained. Mark-free callers exist in practice — the
            # single-sequence eval batches built by seahorse/evaluation/likelihood.py
            # carry no marks — and for those we report the ground-process NLL and
            # simply omit the mark factor.
            tll_seq, mark_log_probs = self.temporal_model.marked_logprob(
                times_shifted, locations, mask, t0_tensor, t1_tensor
            )
            if marks is None:
                mark_log_probs = None
        else:
            tll_seq = self.temporal_model.logprob(
                times_shifted, locations, mask, t0_tensor, t1_tensor
            )
        event_logprob_fn = self._temporal_event_logprob_fn
        tll_mat = (
            None
            if event_logprob_fn is None
            else event_logprob_fn(
                times_shifted,
                locations,
                mask,
                t0_tensor,
                t1_tensor,
            )
        )

        # Spatial log-prob: (B, T) — uses shifted times (time diffs are shift-invariant)
        sll_mat = self.spatial_model.logprob(times_shifted, locations, mask)
        spatial_nll_matrix = -sll_mat
        # A missing per-event temporal term used to be replaced by zeros here. That is
        # not a degraded eventwise NLL, it is a different statistic: nll_matrix became
        # the spatial NLL alone, silently dropping the entire temporal log-likelihood
        # from every benchmark number scored off this matrix. Emit nothing instead, so
        # eventwise consumers either fall back to the (correct) prefix-difference path
        # or fail loudly, rather than reporting a spatial-only quantity as a joint NLL.
        temporal_nll_matrix = None if tll_mat is None else -tll_mat
        nll_matrix = None if temporal_nll_matrix is None else temporal_nll_matrix + spatial_nll_matrix
        next_event_mask = mask.clone()
        if T > 0:
            next_event_mask[:, 0] = 0.0

        n_events_total = mask.sum().clamp(min=1)
        tll = tll_seq.sum() / n_events_total
        sll = (sll_mat * mask).sum() / n_events_total

        # Mark log-prob: (B, T) — log p(k_i | t_i, H_i) gathered at the observed mark.
        # Zero for every branch below when unmarked, so nothing here can perturb the
        # unmarked NLL.
        mark_ll_mat = None
        if mark_log_probs is not None:
            marks_t = marks.to(device).long()
            if marks_t.shape[:2] != (B, T):
                raise ValueError(
                    f"marks shape {tuple(marks_t.shape)} incompatible with times ({B}, {T})."
                )
            valid_marks = marks_t * mask.long()  # padding slots → index 0, masked out below
            if int(valid_marks.max().item() if valid_marks.numel() else 0) >= self.n_marks:
                raise ValueError(
                    f"mark index {int(valid_marks.max().item())} out of range for "
                    f"n_marks={self.n_marks}."
                )
            mark_ll_mat = mark_log_probs.gather(
                -1, valid_marks.unsqueeze(-1)
            ).squeeze(-1) * mask  # (B, T)

        mll = (
            mark_ll_mat.sum() / n_events_total
            if mark_ll_mat is not None
            else tll.new_zeros(())
        )
        mean_nll = -(tll + sll + mll)

        n_per_seq = mask.sum(dim=-1).clamp(min=1)
        nll_per_event = -(tll_seq / n_per_seq + (sll_mat * mask).sum(dim=-1) / n_per_seq)
        if mark_ll_mat is not None:
            nll_per_event = nll_per_event - mark_ll_mat.sum(dim=-1) / n_per_seq
            nll_matrix = nll_matrix - mark_ll_mat

        out = {
            "loss": mean_nll,
            "nll": mean_nll,
            "temporal_nll": float((-tll).item()),   # mean temporal NLL/event
            "spatial_nll": float((-sll).item()),    # mean spatial NLL/event
            "nll_per_event": nll_per_event,
            "total_events": mask.sum(),
            "tll": tll,
            "sll": sll,
            "sll_matrix": sll_mat,
            "spatial_nll_matrix": spatial_nll_matrix,
            "mask": mask,
            "extra_metrics": self.raw_reporting_metrics(
                state=state,
                nll=mean_nll,
                temporal_nll=-tll,
                spatial_nll=-sll,
            ),
        }
        if nll_matrix is not None:
            out["tll_matrix"] = tll_mat
            out["temporal_nll_matrix"] = temporal_nll_matrix
            out["nll_matrix"] = nll_matrix
            out["next_event_mask"] = next_event_mask

        if mark_ll_mat is not None:
            # Everything a downstream accuracy / macro-F1 / ECE metric needs. The
            # flattened *_events views drop padding, so they line up row-for-row
            # and can be concatenated straight across batches.
            valid = mask.bool()
            marks_t = marks.to(device).long()
            out.update({
                # (B, T, M) log p(k | t_i, H_i) — rows sum to 1 in probability space
                "mark_logprob_matrix": mark_log_probs,
                # (n_events, M) — the padding-free view; the ideal metric input
                "mark_logprob_events": mark_log_probs[valid],
                # (n_events,) int64 ground-truth mark per valid event
                "mark_targets_events": marks_t[valid],
                # (n_events,) int64 argmax prediction — accuracy / macro-F1
                "mark_pred_events": mark_log_probs[valid].argmax(dim=-1),
                # (B, T) −log p(k_i), zero on padding
                "mark_nll_matrix": -mark_ll_mat,
                "mark_ll": mll,
                "mark_nll": float((-mll).item()),   # mean mark NLL/event
                "n_marks": self.n_marks,
            })

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
        del state_regularization_terms, x_field_at_events
        if device is None:
            device = times.device
        return self._compute(
            marks=marks,
            times=times,
            locations=locations,
            lengths=lengths,
            state=state,
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
