"""EventModel for the marked (mutually-exciting) multivariate Hawkes baseline.

Model
-----
An ``M``-mark space-time Hawkes process with an ``M x M`` branching matrix.  The
mark-``k`` conditional intensity at time ``t`` and location ``y`` is

    lambda_k(t, y | H_t) = mu_k * rho_k(y)
                           + sum_{i: t_i < t} alpha[k_i, k] * g(t - t_i)
                                              * f_{k_i, k}(y - x_i)

where

  * ``mu_k >= 0``            — background rate of mark ``k``
  * ``rho_k``                — background spatial density of mark ``k``,
                               ``N(b_k, diag(tau_k^2))``, integrates to 1 over R^d
  * ``alpha`` (M x M, >= 0)  — branching matrix; ``alpha[j, k]`` governs how
                               strongly a mark-``j`` event excites mark ``k``
  * ``g(u) = q * exp(-q u)`` — **normalized** exponential decay, ``\\int_0^inf g = 1``
  * ``f_{j,k}``              — displacement density ``N(m_jk, diag(s_jk^2))``,
                               integrates to 1 over R^d

Why ``g`` is normalized
-----------------------
Because ``\\int_0^inf g = 1`` and every ``f_{j,k}`` integrates to 1 over space, the
expected number of direct mark-``k`` offspring of a mark-``j`` event is *exactly*
``alpha[j, k]``.  The branching matrix is therefore the matrix of branching
ratios, on the same scale as the adjacency it is meant to recover — no
post-hoc ``alpha / beta`` rescaling, and the diagonal entries ``alpha[k, k]`` are
directly the per-mark self-excitation ratios ``n_kk``.

Why the decay ``q`` is shared across pairs
------------------------------------------
``q`` is a single scalar shared by all ``M^2`` pairs.  Rationale:

  1. **Identifiability.**  With normalized ``g``, ``alpha[j,k]`` is the branching
     ratio regardless of ``q``.  Per-pair ``q_jk`` adds ``M^2`` parameters that
     trade off against ``alpha`` in the likelihood at modest sample sizes and
     degrade recovery of ``supp(alpha)`` — which is the downstream target.
  2. **Exactness is cheaper.**  A shared ``q`` lets the space-and-mark integral
     collapse onto the *row sums* ``A_j = sum_k alpha[j,k]``, giving an ``O(T^2)``
     compensator instead of ``O(T^2 M)``.
  3. It is the standard choice for multivariate-Hawkes adjacency recovery
     (e.g. ADM4 and the Bacry-Muzy estimators fix a single decay).

Exact compensator
-----------------
Because the mark index is summed and the spatial argument integrated over all of
R^d, both collapse and only the temporal kernel survives::

    \\int_{t0}^{t1} sum_k \\int_{R^d} lambda_k(t, y) dy dt
        = (sum_k mu_k) * (t1 - t0)
          + sum_{i: t_i <= t1} A_{k_i} * [1 - exp(-q (t1 - t_i))]

with ``A_j = sum_k alpha[j, k]``.  The ``q`` in ``g`` cancels against the ``1/q``
from integrating it, which is exactly why the normalized kernel is convenient.

Per-event decomposition (``nll_matrix``)
----------------------------------------
Following the framework convention (cf. ``HawkesProcess.event_logprob_matrix``),
event ``i`` is charged the compensator mass over ``(t_{i-1}, t_i]``::

    nll_matrix[b, i] = -log lambda_{k_i}(t_i, x_i | H)
                       + (sum_k mu_k) * (t_i - t_{i-1})
                       + sum_{j < i} A_{k_j} * [exp(-q (t_{i-1} - t_j))
                                                - exp(-q (t_i     - t_j))]

with ``t_{-1} := t0``.  These telescope, so ``sum_i nll_matrix[b, i]`` is exactly
the negative log-likelihood of sequence ``b`` over ``[t0, t_last]``.  This is a
*coupled* space-time model, so there is no meaningful temporal/spatial split of
``nll_matrix`` and none is reported.

Mark panel and the mark / space-time split
------------------------------------------
Integrating ``lambda_k(t, y)`` over all of ``R^d`` kills both spatial factors
(each integrates to 1) and leaves the space-marginalised per-mark rate::

    lambda_k(t) = mu_k + sum_{j: t_j < t} alpha[k_j, k] g(t - t_j)

from which the per-event mark distribution follows directly::

    p(k | t_i, H_i) = lambda_k(t_i) / sum_l lambda_l(t_i)

This conditions on time and history but **not** on the observed location, which
is the convention the other marked presets report, so the mark panel compares
them like for like.  Reported as ``mark_logprob_matrix`` ``(B, T, M)`` (rows
normalised), the padding-free views ``mark_logprob_events`` ``(n_events, M)`` and
``mark_targets_events`` ``(n_events,)``, and the scalar ``mark_nll``.

Writing ``Lambda*(t) = sum_l lambda_l(t)`` for the ground intensity, the joint
per-event term factorises exactly::

    log lambda_{k_i}(t_i, x_i) = log Lambda*(t_i)
                                 + log p(k_i | t_i, H_i)
                                 + log p(x_i | k_i, t_i, H_i)

so ``spatiotemporal_nll = nll - mark_nll`` strips out the discrete mark factor.
That subtraction is what makes the number comparable across models with
different mark cardinality and against unmarked baselines — the joint marked NLL
is a density over a larger space and is *not* comparable.  For ``M = 1`` the mark
factor is identically zero and ``spatiotemporal_nll == nll``.

Observation window
------------------
``t0``/``t1`` are interpreted in the sequence-relative frame (times are shifted so
the first event of each sequence sits at 0), matching ``FactorizedEventModel``.
``t1=None`` (default) uses the last-event convention, for which the per-event
decomposition above is exact and complete.  When ``t1`` is set explicitly, the
residual mass over ``(t_last, t1]`` is charged to the last valid event so that
``sum_i nll_matrix[b, i]`` remains exactly the sequence NLL.

Purely temporal mode
--------------------
With ``spatial=False`` the ``rho_k`` and ``f_{j,k}`` factors are dropped entirely and
the model is a marked TPP on ``(t, k)``: ``lambda_k(t) = mu_k + sum_i alpha[k_i,k] g(t-t_i)``.
The compensator formula is unchanged.  NLL units differ from the spatial mode
(no per-area density factor), so the two modes are not directly comparable.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..abstractions import EventCapabilities, EventModel, StateContext
from ..model_registry import register_event

_LOG_2PI = math.log(2.0 * math.pi)
_TINY = 1e-38


@register_event("marked_hawkes")
class MarkedHawkesEventModel(EventModel):
    """Marked multivariate space-time Hawkes process with exact likelihood.

    Args:
        n_marks:        number of marks ``M``.  ``0`` or ``1`` gives the unmarked
                        special case (a single 1x1 branching matrix).
        spatial_dim:    ``d``; ignored when ``spatial=False``.
        spatial:        include the spatial factors ``rho_k`` and ``f_{j,k}``.
                        ``False`` gives a purely temporal marked TPP.
        diagonal_only:  hard-mask ``alpha`` to its diagonal.  Off-diagonal entries
                        are *exactly* zero (multiplied by a 0/1 buffer), receive
                        exactly zero gradient, and are excluded from the
                        excitation sum — this is a structural restriction, not a
                        penalty.
        t0, t1:         observation window in sequence-relative time (see module
                        docstring).  ``t1=None`` → last-event convention.
        init_*:         initial parameter values (pre-softplus quantities are
                        derived from these via an inverse softplus).
    """

    def __init__(
        self,
        *,
        n_marks: int = 1,
        spatial_dim: int = 2,
        spatial: bool = True,
        diagonal_only: bool = False,
        t0: float = 0.0,
        t1: Optional[float] = None,
        init_mu: float = 0.5,
        init_alpha: float = 0.2,
        init_decay: float = 1.0,
        init_disp_scale: float = 0.5,
        init_bg_scale: float = 1.0,
        learn_displacement_mean: bool = True,
    ):
        super().__init__()
        self.n_marks = max(1, int(n_marks))
        self.spatial = bool(spatial)
        self.spatial_dim = int(spatial_dim) if self.spatial else 0
        self.diagonal_only = bool(diagonal_only)
        self.t0 = float(t0)
        self.t1 = None if t1 is None else float(t1)

        m = self.n_marks
        d = self.spatial_dim

        # Branching structure ------------------------------------------------
        self._raw_mu = nn.Parameter(torch.full((m,), _inv_softplus(init_mu)))
        self._raw_alpha = nn.Parameter(torch.full((m, m), _inv_softplus(init_alpha)))
        self._raw_decay = nn.Parameter(torch.tensor(_inv_softplus(init_decay)))

        # Hard structural mask: exactly 1.0 on kept entries, exactly 0.0 elsewhere.
        mask = torch.eye(m) if self.diagonal_only else torch.ones(m, m)
        self.register_buffer("alpha_mask", mask)

        # Spatial components -------------------------------------------------
        if self.spatial:
            disp_mean = torch.zeros(m, m, d)
            if learn_displacement_mean:
                self._disp_mean = nn.Parameter(disp_mean)
            else:
                self.register_buffer("_disp_mean", disp_mean)
            self._raw_disp_scale = nn.Parameter(
                torch.full((m, m, d), _inv_softplus(init_disp_scale))
            )
            self._bg_mean = nn.Parameter(torch.zeros(m, d))
            self._raw_bg_scale = nn.Parameter(
                torch.full((m, d), _inv_softplus(init_bg_scale))
            )
        else:
            self.register_parameter("_disp_mean", None)
            self.register_parameter("_raw_disp_scale", None)
            self.register_parameter("_bg_mean", None)
            self.register_parameter("_raw_bg_scale", None)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> EventCapabilities:
        kind = "marked space-time" if self.spatial else "marked temporal"
        return EventCapabilities(
            training_objective="nll",
            metric_key="nll",
            objective_description="exact NLL",
            nll_kind="exact",
            nll_description=f"exact {kind} Hawkes NLL/event (normalized space)",
            supports_raw_reporting=False,
            has_intensity=True,
            has_density=False,
            exposes_eventwise_terms=True,
        )

    # ------------------------------------------------------------------
    # Parameter accessors / diagnostics
    # ------------------------------------------------------------------

    @property
    def mu(self) -> Tensor:
        """Background rates ``mu`` — shape ``(M,)``, non-negative."""
        return F.softplus(self._raw_mu)

    @property
    def decay(self) -> Tensor:
        """Shared exponential decay rate ``q`` — scalar, positive."""
        return F.softplus(self._raw_decay)

    @property
    def alpha(self) -> Tensor:
        """Branching matrix ``alpha`` — shape ``(M, M)``, non-negative.

        ``alpha[j, k]`` is the expected number of direct mark-``k`` offspring of a
        mark-``j`` event.  Under ``diagonal_only=True`` the off-diagonal entries
        are exactly ``0.0``.
        """
        return F.softplus(self._raw_alpha) * self.alpha_mask

    def alpha_matrix(self) -> Tensor:
        """Detached copy of the fitted ``M x M`` branching matrix (for scoring)."""
        return self.alpha.detach().clone()

    @property
    def branching_ratios(self) -> Tensor:
        """Per-mark self-excitation ratios ``n_kk = alpha[k, k]`` — shape ``(M,)``."""
        return torch.diagonal(self.alpha)

    @property
    def spectral_radius(self) -> Tensor:
        """Spectral radius of ``alpha``; the process is stationary iff ``< 1``."""
        eigs = torch.linalg.eigvals(self.alpha.detach().to(torch.float64))
        return eigs.abs().max().to(self.alpha.dtype)

    def branching_diagnostics(self) -> Dict[str, float]:
        """Scalar diagnostics describing the fitted branching structure."""
        a = self.alpha.detach()
        out: Dict[str, float] = {
            "spectral_radius": float(self.spectral_radius.item()),
            "alpha_row_sum_max": float(a.sum(dim=1).max().item()),
            "decay_q": float(self.decay.detach().item()),
        }
        for k in range(self.n_marks):
            out[f"branching_ratio_{k}{k}"] = float(a[k, k].item())
            out[f"mu_{k}"] = float(self.mu.detach()[k].item())
        if self.n_marks <= 12:
            for j in range(self.n_marks):
                for k in range(self.n_marks):
                    out[f"alpha_{j}_{k}"] = float(a[j, k].item())
        return out

    # ------------------------------------------------------------------
    # Core likelihood
    # ------------------------------------------------------------------

    def _resolve_marks(
        self, marks: Optional[Tensor], shape, device, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Coerce ``marks`` to a valid ``(B, T)`` long tensor.

        ``None`` (unmarked data) maps every event to mark 0, which reduces the
        model exactly to the single-mark Hawkes process.  Padded positions are
        forced to 0 *before* validation, since collate zero-fills them anyway and
        they contribute nothing.  Never mutates the caller's tensor.
        """
        if marks is None:
            return torch.zeros(shape, dtype=torch.long, device=device)
        m = marks.to(device=device, dtype=torch.long)
        if m.shape != tuple(shape):
            raise ValueError(
                f"marks has shape {tuple(m.shape)}, expected {tuple(shape)} "
                "(one integer mark per event)."
            )
        if mask is not None:
            m = torch.where(mask > 0, m, torch.zeros_like(m))
        else:
            m = m.clone()
        if m.numel():
            hi = int(m.max().item())
            lo = int(m.min().item())
            if hi >= self.n_marks or lo < 0:
                raise ValueError(
                    f"marks must lie in [0, n_marks-1] = [0, {self.n_marks - 1}]; "
                    f"got range [{lo}, {hi}]."
                )
        return m

    def _log_gaussian(self, diff: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        """Diagonal-Gaussian log-density, summed over the last (spatial) axis."""
        z = (diff - mean) / scale
        return -0.5 * (z * z).sum(-1) - torch.log(scale).sum(-1) - 0.5 * self.spatial_dim * _LOG_2PI

    def _event_terms(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        marks: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Both per-event likelihood terms, sharing one ``O(T^2)`` pass.

        Returns
        -------
        log_lambda : ``(B, T)``
            ``log lambda_{k_i}(t_i, x_i | H_{t_i})`` — the full space-time
            intensity at the observed mark and location.  Computed with
            ``logsumexp`` over {background, each strictly-earlier event} so that
            far-field Gaussian factors underflow gracefully rather than driving
            the total rate to zero.
        mark_logprob : ``(B, T, M)``
            ``log p(k | t_i, H_i)`` with rows normalised over marks, obtained
            from the **space-marginalised** per-mark rates

                lambda_k(t) = \\int_{R^d} lambda_k(t, y) dy
                            = mu_k + sum_{j: t_j < t} alpha[k_j, k] g(t - t_j)

            (the spatial factors drop out exactly because every ``rho_k`` and
            ``f_{j,k}`` integrates to 1), so that
            ``p(k | t, H) = lambda_k(t) / sum_l lambda_l(t)``.  This conditions on
            time and history but **not** on the observed location, matching the
            convention the other marked presets report.
        """
        b, t = times.shape
        device = times.device
        alpha = self.alpha
        q = self.decay

        # src[b,i,j] = mark of the *parent* j ; tgt[b,i,j] = mark of the child i
        src = marks.unsqueeze(1).expand(b, t, t)
        tgt = marks.unsqueeze(2).expand(b, t, t)

        # Strictly-causal, padding-aware pairs.
        lower = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=-1)
        causal = lower.unsqueeze(0) & (mask.unsqueeze(-2) > 0)

        dt = (times.unsqueeze(-1) - times.unsqueeze(-2)).clamp(min=0.0)  # (B,T,T)
        decay_kernel = q * torch.exp(-q * dt)                            # g(t_i - t_j)

        # ---- joint space-time log-intensity at the observed (t_i, x_i, k_i) ----
        valid = causal
        if self.diagonal_only:
            valid = valid & (self.alpha_mask[src, tgt] > 0)

        log_alpha = torch.log(alpha.clamp_min(_TINY))[src, tgt]          # (B,T,T)
        log_terms = log_alpha + torch.log(q) - q * dt

        if self.spatial:
            disp = locations.unsqueeze(2) - locations.unsqueeze(1)       # (B,T,T,d)
            disp_scale = F.softplus(self._raw_disp_scale) + 1e-6
            log_terms = log_terms + self._log_gaussian(
                disp, self._disp_mean[src, tgt], disp_scale[src, tgt]
            )

        # masked_fill zeroes the gradient of excluded pairs exactly.
        log_terms = log_terms.masked_fill(~valid, float("-inf"))

        log_bg = torch.log(self.mu.clamp_min(_TINY))[marks]              # (B,T)
        if self.spatial:
            bg_scale = F.softplus(self._raw_bg_scale) + 1e-6
            log_bg = log_bg + self._log_gaussian(
                locations, self._bg_mean[marks], bg_scale[marks]
            )

        log_lambda = torch.logsumexp(
            torch.cat([log_bg.unsqueeze(-1), log_terms], dim=-1), dim=-1
        )

        # ---- space-marginalised per-mark rates -> p(k | t_i, H_i) ----
        # alpha is already hard-masked, so the diagonal variant needs no extra
        # masking here: its off-diagonal entries contribute exactly 0.
        alpha_rows = alpha[marks]                                        # (B,T,M) rows by parent
        excite = torch.einsum(
            "bij,bjm->bim", decay_kernel * causal.to(times.dtype), alpha_rows
        )                                                                # (B,T,M)
        lam_marks = self.mu.reshape(1, 1, -1) + excite                   # (B,T,M), > 0
        log_lam_marks = torch.log(lam_marks.clamp_min(_TINY))
        mark_logprob = log_lam_marks - torch.logsumexp(
            log_lam_marks, dim=-1, keepdim=True
        )

        return log_lambda, mark_logprob

    def _interval_compensator(
        self,
        *,
        times: Tensor,
        marks: Tensor,
        mask: Tensor,
        t0: Tensor,
    ) -> Tensor:
        """Compensator mass on ``(t_{i-1}, t_i]`` for each event — shape ``(B, T)``.

        The space and mark integrals have already collapsed here: what remains is
        the total rate ``sum_k mu_k`` plus, per earlier event ``j``, its row sum
        ``A_{k_j} = sum_k alpha[k_j, k]`` times the integrated normalized decay.
        """
        b, t = times.shape
        device = times.device
        q = self.decay
        row_sums = self.alpha.sum(dim=1)                                  # (M,)
        mu_total = self.mu.sum()

        prev_times = torch.cat([t0.unsqueeze(-1), times[:, :-1]], dim=-1)  # (B,T)
        comp_base = mu_total * (times - prev_times).clamp(min=0.0)

        lower = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=-1)
        valid = (lower.unsqueeze(0) & (mask.unsqueeze(-2) > 0)).to(times.dtype)

        prev_dt = (prev_times.unsqueeze(-1) - times.unsqueeze(-2)).clamp(min=0.0)
        curr_dt = (times.unsqueeze(-1) - times.unsqueeze(-2)).clamp(min=0.0)
        decayed = torch.exp(-q * prev_dt) - torch.exp(-q * curr_dt)       # (B,T,T)

        comp_excite = (row_sums[marks].unsqueeze(-2) * decayed * valid).sum(dim=-1)
        return comp_base + comp_excite

    def _tail_compensator(
        self,
        *,
        times: Tensor,
        marks: Tensor,
        mask: Tensor,
        t_last: Tensor,
        t1: Tensor,
    ) -> Tensor:
        """Residual mass on ``(t_last, t1]`` — shape ``(B,)``.  Zero when ``t1 is None``."""
        q = self.decay
        row_sums = self.alpha.sum(dim=1)
        mu_total = self.mu.sum()

        base = mu_total * (t1 - t_last).clamp(min=0.0)
        a = (t_last.unsqueeze(-1) - times).clamp(min=0.0)
        b_ = (t1.unsqueeze(-1) - times).clamp(min=0.0)
        excite = (row_sums[marks] * (torch.exp(-q * a) - torch.exp(-q * b_)) * mask).sum(-1)
        return base + excite

    def _compute(
        self,
        *,
        times: Tensor,
        locations: Tensor,
        lengths: Tensor,
        marks: Optional[Tensor],
        state: StateContext,
        device,
    ) -> Dict[str, Tensor]:
        payload = state.payload if state is not None else {}
        times = payload.get("times", times).to(device)
        locations = payload.get("locations", locations).to(device)
        lengths = payload.get("lengths", lengths).to(device)
        if marks is None:
            marks = payload.get("marks")

        b, t = times.shape
        idx = torch.arange(t, device=device)
        mask = (idx.unsqueeze(0) < lengths.unsqueeze(1)).to(times.dtype)  # (B,T)
        marks_idx = self._resolve_marks(marks, (b, t), device, mask=mask)

        if t == 0:
            zero = times.new_zeros(b, 0)
            total = times.new_zeros(())
            extra = self.branching_diagnostics()
            extra["alpha_matrix"] = self.alpha_matrix()
            extra["spatiotemporal_nll"] = 0.0
            extra["mark_nll"] = 0.0
            return {
                "loss": total, "nll": total, "nll_matrix": zero, "mask": zero,
                "next_event_mask": zero, "total_events": times.new_zeros(()),
                "nll_per_event": times.new_zeros(b),
                "log_intensity_matrix": zero, "compensator_matrix": zero,
                "mark_logprob_matrix": times.new_zeros(b, 0, self.n_marks),
                "mark_logprob_events": times.new_zeros(0, self.n_marks),
                "mark_targets_events": torch.zeros(0, dtype=torch.long, device=device),
                "mark_nll": total,
                "extra_metrics": extra,
            }

        # Sequence-relative shift: parametric Hawkes needs t >= 0 and the
        # framework may hand us z-scored (possibly negative) times.  Only
        # differences matter, so this is semantically inert.
        t_shift = times[:, 0:1]
        times_s = (times - t_shift).clamp(min=0.0)

        t0_tensor = torch.full((b,), self.t0, device=device, dtype=times.dtype)

        log_lambda, mark_logprob = self._event_terms(
            times=times_s, locations=locations, marks=marks_idx, mask=mask
        )
        comp = self._interval_compensator(
            times=times_s, marks=marks_idx, mask=mask, t0=t0_tensor
        )

        nll_matrix = (-log_lambda + comp) * mask

        # Residual window mass (t_last, t1], charged to the last valid event so
        # that sum(nll_matrix) stays exactly equal to the sequence NLL.
        if self.t1 is not None:
            last_idx = (lengths - 1).clamp(min=0)
            arange_b = torch.arange(b, device=device)
            t_last = times_s[arange_b, last_idx]
            t1_tensor = torch.full((b,), self.t1, device=device, dtype=times.dtype)
            tail = self._tail_compensator(
                times=times_s, marks=marks_idx, mask=mask, t_last=t_last, t1=t1_tensor
            )
            tail_slot = torch.zeros_like(nll_matrix)
            tail_slot[arange_b, last_idx] = tail
            nll_matrix = nll_matrix + tail_slot * mask

        n_events_total = mask.sum().clamp(min=1)
        mean_nll = nll_matrix.sum() / n_events_total
        n_per_seq = mask.sum(dim=-1).clamp(min=1)
        nll_per_event = nll_matrix.sum(dim=-1) / n_per_seq

        next_event_mask = mask.clone()
        next_event_mask[:, 0] = 0.0

        # ---- mark panel -------------------------------------------------
        # -log p(k_i | t_i, H_i) at the observed mark.
        mark_nll_matrix = -mark_logprob.gather(
            -1, marks_idx.unsqueeze(-1)
        ).squeeze(-1) * mask                                             # (B,T)
        mark_nll = mark_nll_matrix.sum() / n_events_total

        # Padding-free views, flattened row-major over (B, T).
        valid_sel = mask > 0
        mark_logprob_events = mark_logprob[valid_sel]                    # (n_events, M)
        mark_targets_events = marks_idx[valid_sel]                       # (n_events,)

        # Joint minus the mark factor: the only NLL comparable across models
        # with different mark cardinality, and to unmarked baselines.
        spatiotemporal_nll = mean_nll - mark_nll

        extra = self.branching_diagnostics()
        extra["alpha_matrix"] = self.alpha_matrix()
        extra["spatiotemporal_nll"] = float(spatiotemporal_nll.detach())
        extra["mark_nll"] = float(mark_nll.detach())

        return {
            "loss": mean_nll,
            "nll": mean_nll,
            "nll_matrix": nll_matrix,
            "log_intensity_matrix": log_lambda,
            "compensator_matrix": comp,
            "nll_per_event": nll_per_event,
            "total_events": mask.sum(),
            "mask": mask,
            "next_event_mask": next_event_mask,
            "mark_logprob_matrix": mark_logprob,
            "mark_logprob_events": mark_logprob_events,
            "mark_targets_events": mark_targets_events,
            "mark_nll": mark_nll,
            "extra_metrics": extra,
        }

    # ------------------------------------------------------------------
    # EventModel interface
    # ------------------------------------------------------------------

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
            times=times,
            locations=locations,
            lengths=lengths,
            marks=marks,
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

    # ------------------------------------------------------------------
    # Intensity queries
    # ------------------------------------------------------------------

    def intensity_components(
        self,
        *,
        query_times: Tensor,
        query_locations: Optional[Tensor],
        history_times: Tensor,
        history_locations: Optional[Tensor],
        history_marks: Tensor,
        history_mask: Tensor,
    ) -> Tensor:
        """Per-mark intensity ``lambda_k(t, y | H)`` at query points — ``(Q, M)``.

        All inputs are in the same (shifted) frame.  ``history_*`` are ``(T,)`` /
        ``(T, d)`` tensors describing one sequence.  Events at exactly the query
        time are excluded (strict ``t_j < t``).
        """
        q = self.decay
        alpha = self.alpha
        qn = query_times.shape[0]

        dt = query_times.unsqueeze(-1) - history_times.unsqueeze(0)      # (Q,T)
        causal = (dt > 0) & (history_mask.unsqueeze(0) > 0)
        g = q * torch.exp(-q * dt.clamp(min=0.0))                        # (Q,T)

        a_rows = alpha[history_marks]                                    # (T,M)
        contrib = g.unsqueeze(-1) * a_rows.unsqueeze(0)                  # (Q,T,M)

        if self.spatial:
            disp = query_locations.unsqueeze(1) - history_locations.unsqueeze(0)  # (Q,T,d)
            disp_scale = F.softplus(self._raw_disp_scale) + 1e-6
            mean_rows = self._disp_mean[history_marks]                   # (T,M,d)
            scale_rows = disp_scale[history_marks]                       # (T,M,d)
            log_f = self._log_gaussian(
                disp.unsqueeze(2), mean_rows.unsqueeze(0), scale_rows.unsqueeze(0)
            )                                                            # (Q,T,M)
            contrib = contrib * torch.exp(log_f)

        contrib = contrib * causal.unsqueeze(-1).to(contrib.dtype)
        excite = contrib.sum(dim=1)                                      # (Q,M)

        bg = self.mu.unsqueeze(0).expand(qn, -1)                         # (Q,M)
        if self.spatial:
            bg_scale = F.softplus(self._raw_bg_scale) + 1e-6
            log_rho = self._log_gaussian(
                query_locations.unsqueeze(1), self._bg_mean.unsqueeze(0), bg_scale.unsqueeze(0)
            )                                                            # (Q,M)
            bg = bg * torch.exp(log_rho)

        return bg + excite

    def _history_from_state(self, state: StateContext, device):
        payload = state.payload
        times = payload["times"].to(device)
        locations = payload["locations"].to(device)
        lengths = payload["lengths"].to(device)
        marks = payload.get("marks")

        b, t = times.shape
        if b != 1:
            raise ValueError("intensity() requires a single-sequence state (B=1).")
        idx = torch.arange(t, device=device)
        mask = (idx.unsqueeze(0) < lengths.unsqueeze(1)).to(times.dtype)
        marks_idx = self._resolve_marks(marks, (b, t), device, mask=mask)
        t_shift = times[:, 0:1]
        times_s = (times - t_shift).clamp(min=0.0)
        return times_s[0], locations[0], marks_idx[0], mask[0], t_shift.reshape(())

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
        """Conditional intensity at query points.

        Returns the **ground** intensity ``sum_k lambda_k(t, y | H)`` by default.
        Pass ``marks`` as a ``(Q,)`` long tensor to select a specific mark per query.
        """
        if device is None:
            device = query_times.device
        h_t, h_s, h_k, h_m, t_shift = self._history_from_state(state, device)

        q_t = query_times.to(device)
        q_t = (q_t.squeeze(-1) if q_t.ndim > 1 else q_t) - t_shift
        q_t = q_t.clamp(min=0.0)
        q_s = query_locations.to(device) if self.spatial else None

        per_mark = self.intensity_components(
            query_times=q_t,
            query_locations=q_s,
            history_times=h_t,
            history_locations=h_s,
            history_marks=h_k,
            history_mask=h_m,
        )
        if marks is None:
            return per_mark.sum(dim=-1)
        sel = marks.to(device=device, dtype=torch.long).reshape(-1)
        return per_mark.gather(1, sel.unsqueeze(-1)).squeeze(-1)

    def query_surface(
        self,
        *,
        state: StateContext,
        grid_times: Tensor,
        grid_locs: Tensor,
        **kwargs,
    ) -> Tensor:
        """Ground intensity over a spatial grid — ``(G,)``."""
        t = grid_times.unsqueeze(-1) if grid_times.ndim == 1 else grid_times
        return self.intensity(
            state=state,
            query_times=t,
            query_locations=grid_locs,
            marks=kwargs.get("marks"),
            device=grid_times.device,
        )


def _inv_softplus(y: float) -> float:
    """Inverse of ``softplus`` so that ``softplus(_inv_softplus(y)) == y``."""
    y = float(y)
    if y <= 0.0:
        raise ValueError(f"softplus outputs are strictly positive; got {y}.")
    if y > 20.0:  # softplus is numerically the identity up here
        return y
    return math.log(math.expm1(y))
