"""
Likelihood evaluator — single source of truth for NLL/LL metrics.

All experiment scripts must report LL and NLL/event exclusively via
`LikelihoodEvaluator`.  This guarantees:
  - identical masking convention across all models
  - identical event-weighting scheme
  - no accidental use of training-mode stochasticity during eval

Usage
-----
    ev = LikelihoodEvaluator(model, device="cpu")
    result = ev.evaluate(test_dataloader)
    print(result.nll_per_event, result.ll_per_event)

    # Optional parity check (IdentityDynamics only)
    report = ev.parity_check(one_batch)
    assert report.passed

API
---
    EvalResult   — dataclass holding scalar evaluation metrics
    ParityReport — dataclass holding parity-check results
    LikelihoodEvaluator
        .evaluate(dataloader) → EvalResult
        .evaluate_batch(batch) → EvalResult
        .parity_check(batch, tol=1e-4) → ParityReport
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from unified_stpp.models.dynamics.identity import IdentityDynamics


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Scalar likelihood metrics for one evaluation pass."""

    nll_per_event: float
    """Mean negative log-likelihood per event (lower = better)."""

    ll_per_event: float
    """Mean log-likelihood per event (= -nll_per_event)."""

    n_events: int
    """Total number of (non-padded) events evaluated."""

    ll_total: float
    """Sum of per-event log-likelihoods (= ll_per_event * n_events)."""


@dataclass
class ParityReport:
    """
    Result of a parity check between the model's own forward-pass NLL
    and a manually reconstructed NLL (encoder + decoder calls).

    Only meaningful for models with IdentityDynamics.
    """

    passed: bool
    """True if |model_nll - manual_nll| <= tol."""

    model_nll: float
    """NLL returned by model.forward(batch)["nll"]."""

    manual_nll: float
    """NLL reconstructed by calling encoder + decoder directly."""

    max_abs_diff: float
    """Absolute difference |model_nll - manual_nll|."""

    n_events: int
    """Number of valid events in the batch."""

    tol: float
    """Tolerance used for the PASS/FAIL decision."""


# ---------------------------------------------------------------------------
# LikelihoodEvaluator
# ---------------------------------------------------------------------------

class LikelihoodEvaluator:
    """
    Evaluates NLL/LL for any UnifiedSTPP-compatible model.

    Parameters
    ----------
    model  : nn.Module
        Any model whose forward pass accepts the canonical batch keys
        (times, locations, lengths, marks, event_covariates,
        field_covariates) and returns a dict with 'nll' and
        'total_events'.
    device : str
        Target device for evaluation ("cpu", "cuda", "mps", …).
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, dataloader: DataLoader) -> EvalResult:
        """
        Evaluate over a full dataloader.

        Event-weighted NLL across all batches:
            NLL = (Σ_batch  nll_batch * n_events_batch) / Σ_batch n_events_batch

        Batches with non-finite NLL are silently skipped (consistent with
        the LegacyTrainer behaviour).

        Parameters
        ----------
        dataloader : DataLoader
            Produces canonical batches (validate_batch-compliant).

        Returns
        -------
        EvalResult
        """
        self.model.to(self.device)
        self.model.eval()

        total_nll_sum = 0.0
        total_events = 0

        with torch.no_grad():
            for batch in dataloader:
                result = self.evaluate_batch(batch)
                if result.n_events > 0 and torch.isfinite(
                    torch.tensor(result.nll_per_event)
                ):
                    total_nll_sum += result.nll_per_event * result.n_events
                    total_events += result.n_events

        avg_nll = total_nll_sum / max(total_events, 1)
        return EvalResult(
            nll_per_event=avg_nll,
            ll_per_event=-avg_nll,
            n_events=total_events,
            ll_total=-avg_nll * total_events,
        )

    def evaluate_batch(self, batch: Dict[str, Any]) -> EvalResult:
        """
        Evaluate one batch.

        Parameters
        ----------
        batch : dict
            Canonical batch (validate_batch-compliant).  Tensors are
            moved to `self.device` internally.

        Returns
        -------
        EvalResult
        """
        batch = self._to_device(batch)
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            out = self.model(
                times=batch["times"],
                locations=batch["locations"],
                lengths=batch["lengths"],
                marks=batch.get("marks"),
                x_event=batch.get("event_covariates"),
                x_field_at_events=batch.get("field_covariates"),
            )

        nll = float(out["nll"].item())
        n_events = int(out["total_events"].item())
        return EvalResult(
            nll_per_event=nll,
            ll_per_event=-nll,
            n_events=n_events,
            ll_total=-nll * n_events,
        )

    def parity_check(
        self,
        batch: Dict[str, Any],
        tol: float = 1e-4,
    ) -> ParityReport:
        """
        Verify that model.forward NLL matches a manually reconstructed NLL.

        Manually replicates `_forward_batched` (the IdentityDynamics path):
          1. Encode the full sequence  → all_states (B, N, h)
          2. Gather conditioning states z_cond = all_states[:, :L, :]
          3. Call decoder.nll on flattened (z, t, s, t_prev) tuples
          4. Apply the same valid-event mask and compute mean NLL
          5. Compare to model.forward(batch)["nll"]

        Only meaningful for models using IdentityDynamics.  For other
        dynamics a ParityReport with passed=False is returned and
        max_abs_diff is set to float("inf").

        Parameters
        ----------
        batch : dict
            One canonical batch (validate_batch-compliant).
        tol   : float
            Absolute tolerance for PASS/FAIL.

        Returns
        -------
        ParityReport
        """
        batch = self._to_device(batch)
        self.model.to(self.device)
        self.model.eval()

        # Check for IdentityDynamics (parity only defined there)
        dynamics = getattr(self.model, "dynamics", None)
        if not isinstance(dynamics, IdentityDynamics):
            return ParityReport(
                passed=False,
                model_nll=float("nan"),
                manual_nll=float("nan"),
                max_abs_diff=float("inf"),
                n_events=0,
                tol=tol,
            )

        times     = batch["times"]      # (B, N)
        locations = batch["locations"]  # (B, N, d)
        lengths   = batch["lengths"]    # (B,)

        with torch.no_grad():
            # --- model forward path ---
            out = self.model(
                times=times,
                locations=locations,
                lengths=lengths,
                marks=batch.get("marks"),
                x_event=batch.get("event_covariates"),
                x_field_at_events=batch.get("field_covariates"),
            )
            model_nll = float(out["nll"].item())

            # --- manual reconstruction path ---
            events = torch.cat([times.unsqueeze(-1), locations], dim=-1)  # (B, N, 1+d)
            _, all_states = self.model.encode(
                events, lengths, x_event=batch.get("event_covariates")
            )  # (B, N, h)

            B = times.shape[0]
            max_len = int(lengths.max().item())

            if max_len < 2:
                return ParityReport(
                    passed=(model_nll == 0.0 or abs(model_nll) < tol),
                    model_nll=model_nll,
                    manual_nll=0.0,
                    max_abs_diff=abs(model_nll),
                    n_events=0,
                    tol=tol,
                )

            L = max_len - 1

            z_cond   = all_states[:, :L, :]              # (B, L, h)
            t_target = times[:, 1:1 + L].unsqueeze(-1)   # (B, L, 1)
            s_target = locations[:, 1:1 + L, :]          # (B, L, d)
            t_prev   = times[:, :L].unsqueeze(-1)        # (B, L, 1)

            n_idx  = torch.arange(L, device=times.device)
            mask   = (n_idx.unsqueeze(0) < (lengths.unsqueeze(1) - 1)).float()  # (B, L)

            h = z_cond.shape[-1]
            d = s_target.shape[-1]
            z_flat      = z_cond.reshape(B * L, h)
            t_flat      = t_target.reshape(B * L, 1)
            s_flat      = s_target.reshape(B * L, d)
            t_prev_flat = t_prev.reshape(B * L, 1)

            x_field_flat: Optional[Tensor] = None
            x_field = batch.get("field_covariates")
            if x_field is not None:
                x_field_flat = x_field[:, :L, :].reshape(B * L, -1)

            nll_flat = self.model.decoder.nll(
                z_flat, t_flat, s_flat, t_prev_flat, x_field=x_field_flat
            )  # (B*L,)
            nll_all    = nll_flat.reshape(B, L)      # (B, L)
            nll_masked = nll_all * mask              # (B, L)
            manual_nll = float(
                (nll_masked.sum() / mask.sum().clamp(min=1)).item()
            )

        n_events  = int(mask.sum().item())
        abs_diff  = abs(model_nll - manual_nll)
        return ParityReport(
            passed=abs_diff <= tol,
            model_nll=model_nll,
            manual_nll=manual_nll,
            max_abs_diff=abs_diff,
            n_events=n_events,
            tol=tol,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }
