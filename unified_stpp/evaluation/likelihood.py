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

API
---
    EvalResult   — dataclass holding scalar evaluation metrics
    LikelihoodEvaluator
        .evaluate(dataloader) → EvalResult
        .evaluate_batch(batch) → EvalResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }
