"""RunResult — output of a single STPPRunner.fit() call."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class RunResult:
    """Output of one training run.

    Attributes
    ----------
    preset:           Model preset name (e.g. ``"auto_stpp"``).
    dataset_id:       Identifier for the dataset used.
    seed:             Random seed used for this run.
    val_nll:          Best validation NLL/event achieved during training.
    test_nll:         Test NLL/event (``nan`` if no test set was provided).
    train_time_sec:   Wall-clock training time in seconds.
    n_params:         Total number of trainable model parameters.
    effective_config: The config dict actually used (post-HPO or from YAML).
    checkpoint_path:  Path to the saved Lightning checkpoint (if any).
    extra_metrics:    Dict for any additional metrics the caller wants to store.
    """

    preset: str
    dataset_id: str
    seed: int
    val_nll: float
    test_nll: float
    train_time_sec: float
    n_params: int
    effective_config: dict[str, Any]
    checkpoint_path: Optional[Path] = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["checkpoint_path"] is not None:
            d["checkpoint_path"] = str(d["checkpoint_path"])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        d = dict(d)
        if d.get("checkpoint_path"):
            d["checkpoint_path"] = Path(d["checkpoint_path"])
        return cls(**d)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        test_str = (
            f"{self.test_nll:.4f}" if not math.isnan(self.test_nll) else "n/a"
        )
        return (
            f"RunResult(preset={self.preset!r}, dataset={self.dataset_id!r}, "
            f"seed={self.seed}, val_nll={self.val_nll:.4f}, test_nll={test_str}, "
            f"params={self.n_params:,})"
        )
