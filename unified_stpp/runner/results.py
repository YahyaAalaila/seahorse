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

    Three-layer metric architecture
    --------------------------------
    Layer 1 — Objective (``val_objective``, ``val_metric_key``):
        What the model trained on and what drove checkpoint selection.
        ``val_objective`` is the best val score for the model's native objective
        (e.g. score-matching loss for SMASH, ELBO for Diffusion, NLL for exact models).
        ``val_metric_key`` names the metric ("sm", "elbo", "nll").

    Layer 2 — NLL (``test_nll``):
        A benchmark-facing likelihood quantity independent of the training objective.
        Always NLL semantics (exact or approximate).  ``nan`` when no test set or
        ``nll_kind="none"``.  ``nll_kind`` describes accuracy: "exact" | "approx" | "none".
        ``temporal_nll`` / ``spatial_nll`` are per-component breakdowns when available.

    Layer 3 — Sampling-based eval metrics (future):
        ``test_rmse``, ``test_mae``, … populated post-training from model samples.
        Not yet implemented; reserved in ``extra_metrics`` until then.

    Normalization
    -------------
    ``test_nll`` may be reported either in the model's native benchmark-facing
    space or, for exact families that support it, in raw/original data space.
    The exact convention is described by ``nll_description`` and
    ``nll_report_space``.

    Attributes
    ----------
    preset:           Model preset name (e.g. ``"smash"``).
    dataset_id:       Identifier for the dataset used.
    seed:             Random seed used for this run.
    val_objective:    Best validation objective score (Layer 1; e.g. best val/sm).
    val_metric_key:   Name of the val metric: "nll", "elbo", "sm", … (Layer 1).
    test_nll:         Test NLL/event in the documented reporting space
                      (Layer 2; ``nan`` if unavailable).
    nll_kind:         Quality of test_nll: "exact" | "approx" | "none" (Layer 2).
    train_time_sec:   Wall-clock training time in seconds.
    n_params:         Total number of trainable model parameters.
    effective_config: The config dict actually used (post-HPO or from YAML).
    checkpoint_path:  Path to the saved Lightning checkpoint (if any).
    norm_stats:       Normalization stats from training data.
    extra_metrics:    Dict for any additional metrics the caller wants to store.
    """

    preset: str
    dataset_id: str
    seed: int
    val_objective: float       # Layer 1: best val objective score
    test_nll: float            # Layer 2: test NLL (exact or approx)
    train_time_sec: float
    n_params: int
    effective_config: dict[str, Any]
    checkpoint_path: Optional[Path] = None
    norm_stats: dict[str, Any] = field(default_factory=dict)
    extra_metrics: dict[str, Any] = field(default_factory=dict)
    run_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Layer 1 — Objective metadata
    # ------------------------------------------------------------------
    training_objective: str = "nll"
    # ^ mirrors capabilities.training_objective ("nll", "elbo", "score_matching", …)

    val_metric_key: str = "nll"
    # ^ display key of the val metric: "nll", "elbo", "sm", …
    # Matches capabilities.metric_key and the logged f"val/{val_metric_key}" key.

    objective_description: str = ""
    # ^ human-readable: "exact NLL", "variational ELBO (1-step)", "denoising score matching"

    # ------------------------------------------------------------------
    # Layer 2 — NLL metadata
    # ------------------------------------------------------------------
    nll_kind: str = "exact"
    # ^ quality of test_nll: "exact" | "approx" | "none"

    nll_description: str = "exact NLL/event"
    # ^ human-readable description of what test_nll measures

    nll_footnote: str = ""
    # ^ superscript in LaTeX/HTML benchmark tables (e.g. "‡ approx NLL")

    nll_report_space: str = "native"
    # ^ "native" or "raw"; describes the space used by ``test_nll``.

    # ------------------------------------------------------------------
    # Layer 2 — Temporal/spatial NLL breakdowns
    # ------------------------------------------------------------------
    temporal_nll: float = float("nan")   # mean temporal NLL/event (test set)
    spatial_nll: float = float("nan")    # mean spatial NLL/event (test set)

    # ------------------------------------------------------------------
    # Layer 3 — Sampling-based eval metrics (future)
    # ------------------------------------------------------------------
    # test_rmse: float = float("nan")
    # test_mae:  float = float("nan")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["checkpoint_path"] is not None:
            d["checkpoint_path"] = str(d["checkpoint_path"])
        if d["run_dir"] is not None:
            d["run_dir"] = str(d["run_dir"])
        return d

    def to_json(self, path) -> None:
        """Serialise to a JSON file, converting NaN → null."""
        def _nan_to_null(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return v

        raw = self.to_dict()
        cleaned = {k: _nan_to_null(v) for k, v in raw.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cleaned, f, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        d = dict(d)
        if d.get("checkpoint_path"):
            d["checkpoint_path"] = Path(d["checkpoint_path"])
        if d.get("run_dir"):
            d["run_dir"] = Path(d["run_dir"])
        # Tolerate older JSON files that predate the three-layer fields.
        # val_nll was renamed to val_objective — remap transparently.
        if "val_nll" in d and "val_objective" not in d:
            d["val_objective"] = d.pop("val_nll")
        # Filter to only known fields so extra keys don't raise TypeError.
        import dataclasses as _dc
        known = {f.name for f in _dc.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        test_str = (
            f"{self.test_nll:.4f}" if not math.isnan(self.test_nll) else "n/a"
        )
        nll_tag = f" [{self.nll_kind}]" if self.nll_kind != "exact" else ""
        dir_str = f", run_dir={str(self.run_dir)!r}" if self.run_dir is not None else ""
        return (
            f"RunResult(preset={self.preset!r}, dataset={self.dataset_id!r}, "
            f"seed={self.seed}, val_{self.val_metric_key}={self.val_objective:.4f}, "
            f"test_nll={test_str}{nll_tag}, params={self.n_params:,}{dir_str})"
        )
