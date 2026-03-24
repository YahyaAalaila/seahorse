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

    Metric definition
    -----------------
    ``val_nll`` and ``test_nll`` are the **per-event negative log-likelihood**
    in *normalized* coordinates::

        NLL = -1/N Σ_i log p(t_i_norm, s_i_norm | H_i)

    where (t_norm, s_norm) are z-scored with the training-set statistics stored
    in ``norm_stats``.  The values are directly comparable across models IFF all
    models used the same normalization (enforced by ``Benchmark``).

    To convert to NLL in original-space coordinates, subtract the log-Jacobian::

        NLL_original = NLL_normalized - log(time_std × loc_std_x × loc_std_y)

    This constant shift does not affect model ranking.

    For sampling-based metrics (Wasserstein, RMSE, …), denormalize model
    outputs using ``norm_stats`` before computing distances in original space::

        t_orig = t_norm * norm_stats["time_std"] + norm_stats["time_mean"]
        s_orig = s_norm * norm_stats["loc_std"]  + norm_stats["loc_mean"]

    Attributes
    ----------
    preset:           Model preset name (e.g. ``"auto_stpp"``).
    dataset_id:       Identifier for the dataset used.
    seed:             Random seed used for this run.
    val_nll:          Best validation NLL/event (normalized space).
    test_nll:         Test NLL/event (normalized space; ``nan`` if no test set).
    train_time_sec:   Wall-clock training time in seconds.
    n_params:         Total number of trainable model parameters.
    effective_config: The config dict actually used (post-HPO or from YAML).
    checkpoint_path:  Path to the saved Lightning checkpoint (if any).
    norm_stats:       Normalization stats from training data:
                      ``time_mean``, ``time_std``, ``loc_mean`` (list),
                      ``loc_std`` (list), ``normalize`` (bool).
                      Use these to convert model outputs to original space.
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
    norm_stats: dict[str, Any] = field(default_factory=dict)
    extra_metrics: dict[str, Any] = field(default_factory=dict)
    run_dir: Optional[Path] = None

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
        return cls(**d)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        test_str = (
            f"{self.test_nll:.4f}" if not math.isnan(self.test_nll) else "n/a"
        )
        dir_str = f", run_dir={str(self.run_dir)!r}" if self.run_dir is not None else ""
        return (
            f"RunResult(preset={self.preset!r}, dataset={self.dataset_id!r}, "
            f"seed={self.seed}, val_nll={self.val_nll:.4f}, test_nll={test_str}, "
            f"params={self.n_params:,}{dir_str})"
        )
