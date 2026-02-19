"""Config parsing helpers shared by CLI and tests."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def resolve_t_end(data_cfg: Mapping, fallback_t_end: float) -> float:
    """Resolve time horizon from modern ``t_end`` or legacy ``T`` config keys."""
    t_end_cfg = data_cfg.get("t_end", None)
    legacy_t = data_cfg.get("T", None)

    if t_end_cfg is not None:
        if legacy_t is not None and not np.isclose(float(t_end_cfg), float(legacy_t)):
            print(
                "Warning: both data.t_end and legacy data.T are set with different values; "
                f"using data.t_end={float(t_end_cfg):.6g} over data.T={float(legacy_t):.6g}."
            )
        return float(t_end_cfg)

    if legacy_t is not None:
        return float(legacy_t)

    return float(fallback_t_end)


def resolve_optimizer_hparams(
    training_cfg: Mapping,
    *,
    lr_default: float,
    weight_decay_default: float,
    grad_clip_default: float,
) -> tuple[float, float, float]:
    """Resolve optimizer/training scalars with CLI defaults as fallback."""
    lr = float(training_cfg.get("lr", lr_default))
    weight_decay = float(training_cfg.get("weight_decay", weight_decay_default))
    grad_clip = float(training_cfg.get("grad_clip", grad_clip_default))
    return lr, weight_decay, grad_clip

