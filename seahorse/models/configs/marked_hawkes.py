"""MarkedHawkesConfig — construction configs for the marked Hawkes baseline family.

Presets
-------
  - ``marked_hawkes``      : full ``M x M`` branching matrix (all cross-mark terms free)
  - ``marked_hawkes_diag`` : ``alpha`` structurally restricted to its diagonal

The pair is designed for a single controlled comparison: they differ *only* in
whether off-diagonal excitation is permitted.  The diagonal restriction is a hard
structural mask (off-diagonal entries are exactly ``0.0`` and receive exactly zero
gradient), not a penalty or a prior, so a likelihood-ratio / information-criterion
comparison between the two is well defined.

Config parameters
-----------------
n_marks       : number of marks ``M``.  Normally supplied by ``ModelConfig.n_marks``;
                an explicit ``n_marks`` key in the preset override dict wins.
spatial       : ``True`` (default) → marked space-time Hawkes with Gaussian
                displacement kernels;  ``False`` → purely temporal marked TPP.
t0, t1        : observation window in sequence-relative time.  ``t1=None`` (default)
                uses the last-event convention.  See the event-model docstring.
init_mu, init_alpha, init_decay, init_disp_scale, init_bg_scale
              : parameter initialisations (all strictly positive).
learn_displacement_mean
              : when ``False``, offspring displacements are zero-mean (isotropic
                aftershock model) and only the scales are learned.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

from .base import BaseModelConfig, ConfigRegistry

if TYPE_CHECKING:
    from seahorse.models.unified_model import UnifiedSTPP


@dataclasses.dataclass
class MarkedHawkesBaseConfig(BaseModelConfig):
    """Base config for the marked Hawkes family.  Subclasses set ``_DIAGONAL_ONLY``."""

    _DIAGONAL_ONLY: ClassVar[bool] = False
    _STATE_MODEL: ClassVar[str] = "marked_hawkes"
    _EVENT_MODEL: ClassVar[str] = "marked_hawkes"

    n_marks: int = 1
    spatial: bool = True
    t0: float = 0.0
    t1: Optional[float] = None
    init_mu: float = 0.5
    init_alpha: float = 0.2
    init_decay: float = 1.0
    init_disp_scale: float = 0.5
    init_bg_scale: float = 1.0
    learn_displacement_mean: bool = True
    input_transform: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        *,
        hidden_dim: int = 128,
        spatial_dim: int = 2,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        n_marks: int = 0,
    ) -> "MarkedHawkesBaseConfig":
        del event_cov_dim, field_cov_dim
        # An explicit override wins; otherwise take ModelConfig.n_marks.
        marks = int(d.get("n_marks", n_marks) or 0)
        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            n_marks=max(1, marks),
            spatial=bool(d.get("spatial", True)),
            t0=float(d.get("t0", 0.0)),
            t1=float(d["t1"]) if d.get("t1") is not None else None,
            init_mu=float(d.get("init_mu", 0.5)),
            init_alpha=float(d.get("init_alpha", 0.2)),
            init_decay=float(d.get("init_decay", 1.0)),
            init_disp_scale=float(d.get("init_disp_scale", 0.5)),
            init_bg_scale=float(d.get("init_bg_scale", 1.0)),
            learn_displacement_mean=bool(d.get("learn_displacement_mean", True)),
            input_transform=dict(d.get("input_transform", {}) or {}),
        )

    def build_model(self) -> "UnifiedSTPP":
        from seahorse.models.event_models.marked_hawkes import MarkedHawkesEventModel
        from seahorse.models.state_models.marked_hawkes import MarkedHawkesStateModel
        from seahorse.models.unified_model import UnifiedSTPP

        state_model = MarkedHawkesStateModel(input_transform=self.input_transform)
        event_model = MarkedHawkesEventModel(
            n_marks=self.n_marks,
            spatial_dim=self.spatial_dim,
            spatial=self.spatial,
            diagonal_only=self._DIAGONAL_ONLY,
            t0=self.t0,
            t1=self.t1,
            init_mu=self.init_mu,
            init_alpha=self.init_alpha,
            init_decay=self.init_decay,
            init_disp_scale=self.init_disp_scale,
            init_bg_scale=self.init_bg_scale,
            learn_displacement_mean=self.learn_displacement_mean,
        )
        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )


@ConfigRegistry.register("marked_hawkes")
@dataclasses.dataclass
class MarkedHawkesConfig(MarkedHawkesBaseConfig):
    """Full ``M x M`` branching matrix — all cross-mark excitation terms are free."""

    _DIAGONAL_ONLY: ClassVar[bool] = False


@ConfigRegistry.register("marked_hawkes_diag")
@dataclasses.dataclass
class MarkedHawkesDiagConfig(MarkedHawkesBaseConfig):
    """Diagonal-only ``alpha`` — marks excite themselves but never each other."""

    _DIAGONAL_ONLY: ClassVar[bool] = True
