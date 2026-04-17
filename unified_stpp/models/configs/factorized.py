"""FactorizedConfig — construction configs for the factorized baseline family.

GMM presets (GaussianMixtureSpatialModel):
  - poisson_gmm          : HomogeneousPoissonProcess + GaussianMixtureSpatialModel
  - hawkes_gmm           : HawkesProcess             + GaussianMixtureSpatialModel
  - selfcorrecting_gmm   : SelfCorrectingProcess     + GaussianMixtureSpatialModel

CNF presets (IndependentCNF, squash_time=True):
  - poisson_cnf          : HomogeneousPoissonProcess + IndependentCNF
  - hawkes_cnf           : HawkesProcess             + IndependentCNF
  - selfcorrecting_cnf   : SelfCorrectingProcess     + IndependentCNF

GMM config parameters
---------------------
sigma_prior  : initial prior std for the first event's spatial distribution
sigma_kernel : initial bandwidth of the Gaussian mixture kernels
tau          : initial time-decay rate for mixture weights
t0           : observation window start (float, default 0.0)
t1           : observation window end   (float or None; None → last-event convention)
               See FactorizedEventModel docstring for the last-event convention details.

CNF config parameters
---------------------
hidden_dims    : MLP hidden layer widths for the velocity field (tuple of ints)
layer_type     : "concat" (ConcatLinear_v2) or "concatsquash" (ConcatSquashLinear)
actfn          : "softplus" or "swish"
tol            : ODE solver tolerance (rtol = atol = tol)
otreg_strength : OT regularisation coefficient (0 = off)
squash_time    : True = cnf (time-invariant), False = tvcnf (time-varying per event)
t0, t1         : same observation window semantics as GMM configs
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import numpy as np

from unified_stpp.data.transforms import ZScoreTransformArtifact
from .base import BaseModelConfig, ConfigRegistry

if TYPE_CHECKING:
    from unified_stpp.models.unified_model import UnifiedSTPP


@dataclasses.dataclass
class FactorizedConfig(BaseModelConfig):
    """Base config for factorized STPP baselines. Subclasses set _TEMPORAL_TYPE."""

    _TEMPORAL_TYPE: ClassVar[str] = "poisson"  # overridden by registered subclasses

    sigma_prior:  float = 1.0
    sigma_kernel: float = 0.5
    tau:          float = 1.0
    t0:           float = 0.0
    t1:           Optional[float] = None
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
    ) -> "FactorizedConfig":
        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            sigma_prior=d.get("sigma_prior", 1.0),
            sigma_kernel=d.get("sigma_kernel", 0.5),
            tau=d.get("tau", 1.0),
            t0=float(d.get("t0", 0.0)),
            t1=float(d["t1"]) if d.get("t1") is not None else None,
            input_transform=dict(d.get("input_transform", {}) or {}),
        )

    def build_model(self) -> "UnifiedSTPP":
        from unified_stpp.models.temporal_models.parametric_processes import (
            HomogeneousPoissonProcess,
            HawkesProcess,
            SelfCorrectingProcess,
        )
        from unified_stpp.models.temporal_models.neural_temporal import (
            RMTPPTemporalProcess,
            THPTemporalProcess,
        )
        from unified_stpp.models.spatial_models.gaussian_mixture import GaussianMixtureSpatialModel
        from unified_stpp.models.state_models.factorized_state import FactorizedStateModel
        from unified_stpp.models.event_models.factorized_event import FactorizedEventModel
        from unified_stpp.models.unified_model import UnifiedSTPP

        _temporal_cls = {
            "poisson":        HomogeneousPoissonProcess,
            "hawkes":         HawkesProcess,
            "self_correcting": SelfCorrectingProcess,
            "rmtpp":          RMTPPTemporalProcess,
            "thp":            THPTemporalProcess,
        }[self._TEMPORAL_TYPE]

        temporal_model = _temporal_cls(**self._temporal_kwargs())
        spatial_model = GaussianMixtureSpatialModel(
            sigma_prior=self.sigma_prior,
            sigma_kernel=self.sigma_kernel,
            tau=self.tau,
        )
        state_model = FactorizedStateModel(input_transform=self.input_transform)
        event_model = FactorizedEventModel(
            temporal_model=temporal_model,
            spatial_model=spatial_model,
            t0=self.t0,
            t1=self.t1,
        )
        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )

    def _temporal_kwargs(self) -> dict:
        """Extra kwargs forwarded to the temporal model constructor.

        Parametric processes (Poisson, Hawkes, SelfCorrecting) take no arguments,
        so the base implementation returns an empty dict. Subclasses with learnable
        neural temporal models override this to pass their hyperparameters.
        """
        return {}

    @classmethod
    def fit_transform_artifact(cls, dm):
        ds = getattr(dm, "train_dataset", None)
        if ds is None:
            return None
        if getattr(ds, "coordinate_space", None) != "raw":
            return None
        sequences = list(getattr(ds, "sequences", []))
        first_seq = next(iter(sequences), None)
        if first_seq is not None:
            spatial_dim = int(np.asarray(first_seq["locations"]).shape[-1])
            all_times = np.concatenate(
                [np.asarray(seq["times"], dtype=np.float32).reshape(-1) for seq in sequences],
                axis=0,
            )
            all_locs = np.concatenate(
                [np.asarray(seq["locations"], dtype=np.float32).reshape(-1, spatial_dim) for seq in sequences],
                axis=0,
            )
            time_mean = float(all_times.mean())
            time_std = float(all_times.std() + 1e-8)
            loc_mean_arr = all_locs.mean(axis=0).astype(np.float32)
            loc_std_arr = (all_locs.std(axis=0) + 1e-8).astype(np.float32)
        else:
            spatial_dim = 2
            time_mean = 0.0
            time_std = 1.0
            loc_mean_arr = np.zeros(spatial_dim, dtype=np.float32)
            loc_std_arr = np.ones(spatial_dim, dtype=np.float32)
        return ZScoreTransformArtifact(
            normalize_time=True,
            normalize_space=True,
            time_mean=time_mean,
            time_std=time_std,
            loc_mean=tuple(
                float(x) for x in loc_mean_arr.tolist()
            ),
            loc_std=tuple(
                float(x) for x in loc_std_arr.tolist()
            ),
        )


@ConfigRegistry.register("poisson_gmm")
@dataclasses.dataclass
class PoissonGMMConfig(FactorizedConfig):
    _TEMPORAL_TYPE: ClassVar[str] = "poisson"


@ConfigRegistry.register("hawkes_gmm")
@dataclasses.dataclass
class HawkesGMMConfig(FactorizedConfig):
    _TEMPORAL_TYPE: ClassVar[str] = "hawkes"


@ConfigRegistry.register("selfcorrecting_gmm")
@dataclasses.dataclass
class SelfCorrectingGMMConfig(FactorizedConfig):
    _TEMPORAL_TYPE: ClassVar[str] = "self_correcting"


# ---------------------------------------------------------------------------
# CNF-based factorized configs  (IndependentCNF spatial model)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FactorizedCNFConfig(BaseModelConfig):
    """Base config for factorized STPP baselines with IndependentCNF spatial model.

    Parallel to FactorizedConfig; carries CNF-specific hyperparams instead of
    GMM bandwidth parameters.
    """

    _TEMPORAL_TYPE: ClassVar[str]  = "poisson"  # overridden by registered subclasses
    _SQUASH_TIME:   ClassVar[bool] = True        # overridden for tvcnf variants

    # CNF velocity-field params
    hidden_dims:    tuple           = dataclasses.field(default_factory=lambda: (64, 64, 64))
    layer_type:     str             = "concat"
    actfn:          str             = "softplus"
    tol:            float           = 1e-5
    otreg_strength: float           = 0.0
    squash_time:    bool            = True

    # Observation window (same semantics as FactorizedConfig)
    t0: float           = 0.0
    t1: Optional[float] = None
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
    ) -> "FactorizedCNFConfig":
        default_hidden_dims = (64, 64, 64)
        raw = d.get("hidden_dims", default_hidden_dims)
        return cls(
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            hidden_dims=tuple(raw),
            layer_type=d.get("layer_type", "concat"),
            actfn=d.get("actfn", "softplus"),
            tol=float(d.get("tol", 1e-5)),
            otreg_strength=float(d.get("otreg_strength", 0.0)),
            squash_time=bool(d.get("squash_time", cls._SQUASH_TIME)),
            t0=float(d.get("t0", 0.0)),
            t1=float(d["t1"]) if d.get("t1") is not None else None,
            input_transform=dict(d.get("input_transform", {}) or {}),
        )

    def build_model(self) -> "UnifiedSTPP":
        from unified_stpp.models.temporal_models.parametric_processes import (
            HomogeneousPoissonProcess,
            HawkesProcess,
            SelfCorrectingProcess,
        )
        from unified_stpp.models.spatial_models.independent_cnf import IndependentCNF
        from unified_stpp.models.state_models.factorized_state import FactorizedStateModel
        from unified_stpp.models.event_models.factorized_event import FactorizedEventModel
        from unified_stpp.models.unified_model import UnifiedSTPP

        _temporal_cls = {
            "poisson":        HomogeneousPoissonProcess,
            "hawkes":         HawkesProcess,
            "self_correcting": SelfCorrectingProcess,
        }[self._TEMPORAL_TYPE]

        temporal_model = _temporal_cls()
        spatial_model  = IndependentCNF(
            dim=self.spatial_dim,
            hidden_dims=self.hidden_dims,
            layer_type=self.layer_type,
            actfn=self.actfn,
            tol=self.tol,
            otreg_strength=self.otreg_strength,
            squash_time=self.squash_time,
        )
        state_model = FactorizedStateModel(input_transform=self.input_transform)
        event_model = FactorizedEventModel(
            temporal_model=temporal_model,
            spatial_model=spatial_model,
            t0=self.t0,
            t1=self.t1,
        )
        return UnifiedSTPP(
            state_model=state_model,
            event_model=event_model,
            hidden_dim=self.hidden_dim,
        )

    @classmethod
    def fit_transform_artifact(cls, dm):
        return FactorizedConfig.fit_transform_artifact(dm)


@ConfigRegistry.register("poisson_cnf")
@dataclasses.dataclass
class PoissonCNFConfig(FactorizedCNFConfig):
    _TEMPORAL_TYPE: ClassVar[str] = "poisson"


@ConfigRegistry.register("hawkes_cnf")
@dataclasses.dataclass
class HawkesCNFConfig(FactorizedCNFConfig):
    _TEMPORAL_TYPE: ClassVar[str] = "hawkes"


@ConfigRegistry.register("selfcorrecting_cnf")
@dataclasses.dataclass
class SelfCorrectingCNFConfig(FactorizedCNFConfig):
    _TEMPORAL_TYPE: ClassVar[str] = "self_correcting"


# ---------------------------------------------------------------------------
# tvcnf presets (squash_time=False — time-varying per event)
# ---------------------------------------------------------------------------

@ConfigRegistry.register("poisson_tvcnf")
@dataclasses.dataclass
class PoissonTVCNFConfig(FactorizedCNFConfig):
    _TEMPORAL_TYPE: ClassVar[str]  = "poisson"
    _SQUASH_TIME:   ClassVar[bool] = False
    squash_time:    bool           = False


@ConfigRegistry.register("hawkes_tvcnf")
@dataclasses.dataclass
class HawkesTVCNFConfig(FactorizedCNFConfig):
    _TEMPORAL_TYPE: ClassVar[str]  = "hawkes"
    _SQUASH_TIME:   ClassVar[bool] = False
    squash_time:    bool           = False


@ConfigRegistry.register("selfcorrecting_tvcnf")
@dataclasses.dataclass
class SelfCorrectingTVCNFConfig(FactorizedCNFConfig):
    _TEMPORAL_TYPE: ClassVar[str]  = "self_correcting"
    _SQUASH_TIME:   ClassVar[bool] = False
    squash_time:    bool           = False


# ---------------------------------------------------------------------------
# Neural temporal baselines (GMM spatial, neural temporal encoder)
# ---------------------------------------------------------------------------

@ConfigRegistry.register("rmtpp_gmm")
@dataclasses.dataclass
class RMTPPGMMConfig(FactorizedConfig):
    """RMTPP temporal process + GaussianMixture spatial model.

    Temporal: GRU encoder + closed-form exponential intensity (Du et al., KDD 2016).
    Spatial:  same GaussianMixtureSpatialModel as hawkes_gmm.

    Note: this is a factorized STPP construction (temporal × spatial) — the
    RMTPP was originally validated on pure temporal benchmarks only. Label as
    "RMTPP-GMM (extended)" in paper tables. Hyperparameters need a sweep before
    reporting final numbers.
    """

    _TEMPORAL_TYPE: ClassVar[str] = "rmtpp"

    hidden_size: int = 64

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        hidden_dim: int = 128,
        spatial_dim: int = 2,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        n_marks: int = 0,
    ) -> "RMTPPGMMConfig":
        obj = super().from_dict(
            d,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            event_cov_dim=event_cov_dim,
            field_cov_dim=field_cov_dim,
            n_marks=n_marks,
        )
        obj.hidden_size = int(d.get("hidden_size", 64))
        return obj

    def _temporal_kwargs(self) -> dict:
        return {"hidden_size": self.hidden_size}


@ConfigRegistry.register("thp_gmm")
@dataclasses.dataclass
class THPGMMConfig(FactorizedConfig):
    """Transformer Hawkes Process temporal + GaussianMixture spatial model.

    Temporal: causal Transformer encoder + learned β decay (Zuo et al., ICML 2020).
    Spatial:  same GaussianMixtureSpatialModel as hawkes_gmm.
    MC compensator uses fixed 30-sample Monte Carlo (upstream adaptive loop replaced).

    Note: this is a factorized STPP construction — THP was validated on pure temporal
    benchmarks. Label as "THP-GMM (extended)". Hyperparameters need a sweep.
    """

    _TEMPORAL_TYPE: ClassVar[str] = "thp"

    hidden_size: int = 64
    n_heads:     int = 2
    n_layers:    int = 1
    dropout:     float = 0.1

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        hidden_dim: int = 128,
        spatial_dim: int = 2,
        event_cov_dim: int = 0,
        field_cov_dim: int = 0,
        n_marks: int = 0,
    ) -> "THPGMMConfig":
        obj = super().from_dict(
            d,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            event_cov_dim=event_cov_dim,
            field_cov_dim=field_cov_dim,
            n_marks=n_marks,
        )
        obj.hidden_size = int(d.get("hidden_size", 64))
        obj.n_heads     = int(d.get("n_heads", 2))
        obj.n_layers    = int(d.get("n_layers", 1))
        obj.dropout     = float(d.get("dropout", 0.1))
        return obj

    def _temporal_kwargs(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "n_heads":     self.n_heads,
            "n_layers":    self.n_layers,
            "dropout":     self.dropout,
        }
