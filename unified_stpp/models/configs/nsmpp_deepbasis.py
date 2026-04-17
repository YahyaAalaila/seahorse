"""Construction config for the public ``nsmpp`` preset."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, ClassVar, Dict

import numpy as np

from unified_stpp.data.transforms import IdentityTransformArtifact

from .base import BaseModelConfig, ConfigRegistry


def _as_float_tuple(
    value: Any,
    *,
    dim: int,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} values, got {arr.size}.")
    return tuple(float(x) for x in arr.tolist())


@ConfigRegistry.register("nsmpp", status="provisional")
@dataclasses.dataclass
class NSMPPDeepBasisConfig(BaseModelConfig):
    _STATE_MODEL: ClassVar[str] = "nsmpp_deepbasis"
    _EVENT_MODEL: ClassVar[str] = "nsmpp_deepbasis"
    _SUPPORTED_PROTOCOLS: ClassVar[frozenset[str]] = frozenset({"raw", "unified"})

    mu: float = 1e-1
    n_basis: int = 3
    basis_dim: int = 7
    nn_width: int = 8
    int_res: int = 30
    numerical_int: bool = True
    init_gain: float = 5e-1
    init_bias: float = 1e-3
    init_std: float = 5e-1
    init_weight_mean: float = -3.0
    intensity_eps: float = 1e-5
    support_t0: float = 0.0
    support_t1: float = 1.0
    support_space_min: tuple[float, float] = (0.0, 0.0)
    support_space_max: tuple[float, float] = (1.0, 1.0)
    compensator_chunk_size: int = 256

    input_transform: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    _dec_cfg: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

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
    ) -> "NSMPPDeepBasisConfig":
        del hidden_dim, event_cov_dim, field_cov_dim, n_marks
        if spatial_dim != 2:
            raise ValueError(
                f"nsmpp currently requires spatial_dim=2, got {spatial_dim}."
            )

        dec_cfg = copy.deepcopy(d.get("decoder", {}))
        dec_type = dec_cfg.pop("type", "nsmpp_deepbasis")
        if dec_type not in {"nsmpp_deepbasis", "nmstpp_deepbasis"}:
            raise ValueError(
                "NSMPP DeepBasis build expects decoder.type in "
                "{'nsmpp_deepbasis', 'nmstpp_deepbasis'}, "
                f"got '{dec_type}'."
            )

        return cls(
            spatial_dim=spatial_dim,
            mu=float(dec_cfg.pop("mu", 1e-1)),
            n_basis=int(dec_cfg.pop("n_basis", 3)),
            basis_dim=int(dec_cfg.pop("basis_dim", 7)),
            nn_width=int(dec_cfg.pop("nn_width", 8)),
            int_res=int(dec_cfg.pop("int_res", 30)),
            numerical_int=bool(dec_cfg.pop("numerical_int", True)),
            init_gain=float(dec_cfg.pop("init_gain", 5e-1)),
            init_bias=float(dec_cfg.pop("init_bias", 1e-3)),
            init_std=float(dec_cfg.pop("init_std", 5e-1)),
            init_weight_mean=float(dec_cfg.pop("init_weight_mean", -3.0)),
            intensity_eps=float(dec_cfg.pop("intensity_eps", 1e-5)),
            support_t0=float(d.get("support_t0", 0.0)),
            support_t1=float(d.get("support_t1", 1.0)),
            support_space_min=_as_float_tuple(
                d.get("support_space_min"),
                dim=spatial_dim,
                default=(0.0, 0.0),
            ),
            support_space_max=_as_float_tuple(
                d.get("support_space_max"),
                dim=spatial_dim,
                default=(1.0, 1.0),
            ),
            compensator_chunk_size=int(dec_cfg.pop("compensator_chunk_size", 256)),
            input_transform=copy.deepcopy(d.get("input_transform", {})),
            _dec_cfg=dec_cfg,
        )

    def _state_kwargs(self) -> dict:
        return {"input_transform": copy.deepcopy(self.input_transform)}

    def _event_kwargs(self) -> dict:
        if self._dec_cfg:
            raise ValueError(f"Unknown NSMPP DeepBasis decoder overrides: {sorted(self._dec_cfg)}")
        return dict(
            spatial_dim=self.spatial_dim,
            mu=self.mu,
            n_basis=self.n_basis,
            basis_dim=self.basis_dim,
            nn_width=self.nn_width,
            int_res=self.int_res,
            numerical_int=self.numerical_int,
            init_gain=self.init_gain,
            init_bias=self.init_bias,
            init_std=self.init_std,
            init_weight_mean=self.init_weight_mean,
            intensity_eps=self.intensity_eps,
            support_t0=self.support_t0,
            support_t1=self.support_t1,
            support_space_min=self.support_space_min,
            support_space_max=self.support_space_max,
            compensator_chunk_size=self.compensator_chunk_size,
        )

    @classmethod
    def data_init_overrides(cls, dm) -> dict:
        bundle = getattr(dm, "_bundle", None)
        if bundle is None:
            return {}
        train_ds = getattr(bundle, "train_dataset", None)
        sequences = list(getattr(train_ds, "sequences", []) or [])
        if not sequences:
            return {}

        # sequences stores RAW coordinates; normalization is applied on-the-fly
        # in __getitem__. We read raw bounds then apply the dataset's norm stats
        # so the support is set in the same coordinate space the model trains in.
        time_min = float("inf")
        time_max = float("-inf")
        loc_chunks: list[np.ndarray] = []
        for seq in sequences:
            times = np.asarray(seq["times"], dtype=np.float64).reshape(-1)
            locs = np.asarray(seq["locations"], dtype=np.float64).reshape(-1, 2)
            if times.size > 0:
                time_min = min(time_min, float(times.min()))
                time_max = max(time_max, float(times.max()))
            if locs.size > 0:
                loc_chunks.append(locs.astype(np.float64))

        if loc_chunks:
            loc_all = np.concatenate(loc_chunks, axis=0)
            loc_min = loc_all.min(axis=0)
            loc_max = loc_all.max(axis=0)
        else:
            loc_min = np.zeros(2, dtype=np.float64)
            loc_max = np.ones(2, dtype=np.float64)

        t0_raw = time_min if time_min != float("inf") else 0.0
        t1_raw = time_max if time_max != float("-inf") else 1.0

        # Apply dataset normalization (identity when normalize=false, i.e. std=1, mean=0).
        time_mean = float(getattr(train_ds, "time_mean", 0.0))
        time_std  = float(getattr(train_ds, "time_std",  1.0))
        loc_mean  = np.asarray(getattr(train_ds, "loc_mean", [0.0, 0.0]), dtype=np.float64)
        loc_std   = np.asarray(getattr(train_ds, "loc_std",  [1.0, 1.0]), dtype=np.float64)

        t0 = (t0_raw - time_mean) / time_std
        t1 = (t1_raw - time_mean) / time_std
        loc_min_n = (loc_min - loc_mean) / loc_std
        loc_max_n = (loc_max - loc_mean) / loc_std

        width = loc_max_n - loc_min_n
        tiny = width < 1e-6
        if tiny.any():
            loc_min_n = loc_min_n.copy()
            loc_max_n = loc_max_n.copy()
            loc_min_n[tiny] -= 5e-5
            loc_max_n[tiny] += 5e-5

        return {
            "support_t0": float(t0),
            "support_t1": float(max(t1, t0 + 1e-6)),
            "support_space_min": tuple(float(x) for x in loc_min_n.tolist()),
            "support_space_max": tuple(float(x) for x in loc_max_n.tolist()),
        }

    @classmethod
    def fit_transform_artifact(cls, dm):
        bundle = getattr(dm, "_bundle", None)
        if bundle is None:
            return None
        train_ds = getattr(bundle, "train_dataset", None)
        if train_ds is None:
            return None
        if getattr(train_ds, "coordinate_space", None) != "raw":
            return None
        return IdentityTransformArtifact()


ConfigRegistry.register_alias(
    "nsmpp_deepbasis_provisional",
    "nsmpp",
    status="deprecated",
)
