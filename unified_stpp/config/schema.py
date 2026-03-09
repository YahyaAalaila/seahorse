"""
Pydantic v2 configuration schema for the unified STPP framework.

Layers
------
DataConfig    — data loading, normalization protocol, splits
ModelConfig   — preset + dimension params + arbitrary nested overrides
TrainingConfig — optimiser, scheduler, gradient clipping, early stopping
LoggingConfig  — output directory, checkpoint saving, optional W&B
STPPConfig    — top-level envelope with cross-field validators
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Data loading and normalization configuration.

    Extra YAML keys (e.g. ``n_train``, ``T``, ``rg_num_marks``) are accepted
    and passed through so dataset-specific parameters don't need a separate
    schema.
    """

    model_config = ConfigDict(extra="allow")

    protocol: str = "unified"
    """Normalization + batching protocol.
    ``"unified"``           — z-score absolute times + locations.
    ``"paper_autostpp_sthp"`` — MinMax delta-t + sliding windows (AutoSTPP paper).
    """
    normalize: bool = True
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    # Paper-protocol options (ignored for "unified")
    paper_lookback: Optional[int] = None
    paper_lookahead: int = 1
    paper_split_ratio: tuple[int, int, int] = (8, 1, 1)


class ModelConfig(BaseModel):
    """Model preset + dimension parameters.

    Any extra YAML keys (``encoder``, ``decoder``, ``dynamics``, …) are
    accepted and accessible via :attr:`build_overrides` — they are forwarded
    to :func:`unified_stpp.registry.build_model` as the override dict.
    """

    model_config = ConfigDict(extra="allow")

    preset: str
    hidden_dim: int = 128
    spatial_dim: int = 2
    n_marks: int = 0
    event_cov_dim: int = 0
    field_cov_dim: int = 0

    @field_validator("preset", mode="before")
    @classmethod
    def preset_must_exist(cls, v: str) -> str:
        from unified_stpp.registry import PRESETS

        if v not in PRESETS:
            raise ValueError(
                f"Unknown preset '{v}'. Available presets: {sorted(PRESETS)}"
            )
        return v

    @property
    def build_overrides(self) -> dict[str, Any]:
        """Extra YAML keys forwarded to ``build_model(config=...)``."""
        return dict(self.model_extra) if self.model_extra else {}


class TrainingConfig(BaseModel):
    """Optimiser and training loop configuration."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    n_epochs: int = 100
    batch_size: int = 64
    patience: Optional[int] = None
    """Early stopping patience (epochs). ``None`` disables early stopping."""
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    device: str = "auto"
    lr_schedule: str = "constant"
    """LR schedule: ``"constant"`` (ReduceLROnPlateau) or ``"cosine"`` (cosine annealing)."""
    lr_warmup_epochs: int = 0
    """Linear warmup epochs before the main schedule. Only used when ``lr_schedule="cosine"``."""
    lr_step_size: Optional[int] = None
    """If set, use StepLR: multiply lr by ``lr_step_gamma`` every ``lr_step_size`` epochs."""
    lr_step_gamma: float = 0.5
    """Multiplicative decay factor for StepLR."""
    vae_beta: float = 0.0
    """KL weight for VAE regularization (ELBO beta). 0 disables KL (non-VAE mode)."""

    @field_validator("n_epochs", mode="before")
    @classmethod
    def accept_max_epochs_alias(cls, v: Any) -> Any:
        """Accept ``max_epochs`` as a legacy alias for ``n_epochs``."""
        return v


class LoggingConfig(BaseModel):
    """Output and logging configuration."""

    out_dir: str = "runs/"
    experiment_name: Optional[str] = None
    save_checkpoints: bool = True
    wandb: Optional[dict[str, Any]] = None
    """W&B config dict, e.g. ``{"project": "stpp", "entity": "myteam"}``."""


# ---------------------------------------------------------------------------
# Top-level config with cross-field validators
# ---------------------------------------------------------------------------

class STPPConfig(BaseModel):
    """Top-level configuration envelope."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def sync_batch_size(self) -> "STPPConfig":
        """``training.batch_size`` is authoritative; syncs to ``data.batch_size``."""
        self.data.batch_size = self.training.batch_size
        return self

    @model_validator(mode="after")
    def infer_paper_lookback(self) -> "STPPConfig":
        """Auto-infer ``paper_lookback`` from the model's spatial decoder ``seq_len``."""
        if self.data.protocol != "paper_autostpp_sthp":
            return self
        if self.data.paper_lookback is not None:
            return self
        from unified_stpp.registry import PRESETS

        seq_len = (
            self.model.build_overrides.get("decoder", {}).get("spatial", {}).get("seq_len")
            or PRESETS.get(self.model.preset, {})
            .get("decoder", {})
            .get("spatial", {})
            .get("seq_len")
        )
        if seq_len is not None:
            self.data.paper_lookback = int(seq_len)
        return self

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path, sanitize: bool = True) -> "STPPConfig":
        """Load config from a YAML file.

        Parameters
        ----------
        sanitize : bool
            When ``True`` (default), search-space syntax (``{min, max}`` dicts
            and choice lists) is collapsed to scalar defaults so the YAML can
            serve as both a config *and* an HPO search-space template.
            Set to ``False`` when loading a fully-resolved config saved by
            ``to_yaml()``; skipping sanitisation prevents list-valued fields
            (e.g. ``paper_split_ratio: [8, 1, 1]``) from being misread as
            choice sets.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**(_sanitize_search_space(raw) if sanitize else raw))

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # mode="json" converts tuples → lists so yaml.safe_load can round-trip
            yaml.dump(
                self.model_dump(mode="json", exclude_none=False),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_preset(cls, preset: str) -> "STPPConfig":
        """Load the bundled YAML for *preset*, falling back to bare defaults."""
        yaml_path = Path(__file__).parent.parent / "configs" / f"{preset}.yaml"
        if yaml_path.exists():
            return cls.from_yaml(yaml_path)
        # Bare minimum — preset defaults from registry will fill in the model
        return cls(
            data=DataConfig(),
            model=ModelConfig(preset=preset),
            training=TrainingConfig(),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten config to dotted-key dict (used by HPO parser)."""
        return _flatten(self.model_dump(exclude_none=False))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_search_space(d: dict) -> dict:
    """Replace search-space specs with scalar defaults so Pydantic can validate.

    ``{min: x, max: y}``           →  ``x``        (lower bound)
    ``{min: x, max: y, default: z}`` →  ``z``      (explicit default)
    ``[a, b, c]``                  →  ``a``        (first choice)
    """
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if "min" in v and "max" in v:
                out[k] = v.get("default", v["min"])
            else:
                out[k] = _sanitize_search_space(v)
        elif isinstance(v, list) and v and not isinstance(v[0], (dict, list)):
            out[k] = v[0]
        else:
            out[k] = v
    return out


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested dict to ``{a.b.c: value}`` form."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=key))
        else:
            out[key] = v
    return out
