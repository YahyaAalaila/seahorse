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
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from unified_stpp.data.registry import DataBundle

import yaml
import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unified_stpp.config.tuning import TuningConfig
from unified_stpp.utils import deep_update, parse_overrides


class _CompatSafeLoader(yaml.SafeLoader):
    """Safe YAML loader with minimal legacy run-artifact compatibility."""


def _construct_python_tuple_as_list(loader, node):
    """Read legacy ``!!python/tuple`` values as plain lists.

    Older run artifacts may contain tuple tags because they were dumped from
    Python-mode config dicts. Config validation can coerce lists back to tuple
    fields, so loading them as lists is the safest compatibility behavior.
    """
    return loader.construct_sequence(node)


_CompatSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple_as_list,
)


def _yaml_load_compat(stream) -> dict[str, Any]:
    data = yaml.load(stream, Loader=_CompatSafeLoader)
    return data or {}


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

    protocol: str = "standard"
    """Normalization + batching protocol.
    ``"raw"``             — canonical raw/original coordinates.
    ``"standard"``        — legacy z-score absolute times + locations.
    ``"sliding_window"``  — MinMax delta-t + sliding windows (AutoSTPP paper pipeline).
    """
    normalize: bool = True
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    # Paper-protocol options (ignored for "unified")
    paper_lookback: Optional[int] = None
    paper_lookahead: int = 1
    paper_split_ratio: tuple[int, int, int] = (8, 1, 1)
    # Forwarded to DataAdapter constructor. Example: {"max_events": 5000}
    adapter_kwargs: dict = Field(default_factory=dict)
    dataset: Optional[str] = None
    dataset_revision: Optional[str] = None
    splits_dir: Optional[str] = None
    datasets: list[str] = Field(default_factory=list)
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None

    def resolve_data(
        self,
        *,
        mode: Literal["single", "benchmark"] = "single",
        include_test: bool = True,
    ):
        """Resolve dataset sources for CLI flows.

        Keeps the config schema lightweight while delegating actual path/cache
        logic to :mod:`unified_stpp.data.resolution`.
        """
        from unified_stpp.data.resolution import resolve_data_source

        return resolve_data_source(self, mode=mode, include_test=include_test)


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
        from unified_stpp.models.configs import ConfigRegistry

        if not ConfigRegistry.is_registered(v):
            known = ConfigRegistry.preset_names()
            raise ValueError(
                f"Unknown preset '{v}'. Available presets: {sorted(known)}"
            )
        return ConfigRegistry.resolve_name(v)

    @property
    def build_overrides(self) -> dict[str, Any]:
        """Extra YAML keys forwarded to ``build_model(config=...)``."""
        return dict(self.model_extra) if self.model_extra else {}

    def build_model(self, extra_overrides: "dict[str, Any] | None" = None):
        """Build and return a ``UnifiedSTPP`` from this config.

        Parameters
        ----------
        extra_overrides : additional overrides merged on top of ``build_overrides``
                          *after* the YAML-level overrides (e.g. data-derived bbox
                          computed by ``ConfigRegistry.data_init_overrides``).
        """
        from unified_stpp.models.configs import ConfigRegistry
        overrides = dict(self.build_overrides)
        if extra_overrides:
            from unified_stpp.utils import deep_update
            deep_update(overrides, extra_overrides)
        return ConfigRegistry.build(
            self.preset,
            overrides=overrides,
            hidden_dim=self.hidden_dim,
            spatial_dim=self.spatial_dim,
            n_marks=self.n_marks,
            event_cov_dim=self.event_cov_dim,
            field_cov_dim=self.field_cov_dim,
        )


class TrainingConfig(BaseModel):
    """Optimiser and training loop configuration."""

    # TRANSITION: extra="ignore" is explicit here (matches Pydantic v2 default) so the
    # intent is clear.  The model_validator below upgrades silence to a soft warning for
    # known-renamed fields.  Switch this to extra="forbid" once all bundled YAMLs and
    # downstream user configs have been audited — see architecture refactor Phase 1 notes.
    model_config = ConfigDict(extra="ignore")

    # Fields that were renamed at some point; kept here so the validator can
    # emit a targeted warning instead of silently discarding them.
    _RENAMED_FIELDS: dict[str, str] = {
        "max_epochs": "n_epochs",
        "early_stopping_patience": "patience",
    }

    @model_validator(mode="before")
    @classmethod
    def warn_on_renamed_fields(cls, values: object) -> object:
        if not isinstance(values, dict):
            return values
        for old, new in {
            "max_epochs": "n_epochs",
            "early_stopping_patience": "patience",
        }.items():
            if old in values:
                warnings.warn(
                    f"TrainingConfig: '{old}' is not a valid field and will be ignored. "
                    f"Did you mean '{new}'?",
                    UserWarning,
                    stacklevel=2,
                )
        return values

    lr: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    n_epochs: int = 50
    batch_size: int = 64
    patience: Optional[int] = None
    """Early stopping patience (epochs). ``None`` disables early stopping."""
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    device: str = "auto"
    devices: int | str | list[int] | None = None
    """Lightning device selection override, e.g. ``1``, ``"auto"``, or ``[0, 1]``."""
    strategy: Optional[str] = None
    """Distributed strategy passed through to Lightning, e.g. ``"ddp"``."""
    precision: int | str | None = None
    """Numeric precision passed through to Lightning, e.g. ``"16-mixed"`` or ``64``."""
    num_nodes: int = 1
    """Number of cluster nodes for distributed Lightning strategies."""
    lr_schedule: str = "constant"
    """LR schedule: ``"constant"``, ``"cosine"``, ``"linear_decay"``, ``"step"``, or ``"reduce_on_plateau"``."""
    lr_warmup_epochs: int = 0
    """Linear warmup epochs before the main schedule. Used by cosine and linear_decay."""
    lr_final: Optional[float] = None
    """Final learning rate for ``lr_schedule="linear_decay"``. Defaults to 0 when unset."""
    lr_step_size: Optional[int] = None
    """If set, use StepLR: multiply lr by ``lr_step_gamma`` every ``lr_step_size`` epochs."""
    lr_step_gamma: float = 0.5
    """Multiplicative decay factor for StepLR."""
    checkpoint_select: Literal["best", "last"] = "best"
    """Which checkpoint to use for post-fit test and load flows."""
    resume_from_checkpoint: Optional[str] = None
    """Optional Lightning checkpoint path for resuming fit() with optimizer/epoch state."""
    test_nll_space: Literal["native", "raw"] = "raw"
    """Reporting space for test NLL: native/model space or raw/original data space when supported."""
    predictive_test_nll_samples: int = 128
    """Monte Carlo sample count for sample-based held-out next-event test NLL reporting."""
    vae_beta: float = 0.0
    """KL weight for VAE regularization (ELBO beta). 0 disables KL (non-VAE mode)."""

    def build_callbacks(self, run_dir: Path, monitor_key: str = "val/nll") -> list:
        """Construct and return Lightning training callbacks."""
        from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
        ckpt = ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            filename="best",
            monitor=monitor_key,
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks = [ckpt, LearningRateMonitor()]
        if self.patience is not None:
            callbacks.append(
                EarlyStopping(monitor=monitor_key, patience=self.patience, mode="min")
            )
        return callbacks

    def build_optimizer(self, parameters):
        """Construct and return the configured optimizer for *parameters*."""
        import torch
        opt_name = self.optimizer.strip().lower()
        if opt_name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        if opt_name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
            )
        if opt_name == "adadelta":
            return torch.optim.Adadelta(
                parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        raise ValueError(
            f"Unknown optimizer '{self.optimizer}'. Expected 'adam', 'adamw', or 'adadelta'."
        )

    def build_lr_scheduler(self, optimizer, trainer=None, monitor_key: str = "val/nll") -> "dict":
        """Construct and return a Lightning scheduler config dict.

        All schedules — including ``"constant"`` — return a dict ready for
        Lightning's ``{"optimizer": ..., "lr_scheduler": ...}`` return value.
        ``trainer`` is required only for ``"cosine"`` (needs ``max_epochs``);
        it may be ``None`` for all other schedules.
        """
        import math
        import torch

        if self.lr_schedule == "constant" and self.lr_step_size is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
            return {"scheduler": scheduler, "interval": "epoch"}

        if self.lr_schedule == "cosine":
            n_epochs = trainer.max_epochs if trainer is not None else self.n_epochs
            warmup = self.lr_warmup_epochs

            def _schedule(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / max(warmup, 1)
                progress = (epoch - warmup) / max(n_epochs - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_schedule)
            return {"scheduler": scheduler, "interval": "epoch"}

        if self.lr_schedule == "linear_decay":
            n_epochs = trainer.max_epochs if trainer is not None else self.n_epochs
            warmup = self.lr_warmup_epochs
            final_lr = 0.0 if self.lr_final is None else float(self.lr_final)
            final_ratio = final_lr / max(self.lr, 1e-12)

            def _schedule(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / max(warmup, 1)
                progress = (epoch - warmup) / max(n_epochs - warmup - 1, 1)
                progress = min(max(progress, 0.0), 1.0)
                return 1.0 - (1.0 - final_ratio) * progress

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_schedule)
            return {"scheduler": scheduler, "interval": "epoch"}

        if self.lr_schedule == "step" or self.lr_step_size is not None:
            if self.lr_step_size is None:
                raise ValueError(
                    "lr_schedule='step' requires lr_step_size to be set in TrainingConfig."
                )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_step_size, gamma=self.lr_step_gamma
            )
            return {"scheduler": scheduler, "interval": "epoch"}

        if self.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )
            return {"scheduler": scheduler, "monitor": monitor_key, "interval": "epoch"}

        raise ValueError(
            f"Unknown lr_schedule={self.lr_schedule!r}. "
            "Valid options: 'constant', 'cosine', 'linear_decay', 'step', 'reduce_on_plateau'."
        )

    def build_trainer(
        self,
        run_dir: Path,
        accelerator: str,
        loggers: list,
        monitor_key: str = "val/nll",
        extra_callbacks: Optional[list] = None,
    ):
        """Construct and return a Lightning Trainer.

        ``inference_mode=False`` is required so decoders using
        ``torch.autograd.grad`` internally (e.g. AutoInt) can call
        ``enable_grad()`` during validation/test.
        """
        import pytorch_lightning as pl
        callbacks = self.build_callbacks(run_dir, monitor_key=monitor_key)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        trainer_kwargs = dict(
            max_epochs=self.n_epochs,
            accelerator=accelerator,
            callbacks=callbacks,
            logger=loggers,
            enable_progress_bar=True,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            inference_mode=False,
        )
        if self.devices is not None:
            trainer_kwargs["devices"] = self.devices
        if self.strategy is not None:
            trainer_kwargs["strategy"] = self.strategy
        if self.precision is not None:
            trainer_kwargs["precision"] = self.precision
        if self.num_nodes != 1:
            trainer_kwargs["num_nodes"] = self.num_nodes
        return pl.Trainer(**trainer_kwargs)

class LoggingConfig(BaseModel):
    """Output and logging configuration."""

    out_dir: str = "artifacts/"
    experiment_name: Optional[str] = None
    save_checkpoints: bool = True
    wandb: Optional[dict[str, Any]] = None
    """W&B config dict, e.g. ``{"project": "stpp", "entity": "myteam"}``."""

    def run_dir(self, preset_name: str, run_id: str) -> Path:
        """Return the unique run directory for this experiment.

        Layout: ``{out_dir}/fit/{experiment_name or preset}/{run_id}/``
        """
        return Path(self.out_dir) / "fit" / (self.experiment_name or preset_name) / run_id

    def build_loggers(self, run_dir: Path) -> list:
        """Construct and return Lightning loggers.

        ``CSVLogger`` writes ``metrics.csv`` directly into *run_dir*
        (not buried under ``lightning_logs/version_N/``).
        """
        from pytorch_lightning.loggers import CSVLogger
        loggers: list = [
            CSVLogger(save_dir=str(run_dir.parent), name=run_dir.name, version="")
        ]
        if self.wandb:
            try:
                from pytorch_lightning.loggers import WandbLogger
                loggers.append(WandbLogger(name=run_dir.name, **self.wandb))
            except ImportError:
                pass
        return loggers


# ---------------------------------------------------------------------------
# Top-level config with cross-field validators
# ---------------------------------------------------------------------------

class STPPConfig(BaseModel):
    """Top-level configuration envelope."""

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig = LoggingConfig()
    tuning: Optional[TuningConfig] = None
    """HPO procedure config. ``None`` means no tuning section (the common case)."""

    @model_validator(mode="after")
    def sync_batch_size(self) -> "STPPConfig":
        """``training.batch_size`` is authoritative; syncs to ``data.batch_size``."""
        self.data.batch_size = self.training.batch_size
        return self

    @model_validator(mode="after")
    def infer_paper_lookback(self) -> "STPPConfig":
        """Auto-infer ``paper_lookback`` from the model's spatial decoder ``seq_len``."""
        if self.data.protocol != "sliding_window":
            return self
        if self.data.paper_lookback is not None:
            return self
        seq_len = self.model.build_overrides.get("decoder", {}).get("spatial", {}).get("seq_len")
        if seq_len is not None:
            self.data.paper_lookback = int(seq_len)
        return self

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def raw_source_dict(
        cls,
        preset: "str | None" = None,
        config: "str | None" = None,
    ) -> dict[str, Any]:
        """Return the unsanitized raw config dict for a preset or YAML path.

        ``config`` takes precedence when both parameters are provided. For a
        preset without a bundled YAML file, returns the same minimal dict shape
        that :meth:`from_preset` historically expanded through schema defaults.
        """
        if config:
            with open(config) as f:
                return _yaml_load_compat(f)
        if preset:
            from unified_stpp.models.configs import ConfigRegistry

            source_preset = (
                ConfigRegistry.resolve_name(preset)
                if ConfigRegistry.is_registered(preset)
                else preset
            )
            yaml_path = Path(__file__).parent.parent / "configs" / f"{source_preset}.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    return _yaml_load_compat(f)
            return {"data": {}, "model": {"preset": source_preset}, "training": {}}
        raise ValueError("Either 'preset' or 'config' must be provided.")

    @classmethod
    def from_source(
        cls,
        preset: "str | None" = None,
        config: "str | None" = None,
        *,
        cli_values: Optional[dict[str, Any]] = None,
        override_list: Optional[list[str]] = None,
        sanitize: bool = True,
    ) -> "STPPConfig":
        """Build config from a preset/YAML source plus CLI and dotted overrides.

        Merge precedence is:
        schema defaults ← preset/YAML ← explicit CLI values ← ``--override``.
        """
        raw = cls.raw_source_dict(preset=preset, config=config)
        if sanitize and _has_search_space_syntax(raw):
            src = config if config else f"preset '{preset}'"
            warnings.warn(
                f"{src!r} contains HPO search-space syntax (list choices or "
                "{min/max} ranges) that will be collapsed to scalar defaults. "
                "To run HPO over this config, use "
                "'python -m unified_stpp tune' instead.",
                UserWarning,
                stacklevel=2,
            )
        merged = _sanitize_search_space(raw) if sanitize else raw
        if cli_values:
            deep_update(merged, cli_values)
        parsed_overrides = parse_overrides(override_list or [])
        if parsed_overrides:
            deep_update(merged, parsed_overrides)
        return cls(**merged)

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
        return cls.from_source(config=str(path), sanitize=sanitize)

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
        return cls.from_source(preset=preset)

    @staticmethod
    def split_tuning_dict(raw: dict) -> "tuple[dict, dict]":
        """Pop and return the ``tuning:`` section from a raw config dict.

        Returns ``(config_raw, tuning_raw)`` where ``tuning_raw`` is the raw
        tuning/search-space section (or ``{}``) and ``config_raw`` is the
        remainder.  Mutates *raw* in-place (removes ``"tuning"``).

        Used by the ``tune`` CLI so the HPO search-space dict is isolated from
        the model config dict before being handed to :func:`run_hpo`.
        """
        tuning_raw = raw.pop("tuning", None) or {}
        return raw, tuning_raw

    @classmethod
    def yaml_path_for_source(cls, preset: "str | None", config: "str | None") -> Path:
        """Return the resolved YAML path for a preset name or a user-supplied path.

        Used by commands (e.g. ``tune``) that need the raw YAML dict rather than
        a fully-validated ``STPPConfig``.  Raises ``FileNotFoundError`` if a
        preset name is given but no bundled YAML exists for it.
        """
        if config:
            return Path(config)
        from unified_stpp.models.configs import ConfigRegistry

        source_preset = (
            ConfigRegistry.resolve_name(preset)
            if preset is not None and ConfigRegistry.is_registered(preset)
            else preset
        )
        yaml_path = Path(__file__).parent.parent / "configs" / f"{source_preset}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"No bundled YAML for preset '{preset}' at {yaml_path}"
            )
        return yaml_path

    def build_data_bundle(
        self,
        train_seqs: list,
        val_seqs: list,
        test_seqs=None,
    ) -> "DataBundle":
        """Build the resolved data components for already-split sequences.

        Fully registry-driven — no if/else branching here.
        Protocol validation, dataset construction, collate resolution, and
        train-loader policy are all delegated to DataRegistry via name lookups
        declared on each model config class.

        Returns a ``DataBundle``; callers wrap it in ``STPPDataModule`` to get
        the Lightning interface.
        """
        from unified_stpp.data.registry import DataBundle, DataRegistry
        from unified_stpp.models.configs import ConfigRegistry

        preset   = self.model.preset
        protocol = self.data.protocol

        # 1. Validate — supported protocols declared on model config, not hardcoded
        supported = ConfigRegistry.get_supported_protocols(preset)
        if supported and protocol not in supported:
            raise ValueError(
                f"Protocol {protocol!r} is not supported by preset {preset!r}. "
                f"Supported: {sorted(supported)}"
            )

        # 2. Build datasets — all logic lives in the registered builder
        train_ds, val_ds, test_ds = DataRegistry.build_datasets(
            protocol, self.data, train_seqs, val_seqs, test_seqs
        )

        # 3. Resolve collate — name declared on model config
        collate = DataRegistry.get_collate(ConfigRegistry.get_collate_key(preset))

        # 4. Build train loader policy — name declared on model config
        loader_kwargs = dict(getattr(self.data, "adapter_kwargs", None) or {})
        train_batch_sampler = DataRegistry.build_train_loader(
            ConfigRegistry.get_train_loader_key(preset), train_ds, **loader_kwargs
        )

        return DataBundle(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            collate_fn=collate,
            train_batch_sampler=train_batch_sampler,
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten config to dotted-key dict (used by HPO parser)."""
        return _flatten(self.model_dump(exclude_none=False))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_search_space_syntax(d: dict) -> bool:
    """Return True if *d* contains any HPO search-space syntax.

    Detects ``{min: x, max: y}`` dicts and list-valued fields whose first
    element is a scalar (discrete choice sets), both of which are collapsed
    by :func:`_sanitize_search_space`.
    """
    for v in d.values():
        if isinstance(v, dict):
            if "min" in v and "max" in v:
                return True
            if _has_search_space_syntax(v):
                return True
        elif isinstance(v, list) and v and not isinstance(v[0], (dict, list)):
            return True
    return False


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
