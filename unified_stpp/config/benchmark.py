"""Benchmark procedure configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from unified_stpp.config.schema import STPPConfig

from pydantic import BaseModel, Field, field_validator

from unified_stpp.config.tuning import TuningConfig


class BenchmarkConfig(BaseModel):
    """Benchmark policy for HPO, evaluation, and scalar reporting.

    This config intentionally excludes benchmark inputs such as ``presets``,
    resolved ``splits``, output paths, and override dicts. Those remain
    invocation-level concerns owned by the caller.
    """

    run_hpo: bool = False
    tuning: Optional[TuningConfig] = None
    tune_dataset: Optional[str] = None

    seeds: list[int] = Field(default_factory=lambda: [42])
    backend: Literal["sequential", "joblib", "ray"] = "joblib"
    n_workers: int = 1

    protocol: Literal["raw", "standard"] = "raw"
    normalize: bool = False

    primary_metric: str = "test_nll"
    """Reporting/aggregation metric used by benchmark tables and reports only."""

    @field_validator("seeds")
    @classmethod
    def seeds_must_be_non_empty(cls, seeds: list[int]) -> list[int]:
        if not seeds:
            raise ValueError("BenchmarkConfig.seeds must contain at least one seed.")
        return seeds

    @field_validator("n_workers")
    @classmethod
    def n_workers_must_be_positive(cls, n_workers: int) -> int:
        if n_workers < 1:
            raise ValueError("BenchmarkConfig.n_workers must be >= 1.")
        return n_workers

    def resolved_tuning(self) -> TuningConfig:
        """Return the configured HPO procedure or default tuning settings."""
        return self.tuning or TuningConfig()

    def apply_to_config(self, cfg: "STPPConfig") -> "STPPConfig":
        """Return a copy of *cfg* with the benchmark data contract enforced.

        Forces ``data.protocol`` and ``data.normalize`` to match the benchmark
        policy regardless of what the preset YAML declares.  All other fields
        are preserved. The raw-first benchmark default keeps the canonical
        batch contract in original dataset coordinates while legacy
        ``protocol="standard"`` remains available for compatibility.
        """
        from unified_stpp.config.schema import STPPConfig
        raw = cfg.model_dump(mode="json")
        raw.setdefault("data", {})
        raw["data"]["protocol"] = self.protocol
        raw["data"]["normalize"] = self.normalize
        return STPPConfig(**raw)


__all__ = ["BenchmarkConfig"]
