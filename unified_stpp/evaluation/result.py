"""Metric base class, MetricResult, and Report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .context import EvalContext


class Metric:
    """Base class for all evaluation metrics.

    Subclasses must define:
        name        — snake_case string identifier (e.g. "temporal_crps")
        requires    — frozenset of capability/artifact strings from the fixed vocabulary
        compute(ctx) → MetricResult

    Optional class attributes:
        catalog_id  — cross-reference to metrics_catalog.md (e.g. "M6"), None for custom metrics
        produces    — dict mapping MetricResult field names to their types (documentation only)
        artifact_families — Evaluation artifacts this metric may consume.
        cost_class  — Human-readable cost class used by profile planning.
    """

    name: str
    catalog_id: str | None = None
    requires: frozenset[str] = frozenset()
    artifact_families: frozenset[str] = frozenset()
    cost_class: str = "core"
    produces: dict[str, type] = {}

    def compute(self, ctx: "EvalContext") -> "MetricResult":
        raise NotImplementedError(f"{type(self).__name__}.compute() not implemented")


@dataclass
class MetricResult:
    """Structured output from a single metric computation.

    Attributes
    ----------
    value:      Primary scalar result (None when unavailable or metric only produces a curve).
    per_event:  Optional (N,) float64 array of per-event values, saved to eval/<name>_per_event.npy.
    curve:      Optional x→y mapping (e.g. recall curves, context-sensitivity curves).
    method:     How the metric was computed: "exact" | "quadrature" | "kde" | "vb" | "thinning" | "native".
    available:  False when the metric was skipped (missing requirements or error).
    reason:     Human-readable explanation when available=False, or None.
    """

    value: float | None
    per_event: np.ndarray | None = None
    curve: dict[str | float, float] | None = None
    method: str = "exact"
    available: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "value": self.value,
            "method": self.method,
            "available": self.available,
        }
        if self.reason is not None:
            d["reason"] = self.reason
        if self.curve is not None:
            d["curve"] = {str(k): float(v) for k, v in self.curve.items()}
        return d


@dataclass
class Report:
    """Collection of MetricResults from one evaluation run."""

    results: dict[str, MetricResult] = field(default_factory=dict)
    artifact_events: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, name: str) -> MetricResult:
        return self.results[name]

    def __contains__(self, name: str) -> bool:
        return name in self.results

    def to_dict(self) -> dict[str, Any]:
        return {name: r.to_dict() for name, r in self.results.items()}

    def scalars(self) -> dict[str, float | None]:
        """Return a flat dict of available scalar values, keyed by metric name."""
        return {
            name: r.value
            for name, r in self.results.items()
            if r.available and r.value is not None
        }

    def summary(self) -> str:
        """Pretty-print all metric results."""
        lines = []
        for name in sorted(self.results):
            r = self.results[name]
            if not r.available:
                lines.append(f"  {name:<40s} —  ({r.reason})")
            elif r.value is not None:
                lines.append(f"  {name:<40s} {r.value:>12.5g}  [{r.method}]")
            else:
                lines.append(f"  {name:<40s} (curve/array only)  [{r.method}]")
        return "\n".join(lines)

    def save(self, out_dir: str | Path) -> None:
        """Write metrics.json and per-event .npy arrays to out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        for name, r in self.results.items():
            if r.per_event is not None:
                np.save(out_dir / f"{name}_per_event.npy", r.per_event.astype(np.float64))
