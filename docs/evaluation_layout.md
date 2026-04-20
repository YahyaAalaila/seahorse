# Why `evaluation/` Has So Many Files

`unified_stpp/evaluation/` has ~15 source files. That's not bloat — it reflects several distinct responsibilities that genuinely cannot live in the same file.

## The main split

**Metric/report system** (`result.py`, `registry.py`, `evaluator.py`, `profiles.py`, `metrics/`)  
The registry-based evaluation API. Each metric is a class declaring what it needs (`requires`) and computing a `MetricResult`. `evaluate()` in `evaluator.py` is the public entry point. The `metrics/` subpackage groups metrics by artifact family: NLL-based, predictive-sample-based, generative-rollout-based, and intensity-grid-based.

**Predictive comparison** (`predictive_compare.py`, `predictive_sampling.py`, `evaluation_helpers.py`, `artifacts.py`, `context.py`)  
Next-event and full-sequence sampling against held-out test data. `EvalContext` lazily materializes shared artifacts (samples, rollouts, intensity grids) so metrics that share an artifact compute it once. `evaluation_helpers.py` is the internal compute layer behind `EvalContext`'s `@cached_property` attrs.

**Surface diagnostics** (`surface.py`, `surface_compute.py`, `intensity.py`)  
Spatiotemporal intensity surface evaluation and diagnostic visualization bundles. Separate from the metric registry because these produce structured multi-field objects (`SurfaceResult`, `SurfaceDiagnosticResult`) consumed directly by the viz layer.

**Bundle I/O** (`io.py`, `common.py`)  
Read/write routines for predictive and surface bundles. `common.py` holds low-level sequence manipulation utilities shared across the package.

## Why it's kept flat

A single `evaluation/` package with a `metrics/` subpackage was a deliberate call to avoid premature nesting. Adding more subpackages (`evaluation/surface/`, `evaluation/sampling/`) would buy cleaner boundaries at the cost of deeper import paths and more `__init__.py` routing — not worth it at this stage.

## A few files are historical

`surface_metrics.py` (40 lines) holds two helper functions originally in `metrics.py` before that name was taken by the `metrics/` package. `predictive_compare.py` pre-dates the registry system and provides the `PredictiveComparator` workflow. Both are still used by scripts and the legacy public API and are kept for backward compatibility.

The current shape is driven by the actual `evaluate()` workflow, not by accumulation.
