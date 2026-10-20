"""Exact, compensator-based evaluation layer (``seahorse.evaluation.v1``).

This package adds exact, compensator-based and proper-score evaluation
alongside the legacy metrics in :mod:`seahorse.evaluation.metrics`. Nothing in
the legacy path is modified; the metric definitions are documented in
``docs/reference/exact-evaluation.md``. Run it with
``python -m seahorse.evaluation.v1.run_eval --run <run_dir> --test <test.jsonl> --out <dir>``.

Conventions used throughout this package
----------------------------------------
* All reported quantities are in **raw data units** (raw time, raw location),
  regardless of the coordinate transform a preset was trained in.
* A *context* ``k`` of a test sequence is the teacher-forced history
  ``{(t_j, s_j)}_{j<=k}``; its target is event ``k+1``.
* ``dLambda`` of event ``i`` is the model compensator increment
  ``int_{t_{i-1}}^{t_i} int_S lambda(u, s) ds du`` over the spatial domain ``S``
  that the preset's own NLL integrates over (``R^2`` for density-factorised
  presets, the preset's support box for joint-intensity presets).
"""

from .loading import load_runner_compat

__all__ = ["load_runner_compat"]
