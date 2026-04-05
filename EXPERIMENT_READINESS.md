# Experiment Readiness

This file defines the repo's experiment-freeze target and benchmark contract.
It is execution-facing: it records what is already in place, what is still not
frozen, and what must be true before launching new benchmark experiments.

## Current state

### Implemented
- Raw-first benchmark path and transform-artifact infrastructure are in place.
- One canonical package/import tree is now enforced: repo-root
  `unified_stpp/` is the live package tree, and `archive/` is reference-only.
- Canonical preset IDs and explicit preset statuses are now registry-owned and
  config-validated.
- New configs and runs canonicalize deprecated aliases to final preset IDs, and
  run artifacts persist preset status.
- `auto_stpp` now names the faithful AutoSTPP variant; `auto_stpp_faithful` is
  a deprecated alias, and the older coarse implementation is `auto_stpp_legacy`.
- A package-side surface evaluation and visualization workflow exists.
- Exact-vs-approx metric labeling exists in the benchmark/reporting stack.

### Not frozen
- One canonical predictive-comparison/evaluation path is not yet selected.
- README and benchmark-facing docs are not yet aligned with canonical preset
  names, statuses, and benchmark eligibility.

### Provisional
- Neural STPP presets remain provisional:
  `neural_cond_gmm`, `neural_jumpcnf`, `neural_attncnf`.
- Exclude them from headline experiments until they pass finite and stable
  benchmark-style smoke and surface checks.

### Legacy
- Benchmark artifacts from March 30, 2026 are legacy.
- They were generated with `protocol: standard` and `normalize: true`, so they
  should not be treated as the post-freeze benchmark record.

## Freeze priorities

1. Sync README and benchmark docs to the frozen preset/status contract.
2. Freeze one canonical evaluation/predictive-comparison path.
3. Rerun the benchmark under the frozen contract.
4. Add secondary metrics, broader data access, and wider synthetic coverage
   after first experiments.

## Must be true before experiments

- One live package tree.
- Canonical preset IDs only in new configs and results.
- Explicit preset statuses: `canonical`, `provisional`, `deprecated`, `legacy`.
- Raw-first artifact metadata persisted in results.
- Exact-model comparable NLL reporting path is active for benchmark-eligible
  presets.
- Exact-vs-approx labels are visible in reports.
- One canonical evaluation path is selected.
- Benchmark-eligible presets pass finite smoke and surface checks.
- README and benchmark docs reflect the current contract.

## After first experiments

- Add HF/public dataset access.
- Add secondary metrics.
- Broaden the synthetic benchmark suite.
- Reconsider provisional presets only after stabilization.

## Notes

- Do not describe the raw-first benchmark path or transform-artifact
  infrastructure as merely planned; they already exist.
- Keep first-experiment blockers separate from follow-up work.
- Historical run directories remain in place and are labeled legacy rather than
  renamed in place.
- `archive/` remains in the repo as reference-only code and is not part of the
  supported live import or packaging surface.
