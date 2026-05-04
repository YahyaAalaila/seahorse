# Experiment Readiness

This file defines the repo's experiment-freeze target and benchmark contract.
It is execution-facing: it records what is already in place, what is still not
frozen, and what must be true before launching new benchmark experiments.

For the concrete Pegasus launch workflow, grouped job policy, and rerunnable
sbatch templates, see [PEGASUS_CAMPAIGN.md](PEGASUS_CAMPAIGN.md).

## Current state

### Implemented
- Raw-first benchmark path and transform-artifact infrastructure are in place.
- One canonical package/import tree is now enforced: repo-root
  `unified_stpp/` is the live package tree, and `archive/` is reference-only.
- Canonical preset IDs and explicit preset statuses are now registry-owned and
  config-validated.
- New configs and runs canonicalize deprecated aliases to final preset IDs, and
  run artifacts persist preset status.
- `auto_stpp` is the canonical AutoSTPP preset, and the older coarse
  implementation is `auto_stpp_legacy`.
- README and benchmark-facing docs now reflect canonical preset names, status
  metadata, all-model table inclusion, and run/report artifact semantics.
- The packaged post-fit evaluation stack is now split into explicit lanes:
  `evaluate metrics --metric-profile predictive` for benchmark-aligned held-out
  next-event predictive artifacts,
  `evaluate predictive-compare` for qualitative future-window comparison,
  and `evaluate surface` as the secondary exact/factorized diagnostic path.
- Predictive comparison now persists sampled future-event payloads as the
  primary artifact, with KDE rate surfaces stored only as derived readouts.
- The old runner-owned `surface_viz` workflow is no longer the packaged
  evaluation interface.
- Exact-vs-approx metric labeling exists in the benchmark/reporting stack.
- Predictive comparison bundle metadata now persists the comparison seed policy,
  `preset_status`, `nll_kind`, and `nll_report_space`.

### Not frozen
- Benchmark results have not yet been rerun under the new frozen evaluation
  contract.
- Temp `temp_*` evaluation scripts still exist as parity references and can be
  retired once they are no longer needed.

### Neural Exact Families
- Neural exact-family presets are benchmark-supported:
  `njsde`, `neural_jumpcnf`, `neural_attncnf`.
- The packaged `surface --profile future_exact` lane remains a diagnostic path
  and should be validated on real runs before relying on surface artifacts.

### Legacy
- Benchmark artifacts from March 30, 2026 are legacy.
- They were generated with `protocol: standard` and `normalize: true`, so they
  should not be treated as the post-freeze benchmark record.

## Freeze priorities

1. Rerun the benchmark under the frozen contract.
2. Retire the temp evaluation reference scripts once the new packaged paths are
   the only ones still needed.
3. Add secondary metrics, broader data access, and wider synthetic coverage
   after first experiments.

## Must be true before experiments

- One live package tree.
- Canonical preset IDs only in new configs and results.
- Explicit preset statuses: `canonical`, `provisional`, `deprecated`, `legacy`.
- Raw-first artifact metadata persisted in results.
- Exact-model comparable NLL reporting path is active for benchmark-eligible
  presets.
- Exact-vs-approx labels are visible in reports.
- The supported evaluation interface is the packaged split:
  `evaluate metrics --metric-profile predictive` for benchmark-aligned
  predictive artifacts,
  `evaluate predictive-compare` for qualitative future-window comparison,
  and `evaluate surface` for diagnostics.
- Benchmark-eligible presets pass finite smoke and surface checks.
- README and benchmark docs reflect the current contract.

## After first experiments

- Add HF/public dataset access.
- Add secondary metrics.
- Broaden the synthetic benchmark suite.
- Revisit diagnostic surface coverage after stabilization.

## Notes

- Do not describe the raw-first benchmark path or transform-artifact
  infrastructure as merely planned; they already exist.
- Keep first-experiment blockers separate from follow-up work.
- Historical run directories remain in place and are labeled legacy rather than
  renamed in place.
- `archive/` remains in the repo as reference-only code and is not part of the
  supported live import or packaging surface.
