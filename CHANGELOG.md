# Changelog

## 1.0.0 — 2026-10-20

### Fixed
- NSMPP: the per-event NLL includes the compensator term (omitted from `nll_matrix` in 0.1.0).
- THP-GMM, RMTPP-GMM: the temporal log-likelihood term is no longer dropped from the reported NLL.
- RMTPP: the compensator uses `expm1`; float32 `1 - exp(-x)` underflowed for small decay rates.
- AutoSTPP, DeepSTPP: targets the lookback window cannot score are excluded from the next-event NLL instead of entering as 0; `training.loss_target_scope="last"` scores one common target per context.
- Rollout coherence is reported as a distance (lower is better) and deprecated in favour of the energy scores.
- Intensity recovery is evaluated on all test sequences, not only the first.
- HPO config sanitiser no longer collapses `model.support` and `model.grid` vectors.
- `config/tuning.py` imports `Optional`.

### Changed
- `EventCapabilities.nll_path` and `nll_label` distinguish closed-form, numerical-quadrature (NSMPP, THP-GMM) and sample-based likelihoods; DSTPP's benchmark value is labelled a joint sample-KDE estimate, not a variational bound.
- Intensity thinning requires declared proposal bounds and emits a `DeprecationWarning` for predictive metrics; factorized GMM presets sample by inverse temporal hazard.
- `njsde` is documented as the Neural-STPP jump-ODE model with a conditional-GMM decoder (alias `neural_stpp_gmm`).
- Run ids carry a process tag; `save()` writes the effective config and `norm_stats.json`.
- Prepared-data and campaign roots come from `SEAHORSE_*` environment variables; no default paths.
- Observation metadata (ids, window bounds, time axis) is carried through datasets and CLI evaluation.

### Added
- Exact compensator-based evaluation (`seahorse.evaluation.v1`): time-rescaling PIT, threshold-weighted CRPS, HPD-PIT, energy scores, inverse-transform samplers, 0.1.0 checkpoint compatibility loader.
- Calibration metrics from the NLL compensator (temporal PIT/CRPS, spatial HPD-PIT): `seahorse campaign calibrate`.
- `seahorse campaign`: frozen campaign compilation, execution, Slurm dispatch, numerical NLL checks, common target sets, campaign specs under `experiments/campaign_2026/`.
- Statistical Poisson/Hawkes presets (NumPy/SciPy, vendored TheHawkesPackage, MIT) and the `prepared:` dataset protocol.
- Parametric `static_hawkes` and `gaussian_poisson` baselines; `dnsk` and `glide` presets (available; not benchmarked in the paper).
- Author-baseline adapters (ETAS, WSM, TriTPP) with pinned provenance.
- Released Hugging Face datasets loadable by short name with pinned revisions and card-hash verification (`seahorse.data.hf`).
- Cross-route NLL consistency test; temporal RMSE metric.
- CI (pytest on 3.10–3.12, ruff) and PyPI trusted-publishing workflows.

### Removed
- Unpublished in-house presets and the marked-process presets are not part of the release.

### Licensing
- `seahorse/models/spatial_models/independent_cnf.py` is attributed to neural_stpp (CC BY-NC 4.0); the package licence remains Apache-2.0.

Results produced with 0.1.0 for NSMPP, THP-GMM, RMTPP-GMM, AutoSTPP and DeepSTPP should be recomputed.

## 0.1.0 — 2026-06-30
- Initial release.
