# Changelog

## 1.0.1 — 2026-11-06

### Fixed
- SMASH: the framework surrogate NLL (`ScoreMatchingProcess.NLL_cal`, `eval_nll`) queried the network on the `[0, 1]` tokens, while the network is trained on `y = 2u − 1`. The intensity and the location score are now evaluated in the training coordinates, the compensator is integrated from `y = −1`, and the temporal term carries the `log 2` Jacobian back to the `[0, 1]` token. On the homogeneous Poisson control the temporal NLL in raw units moves from −2.51 to −3.37 (true −3.38). Per-event terms are available from `ScoreMatchingProcess.nll_terms`.
- SMASH defines no likelihood. The spatial Tweedie term of the framework value is not a density (it is dominated by the fixed σ constant), so the reported value stays labelled as a surrogate (`nll_kind="approx"`, `nll_path="surrogate"`); capability descriptions and docstrings say so.

### Added
- `scripts/approx_nll/`: evaluation-only scripts for the camera-ready DSTPP and SMASH NLL rows. `dstpp_bound.py` computes DSTPP's full variational bound per target in raw units (summed over token dimensions, with the Jacobians of the token normalisation and the campaign coordinates); `smash_grid_density.py` computes SMASH's temporal NLL and a grid estimate of the spatial density of its score model (least-squares Poisson solve on a regular grid, normalised), with a stratified subsample option. Formulas in the directory README.
- `load_released_dataset("hawkesnest_entanglement", config="hawkesnest_entanglement_L0" … "_L3")`: the HawkesNest entanglement suite in `seahorse-stpp/seahorse_synthetic_stpp` (folder `hawkesnest_entanglement/`), pinned revision, splits verified against the suite's `SHA256SUMS`.

SMASH values produced with 1.0.0 `eval_nll` should be recomputed.

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
- `njsde` is documented as the Neural Jump SDE baseline of Jia & Benson (2019) as implemented in the Neural STPP codebase (jump-ODE state model + conditional-GMM decoder; alias `neural_stpp_gmm`); the preset wraps that implementation, not the original NJSDE code.
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
