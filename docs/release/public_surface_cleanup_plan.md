# Public Surface Cleanup Plan

Date: 2026-05-04

Branch: `release/v1-integration`

Scope: prepare the private Seahorse / `unified-stpp` tree for a clean public v1 release. The public surface should contain the installable Python package, stable module CLI, minimal examples, benchmark/example configs, tests, release docs, and paper reproducibility docs. Generated outputs, private run logs, local cluster launch state, and stale planning notes should not sit in the public root.

## Top-Level Classification

| Path | Current status | Recommendation | Reason | Risk | Exact Git action |
|---|---|---|---|---|---|
| `.github/` | tracked | KEEP public v1 | CI is public-facing release hygiene. | Low | None |
| `.gitignore` | tracked, modified | KEEP public v1 | Needs generated-output ignores. | Low | Edit and commit |
| `.pre-commit-config.yaml` | tracked | KEEP public v1 | Developer hygiene. | Low | None |
| `LICENSE` | tracked | KEEP public v1 | Required metadata. | Low | None |
| `Makefile` | tracked | KEEP but document | Useful developer commands if accurate. | Low | Re-audit later |
| `README.md` | tracked | KEEP public v1 | Primary public landing doc. | Low | Keep public CLI/data wording current |
| `pyproject.toml` | tracked | KEEP public v1 | Package metadata and dependencies. | Low | None in this pass |
| `pytest.ini` | tracked | KEEP public v1 | Test configuration. | Low | None |
| `requirements.txt` | tracked | KEEP but document | Useful compatibility file, but `pyproject.toml` should remain authoritative. | Low | Document in release metadata plan |
| `unified_stpp/` | tracked | KEEP public v1 | Live package tree and stable CLI. | High | None in cleanup pass |
| `tests/` | tracked plus untracked candidate tests | KEEP public v1 | Release validation depends on tests. | Medium | Do not delete; review untracked tests separately |
| `docs/` | tracked plus untracked internal candidates | KEEP public v1 with internal split | Public docs stay under `docs/`; private paper notes move under `docs/internal/`. | Medium | Move stale root docs into `docs/internal/`; add cleanup plan/examples |
| `docs/release/` | tracked | KEEP public v1 | Release audit, validation, and paper reproducibility docs. | Low | Add this plan |
| `docs/internal/` | tracked plus untracked internal candidates | KEEP but document | Internal paper drafts and private operational notes should be clearly separated from public docs. | Medium | Keep internal; do not advertise as public API |
| `archive/` | tracked, ~4.4GB working-tree size | REMOVE from Git tracking / ignore | Reference-only legacy package code is not part of public import surface; treat as a local bin/archive dump. | Medium | `git rm --cached -r archive`; keep local files |
| `bench_out/` | tracked generated benchmark output | REMOVE from Git tracking / ignore | Contains stale generated HTML/JSON/CSV and hardcoded local checkpoint paths. | Low | `git rm --cached -r bench_out`; keep local files |
| `logs/` | tracked Lightning/TensorBoard logs, ~198MB | REMOVE from Git tracking / ignore | Generated training logs are not package/tests/examples. | Low | `git rm --cached -r logs`; keep local files |
| `runs/paper_readiness_latest/` | tracked generated report | REMOVE from Git tracking / ignore | Generated readiness snapshot includes cluster-specific paths. | Low | `git rm --cached -r runs/paper_readiness_latest`; keep local files |
| `runs/`, `runs2/`, `runs3/` | ignored/untracked generated outputs | REMOVE from Git tracking / ignore | Local run outputs should not be public source. | Low | Already ignored; keep local |
| `artifacts/`, `checkpoints/` | ignored/untracked generated outputs | REMOVE from Git tracking / ignore | Checkpoints and run artifacts are generated. | Low | Already ignored; keep local |
| `data/` | tracked, ~643MB | REMOVE from public tracking after data policy replacement | Real datasets and processed synthetic suites belong on Hugging Face; keep only tiny example data in repo. | High | No action in safe pass; add data publication policy and later replace with tiny examples |
| `best_config.yaml` | tracked generated HPO output | REMOVE from Git tracking / ignore | Single-run best config belongs in run artifacts, not public root. | Low | `git rm --cached best_config.yaml`; keep local file |
| `best_nsmpp_config.yaml` | untracked | REMOVE from Git tracking / ignore | Local generated best config. | Low | Ignore pattern; do not add |
| `configs` | tracked symlink to `unified_stpp/configs` | KEEP but document | Backward-compatible convenience path; package configs live under `unified_stpp/configs`. | Medium | Keep for v1 unless packaging audit says otherwise |
| `train.py` | tracked legacy wrapper | REMOVE from Git tracking / ignore | Public v1 CLI is `python -m unified_stpp ...`; root wrapper is old and not part of the public surface. | Low | `git rm --cached train.py`; keep local file |
| `temp_intensity_viz.py` | tracked temp script | REMOVE from Git tracking / ignore | Temporary visualization script is not stable public CLI or paper reproduction surface. | Low | `git rm --cached temp_intensity_viz.py`; keep local file |
| `EXPERIMENT_READINESS.md` | tracked root stale duplicate | MOVE to docs/internal | Root release-planning note should not be public landing surface. | Low | `git mv EXPERIMENT_READINESS.md docs/internal/EXPERIMENT_READINESS_LEGACY.md` |
| `REPO_CLEANUP_PLAN.md` | tracked root stale duplicate | MOVE to docs/internal | Stale private cleanup planning should not sit in public root. | Low | `git mv REPO_CLEANUP_PLAN.md docs/internal/REPO_CLEANUP_PLAN_LEGACY.md` |
| `docs/REPO_CLEANUP_PLAN.md` | tracked stale planning doc | MOVE to docs/internal | Private cleanup plan includes old provisional naming and should not be a public doc. | Low | `git mv docs/REPO_CLEANUP_PLAN.md docs/internal/REPO_CLEANUP_PLAN.md` |
| `docs/PEGASUS_CAMPAIGN.md` | tracked cluster doc with local paths | MOVE to docs/internal | Pegasus-specific operational workflow is internal and contains machine-specific examples. | Medium | `git mv docs/PEGASUS_CAMPAIGN.md docs/internal/PEGASUS_CAMPAIGN.md`; update public links |
| `scripts/*.sbatch`, `scripts/submit_*.sh`, `scripts/tune_*_gpu.sbatch` | tracked cluster scripts, several hardcoded `/home`, `/netscratch`, and Pegasus assumptions | REMOVE from Git tracking / ignore | Cluster launch wrappers are site-specific and not minimal public CLI examples. | Medium | `git rm --cached` for the scripts listed in the script audit; keep local files |
| Generated root HTML/JSON/SVG reports | untracked | DELETE if obvious junk or ignore | Local reports should not be added. | Low | Ignore root report patterns; do not add |
| `notebooks/` | untracked | KEEP but document or MOVE after audit | HawkesNest generation notebook is planned; exploratory notebooks should stay internal. | Medium | Do not add until curated |

## Tracked Script Audit

This follow-up pass keeps scripts only when they support benchmark/paper reproduction, are referenced by tracked tests/docs, or provide data-generation utilities that still need a public command map. Site-specific cluster launchers are removed from Git tracking and ignored, with local copies left on disk.

| Script | Current status | Classification | Reason | Exact Git action |
|---|---|---|---|---|
| `scripts/aggregate_hawkesnest_predictive_kde.py` | tracked | KEEP public v1 | Paper-facing predictive KDE aggregation; covered by tracked tests. | None |
| `scripts/build_campaign_index.py` | tracked | KEEP public v1 | Builds reproducibility indexes over HPO, bench, and evaluation outputs. | None |
| `scripts/gen_sthp_splits.py` | tracked | KEEP public v1 | Recreates small STHP reference splits and documents the synthetic JSONL shape. | None |
| `scripts/generate_hawkesnest_suite_ground_truth.py` | tracked | KEEP public v1 | HawkesNest/synthetic ground-truth utility; covered by tracked tests. | None |
| `scripts/hawkesnest_predictive_family_metrics.py` | tracked | KEEP public v1 | Paper-facing predictive-family metrics; covered by tracked tests. | None |
| `scripts/make_suite3_paper_artifacts.py` | tracked | KEEP public v1 | Regenerates suite 3 paper-facing artifacts. | None |
| `scripts/make_training_time_diagnostics.py` | tracked | KEEP public v1 | Regenerates paper-facing training-time diagnostics. | None |
| `scripts/paper_readiness_report.py` | tracked, modified in working tree | KEEP public v1 | Paper-readiness report generator. Existing local modification is not part of this cleanup commit unless separately reviewed. | None |
| `scripts/reprocess_hawkesnest_easy_hard_v2.py` | tracked | KEEP public v1 | Data-processing reproduction utility; covered by tracked tests. | None |
| `scripts/resolve_bench_eval_targets.py` | tracked | KEEP public v1 | Resolves benchmark evaluation targets; covered by tracked tests. | None |
| `scripts/resolve_hawkesnest_campaign_eval_targets.py` | tracked | KEEP public v1 | Resolves HawkesNest campaign evaluation targets; covered by tracked tests. | None |
| `scripts/resolve_hawkesnest_predictive_kde_targets.py` | tracked | KEEP public v1 | Resolves predictive KDE campaign targets for paper analysis. | None |
| `scripts/run_eval_metrics.py` | tracked | KEEP public v1 | Thin paper-evaluation helper over packaged evaluation APIs. | None |
| `scripts/run_realdata_bench_with_curve.py` | tracked | KEEP public v1 | Regenerates real-data benchmark cells with training-curve output. | None |
| `scripts/synthetic_suite_campaign.py` | tracked | KEEP public v1 | Synthetic campaign runner; covered by tracked tests. | None |
| `scripts/pegasus_bench_group.sbatch` | tracked | UNTRACK / ignore | Pegasus-specific batch launcher; not a generic public CLI example. | `git rm --cached scripts/pegasus_bench_group.sbatch` |
| `scripts/pegasus_campaign_env.sh` | tracked | UNTRACK / ignore | Pegasus-specific shared environment wrapper. | `git rm --cached scripts/pegasus_campaign_env.sh` |
| `scripts/pegasus_eval_metrics.sbatch` | tracked | UNTRACK / ignore | Pegasus-specific evaluation launcher. | `git rm --cached scripts/pegasus_eval_metrics.sbatch` |
| `scripts/pegasus_eval_predictive_compare.sbatch` | tracked | UNTRACK / ignore | Pegasus-specific predictive-compare launcher. | `git rm --cached scripts/pegasus_eval_predictive_compare.sbatch` |
| `scripts/pegasus_eval_surface.sbatch` | tracked | UNTRACK / ignore | Pegasus-specific surface-evaluation launcher. | `git rm --cached scripts/pegasus_eval_surface.sbatch` |
| `scripts/pegasus_hawkesnest_predictive_kde.sbatch` | tracked | UNTRACK / ignore | Pegasus-specific HawkesNest KDE launcher. | `git rm --cached scripts/pegasus_hawkesnest_predictive_kde.sbatch` |
| `scripts/pegasus_hawkesnest_suite_campaign.sbatch` | tracked | UNTRACK / ignore | Hardcoded `/home`/`/netscratch` HawkesNest launcher. | `git rm --cached scripts/pegasus_hawkesnest_suite_campaign.sbatch` |
| `scripts/pegasus_realdata_bench_curve.sbatch` | tracked | UNTRACK / ignore | Pegasus-specific real-data curve launcher; public script is `run_realdata_bench_with_curve.py`. | `git rm --cached scripts/pegasus_realdata_bench_curve.sbatch` |
| `scripts/pegasus_tune_preset.sbatch` | tracked | UNTRACK / ignore | Pegasus-specific tuning launcher. | `git rm --cached scripts/pegasus_tune_preset.sbatch` |
| `scripts/print_hawkesnest_easy_hard_submit_commands.sh` | tracked | UNTRACK / ignore | Hardcoded local project paths and one-off submit command generation. | `git rm --cached scripts/print_hawkesnest_easy_hard_submit_commands.sh` |
| `scripts/slurm_eval_metrics_batched_validation.sbatch` | tracked | UNTRACK / ignore | Hardcoded container/workdir paths; validation command now lives in docs. | `git rm --cached scripts/slurm_eval_metrics_batched_validation.sbatch` |
| `scripts/slurm_eval_metrics_exact_all_datasets.sbatch` | tracked | UNTRACK / ignore | Hardcoded container/workdir paths and run IDs; not a generic public template. | `git rm --cached scripts/slurm_eval_metrics_exact_all_datasets.sbatch` |
| `scripts/slurm_eval_metrics_exact_validation.sbatch` | tracked | UNTRACK / ignore | Contains fixed local run/data examples; use documented CLI validation instead. | `git rm --cached scripts/slurm_eval_metrics_exact_validation.sbatch` |
| `scripts/slurm_eval_metrics_native_validation.sbatch` | tracked | UNTRACK / ignore | Contains fixed local run/data examples and legacy wording; use documented CLI validation instead. | `git rm --cached scripts/slurm_eval_metrics_native_validation.sbatch` |
| `scripts/slurm_job_summary.py` | tracked | UNTRACK / ignore | Supports the removed site-specific SLURM wrappers. | `git rm --cached scripts/slurm_job_summary.py` |
| `scripts/submit_bench_eval_metrics.sh` | tracked | UNTRACK / ignore | Site-specific batch submission wrapper. | `git rm --cached scripts/submit_bench_eval_metrics.sh` |
| `scripts/submit_hawkesnest_campaign_eval.sh` | tracked | UNTRACK / ignore | Site-specific batch submission wrapper. | `git rm --cached scripts/submit_hawkesnest_campaign_eval.sh` |
| `scripts/submit_hawkesnest_predictive_kde.sh` | tracked | UNTRACK / ignore | Site-specific batch submission wrapper. | `git rm --cached scripts/submit_hawkesnest_predictive_kde.sh` |
| `scripts/submit_hawkesnest_suite_family.sh` | tracked | UNTRACK / ignore | Site-specific batch submission wrapper. | `git rm --cached scripts/submit_hawkesnest_suite_family.sh` |
| `scripts/submit_realdata_family.sh` | tracked | UNTRACK / ignore | Site-specific batch submission wrapper. | `git rm --cached scripts/submit_realdata_family.sh` |
| `scripts/tiny_train.sh` | tracked | UNTRACK / ignore | Calls old root `train.py`; public examples use `python -m unified_stpp fit`. | `git rm --cached scripts/tiny_train.sh` |
| `scripts/tune_factorized_cnf_gpu.sbatch` | tracked | UNTRACK / ignore | Hardcoded local root and cluster mounts. | `git rm --cached scripts/tune_factorized_cnf_gpu.sbatch` |
| `scripts/tune_factorized_gmm_cpu.sbatch` | tracked | UNTRACK / ignore | Hardcoded local root and cluster mounts. | `git rm --cached scripts/tune_factorized_gmm_cpu.sbatch` |
| `scripts/tune_gpu_preset.sbatch` | tracked | UNTRACK / ignore | Hardcoded local root and cluster mounts. | `git rm --cached scripts/tune_gpu_preset.sbatch` |
| `scripts/tune_temporal_gmm_gpu.sbatch` | tracked | UNTRACK / ignore | Hardcoded local root and cluster mounts. | `git rm --cached scripts/tune_temporal_gmm_gpu.sbatch` |

## Data Publication Policy

- Public v1 should not track full real datasets or processed HawkesNest suite exports.
- Real datasets are or will be hosted on Hugging Face.
- Synthetic suites 3 and 4 will be uploaded to Hugging Face.
- HawkesNest generation will be documented separately, preferably as a curated notebook plus command map.
- The repository may keep a tiny toy JSONL dataset under `examples/tiny_jsonl/` for CLI smoke examples.
- Existing tracked `data/` is classified but not removed in this safe cleanup pass because tests/examples may still reference some paths and the HF repo IDs are not finalized.

## Legacy, Provisional, And Local-Path Wording

| Pattern | Current occurrence class | Action in this pass | Justification |
|---|---|---|---|
| `Local Developer` | Not present in tracked package/release docs after metadata cleanup. | No action | Already fixed in `pyproject.toml`. |
| `provisional` | Public docs, tests, registry vocabulary, surface diagnostics, internal docs. | Remove public preset downgrades from `docs/BENCHMARK.md`; leave registry/test/internal diagnostic vocabulary. | Current paper presets are benchmark-supported; code still needs compatibility vocabulary for aliases and diagnostic result fields. |
| `legacy` | Registry/tests, AutoSTPP compatibility docs, stale planning docs. | Move stale root cleanup/readiness docs internal; keep compatibility wording in code/tests. | `auto_stpp_legacy` is an intentional public compatibility preset. |
| `deprecated` | Registry/tests/docs. | Keep where it describes accepted aliases. | Backward compatibility requires deprecated alias metadata. |
| `experimental` | Mostly internal docs or reference-only synthetic notes. | Keep internal; do not surface as public support status. | Internal paper notes are not public release docs. |
| `Pegasus` | README/docs/scripts. | Move `docs/PEGASUS_CAMPAIGN.md` internal and remove README promotion. | Cluster workflow is private operational material, not public quickstart. |
| `/Users/`, `/home/`, `/netscratch/` | Generated outputs, cluster docs/scripts, release validation grep commands. | Remove generated tracked outputs; move Pegasus doc internal; leave scripts for later parameterization. | Cluster scripts need a separate audit before moving/removing. |

## Safe Cleanup To Apply Now

1. Update `.gitignore` for root generated reports, best-config outputs, `archive/`, old root entrypoints, and local/internal cluster launch wrappers.
2. Untrack, but do not delete locally:
   - `bench_out/`
   - `logs/`
   - `runs/paper_readiness_latest/`
   - `best_config.yaml`
   - `archive/`
   - `train.py`
   - `temp_intensity_viz.py`
   - site-specific scripts listed as `UNTRACK / ignore` above
3. Move stale public-root planning docs into `docs/internal/`.
4. Move `docs/PEGASUS_CAMPAIGN.md` into `docs/internal/` and stop promoting it from README.
5. Add public CLI examples using `examples/tiny_jsonl/`.

## Deferred Cleanup

1. Replace tracked `data/` with HF dataset docs plus tiny example data.
2. Replace remaining paper-reproduction scripts with a smaller command map once the final paper artifact inventory is frozen.
3. Move any future clean cluster template under docs/examples only after removing personal paths and fixed local run IDs.
4. Review untracked paper scripts/tests and decide whether they belong under `scripts/`, `tests/`, or `docs/internal/`.
