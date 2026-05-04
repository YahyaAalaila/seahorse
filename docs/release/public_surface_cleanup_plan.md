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
| `archive/` | tracked, ~4.4GB working-tree size | MOVE to archive/internal or REMOVE from public tracking after audit | Reference-only legacy package code is not part of public import surface. | High | No action in safe pass; future `git rm --cached -r archive` or external archive |
| `bench_out/` | tracked generated benchmark output | REMOVE from Git tracking / ignore | Contains stale generated HTML/JSON/CSV and hardcoded local checkpoint paths. | Low | `git rm --cached -r bench_out`; keep local files |
| `logs/` | tracked Lightning/TensorBoard logs, ~198MB | REMOVE from Git tracking / ignore | Generated training logs are not package/tests/examples. | Low | `git rm --cached -r logs`; keep local files |
| `runs/paper_readiness_latest/` | tracked generated report | REMOVE from Git tracking / ignore | Generated readiness snapshot includes cluster-specific paths. | Low | `git rm --cached -r runs/paper_readiness_latest`; keep local files |
| `runs/`, `runs2/`, `runs3/` | ignored/untracked generated outputs | REMOVE from Git tracking / ignore | Local run outputs should not be public source. | Low | Already ignored; keep local |
| `artifacts/`, `checkpoints/` | ignored/untracked generated outputs | REMOVE from Git tracking / ignore | Checkpoints and run artifacts are generated. | Low | Already ignored; keep local |
| `data/` | tracked, ~643MB | REMOVE from public tracking after data policy replacement | Real datasets and processed synthetic suites belong on Hugging Face; keep only tiny example data in repo. | High | No action in safe pass; add data publication policy and later replace with tiny examples |
| `best_config.yaml` | tracked generated HPO output | REMOVE from Git tracking / ignore | Single-run best config belongs in run artifacts, not public root. | Low | `git rm --cached best_config.yaml`; keep local file |
| `best_nsmpp_config.yaml` | untracked | REMOVE from Git tracking / ignore | Local generated best config. | Low | Ignore pattern; do not add |
| `configs` | tracked symlink to `unified_stpp/configs` | KEEP but document | Backward-compatible convenience path; package configs live under `unified_stpp/configs`. | Medium | Keep for v1 unless packaging audit says otherwise |
| `train.py` | tracked legacy wrapper | MOVE to docs/internal or REMOVE after CLI migration note | Public v1 CLI is `python -m unified_stpp ...`; root wrapper is legacy. | Medium | No action in safe pass; future deprecation/removal |
| `temp_intensity_viz.py` | tracked temp script | MOVE to docs/internal or REMOVE after packaged evaluate coverage | Temporary visualization script is not stable public CLI. | Medium | No action in safe pass; document as internal/temp |
| `EXPERIMENT_READINESS.md` | tracked root stale duplicate | MOVE to docs/internal | Root release-planning note should not be public landing surface. | Low | `git mv EXPERIMENT_READINESS.md docs/internal/EXPERIMENT_READINESS_LEGACY.md` |
| `REPO_CLEANUP_PLAN.md` | tracked root stale duplicate | MOVE to docs/internal | Stale private cleanup planning should not sit in public root. | Low | `git mv REPO_CLEANUP_PLAN.md docs/internal/REPO_CLEANUP_PLAN_LEGACY.md` |
| `docs/REPO_CLEANUP_PLAN.md` | tracked stale planning doc | MOVE to docs/internal | Private cleanup plan includes old provisional naming and should not be a public doc. | Low | `git mv docs/REPO_CLEANUP_PLAN.md docs/internal/REPO_CLEANUP_PLAN.md` |
| `docs/PEGASUS_CAMPAIGN.md` | tracked cluster doc with local paths | MOVE to docs/internal | Pegasus-specific operational workflow is internal and contains machine-specific examples. | Medium | `git mv docs/PEGASUS_CAMPAIGN.md docs/internal/PEGASUS_CAMPAIGN.md`; update public links |
| `scripts/*.sbatch`, `scripts/submit_*.sh`, `scripts/tune_*_gpu.sbatch` | tracked cluster scripts, several hardcoded `/home` and `/netscratch` paths | KEEP but document internal for now | Cluster scripts may still be useful for paper reproduction but are not generic public examples. | Medium | No safe-pass deletion; later move under `scripts/internal/cluster/` or parameterize |
| Generated root HTML/JSON/SVG reports | untracked | DELETE if obvious junk or ignore | Local reports should not be added. | Low | Ignore root report patterns; do not add |
| `notebooks/` | untracked | KEEP but document or MOVE after audit | HawkesNest generation notebook is planned; exploratory notebooks should stay internal. | Medium | Do not add until curated |

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

1. Update `.gitignore` for root generated reports and best-config outputs.
2. Untrack, but do not delete locally:
   - `bench_out/`
   - `logs/`
   - `runs/paper_readiness_latest/`
   - `best_config.yaml`
3. Move stale public-root planning docs into `docs/internal/`.
4. Move `docs/PEGASUS_CAMPAIGN.md` into `docs/internal/` and stop promoting it from README.
5. Add public CLI examples using `examples/tiny_jsonl/`.

## Deferred Cleanup

1. Replace tracked `data/` with HF dataset docs plus tiny example data.
2. Move or remove `archive/` from public tracking after confirming no tests import it.
3. Move cluster launch scripts into an internal path or parameterize them completely.
4. Retire `train.py` and `temp_intensity_viz.py` after confirming packaged CLI/evaluate coverage.
5. Review untracked paper scripts/tests and decide whether they belong under `scripts/`, `tests/`, or `docs/internal/`.
