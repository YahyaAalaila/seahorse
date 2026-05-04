# v1 Release Audit

Audit date: 2026-05-04

Audit branch: `release/v1-integration`

Source of truth: current repository state. Legacy/history concerns are intentionally ignored for this release audit.

## Current Release Readiness

The repository is not ready to tag as a clean public v1. The package and stable CLI exist, and `python -m unified_stpp` exposes the expected `fit`, `tune`, `bench`, and `evaluate` subcommands. The blockers are remaining release metadata, public docs, data access documentation, reproducibility of paper outputs, hardcoded local paths, tracked junk artifacts, smoke tests, and the final release checklist.

The absence of a `seahorse` console script is not a blocker. A normal-user Python wrapper is also not a v1 blocker and should be tracked as future/v1.1 work. Current paper presets are benchmark-supported for v1; runtime and stability caveats should be discussed in the paper rather than encoded as release-support downgrades.

## Blocking Before v1

| File/path | Issue | Recommended action | Effort | Risk |
|---|---|---:|---:|---:|
| working tree | Current branch has modified files and many untracked outputs, scripts, tests, notebooks, HTML reports, and run summaries. | Freeze a release candidate with an intentional tracked-file set. Decide which generated/research artifacts are included, ignored, or removed in normal cleanup commits. | M | High |
| `pyproject.toml` | Author metadata has been replaced; version/name/description/project URL polish is still needed for a public release. | Set final version/name/description policy and add project URLs. | S | Medium |
| `CITATION.cff`, `AUTHORS.md`, `CONTRIBUTING.md`, `CHANGELOG.md` | These release metadata files are absent. | Add citation metadata, authors/contributors, contribution guidelines, changelog, and release notes template before tagging. | M | High |
| `README.md`, `docs/`, `pyproject.toml` | Public naming is inconsistent across `unified_stpp`, `unified-stpp`, Seahorse, and `uni-stpp`. | Define public v1 naming policy while keeping `unified_stpp` as the import name and `python -m unified_stpp ...` as the stable CLI. | M | High |
| dataset docs | Data policy is decided but not documented clearly enough: real datasets are on Hugging Face; suite 3 and suite 4 synthetic datasets will be manually uploaded to Hugging Face; HawkesNest generation will be documented in a notebook. | Add public HF download paths/revisions, dataset schema docs, suite 3/4 upload notes, and HawkesNest generation notebook instructions. | M | High |
| dataset schema docs | README has a JSONL sketch, but there is no standalone, release-grade schema and validation guide. | Add a dataset schema doc covering `times`, `locations`, optional fields, split layout, HF/local loading, validation, and examples. | S | High |
| `docs/PEGASUS_CAMPAIGN.md`, `scripts/*.sbatch`, `scripts/*.sh` | Release-facing docs/scripts include `/Users/aalaila/...`, `/home/aalaila/projects/uni-stpp`, and Pegasus-specific assumptions. | Convert local paths to variables/examples, clearly mark cluster scripts as optional templates, and ensure public docs do not require the author's filesystem. | M | High |
| `README.md`, `docs/BENCHMARK.md`, scripts | Benchmark figures/tables can likely be regenerated through scripts, but there is no single public reproduction guide mapping each table/figure to commands and inputs. | Add a benchmark reproduction document or release appendix with exact scripts, inputs, outputs, expected runtime, and HF data dependencies. | L | High |
| `bench_out/`, `=2.0.0`, generated reports | Tracked generated artifacts and a stray empty tracked file (`=2.0.0`) make the release tree look accidental. | Decide whether each tracked artifact is intentional. Remove, relocate, or document artifacts in normal cleanup commits. | S | Medium |
| validation state | Clean install, full tests, and minimal fit/evaluate smoke were not run from a fresh environment during this audit. | Execute the documented validation matrix on the release candidate before tagging. | M | High |

## Should Fix Before v1

| File/path | Issue | Recommended action | Effort | Risk |
|---|---|---:|---:|---:|
| `README.md` | Quickstart should reflect the actual v1 path: HF data access, local JSONL schema, and `python -m unified_stpp` commands. | Rewrite quickstart around install, HF dataset download, tiny local smoke, fit, bench, and evaluate. | M | Medium |
| `pyproject.toml` | Optional extras exist (`dev`, `hpo`, `all`) and should be documented, not flagged as missing. | Document intended install commands for base, dev, hpo, and all extras. | S | Low |
| `unified_stpp/__init__.py` | Public package has no `__version__`; import smoke reports `None`. | Expose package version from package metadata or a single source of truth. | S | Medium |
| `.gitignore` | Ignores `runs/`, artifacts, checkpoints, images, logs, and `temp_*.py`, but current untracked outputs include `runs2/`, `runs3/`, reports, notebooks, and generated scripts. | Extend ignore policy for local generated outputs once decisions are made on current untracked files. | S | Medium |
| `notebooks/` (untracked) | Notebook policy is not settled. One curated notebook is needed for HawkesNest generation, but exploratory notebooks should not be mistaken for release docs. | Add/identify the HawkesNest generation notebook and exclude or label exploratory notebooks. | M | Medium |
| tests | Test suite is broad, but release smoke tests are not separated from long/research tests. | Add/mark a fast release smoke subset and document it in CI/release checklist. | M | Medium |
| `.github/workflows/ci.yml` | CI installs `.[dev]` and runs `ruff check tests scripts` plus `pytest`; package import and CLI smoke are implicit, not explicit. | Add explicit import/CLI smoke and release validation jobs before v1. | S | Medium |

## Nice To Have

| File/path | Issue | Recommended action | Effort | Risk |
|---|---|---:|---:|---:|
| `pyproject.toml` | No stable console alias. | Add a console-script alias only after v1 naming is finalized. | S | Low |
| docs | API docs are narrative only; no generated API reference or docs build config was found. | Add MkDocs/Sphinx later, or document that v1 docs are Markdown-only. | M | Low |
| examples | README has CLI examples, but no curated `examples/` directory. | Add tiny example configs and scripts for local JSONL fit/evaluate. | M | Medium |
| model docs | README preset table is useful but not enough for users choosing presets. | Add a preset guide with benchmark support, runtime expectations, and NLL semantics. | M | Medium |
| release automation | No release notes template or release checklist automation exists. | Add templates after v1 if manual checklist is sufficient for first release. | S | Low |
| normal-user Python wrapper | A polished model-by-model wrapper is not implemented as a tested public module. | Defer to v1.1 unless implemented as a thin tested wrapper over existing config/preset/runner layers. | M | Low |

## Defer After v1

| File/path | Issue | Recommended action | Effort | Risk |
|---|---|---:|---:|---:|
| console-script alias | A `seahorse` or `unified-stpp` executable can improve ergonomics. | Add after the stable module CLI and naming policy are settled. | S | Low |
| normal-user Python wrapper | The wrapper can help notebook users but should not create a parallel pathway. | Implement in v1.1 as a thin tested layer over existing config/preset/runner APIs. | M | Medium |
| broader docs automation | Generated API docs and richer examples would help long-term maintenance. | Add after v1 once core release docs are stable. | M | Low |
