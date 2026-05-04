# v1 Release Summary

## Releasability Verdict

The repo is not currently releasable as a clean public v1. It has the expected stable CLI and current paper presets should be treated as benchmark-supported, but public release hygiene and documentation are incomplete. The release should proceed only after the blockers below are addressed and the validation matrix passes in a clean environment.

## Top 10 Blockers

1. Dirty working tree with modified files and untracked research artifacts that need intentional disposition.
2. Missing public metadata: citation, authors/contributors, contributing guide, changelog, and release notes template.
3. Package version/name/description/project URL polish remains.
4. Naming inconsistency across Seahorse, `unified_stpp`, `unified-stpp`, and `uni-stpp`.
5. Data access docs are incomplete for real HF datasets, suite 3/4 HF uploads, and HawkesNest generation.
6. Dataset schema docs need to be promoted from README sketch to release-grade documentation.
7. Hardcoded local and cluster-specific paths remain in internal docs and local-only scripts.
8. Full tracked `data/` still needs replacement with Hugging Face access docs plus tiny public examples.
9. Paper figure/table reproduction is not documented as a public command map with inputs, outputs, and HF data dependencies.
10. Clean install, smoke fit/evaluate, focused tests, and full tests still need to be run on the release candidate.

## Recommended v1 Scope

Stable:

- `python -m unified_stpp fit`
- `python -m unified_stpp tune`
- `python -m unified_stpp bench`
- `python -m unified_stpp evaluate`
- Local JSONL dataset loading.
- Hugging Face dataset loading when exact repo IDs/revisions are documented.
- Suite 3 and suite 4 synthetic datasets after manual HF upload.
- HawkesNest generation notebook and reproduction commands.
- All current paper presets as benchmark-supported.
- Core run artifacts and evaluation outputs.

Not a v1 blocker:

- Missing `seahorse` console script.
- Normal-user model-by-model Python wrapper.
- Historical branch consolidation.
- Registry wording that could imply paper presets are not benchmark-supported.

Excluded:

- Stale notebooks and untracked generated reports.
- Private machine paths as public examples.
- Cluster-only workflows as required public reproduction steps.

## Must Be Done Before Tagging v1

- Make the release branch clean and intentional.
- Add missing metadata files.
- Polish package version/name/description/project URLs.
- Set naming policy in README and package metadata.
- Document HF download paths for real datasets and suite 3/4 uploads.
- Add dataset schema docs.
- Add HawkesNest generation notebook or notebook reference.
- Remove, relocate, or document tracked junk artifacts through normal commits.
- Parameterize or mark cluster/local paths.
- Add public benchmark reproduction guide for paper figures/tables.
- Run and record validation commands from `validation_commands.md`.

## Can Wait Until v1.1

- `seahorse` or `unified-stpp` console script alias.
- Generated API documentation.
- Curated `examples/` directory beyond minimal v1 quickstart.
- Stable normal-user Python wrapper.
- More polished benchmark automation beyond the paper reproduction commands needed for v1.

## Public-Surface Cleanup Status

The release branch now removes the first wave of public-inappropriate generated outputs, `archive/`, old root entrypoints, and site-specific cluster wrappers from Git tracking while keeping local files on disk.

Remaining public-surface blockers are the dataset publication/docs handoff, final paper artifact command map, missing metadata files, naming polish, and review of untracked paper-analysis scripts/tests.
