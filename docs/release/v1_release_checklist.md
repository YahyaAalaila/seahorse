# v1 Release Execution Checklist

## Phase 0: Freeze Current State

Tasks:

- Use `release/v1-integration` as the single integration branch.
- Record current branch, commit, dirty state, and ignored/generated artifacts.
- Do not fetch, merge, reset, rewrite history, or touch remotes.

Expected outputs:

- Release audit docs under `docs/release/`.
- A documented decision list for dirty/untracked/generated files.

Validation commands:

```bash
git branch --show-current
git status --short --branch
git rev-parse --short HEAD
```

Rollback strategy:

- If release docs need to be abandoned, revert only the `docs/release/` additions on the release branch. Do not reset shared history.

## Phase 1: Clean Package, Imports, And Configs

Tasks:

- Polish package metadata: version/name/description, URLs, and missing release metadata files.
- Keep `python -m unified_stpp` as the stable CLI.
- Document all current paper presets as benchmark-supported.
- Document existing optional extras (`dev`, `hpo`, `all`) in install docs.
- Document HF dataset access, suite 3/4 upload paths, and HawkesNest generation notebook.
- Decide whether tracked generated outputs belong in the public repo.
- Remove or relocate accidental tracked artifacts through normal commits only.

Expected outputs:

- Clean `pyproject.toml` metadata.
- Updated README naming policy.
- Explicit v1 benchmark-supported preset table.
- Public data access guide with HF paths and dataset schema.
- Clean file inventory for release candidate.

Validation commands:

```bash
python -m pip install -e ".[all,dev]"
python -m unified_stpp --help
python -m unified_stpp fit --help
python -m unified_stpp tune --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
```

Rollback strategy:

- Revert individual cleanup commits if metadata or packaging changes break install/import. Keep docs and code changes in separate commits.

## Phase 2: Tests And Smoke Runs

Tasks:

- Run import smoke, CLI smoke, tiny data-loader check, minimal CPU fit, and minimal evaluation.
- Run focused pytest suites for release-critical contracts.
- Separate fast release checks from long research/benchmark tests.

Expected outputs:

- Passing smoke logs.
- Documented failing tests or explicit deferrals.
- Release-candidate validation record.

Validation commands:

```bash
pytest tests/test_smoke.py
pytest tests/test_registry_compat.py tests/test_config_resolution.py
pytest tests/test_data_resolution.py tests/test_data_hub.py
pytest tests/test_evaluate_cli.py tests/test_eval_artifacts.py
pytest tests/test_benchmark_config.py
```

Rollback strategy:

- If a cleanup breaks tests, revert the smallest cleanup commit. If a test is long or flaky, mark it outside the release smoke set rather than weakening coverage silently.

## Phase 3: Docs And Examples

Tasks:

- Make README public-v1 clear: install extras, quickstart, data schema, HF dataset access, stable CLI, paper presets, evaluation outputs, and limitations.
- Add or update dataset schema, HF data access guide, preset guide, HawkesNest generation notebook reference, and benchmark reproduction guide.
- Keep cluster/Pegasus material clearly optional and parameterized.
- Add tiny example data/configs if in scope.

Expected outputs:

- Public README ready for first-time users.
- Dataset and preset docs.
- Benchmark reproduction notes.
- Optional examples that run on CPU.

Validation commands:

```bash
rg -n "/Users/|/home/aalaila|192\\.168|PEGASUS|Pegasus|pegasus" README.md docs scripts
rg -n "python -m unified_stpp (fit|tune|bench|evaluate)" README.md docs
```

Rollback strategy:

- Revert doc-only commits independently from code. Preserve generated release audit files as historical planning unless superseded.

## Phase 4: Metadata Files

Tasks:

- Add `CITATION.cff`.
- Add `AUTHORS.md` or `CONTRIBUTORS.md`.
- Add `CONTRIBUTING.md`.
- Add `CHANGELOG.md`.
- Add release notes template.
- Confirm MIT license text is correct and compatible with bundled data/code.

Expected outputs:

- Complete public metadata set.
- Clear citation and contribution instructions.

Validation commands:

```bash
test -f LICENSE
test -f CITATION.cff
test -f AUTHORS.md -o -f CONTRIBUTORS.md
test -f CONTRIBUTING.md
test -f CHANGELOG.md
```

Rollback strategy:

- Revert metadata file additions if naming/legal content needs correction, then replace with reviewed versions.

## Phase 5: v1 Release Candidate

Tasks:

- Create an RC commit with only intentional tracked files.
- Run full release validation.
- Generate release summary and known limitations.
- Confirm no secrets, private paths, large accidental files, or stale generated outputs remain.

Expected outputs:

- `v1.0.0-rc` candidate commit.
- Validation logs.
- Final blocker list.

Validation commands:

```bash
git status --short --branch
git ls-files | xargs -n 200 du -h 2>/dev/null | sort -hr | sed -n '1,80p'
rg -n "HF_TOKEN|WANDB_API_KEY|api[_-]?key|access[_-]?token|password|secret|PRIVATE KEY" .
rg -n "/Users/|/home/aalaila|192\\.168" README.md docs scripts unified_stpp tests
pytest
```

Rollback strategy:

- Revert the RC commit or split it into smaller commits. Do not rewrite public history after publishing.

## Phase 6: Public Release

Tasks:

- Tag only after install, CLI, smoke tests, metadata, docs, and hygiene checks pass.
- Publish release notes with stable scope, data access instructions, and paper-runtime caveats.
- Do not touch remotes until explicitly authorized.

Expected outputs:

- v1 tag.
- Release notes.
- Public docs aligned with package state.

Validation commands:

```bash
git status --short --branch
python -m unified_stpp --help
python -m pip check
pytest
```

Rollback strategy:

- If a public tag is incorrect, create a follow-up patch release. Do not rewrite public tags unless the repository owner explicitly chooses that policy.
