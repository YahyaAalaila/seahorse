# Documentation And Metadata Plan

## Current State

Present:

- `README.md`
- `LICENSE`
- `requirements.txt`
- Markdown docs under `docs/`
- Internal/paper-facing Seahorse notes under `docs/internal/`
- CI config under `.github/workflows/ci.yml`
- Optional extras in `pyproject.toml`: `dev`, `hpo`, and `all`

Missing or not found:

- `CITATION.cff`
- `AUTHORS.md` or `CONTRIBUTORS.md`
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- Release notes template
- Standalone installation guide
- Standalone dataset schema guide
- Public HF dataset access guide
- Public benchmark reproduction guide
- Generated API docs config

## README Plan

Required before v1:

- State the public name and package/import name clearly.
- Keep `python -m unified_stpp fit/tune/bench/evaluate` as the stable CLI.
- Explain that `seahorse` console alias is not required for v1.
- Document existing install extras: base, `.[dev]`, `.[hpo]`, and `.[all]`.
- Add a minimal CPU quickstart using tiny local JSONL data.
- Add a Hugging Face dataset quickstart using exact repo IDs/revisions.
- Present all current paper presets as benchmark-supported.
- Explain exact vs approximate NLL reporting in plain language.
- Link to dataset schema, benchmark reproduction docs, and release limitations.

## Installation Plan

Required before v1:

- Keep editable install command: `pip install -e ".[all,dev]"`.
- Document base, dev, hpo, and all-extra install modes.
- Document supported Python versions. `pyproject.toml` declares `>=3.10`; CI covers 3.10 and 3.11.
- Decide whether Python 3.12/3.13 are supported or simply untested.
- Align `requirements.txt` with `pyproject.toml` or document that `pyproject.toml` is authoritative.

## Dataset And HF Access Plan

Required before v1:

- Add a standalone JSONL schema doc:
  - required `times`
  - required `locations`
  - optional `marks`
  - optional `event_covariates`
  - optional `field_covariates`
  - expected split layout
- Document validation behavior from `unified_stpp.data.contract`.
- Document local path and Hugging Face dataset resolution.
- Publish exact HF repo IDs and revisions/tags for real datasets.
- Publish exact HF paths for suite 3 and suite 4 synthetic datasets after manual upload.
- Add HawkesNest generation notebook instructions and link the notebook from the docs.
- Add data availability and license notes for all referenced datasets.

## Model Preset Plan

Required before v1:

- Publish a preset guide that treats all current paper presets as benchmark-supported.
- For each preset, document runtime expectations, NLL kind, reporting space, and evaluation capability.
- Move runtime/stability caveats into paper-facing discussion and benchmark notes rather than release-support downgrades.

## Benchmark Reproduction Plan

Required before v1 if benchmark figures/tables are public:

- Add one guide mapping each public table/figure to:
  - HF input data or run artifact location
  - script
  - command
  - output path
  - expected runtime and hardware
- Mark Pegasus-only workflows as optional cluster templates.
- Avoid hardcoded author-local paths in public docs.

## API Docs Plan

For v1:

- Treat CLI and artifact formats as stable.
- Do not make a normal-user model-by-model Python wrapper a v1 blocker.
- Do not publish `temp_evaluate_api.py` as API.

For v1.1:

- Add generated API reference or a curated Python API guide.
- Add a tested normal-user wrapper only if it delegates to the existing config/preset/runner system.

## Metadata Plan

Required before v1:

- Keep the replaced author metadata in `pyproject.toml` and confirm the final author/contributor list.
- Polish package version/name/description and project URLs in `pyproject.toml`.
- Add `CITATION.cff`.
- Add `AUTHORS.md` or `CONTRIBUTORS.md`.
- Add `CONTRIBUTING.md`.
- Add `CHANGELOG.md`.
- Add release notes template.
- Confirm license coverage for source code, bundled examples, and referenced datasets.
