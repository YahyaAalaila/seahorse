# Minimal Public v1 Scope

## Core Scope

v1 should ship Seahorse / `unified_stpp` as a research framework with a stable command-line workflow.

Stable command interface:

```bash
python -m unified_stpp fit ...
python -m unified_stpp tune ...
python -m unified_stpp bench ...
python -m unified_stpp evaluate ...
```

The absence of a `seahorse` console script is not a blocker. A console alias can be added later as a usability improvement.

## Included In v1

- Package import path: `unified_stpp`.
- Editable install targets using existing extras: base install, `.[dev]`, `.[hpo]`, and `.[all]`.
- Core JSONL data contract with one sequence per line and required `times` and `locations` fields.
- Local JSONL split loading through `--train`, `--val`, `--test`.
- Dataset directory and Hugging Face-style dataset resolution through `--dataset` and `--dataset-revision`.
- Stable CLI commands: `fit`, `tune`, `bench`, and `evaluate`.
- Evaluation subcommands currently exposed by help: `metrics`, `predictive-compare`, `surface`, and `merge-artifacts`.
- Run artifacts: config snapshots, resolved config, run result metadata, metric outputs, checkpoints, and benchmark tables as described in README.

## Supported Preset Scope

All current paper presets are benchmark-supported for v1. Do not encode paper runtime or stability caveats as release-support downgrades; discuss those caveats in the paper and benchmark notes.

Benchmark-supported paper preset families include:

- Factorized exact families: `poisson_gmm`, `hawkes_gmm`, `selfcorrecting_gmm`, `poisson_cnf`, `hawkes_cnf`, `selfcorrecting_cnf`, `poisson_tvcnf`, `hawkes_tvcnf`, `selfcorrecting_tvcnf`.
- Temporal neural baselines with GMM spatial component: `rmtpp_gmm`, `thp_gmm`.
- Paper-faithful neural baselines: `deep_stpp`, `auto_stpp`.
- Sample-based benchmark families: `smash`, `diffusion_stpp`.
- Neural/paper benchmark families: `nsmpp`, `njsde`, `neural_jumpcnf`, `neural_attncnf`.

The v1 docs should focus on benchmark support, runtime expectations, NLL semantics, and required data/config inputs rather than historical registry status labels.

## Supported Dataset Scope

Stable v1 data support:

- User-provided local JSONL datasets with `train.jsonl`, `val.jsonl`, and optional `test.jsonl`.
- Real datasets hosted on Hugging Face, documented with exact repo IDs, revisions/tags, and example `--dataset` values.
- Suite 3 and suite 4 synthetic datasets after manual Hugging Face upload, documented with exact download paths.
- HawkesNest generation documented through a curated notebook and linked reproduction commands.
- Tiny local sample data for smoke tests.

Required before tagging:

- Publish or record all HF dataset paths.
- Document dataset schema and split layout.
- Document paper figure/table input data locations.
- Document which commands regenerate benchmark tables/figures and which commands only validate the packaged path.

Excluded from v1 stable scope:

- Private machine paths.
- Untracked `runs2/`, `runs3/`, ad hoc HTML reports, and exploratory notebooks.
- Cluster-only commands as required public reproduction steps.

## Public API Position

Stable in v1:

- CLI commands listed above.
- Config and artifact formats documented in README and release docs.

Available but not the primary v1 promise:

- `STPPRunner`
- `STPPConfig.from_preset`
- `ConfigRegistry`
- `unified_stpp.registry.build_model`
- Evaluation Python helpers exposed under `unified_stpp.evaluation`

Future/v1.1:

- A polished normal-user model-by-model wrapper. The ignored `temp_evaluate_api.py` is a local smoke script, not a public wrapper. Any future wrapper should be thin over `STPPConfig`, `ConfigRegistry`, and `STPPRunner`, with tests proving it does not create a parallel model construction path.

## Excluded From v1

- A stable `seahorse` executable alias.
- External repository merges.
- Long experiment reruns as part of install validation.
- Public support requirements for Pegasus-specific launch scripts.
- Stable normal-user Python wrapper.
- Stale exploratory notebooks.
