# Seahorse / unified-stpp

Seahorse (`unified-stpp`, imported as `unified_stpp`) is a research framework
for spatio-temporal point process benchmarks. It keeps different model families
behind one training, benchmark, data, and post-fit evaluation interface while
preserving each model's own internal objective and preprocessing choices.

The current public workflow is:

```text
data -> tune -> bench -> evaluate
```

The benchmark path is raw-first: datasets enter in original coordinates, models
may transform internally, and benchmark-facing NLL is reported in the selected
canonical space whenever the model can support it.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Development and release checks:

```bash
python -m pip install -e ".[dev]"
```

All optional extras, including HPO:

```bash
python -m pip install -e ".[all]"
```

HPO only:

```bash
python -m pip install -e ".[hpo]"
```

## Public CLI

```bash
python -m unified_stpp fit      ...
python -m unified_stpp tune     ...
python -m unified_stpp bench    ...
python -m unified_stpp evaluate ...
```

Use `--help` on any command for the full argument list.

## Data Contract

The canonical dataset format is JSONL, one sequence per line:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
{"times": [0.2, 0.9], "locations": [[0.5, 0.5], [0.4, 0.6]], "marks": [0, 1]}
```

Required fields:

- `times`: event times, length `T`
- `locations`: event locations, shape `T x d`

Optional fields:

- `marks`: integer event marks
- `event_covariates`: event-level covariates
- `field_covariates`: covariates pre-evaluated at events

Benchmark-ready datasets should expose:

```text
dataset_root/
  train.jsonl
  val.jsonl
  test.jsonl
  dataset_meta.json      # optional but recommended
  splits.json            # optional, for extra split provenance
```

## Dataset Loading

The data API supports both Hugging Face-backed datasets and direct local paths:

```python
from unified_stpp.data import download_dataset, load_dataset

root = download_dataset("yahya021/covid-stpp")
splits = load_dataset("yahya021/covid-stpp")
train = load_dataset("yahya021/covid-stpp", split="train")

local_splits = load_dataset("/path/to/local/dataset")
```

`revision` means a Hugging Face dataset repo revision: tag, branch, commit, or
SHA. For paper releases, use explicit tags such as `v1` after creating the tag
on the HF dataset repo:

```python
root = download_dataset("yahya021/covid-stpp", revision="v1")
```

For the v1/paper release, real datasets are hosted on Hugging Face. Processed
HawkesNest suite 3 and suite 4 synthetic datasets will also be uploaded to
Hugging Face with documented repo paths and revisions. Synthetic HawkesNest
generation is documented through a release notebook; the repository data
contract remains the JSONL split layout shown above.

The CLI accepts the same source style:

```bash
python -m unified_stpp fit \
  --preset deep_stpp \
  --dataset yahya021/covid-stpp \
  --out runs/smoke_deep

python -m unified_stpp fit \
  --preset deep_stpp \
  --train /path/train.jsonl \
  --val /path/val.jsonl \
  --test /path/test.jsonl \
  --out runs/smoke_deep_local
```

## Presets

Current public preset names:

| Preset | Family | Status | NLL lane |
|---|---|---:|---|
| `poisson_gmm` | factorized Poisson + GMM spatial | canonical | exact |
| `hawkes_gmm` | factorized Hawkes + GMM spatial | canonical | exact |
| `selfcorrecting_gmm` | factorized self-correcting + GMM spatial | canonical | exact |
| `poisson_cnf` | factorized Poisson + CNF spatial | canonical | exact |
| `hawkes_cnf` | factorized Hawkes + CNF spatial | canonical | exact |
| `selfcorrecting_cnf` | factorized self-correcting + CNF spatial | canonical | exact |
| `poisson_tvcnf` | factorized Poisson + time-varying CNF spatial | canonical | exact |
| `hawkes_tvcnf` | factorized Hawkes + time-varying CNF spatial | canonical | exact |
| `selfcorrecting_tvcnf` | factorized self-correcting + time-varying CNF spatial | canonical | exact |
| `rmtpp_gmm` | RMTPP temporal + GMM spatial | canonical | exact |
| `thp_gmm` | THP temporal + GMM spatial | canonical | exact |
| `deep_stpp` | DeepSTPP-style neural Hawkes/GMM model | canonical | exact/reportable |
| `auto_stpp` | AutoSTPP faithful AutoInt model | canonical | exact/reportable |
| `smash` | SMASH score-based model | canonical | approximate |
| `diffusion_stpp` | diffusion STPP model | canonical | approximate/surrogate |
| `nsmpp` | NSMPP DeepBasis implementation | benchmark-supported | exact/reportable |
| `njsde` | NJSDE + GMM | benchmark-supported | exact/reportable |
| `neural_jumpcnf` | Neural STPP JumpCNF | benchmark-supported | exact/reportable |
| `neural_attncnf` | Neural STPP attentive CNF | benchmark-supported | exact/reportable |
| `auto_stpp_legacy` | older AutoSTPP implementation | legacy | compatibility |

Deprecated legacy aliases remain accepted in the registry for old runs, but new
configs, docs, and benchmark tables should use the canonical names above.

## Training One Model

```bash
python -m unified_stpp fit \
  --preset auto_stpp \
  --dataset yahya021/covid-stpp \
  --out runs/fit_covid_auto
```

Or with explicit splits:

```bash
python -m unified_stpp fit \
  --preset hawkes_gmm \
  --train data/sthp0/train.jsonl \
  --val data/sthp0/val.jsonl \
  --test data/sthp0/test.jsonl \
  --out runs/fit_hawkes_gmm
```

Use dotted overrides for quick smoke runs:

```bash
python -m unified_stpp fit \
  --preset deep_stpp \
  --dataset yahya021/covid-stpp \
  --out /tmp/deep_stpp_smoke \
  --override \
    training.n_epochs=1 \
    training.batch_size=16 \
    training.device=cpu \
    data.num_workers=0
```

Run directories use:

```text
<out>/fit/<preset>/<run_id>/
  config.yaml
  resolved_config.yaml
  run_result.json
  artifacts.json
  metrics.csv
  checkpoints/
```

`run_result.json` records the model objective, validation objective, benchmark
test NLL metadata, raw/native reporting space, preset status, selected
checkpoint, data fingerprint/provenance, and run directory.

## Benchmark NLL Semantics

The benchmark-facing `test_nll` is not the old mixed full-sequence/windowed
quantity. It is now a held-out next-event predictive score:

For each test sequence `(e_1, ..., e_T)`, the scored contexts are:

```text
H_i = (e_1, ..., e_{i-1}) -> target e_i, for i = 2, ..., T
```

The reported value is the mean per scored next-event context.

Important distinctions:

- `val_objective`: model-native validation objective used for checkpointing.
- `test_nll`: benchmark-facing held-out next-event NLL.
- `nll_kind`: `exact`, `approx`, or `none`.
- `nll_report_space`: `raw` or `native`.

Exact families should report comparable raw-space NLL when their transform
artifact supports the required Jacobian correction. Approximate families such
as `smash` and `diffusion_stpp` keep explicit approximate/surrogate metadata.

## Hyperparameter Tuning

HPO uses Ray Tune and reads search spaces from YAML. Scalars are fixed values;
lists are choices; `{min, max}` defines numeric search ranges.

```bash
python -m unified_stpp tune \
  --config unified_stpp/configs/auto_stpp_hpo.yaml \
  --dataset yahya021/covid-stpp \
  --seed 42 \
  --n_trials 30 \
  --max-concurrent-trials 1 \
  --out runs/exp1/covid-stpp/tune/auto_stpp/auto_stpp_best.yaml
```

The tuning command writes:

```text
*_best.yaml
*.data_manifest.json
*.trials.json
*.trials.csv
*.hpo_manifest.json
```

For production benchmark runs, tune first, freeze the best YAMLs, then run
`bench` with `--hpo_configs_dir`. Avoid mixing fresh HPO and pre-tuned configs
inside one benchmark table unless you explicitly allow that policy.

## Benchmark Runs

Single HF-backed dataset:

```bash
python -m unified_stpp bench \
  --presets auto_stpp deep_stpp smash diffusion_stpp \
  --dataset yahya021/covid-stpp \
  --seeds 42 \
  --out runs/bench_covid \
  --hpo_configs_dir runs/exp1/covid-stpp/tune \
  --n_workers 1 \
  --no-normalize
```

Local split collection:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm selfcorrecting_gmm \
  --splits_dir data/bench_splits \
  --seeds 42 43 44 \
  --out runs/bench_factorized \
  --no-normalize
```

Benchmark outputs include:

```text
bench_meta.json
data_manifest.json
cell_index.json
results.json
report.html
table_*_all.csv
table_*_exact.csv
fit/<preset>/<run_id>/...
```

## Post-Fit Evaluation

### Metric Profiles

The main packaged evaluation path is:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/bench_covid/fit/deep_stpp/<run_id> \
  --data /path/to/test.jsonl \
  --split test \
  --metric-profile predictive \
  --k-pred 128 \
  --seed 42 \
  --out runs/eval/deep_stpp_predictive
```

Profiles:

- `core`: lightweight scalar metrics
- `nll`: NLL-focused metrics
- `predictive`: benchmark-aligned held-out next-event predictive artifacts
- `generative`: rollout-based metrics
- `surface`: intensity-grid artifacts
- `full`: broader packaged metric set

The predictive profile writes context-level artifacts such as:

```text
next_event_context_index.npz
next_event_benchmark_summary.json
scores/<metric>_per_context.npy
scores/<metric>_per_sequence_mean.npy
scores/<metric>_last_context_per_sequence.npy
```

This preserves within-run context variation, within-run sequence variation, and
per-sequence last-context scores for later seed-level aggregation.

### Predictive Compare

`predictive-compare` is qualitative future-window visualization, not the
primary benchmark metric path:

```bash
python -m unified_stpp evaluate predictive-compare \
  --run runs/bench_covid/fit/deep_stpp/<run_id> \
  --history /path/to/test.jsonl \
  --seq-idx 0 \
  --start-event-idx 20 \
  --rollout-mode teacher_forced \
  --horizon 1.0 \
  --n-rollouts 32 \
  --plot-style both \
  --out runs/eval/predictive_compare
```

### Surface Diagnostics

`surface` is a secondary diagnostic path for intensity surfaces:

```bash
python -m unified_stpp evaluate surface \
  --run runs/bench_covid/fit/hawkes_gmm/<run_id> \
  --history /path/to/test.jsonl \
  --seq-idx 0 \
  --profile history_frame \
  --out runs/eval/hawkes_surface
```

## Examples

Minimal public CLI examples are in `docs/examples/cli_examples.md`. They use
the tiny synthetic smoke-test JSONL files under `examples/tiny_jsonl/` and
cover `fit`, `tune`, `bench`, and `evaluate`. These files are only for
quickstart and CLI smoke checks; they are not benchmark data.

Cluster launch templates are internal operational material and are not required
for the public v1 workflow.

## Framework Internals

Every model family is represented by:

- a `StateModel`, which encodes observed history into a `StateContext`
- an `EventModel`, which computes loss/NLL/intensity-facing outputs from that state
- a config class registered by preset name

The outer model wrapper is `UnifiedSTPP`.

```python
ctx = state_model.encode_history(
    times=times,
    locations=locations,
    lengths=lengths,
)

out = event_model.training_loss(
    times=times,
    locations=locations,
    lengths=lengths,
    state=ctx,
)
```

Different families can keep different internals: Hawkes kernels, GRUs, CNFs,
AutoInt monotone integrators, score networks, and diffusion decoders. The
framework standardizes only the benchmark-facing contracts.

## Repository Layout

```text
unified_stpp/
  cli/                 # fit, tune, bench, evaluate commands
  config/              # Pydantic config schema and tuning config
  data/                # JSONL datasets, HF/local loading, transforms, collate
  benchmark/           # benchmark grid, HPO, report aggregation
  runner/              # STPPRunner, saved run artifacts, RunResult
  training/            # Lightning module and data module
  models/
    configs/           # preset registry and family configs
    state_models/      # framework-facing history encoders
    event_models/      # framework-facing objectives/intensities
    temporal_models/   # internal temporal processes
    spatial_models/    # internal spatial densities/flows
  evaluation/
    metrics/           # metric implementations
    predictive/        # predictive rollout/summary workflows
    surface/           # surface query/diagnostic workflows
```

## Development Checks

```bash
pytest
```

Useful focused checks:

```bash
pytest tests/test_config_resolution.py
pytest tests/test_benchmark_config.py
pytest tests/test_evaluate_cli.py
pytest tests/test_evaluation_import_audit.py
```

## Documentation Pointers

- `docs/BENCHMARK.md`: benchmark contract and reporting semantics
- `docs/EXPERIMENT_READINESS.md`: experiment freeze checklist
- `docs/metrics_catalog.md`: metric definitions
- `docs/release/`: v1 release audit, validation commands, and paper artifact reproducibility notes
- `docs/examples/cli_examples.md`: minimal public CLI examples
- `docs/internal/`: paper-facing summaries and locked model-parameter notes
