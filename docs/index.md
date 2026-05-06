# Seahorse

Seahorse, packaged as `unified-stpp`, is a framework for spatio-temporal point
process experiments. It is intended to support two user-facing interfaces:

- A Python-first API for normal users training and evaluating one model
  programmatically.
- A CLI/config interface for reproducible runs, HPO, benchmark campaigns, and
  paper-style artifacts.

## Python-First Quickstart

Run one model programmatically:

```python
from unified_stpp import AutoSTPP, PoissonGMM, load_jsonl

train = load_jsonl("path/to/train.jsonl")
val = load_jsonl("path/to/val.jsonl")
test = load_jsonl("path/to/test.jsonl")

model = AutoSTPP(device="cpu")
baseline = PoissonGMM()

model.fit(train, val, test, epochs=10, batch_size=64)
scores = model.evaluate(test)
samples = model.predict_next(test, n_samples=32)
```

See [Python API](python-api.md) for method details and current limits.

## CLI Quickstart

Use the CLI when you need reproducible run artifacts:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --test path/to/test.jsonl \
  --out runs/quickstart \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

Then evaluate the saved run:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/quickstart/fit/poisson_gmm/<run_id> \
  --data path/to/test.jsonl \
  --split test \
  --metric-profile core
```

Replace `<run_id>` with the run directory created by `fit`.

## What Seahorse Provides

- `fit`: train one preset or YAML config.
- `tune`: run HPO for one preset or YAML config.
- `bench`: run benchmark grids across presets, datasets, and seeds.
- `evaluate`: run post-fit metrics, predictive comparisons, surfaces, or artifact merges.

## Where To Go Next

- [Python API](python-api.md) for the normal-user model API.
- [Getting Started](getting-started.md) for installation and supported commands.
- [Data Format](data-format.md) for JSONL records and dataset layouts.
- [CLI Reference](cli.md) for the supported module CLI.
- [Benchmarks](benchmarks.md) for benchmark grids and outputs.
- [Evaluation](evaluation.md) for post-fit metrics and diagnostics.
- [Adding a Model](adding-a-model.md) for the registry-based extension path.
