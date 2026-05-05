# Seahorse

Seahorse, packaged as `unified-stpp`, is a framework for spatio-temporal point
process experiments. It gives you one CLI for training, tuning, benchmarking,
and evaluating STPP models.

Start with a local smoke run:

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

- [Getting Started](getting-started.md) for installation and first commands.
- [Data Format](data-format.md) for JSONL records and dataset layouts.
- [CLI Reference](cli.md) for the supported module CLI.
- [Benchmarks](benchmarks.md) for benchmark grids and outputs.
- [Evaluation](evaluation.md) for post-fit metrics and diagnostics.
- [Adding a Model](adding-a-model.md) for the registry-based extension path.
