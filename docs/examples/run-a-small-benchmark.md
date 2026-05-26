# Run A Small Benchmark

## Goal

This example runs a small Seahorse benchmark across several model presets on
one dataset. It uses the CLI/config path so the run records presets, data
source, seed, overrides, and benchmark artifacts.

The default command uses one seed and writes outputs under
`runs/examples/small_benchmark`.

## When To Use This Instead Of The Python API

Use this benchmark workflow when you want to:

- compare several presets on the same split.
- keep reproducible run directories and benchmark reports.
- use the same execution path as paper-style experiments.
- scale later to more datasets, presets, seeds, or HPO.

Use the Python API when you only want to train and inspect one model in a script
or notebook.

## Input Data Layout

For a local one-dataset benchmark, create a directory with JSONL split files:

```text
data/my_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
```

Each JSONL line is one sequence:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

The benchmark CLI receives the dataset directory with `--dataset`; it does not
take separate `--train`, `--val`, or `--test` flags.

For Hugging Face-backed data, use a dataset repository placeholder:

```text
owner/repo[/subdir]
```

Pin `--dataset-revision` when the benchmark should be reproducible.

## Run The Benchmark

Run four presets on one local dataset with one seed:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp \
  --dataset data/my_dataset \
  --seeds 1 \
  --out runs/examples/small_benchmark \
  --n_workers 1
```

For a short smoke test, add conservative training overrides:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp \
  --dataset data/my_dataset \
  --seeds 1 \
  --out runs/examples/small_benchmark \
  --n_workers 1 \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

For a Hugging Face dataset source:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --seeds 1 \
  --out runs/examples/small_benchmark \
  --n_workers 1
```

## Inspect Output Directory

The benchmark writes campaign-level files under:

```text
runs/examples/small_benchmark/
```

Expected campaign artifacts include:

- `bench_meta.json`
- `cell_index.json`
- `results.json`
- `report.html`
- `table_test_nll_all.csv` when available
- `table_test_nll_exact.csv` when available

Each benchmark cell also writes a saved run directory for its preset, dataset,
and seed. Use `cell_index.json` to map benchmark cells to run directories.

## Generate/Evaluate Metrics If Needed

The benchmark report includes the benchmark table produced by the run. If you
need an additional post-fit metric profile for one saved run, use the run
directory recorded in `cell_index.json`:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/examples/small_benchmark/evaluate_core
```

Use `python -m unified_stpp evaluate metrics --help` to see the metric profiles
available in the installed version.

## Common Errors

`error: ... train.jsonl ... val.jsonl`

: Check that `--dataset` points to a directory containing `train.jsonl` and
  `val.jsonl`. `test.jsonl` is normally needed for benchmark evaluation.

`Unknown model preset`

: Check preset spelling. This example uses `poisson_gmm`, `hawkes_gmm`,
  `auto_stpp`, and `deep_stpp`.

`Out of memory`

: Start with one seed, `--n_workers 1`, smaller batch size, or fewer presets.

`ImportError` for HPO dependencies

: This example does not use HPO. Remove `--tune` if you added it, or install the
  HPO extra before running tuned benchmarks.

Existing output directory contains older files

: Choose a fresh `--out` directory when you want a clean benchmark artifact
  bundle.

## Scaling Up: More Presets, More Seeds, HPO

Add more presets to `--presets`:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp nsmpp neural_jumpcnf \
  --dataset data/my_dataset \
  --seeds 1 \
  --out runs/examples/larger_benchmark \
  --n_workers 1
```

Use more seeds when you want variability estimates:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp deep_stpp \
  --dataset data/my_dataset \
  --seeds 1 2 3 \
  --out runs/examples/small_benchmark_seeds \
  --n_workers 1
```

Use HPO only when you have selected a tuning dataset:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --dataset data/my_dataset \
  --tune \
  --tune-dataset my_dataset \
  --n_trials 20 \
  --seeds 1 2 3 \
  --out runs/examples/small_benchmark_hpo \
  --n_workers 1
```

For multi-dataset benchmark collections, use `--splits_dir` instead of
`--dataset`; see [Benchmark Campaigns](../benchmarks.md).
