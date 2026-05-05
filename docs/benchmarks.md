# Benchmarks

Run a benchmark grid with one or more presets, one or more datasets, and one or
more seeds:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --seeds 1 2 3 \
  --out runs/bench \
  --n_workers 1
```

## Inputs

For one dataset, pass a local dataset directory or Hugging Face dataset source:

```bash
python -m unified_stpp bench \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --seeds 1 \
  --out runs/bench_one
```

For a collection of datasets, use `--splits_dir`:

```text
splits_root/
  dataset_a/train.jsonl
  dataset_a/val.jsonl
  dataset_a/test.jsonl
  dataset_b/train.jsonl
  dataset_b/val.jsonl
  dataset_b/test.jsonl
```

Filter a collection with `--datasets`:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --datasets dataset_a dataset_b \
  --seeds 1 \
  --out runs/bench_subset
```

## Benchmark Policy

The benchmark config defaults to raw input coordinates and raw-space test NLL
reporting. Use `--normalize` only when you intentionally want the benchmark
policy to z-score time and space for all presets.

Use `--override` for cross-preset config values that are part of the invocation:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --seeds 1 \
  --out runs/bench_short \
  --override training.n_epochs=10 training.batch_size=64
```

## HPO Inside A Benchmark

Benchmark HPO requires an explicit tuning dataset:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --tune \
  --tune-dataset dataset_a \
  --n_trials 20 \
  --seeds 1 2 3 \
  --out runs/bench_hpo
```

To reuse previously tuned configs, pass a directory containing
`{preset}_best.yaml` files:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --hpo_configs_dir runs/hpo \
  --seeds 1 \
  --out runs/bench_reuse_hpo
```

## Outputs

A benchmark run writes run-level artifacts and benchmark-level reports under
the output directory. Public outputs include:

- `bench_meta.json`: benchmark configuration and provenance.
- `cell_index.json`: index of benchmark cells and run directories.
- `results.json`: serialized run results.
- `report.html`: self-contained benchmark report.
- `table_test_nll_all.csv`: table over all reported runs when available.
- `table_test_nll_exact.csv`: exact/raw-space NLL table when available.

Each fit run under the benchmark also writes its own config, metrics, checkpoint,
and `run_result.json` artifacts.
