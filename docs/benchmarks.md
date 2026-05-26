# Benchmark Campaigns

Benchmarks are a CLI/config workflow. Use this path when you need reproducible
campaigns, HPO, benchmark reports, and paper-style artifacts.

Run a benchmark grid with one or more presets, one or more datasets, and one or
more seeds:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
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
  --dataset-revision main \
  --seeds 1 \
  --out runs/bench_one
```

For a collection of datasets, use `--splits_dir`:

```text
splits/
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
  --splits_dir splits \
  --datasets dataset_a dataset_b \
  --seeds 1 \
  --out runs/bench_subset
```

## Model Families

Seahorse presets cover several public families:

| Family | Example presets or aliases | Notes |
| --- | --- | --- |
| Parametric baselines | `poisson_gmm`, `hawkes_gmm`, `selfcorrecting_gmm` | Compact temporal process plus GMM spatial components. |
| Flow spatial baselines | `poisson_cnf`, `hawkes_cnf`, `selfcorrecting_cnf` | Continuous normalizing-flow spatial components. |
| Time-varying CNF baselines | `poisson_tvcnf`, `hawkes_tvcnf`, `selfcorrecting_tvcnf` | Time-conditioned flow spatial components. |
| Neural temporal baselines | `rmtpp_gmm`, `thp_gmm` | Neural temporal model with GMM spatial output. |
| Neural STPP variants | `njsde`, `neural_jumpcnf`, `neural_attncnf`, `neural_stpp_attn_sc` | Neural conditioning with sampler or exact-density variants depending on preset. |
| Paper-style families | `auto_stpp`, `deep_stpp`, `smash`, `diffusion_stpp`, `nsmpp` | Wrapped behind registered presets and Python aliases. |

Use `python -m unified_stpp fit --help` and the registered preset list in the
installed package as the source of truth for the current build.

## Capability Notes

Metric and visualization support depends on model capabilities:

- Exact likelihood paths support NLL-family metrics.
- Native or exact sampling paths support next-event predictive metrics.
- Surface diagnostics require an intensity grid path or an approximation path
  planned by the selected evaluation profile.
- Unsupported combinations fail explicitly rather than silently producing fake
  metrics.

## Benchmark Policy

The benchmark config defaults to raw input coordinates and raw-space test NLL
reporting. Use `--normalize` only when you intentionally want the benchmark
policy to z-score time and space for all presets.

Use `--override` for cross-preset config values that are part of the invocation:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --seeds 1 \
  --out runs/bench_short \
  --override training.n_epochs=10 training.batch_size=64
```

## HPO Inside A Benchmark

Benchmark HPO requires an explicit tuning dataset:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
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
  --splits_dir splits \
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

Each fit run under the benchmark also writes its own config, metrics,
checkpoint, and `run_result.json` artifacts.

## Next Steps

- Use [Run A Small Benchmark](examples/run-a-small-benchmark.md) for a concrete example.
- Use [Evaluation And Visualization](evaluation.md) for post-fit metrics and
  plots.
