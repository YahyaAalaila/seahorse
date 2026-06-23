# Benchmark Campaigns

!!! tip "When to use this"
    Use the benchmark path for reproducible multi-preset, multi-dataset, multi-seed runs
    with saved artifacts. For a single model in a notebook, the [Python API](python-api.md) is simpler.

```bash
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --seeds 1 2 3 \
  --out runs/bench \
  --n_workers 1
```

## Inputs

For a collection of datasets, point `--splits_dir` at a directory:

```text
splits/
  dataset_a/train.jsonl  dataset_a/val.jsonl  dataset_a/test.jsonl
  dataset_b/train.jsonl  dataset_b/val.jsonl  dataset_b/test.jsonl
```

??? example "Show CLI command — single dataset or Hugging Face source"
    ```bash
    python -m seahorse bench \
      --preset poisson_gmm \
      --dataset owner/repo[/subdir] \
      --dataset-revision main \
      --seeds 1 \
      --out runs/bench_one
    ```

??? example "Show CLI command — filter specific datasets from a collection"
    ```bash
    python -m seahorse bench \
      --presets poisson_gmm hawkes_gmm \
      --splits_dir splits \
      --datasets dataset_a dataset_b \
      --seeds 1 \
      --out runs/bench_subset
    ```

## Model Families

| Family | Example presets | Notes |
| --- | --- | --- |
| Parametric baselines | `poisson_gmm`, `hawkes_gmm`, `selfcorrecting_gmm` | Compact temporal + GMM spatial. |
| Flow spatial | `poisson_cnf`, `hawkes_cnf`, `selfcorrecting_cnf` | CNF spatial components. |
| Time-varying CNF | `poisson_tvcnf`, `hawkes_tvcnf`, `selfcorrecting_tvcnf` | Time-conditioned flow spatial. |
| Neural temporal | `rmtpp_gmm`, `thp_gmm` | Neural temporal + GMM spatial output. |
| Neural STPP | `njsde`, `neural_jumpcnf`, `neural_attncnf` | Exact-density variants. |
| Paper families | `auto_stpp`, `deep_stpp`, `smash`, `diffusion_stpp`, `nsmpp` | Registered presets and Python aliases. |

Use `python -m seahorse fit --help` for the current preset list.

## Capability Notes

- Exact likelihood paths support NLL-family metrics.
- Native or exact sampling paths support next-event predictive metrics.
- Surface diagnostics require an intensity grid or approximation path.
- Unsupported combinations fail explicitly rather than silently.

## Benchmark Policy

The benchmark config defaults to raw input coordinates and raw-space test NLL.
Use `--normalize` only when you intentionally want z-scored time and space.

??? example "Show CLI command — short run with training overrides"
    ```bash
    python -m seahorse bench \
      --presets poisson_gmm hawkes_gmm \
      --splits_dir splits \
      --seeds 1 \
      --out runs/bench_short \
      --override training.n_epochs=10 training.batch_size=64
    ```

## HPO Inside A Benchmark

Benchmark HPO requires an explicit tuning dataset:

??? example "Show CLI command — HPO benchmark"
    ```bash
    python -m seahorse bench \
      --presets poisson_gmm hawkes_gmm \
      --splits_dir splits \
      --tune \
      --tune-dataset dataset_a \
      --n_trials 20 \
      --seeds 1 2 3 \
      --out runs/bench_hpo
    ```

??? example "Show CLI command — reuse previously tuned configs"
    ```bash
    python -m seahorse bench \
      --presets poisson_gmm hawkes_gmm \
      --splits_dir splits \
      --hpo_configs_dir runs/hpo \
      --seeds 1 \
      --out runs/bench_reuse_hpo
    ```

## Outputs

| File | Contents |
| --- | --- |
| `bench_meta.json` | Benchmark configuration and provenance |
| `cell_index.json` | Index of benchmark cells and run directories |
| `results.json` | Serialized run results |
| `report.html` | Self-contained benchmark report |
| `table_test_nll_all.csv` | NLL table over all reported runs |
| `table_test_nll_exact.csv` | Exact/raw-space NLL table |

Each fit run also writes its own config, metrics, checkpoint, and `run_result.json`.

## Next Steps

- [Run A Small Benchmark](examples/run-a-small-benchmark.md) — concrete walkthrough.
- [Evaluation And Visualization](evaluation.md) — post-fit metrics and plots.
