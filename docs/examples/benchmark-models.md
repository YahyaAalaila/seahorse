# Benchmark Models

Use the CLI for benchmark campaigns. This keeps model presets, data sources,
seeds, overrides, and output artifacts explicit.

## Notebook

Use <a href="../../notebooks/02_benchmark_models_cli.ipynb">02 Benchmark Models With The CLI</a>
for an executable walkthrough. Open in Colab badges will be added after public
release.

## Benchmark A Split Collection

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm neural_attncnf \
  --splits_dir splits \
  --datasets dataset_a dataset_b \
  --seeds 1 2 3 \
  --out runs/bench \
  --n_workers 1
```

Use `--override` for campaign-wide training settings:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --seeds 1 \
  --out runs/bench_smoke \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

## Benchmark One Hugging Face Dataset

```bash
python -m unified_stpp bench \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --seeds 1 \
  --out runs/bench_hf
```

Pin `--dataset-revision` for reproducible comparisons.

## Outputs To Inspect

Benchmark runs write campaign-level files under `--out`, including:

- `bench_meta.json`
- `cell_index.json`
- `results.json`
- `report.html`
- `table_test_nll_all.csv` when available
- `table_test_nll_exact.csv` when available

Each benchmark cell also writes a normal saved run directory that can be passed
to `python -m unified_stpp evaluate ...`.
