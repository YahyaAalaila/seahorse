# Compare Results

After running a benchmark you typically want to compare models by NLL, inspect predictive quality on specific sequences, and understand which families differ meaningfully.

## Reading the Benchmark Tables

`bench` writes summary tables to the campaign directory:

```text
runs/bench/
  table_test_nll_all.csv   ← test NLL for every (preset, dataset, seed) cell
  table_test_nll_exact.csv ← exact-NLL families only
  report.html              ← self-contained HTML report with tables and metadata
```

Open `report.html` in a browser for a quick interactive view. Use the CSV files for downstream analysis.

## Programmatic Comparison

Load benchmark results with the `BenchmarkTable` class:

```python
from seahorse import BenchmarkTable

table = BenchmarkTable.from_bench_dir("runs/bench")
df = table.to_dataframe()
print(df.pivot_table(index="preset", columns="dataset", values="test_nll"))
```

## Predictive Comparison (CLI)

Visualize where two models place their predictions on a single sequence:

```bash
python -m seahorse evaluate predictive-compare \
  --run runs/bench/fit/auto_stpp/dataset_a/seed_1/<run_id> \
  --run runs/bench/fit/deep_stpp/dataset_a/seed_1/<run_id> \
  --label auto_stpp \
  --label deep_stpp \
  --history data/my_dataset/test.jsonl \
  --split test \
  --seq-idx 0 \
  --horizon 1.0 \
  --out runs/compare/auto_vs_deep
```

Look up run directories for specific cells in `cell_index.json`.

## Quantitative Predictive Metrics

For a benchmark-aligned predictive comparison across multiple models, run `evaluate metrics` on each cell separately and collect the results:

```bash
python -m seahorse evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile predictive \
  --k-pred 32 \
  --out runs/evaluate/predictive/auto_stpp_dataset_a_seed1
```

Then read `metrics.json` from each output directory to compare CRPS, energy score, MAE, and RMSE across presets.

## Interpreting NLL Differences

- NLL values are only directly comparable across presets that share the **same dataset, normalization policy, and metric definition** — which `bench` enforces via the [benchmark contract](../learn/execution-contract.md).
- Exact-NLL families (`auto_stpp`, `deep_stpp`, `poisson_gmm`, ...) are comparable with each other.
- Approximate families (`smash`, `diffusion_stpp`) optimize surrogate objectives; note this when including them in comparison tables.
- See the [Model Capability Matrix](../model-capability-matrix.md) for the NLL type of each preset.

## Common Pitfalls

- **Using the wrong run directory**: always look up the cell in `cell_index.json` rather than guessing the path.
- **Comparing NLL across different normalization settings**: `bench` prevents this by default, but check `run_result.json → norm_stats` if comparing runs from separate campaigns.
- **Missing metrics**: a metric marked unavailable in `metrics.json` means the model lacks the capability, not that evaluation failed.
