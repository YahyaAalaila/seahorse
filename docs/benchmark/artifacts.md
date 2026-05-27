# Artifacts and Run Directories

Every `fit`, `tune`, and `bench` call writes deterministic artifacts to disk. This page explains what gets saved and where to find it.

## Single-run Directory (fit)

`fit` writes under `{--out}/fit/{preset}/{run_id}/`:

```text
runs/fit/auto_stpp/<run_id>/
  config.yaml          ← resolved STPPConfig used for this run
  run_result.json      ← RunResult with val_nll, test_nll, norm_stats
  checkpoint.ckpt      ← PyTorch Lightning checkpoint (best val epoch)
```

`run_id` is a timestamp-based identifier. Use it when re-loading a run:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/fit/auto_stpp/<run_id> \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core
```

## Benchmark Directory (bench)

`bench` writes top-level campaign artifacts plus one subdirectory per benchmark cell (preset × dataset × seed):

```text
runs/bench/
  bench_meta.json          ← benchmark configuration and provenance
  cell_index.json          ← maps (preset, dataset, seed) → run directory path
  results.json             ← serialised RunResult for each cell
  report.html              ← self-contained benchmark report
  table_test_nll_all.csv   ← NLL table over all reported runs
  table_test_nll_exact.csv ← Exact/raw-space NLL table
  fit/
    poisson_gmm/
      dataset_a/
        seed_1/
          <run_id>/        ← same layout as single-run directory
    hawkes_gmm/
      ...
```

Use `cell_index.json` to look up the run directory for a specific (preset, dataset, seed) combination when running follow-up evaluation commands.

## run_result.json Fields

| Field | Type | Description |
| --- | --- | --- |
| `val_nll` | float | Per-event NLL on the validation split at the best checkpoint |
| `test_nll` | float | Per-event NLL on the test split |
| `norm_stats` | dict | `{normalize, time_mean, time_std, loc_mean, loc_std}` |
| `config` | dict | Full resolved `STPPConfig` |

`norm_stats` lets you convert normalized NLL back to original-coordinate NLL:

```
NLL_original = NLL_normalised − log(time_std × loc_std_x × loc_std_y)
```

## Evaluate Artifacts

Running `evaluate metrics` adds a timestamped directory under `--out`:

```text
runs/evaluate/core_test/
  metrics.json              ← per-metric result with availability, value, method
  evaluation_manifest.json  ← run metadata and evaluation settings
  *_per_event.npy           ← per-event arrays for offline analysis
```

For predictive, generative, or surface profiles, additional artifact families are written under an `artifacts/` subdirectory. These can be merged across shards using `evaluate merge-artifacts`.

## Reloading A Run

Load a saved run through the Python API:

```python
from unified_stpp import AutoSTPP

model = AutoSTPP.load("runs/fit/auto_stpp/<run_id>")
scores = model.evaluate(test)
```

Or through the base estimator class:

```python
from unified_stpp import STPPEstimator

model = STPPEstimator.load("runs/fit/auto_stpp/<run_id>")
```
