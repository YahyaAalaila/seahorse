# Evaluate Results

## Goal

This example shows how to inspect a completed Seahorse run, compute additional
metric reports, and create basic visualization artifacts with the CLI.

Use this after running a single `fit` command or after a benchmark such as
[Run A Small Benchmark](run-a-small-benchmark.md).

## Input: A Fitted Run Or Benchmark Directory

Evaluation commands need one saved run directory. For a benchmark, look up the
run directory for the preset, dataset, and seed you want in `cell_index.json`.

Use that path anywhere this page shows `path/to/run_dir`.

## Run Evaluation From CLI

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/examples/small_benchmark/evaluate/core
```

??? example "Show CLI command — predictive metrics"
    ```bash
    python -m unified_stpp evaluate metrics \
      --run path/to/run_dir \
      --data data/my_dataset/test.jsonl \
      --split test \
      --metric-profile predictive \
      --k-pred 32 \
      --out runs/examples/small_benchmark/evaluate/predictive
    ```

??? example "Show CLI command — cap sequences for quick inspection"
    ```bash
    python -m unified_stpp evaluate metrics \
      --run path/to/run_dir \
      --data data/my_dataset/test.jsonl \
      --split test \
      --metric-profile core \
      --max-seqs 10 \
      --out runs/examples/small_benchmark/evaluate/core_10seq
    ```

## Inspect Metrics/Artifacts

`metrics.json` is the first file to inspect. It records each metric result,
availability, scalar value, method, and any reason a metric was unavailable.

??? example "Show expected output — artifact layout"
    ```text
    runs/examples/small_benchmark/evaluate/core/
      metrics.json
      evaluation_manifest.json
      *_per_event.npy
    ```

## Plot Or Visualize Results

```bash
python -m unified_stpp evaluate predictive-compare \
  --run path/to/run_dir \
  --label auto_stpp \
  --history data/my_dataset/test.jsonl \
  --split test \
  --seq-idx 0 \
  --horizon 1.0 \
  --out runs/examples/small_benchmark/visualize/predictive_compare
```

??? example "Show CLI command — compare two runs"
    ```bash
    python -m unified_stpp evaluate predictive-compare \
      --run path/to/auto_stpp_run_dir \
      --run path/to/deep_stpp_run_dir \
      --label auto_stpp \
      --label deep_stpp \
      --history data/my_dataset/test.jsonl \
      --split test \
      --seq-idx 0 \
      --horizon 1.0 \
      --out runs/examples/small_benchmark/visualize/predictive_compare_two_models
    ```

??? example "Show CLI command — surface diagnostics"
    ```bash
    python -m unified_stpp evaluate surface \
      --run path/to/run_dir \
      --history data/my_dataset/test.jsonl \
      --split test \
      --seq-idx 0 \
      --profile history_frame \
      --out runs/examples/small_benchmark/visualize/surface
    ```

    `history_frame` supports `auto_stpp` and `deep_stpp`. `future_exact` supports
    the neural exact families listed by `python -m unified_stpp evaluate surface --help`.

## Interpret Common Outputs

`nll`, `temporal_nll`, `spatial_nll`
: Lower values are better when comparing runs on the same dataset, split, and
  metric definition. Be careful comparing exact and approximate NLL families.

Predictive metrics
: Temporal CRPS, spatial energy score, MAE, RMSE, and coverage use sampled
  next-event predictions. Availability depends on model sampling capability.

Surface diagnostics
: Qualitative diagnostics over an intensity or density grid. Useful for
  inspecting where a model places event mass over space and time.

Unavailable metrics
: An unavailable metric in `metrics.json` is not a failed run — it usually
  means the profile needs a capability the model does not expose.

## Common Errors

`Path ... is not a saved run directory`
: Use a per-model run directory, not the top-level benchmark directory.
  Look up the run directory in `cell_index.json`.

`FileNotFoundError` for `test.jsonl`
: Check the `--data` or `--history` path.

`Requested metrics require unplanned heavy artifacts`
: Use a profile such as `predictive`, `generative`, `surface`, or `full` when
  requesting metrics that require sampling or grid artifacts.

`missing capabilities` or unavailable metrics in `metrics.json`
: The model does not support the requested metric family. Try `core` or choose
  a model with the required capability.

`predictive-compare` requires `--horizon`
: Pass a positive prediction window duration, such as `--horizon 1.0`.

## Next Steps

- [Model Capability Matrix](../model-capability-matrix.md) — choose profiles by model family.
- [Evaluation And Visualization](../evaluation.md) — full CLI reference overview.
- [Paper Reproduction](../paper-reproduction.md) — reproducible artifact bundles.
