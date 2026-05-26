# Evaluate Results

## Goal

This example shows how to inspect a completed Seahorse run, compute additional
metric reports, and create basic visualization artifacts with the CLI.

Use this after running a single `fit` command or after a benchmark such as
[Run A Small Benchmark](run-a-small-benchmark.md).

## Input: A Fitted Run Or Benchmark Directory

Evaluation commands need one saved run directory. A benchmark directory is a
collection of saved runs, so first identify the run you want to inspect.

From the small benchmark example, start here:

```text
runs/examples/small_benchmark/
```

Open `runs/examples/small_benchmark/cell_index.json` and find the run directory
for the preset, dataset, and seed you want to evaluate. Use that path anywhere
this page shows:

```text
path/to/run_dir
```

The test split used below is:

```text
data/my_dataset/test.jsonl
```

## Run Evaluation From CLI

Run the core metric profile:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/examples/small_benchmark/evaluate/core
```

Run predictive metrics when the model supports predictive sampling:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile predictive \
  --k-pred 32 \
  --out runs/examples/small_benchmark/evaluate/predictive
```

Use a smaller subset for quick inspection:

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

Metric evaluation writes files under the `--out` directory. Common files include:

```text
runs/examples/small_benchmark/evaluate/core/
  metrics.json
  evaluation_manifest.json
  *_per_event.npy
```

Predictive profiles can also write sampling artifacts under the evaluation
artifact directory. The exact artifact set depends on the metric profile and
model capabilities.

`metrics.json` is the first file to inspect. It records each metric result,
whether the metric was available, the scalar value when one exists, the method,
and any reason a metric was unavailable.

## Plot Or Visualize Results If Supported

Create a qualitative predictive comparison for one or more saved runs:

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

Compare two runs by repeating `--run` and `--label`:

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

For supported surface diagnostics, run:

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
the neural exact families listed by `python -m unified_stpp evaluate surface
--help`.

## Interpret Common Outputs

`nll`, `temporal_nll`, `spatial_nll`

: Lower values are better when comparing runs evaluated on the same dataset,
  split, and metric definition. Be careful comparing exact and approximate NLL
  families.

Predictive metrics

: Metrics such as temporal CRPS, spatial energy score, MAE, RMSE, and coverage
  use sampled next-event predictions. Availability depends on the model's
  sampling capability.

Surface diagnostics

: Surface outputs are qualitative diagnostics over an intensity or density grid.
  They are useful for inspecting where a model places event mass over space and
  time, but support is model-family specific.

Unavailable metrics

: An unavailable metric in `metrics.json` is not automatically a failed run. It
  usually means the requested profile needs a capability the model does not
  expose.

## Common Errors

`Path ... is not a saved run directory`

: Use a per-model run directory, not the top-level benchmark directory. For
  benchmarks, look up the run directory in `cell_index.json`.

`FileNotFoundError` for `test.jsonl`

: Check the `--data` or `--history` path. This example assumes
  `data/my_dataset/test.jsonl`.

`Requested metrics require unplanned heavy artifacts`

: Use a metric profile such as `predictive`, `generative`, `surface`, or `full`
  when requesting metrics that require sampling or grid artifacts.

`missing capabilities` or unavailable metrics in `metrics.json`

: The model does not support the requested metric family. Try `core` metrics or
  choose a model with the required sampling or surface capability.

`surface --profile history_frame currently supports only auto_stpp and deep_stpp`

: Use an `auto_stpp` or `deep_stpp` run for `history_frame`, or choose the
  supported profile for the model family.

`predictive-compare` requires `--horizon`

: Pass a positive prediction window duration, such as `--horizon 1.0`.

## Next Steps

- Use [Model Capability Matrix](../model-capability-matrix.md) to choose metric
  profiles that match each model family.
- Use [Evaluation And Visualization](../evaluation.md) for the full CLI
  reference-style overview.
- Use [Paper Reproduction](../paper-reproduction.md) when these outputs need to
  become part of a reproducible artifact bundle.
