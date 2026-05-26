# Evaluation And Visualization

Use `evaluate` after `fit` or `bench` has produced a saved run directory. This
is the supported path for benchmark-aligned metrics and diagnostic artifacts.

For one fitted model in Python, use `model.evaluate(test)`. Use this CLI page
when you need artifact-backed evaluation profiles, benchmark-aligned outputs,
or visual diagnostics.

## Metric Profiles

The installed evaluation registry defines these profile families:

| Profile | Purpose | Heavy artifacts |
| --- | --- | --- |
| `core` | Basic NLL/report metrics | none |
| `nll` | Extended NLL-family checks | none |
| `predictive` | Next-event predictive scores | predictive samples |
| `generative` | Full-rollout distribution metrics | generative rollouts |
| `autoregressive` | Fixed-prefix autoregressive degradation | generative rollouts |
| `surface` | Intensity-grid diagnostics | intensity grids or rollout approximations |
| `full` | All registered benchmark metrics | all planned heavy artifact families |

Run `python -m unified_stpp evaluate metrics --help` for the exact metric names
and controls in the installed version.

## Core Metric Report

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core_test
```

Use `--metric` when you want explicit metric names:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric nll \
  --out runs/evaluate/nll_test
```

Useful controls:

- `--max-seqs`: cap the number of sequences.
- `--max-events`: cap events per sequence.
- `--k-pred`: next-event sample count.
- `--k-gen`: full-rollout sample count.
- `--n-context-events`: observed prefix length for rollout metrics.
- `--device`: `auto`, `cpu`, `cuda`, `cuda:0`, `mps`, or another supported device string.
- `--artifact-dir`: root directory for persisted evaluation artifacts.

## Output Artifacts

Evaluation outputs depend on the chosen profile and model capabilities. Common
outputs include:

- metric summary files under the `--out` directory.
- predictive sample artifacts for `predictive` profiles.
- generative rollout artifacts for `generative` and `autoregressive` profiles.
- intensity-grid artifacts for `surface` profiles.
- rendered HTML or image files for visualization commands.

Heavy artifacts are intentionally profile-gated so expensive sampling or grid
work is explicit.

## Sharded Metric Evaluation

Evaluate sequence ranges with `--seq-shard`:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile predictive \
  --seq-shard 0:50 \
  --out runs/evaluate/shard_0
```

Merge predictive sample artifacts after running shards:

```bash
python -m unified_stpp evaluate merge-artifacts \
  --artifact-dir runs/evaluate/shard_0/artifacts \
  --artifact-dir runs/evaluate/shard_1/artifacts \
  --out runs/evaluate/merged_artifacts
```

Repeat `--artifact-dir` in shard order.

## Predictive Comparison

Compare sampled future windows across one or more saved runs:

```bash
python -m unified_stpp evaluate predictive-compare \
  --run path/to/run_a \
  --run path/to/run_b \
  --label model_a \
  --label model_b \
  --history data/my_dataset/test.jsonl \
  --split test \
  --horizon 1.0 \
  --out runs/evaluate/predictive_compare
```

`predictive-compare` is a qualitative visualization workflow. Use
`evaluate metrics --metric-profile predictive` for benchmark-aligned predictive
metric artifacts.

## Surface Diagnostics

Render a surface diagnostic for one saved run:

```bash
python -m unified_stpp evaluate surface \
  --run path/to/run_dir \
  --history data/my_dataset/test.jsonl \
  --split test \
  --seq-idx 0 \
  --profile history_frame \
  --out runs/evaluate/surface
```

The `surface` command supports profiles such as `history_frame` and
`future_exact`. Run `python -m unified_stpp evaluate surface --help` for the
exact options in your installed version.

## Python Visualization Helpers

For one fitted estimator, the Python API exposes lightweight helpers:

```python
surface = model.plot_intensity(test[0], output_path="runs/plots/intensity")
kde = model.plot_kde_surface(test[0], n_samples=128, output_path="runs/plots/kde")
```

Use CLI visualization commands when the output needs to line up with benchmark
artifacts or paper reproduction.
