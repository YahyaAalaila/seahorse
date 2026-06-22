# Evaluation And Visualization

!!! tip "Quick Python evaluation"
    For a single fitted model, use `model.evaluate(test)` — no run directory needed.
    This page covers the CLI path for artifact-backed metric profiles and visual diagnostics.

## Metric Profiles

| Profile | Purpose | Heavy artifacts |
| --- | --- | --- |
| `core` | Basic NLL/report metrics | none |
| `nll` | Extended NLL-family checks | none |
| `predictive` | Next-event predictive scores | predictive samples |
| `generative` | Full-rollout distribution metrics | generative rollouts |
| `autoregressive` | Fixed-prefix autoregressive degradation | generative rollouts |
| `surface` | Intensity-grid diagnostics | intensity grids or approximations |
| `full` | All registered benchmark metrics | all heavy artifact families |

Run `python -m seahorse evaluate metrics --help` for the exact metric names
in the installed version.

## Core Metric Report

```bash
python -m seahorse evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core_test
```

??? example "Show CLI command — explicit metric names"
    ```bash
    python -m seahorse evaluate metrics \
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
- `--device`: `auto`, `cpu`, `cuda`, `cuda:0`, `mps`, or another device string.
- `--artifact-dir`: root directory for persisted evaluation artifacts.

## Output Artifacts

Outputs depend on the chosen profile and model capabilities:

- metric summary files under the `--out` directory.
- predictive sample artifacts for `predictive` profiles.
- generative rollout artifacts for `generative` and `autoregressive` profiles.
- intensity-grid artifacts for `surface` profiles.
- rendered HTML or image files for visualization commands.

Heavy artifacts are profile-gated so expensive sampling or grid work is explicit.

## Sharded Metric Evaluation

Evaluate large test sets in sequence ranges:

```bash
python -m seahorse evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile predictive \
  --seq-shard 0:50 \
  --out runs/evaluate/shard_0
```

??? example "Show CLI command — merge shard artifacts"
    ```bash
    python -m seahorse evaluate merge-artifacts \
      --artifact-dir runs/evaluate/shard_0/artifacts \
      --artifact-dir runs/evaluate/shard_1/artifacts \
      --out runs/evaluate/merged_artifacts
    ```

    Repeat `--artifact-dir` in shard order.

## Predictive Comparison

```bash
python -m seahorse evaluate predictive-compare \
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
`evaluate metrics --metric-profile predictive` for benchmark-aligned artifacts.

## Surface Diagnostics

??? example "Show CLI command"
    ```bash
    python -m seahorse evaluate surface \
      --run path/to/run_dir \
      --history data/my_dataset/test.jsonl \
      --split test \
      --seq-idx 0 \
      --profile history_frame \
      --out runs/evaluate/surface
    ```

    The `surface` command supports `history_frame` (`auto_stpp`, `deep_stpp`) and
    `future_exact` (neural exact families). Run
    `python -m seahorse evaluate surface --help` for the full option list.

## Python Visualization Helpers

```python
surface = model.plot_intensity(test[0], output_path="runs/plots/intensity")
kde = model.plot_kde_surface(test[0], n_samples=128, output_path="runs/plots/kde")
```

Use CLI visualization commands when outputs need to align with benchmark artifacts.
