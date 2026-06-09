# Evaluate a Model

Seahorse has two evaluation paths: a lightweight Python method for quick likelihood scores, and the CLI `evaluate` command for full benchmark-aligned metric profiles backed by saved artifacts.

## Quick Python Evaluation

After fitting, `evaluate()` returns likelihood metrics without requiring a run directory:

```python
from unified_stpp import AutoSTPP, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val   = load_jsonl("data/my_dataset/val.jsonl")
test  = load_jsonl("data/my_dataset/test.jsonl")

model = AutoSTPP(device="cpu", seed=42)
model.fit(train, val, test, epochs=10, batch_size=64)

scores = model.evaluate(test)
print(scores)
# {"test_nll": ..., "mean_seq_nll": ...}
```

The default `core` profile returns `test_nll` and `mean_seq_nll`. You can request specific metrics explicitly:

```python
scores = model.evaluate(test, metrics=["test_nll"])
```

!!! note "Python evaluation scope"
    `model.evaluate()` covers implemented likelihood metrics only. For predictive,
    generative, or surface metrics, use the CLI path below.

## CLI Evaluation (Full Profiles)

The CLI path requires a saved run directory produced by `fit`, `tune`, or `bench`.

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/evaluate/core_test
```

### Metric Profiles

| Profile | What it computes | Heavy work |
| --- | --- | --- |
| `core` | Per-event NLL and report metrics | none |
| `nll` | Extended NLL-family checks | none |
| `predictive` | Next-event CRPS, MAE, RMSE, coverage | predictive sampling |
| `generative` | Full-rollout distribution metrics | generative rollouts |
| `autoregressive` | Fixed-prefix autoregressive degradation | generative rollouts |
| `surface` | Intensity/density grid diagnostics | intensity grids |
| `full` | All registered benchmark metrics | all of the above |

Heavy artifact families are explicit — a `core` run never pays for sampling.

### Useful Flags

- `--max-seqs N`: evaluate only the first N test sequences (quick inspection).
- `--max-events N`: cap events per sequence.
- `--k-pred N`: next-event sample count for predictive metrics.
- `--k-gen N`: full-rollout sample count.
- `--device auto|cpu|cuda|mps`: override the compute device.
- `--seq-shard 0:50`: evaluate a slice of the test set (for parallelism).

### Output Layout

```text
runs/evaluate/core_test/
  metrics.json               ← per-metric results with availability and scalar values
  evaluation_manifest.json   ← run metadata
  *_per_event.npy            ← per-event arrays for offline analysis
```

`metrics.json` records each metric's `available` flag, scalar value, method, and a human-readable reason when the metric was skipped. An unavailable metric is not a failed run — it means the model does not expose the required capability.

## Which Profile To Use

See the [Model Capability Matrix](../model-capability-matrix.md) to match presets to profiles.

| Goal | Profile |
| --- | --- |
| Check that a run trained correctly | `core` |
| Compare NLL across presets | `core` or `nll` |
| Evaluate next-event prediction quality | `predictive` |
| Visual diagnostic on one sequence | `surface` |
| Full paper table | `full` or explicit metrics |
