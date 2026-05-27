# Predict / Sample Events

Seahorse exposes next-event predictive sampling through `predict_next` in the Python API and through `evaluate metrics --metric-profile predictive` and `evaluate predictive-compare` in the CLI.

## predict_next (Python API)

```python
from unified_stpp import AutoSTPP, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val   = load_jsonl("data/my_dataset/val.jsonl")
test  = load_jsonl("data/my_dataset/test.jsonl")

model = AutoSTPP(device="cpu", seed=42)
model.fit(train, val, test, epochs=10, batch_size=64)

samples = model.predict_next(test, n_samples=32)
```

The returned dictionary contains:

| Key | Description |
| --- | --- |
| `next_times` | Sampled next-event times |
| `next_locations` | Sampled next-event locations |
| `true_next_times` | Ground-truth next-event times |
| `true_next_locations` | Ground-truth next-event locations |
| `sequence_index` | Which test sequence each row belongs to |
| `target_event_index` | Which event within the sequence |
| `sampling_succeeded` | Boolean mask for successful samples |
| `sampling_backend` | Which sampling path was used |

!!! note "Capability requirement"
    `predict_next` raises `NotImplementedError` when the fitted model does not
    support next-event sampling. Check the [Model Capability Matrix](../model-capability-matrix.md)
    before calling this method on a new preset.

## Predictive Metrics (CLI)

For benchmark-aligned predictive evaluation, use the CLI after saving a run:

```bash
python -m unified_stpp evaluate metrics \
  --run path/to/run_dir \
  --data data/my_dataset/test.jsonl \
  --split test \
  --metric-profile predictive \
  --k-pred 32 \
  --out runs/evaluate/predictive_test
```

This computes temporal CRPS, spatial energy score, MAE, RMSE, and coverage using `n_samples` sampled next-event predictions. Results land in `metrics.json` under the output directory.

## Predictive Comparison (CLI)

`predictive-compare` is a qualitative visualization workflow that overlays predictions from one or two models against observed events on a single sequence:

```bash
python -m unified_stpp evaluate predictive-compare \
  --run path/to/run_a \
  --run path/to/run_b \
  --label model_a \
  --label model_b \
  --history data/my_dataset/test.jsonl \
  --split test \
  --seq-idx 0 \
  --horizon 1.0 \
  --out runs/evaluate/predictive_compare
```

Key options:

- `--run`: one or two run directories; repeat the flag for a two-model comparison.
- `--label`: display name matching each `--run` in order.
- `--seq-idx`: which test sequence to visualize.
- `--horizon`: prediction window duration (required; e.g. `1.0`).

!!! note "Qualitative only"
    `predictive-compare` is for visual inspection. Use `evaluate metrics --metric-profile predictive`
    for benchmark-aligned quantitative scores.
