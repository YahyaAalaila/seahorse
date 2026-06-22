# Add Your Dataset

This page walks through preparing an event sequence dataset for use with Seahorse.

## Format Requirements

Each JSONL split file must have one JSON object per line. Each object represents one event sequence:

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

Rules:
- `times` — list of numeric event times, monotonically increasing within each sequence.
- `locations` — list of 2D coordinate pairs `[x, y]`, same length as `times`.
- Optional arrays (`marks`, `event_covariates`, `field_covariates`) must align with `times`.
- Each split is a separate file (`train.jsonl`, `val.jsonl`, `test.jsonl`).
- Do **not** use a single JSON array; each line must be a standalone JSON object.

## Preparation Checklist

- [ ] Times are numeric (float or int), not strings or datetimes.
- [ ] `times` within each sequence are monotonically non-decreasing.
- [ ] `locations` is a list of `[x, y]` pairs (2D only; 3D coordinates are not supported).
- [ ] `times` and `locations` have the same length in every sequence.
- [ ] Optional per-event arrays have the same length as `times`.
- [ ] `train.jsonl`, `val.jsonl`, and `test.jsonl` are kept in separate files.
- [ ] Coordinate ranges are consistent across all splits.
- [ ] Splits do not overlap (no sequence in both train and test).

## Split Ratios

There is no enforced ratio, but typical practice:
- Training: 70–80% of sequences.
- Validation: 10–15%.
- Test: 10–15%.

For short datasets, ensure the validation set has enough sequences for stable early stopping (at least 50–100 sequences is a reasonable target).

## Local Validation

Read the splits and print basic stats before fitting:

```python
from seahorse import load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
seq = train[0]

print(f"Sequences: {len(train)}")
print(f"Keys: {list(seq.keys())}")
print(f"Events in first seq: {len(seq['times'])}")
print(f"First 3 times: {seq['times'][:3]}")
print(f"First 3 locations: {seq['locations'][:3]}")
```

## Smoke Test

Run a one-epoch fit with a lightweight preset to verify the data loads correctly:

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --train data/my_dataset/train.jsonl \
  --val data/my_dataset/val.jsonl \
  --test data/my_dataset/test.jsonl \
  --out runs/smoke \
  --override training.n_epochs=1 training.batch_size=4 data.num_workers=0
```

A successful run writes `run_result.json` under the output directory.

## Publishing to HuggingFace

To make the dataset usable with `--dataset owner/repo`:

1. Create a HuggingFace dataset repository.
2. Upload `train.jsonl`, `val.jsonl`, and `test.jsonl` to the repository root (or a named subdirectory).
3. Tag a release or note a commit hash to use as `--dataset-revision`.

See [Ready-to-use HF Datasets](hf-datasets.md) for how Seahorse loads HuggingFace data.
