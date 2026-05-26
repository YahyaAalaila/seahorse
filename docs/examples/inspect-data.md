# Inspect Data

Use this example before fitting or benchmarking. Seahorse expects JSONL split
files where each line is one event sequence.

## Placeholder Notebook

[Colab badge placeholder]

A Colab notebook can be linked here when a maintained notebook exists.

## Local Split Layout

For a single run, keep the data in one directory:

```text
data/my_dataset/
  train.jsonl
  val.jsonl
  test.jsonl
```

For benchmark campaigns, use one directory per dataset:

```text
splits/
  dataset_a/
    train.jsonl
    val.jsonl
    test.jsonl
  dataset_b/
    train.jsonl
    val.jsonl
    test.jsonl
```

## Read A Split In Python

```python
from unified_stpp import load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
first = train[0]

print(first.keys())
print(len(first["times"]), len(first["locations"]))
print(first["times"][:3])
print(first["locations"][:3])
```

Every record needs matching `times` and `locations` arrays. Optional arrays such
as `marks`, `event_covariates`, and `field_covariates` must align with
`times`.

## Quick Validation Checklist

- `train.jsonl` and `val.jsonl` exist.
- `test.jsonl` exists when you plan to evaluate held-out performance.
- Each line is valid JSON.
- `times` are numeric event times.
- `locations` is a list of coordinate pairs.
- Every per-event array has the same length as `times`.

## Hugging Face Sources

CLI workflows can resolve a dataset repository directly:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --out runs/hf_fit
```

Use a pinned `--dataset-revision` for reproducible benchmark or paper runs.
