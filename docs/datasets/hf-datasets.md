# Ready-to-use HF Datasets

Seahorse can load datasets directly from the Hugging Face Hub when they follow the JSONL split convention. No manual download or conversion is needed.

## Using a Hugging Face Dataset

Pass a repository identifier with optional subdirectory:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --out runs/hf_fit
```

The resolved repository must expose `train.jsonl` and `val.jsonl`. `test.jsonl` is used when present.

!!! tip "Pin the revision for reproducibility"
    Always pass `--dataset-revision` with a tag or commit hash when running benchmark
    or paper-reproduction commands. `main` moves with new commits and will break reproducibility.

## In a Benchmark

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm auto_stpp \
  --dataset owner/repo[/subdir] \
  --dataset-revision <revision> \
  --seeds 1 2 3 \
  --out runs/bench_hf
```

## Python API

The Python API does not download HuggingFace datasets automatically. Download and cache the splits first, then load with `load_jsonl`:

```python
from unified_stpp import load_jsonl

# After downloading splits to a local directory:
train = load_jsonl("cache/my_dataset/train.jsonl")
val   = load_jsonl("cache/my_dataset/val.jsonl")
test  = load_jsonl("cache/my_dataset/test.jsonl")
```

## Hosting Your Own Dataset on HuggingFace

To make a dataset work with `--dataset`, the repository must:

1. Contain `train.jsonl`, `val.jsonl`, and `test.jsonl` at the repository root or a named subdirectory.
2. Use the Seahorse JSONL format: one JSON object per line, each with `times` and `locations` arrays of equal length.

See [Conversion Standard](conversion.md) for format details and [Add Your Dataset](add-dataset.md) for the preparation checklist.

## Dataset Catalog

See [Dataset Catalog](catalog.md) for a list of known-working public datasets.
