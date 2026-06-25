# Ready-to-use HF Datasets

Seahorse can load datasets directly from the Hugging Face Hub when they follow the JSONL split convention. No manual download or conversion is needed.

## Using a Hugging Face Dataset

Pass a repository identifier with optional subdirectory:

```bash
python -m seahorse fit \
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
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm auto_stpp \
  --dataset owner/repo[/subdir] \
  --dataset-revision <revision> \
  --seeds 1 2 3 \
  --out runs/bench_hf
```

## Python API

`load_dataset` resolves a HuggingFace repo id (or a curated `seahorse-stpp`
name), downloads the splits, caches them, and returns parsed sequences:

```python
from seahorse.data import load_dataset

splits = load_dataset("seahorse-stpp/citibike-stpp")
train, val, test = splits["train"], splits["val"], splits["test"]
```

For splits already on local disk, use `load_jsonl` directly:

```python
from seahorse import load_jsonl

train = load_jsonl("cache/my_dataset/train.jsonl")
```

## Hosting Your Own Dataset on HuggingFace

To make a dataset work with `--dataset`, the repository must:

1. Contain `train.jsonl`, `val.jsonl`, and `test.jsonl` at the repository root or a named subdirectory.
2. Use the Seahorse JSONL format: one JSON object per line, each with `times` and `locations` arrays of equal length.

See [Conversion Standard](conversion.md) for format details and [Add Your Dataset](add-dataset.md) for the preparation checklist.

## Dataset Catalog

Seahorse curates **13 ready-to-use datasets** in the
[`seahorse-stpp`](https://huggingface.co/seahorse-stpp) organization, spanning
urban mobility, crime, natural hazards, public health, social check-ins, and
neuroimaging. Browse them — with load snippets and per-dataset space/time axes —
in the **[Dataset Catalog](catalog.md)**.
