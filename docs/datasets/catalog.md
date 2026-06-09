# Dataset Catalog

This page will list publicly available STPP datasets that are formatted for Seahorse and available on the Hugging Face Hub.

!!! note "Work in progress"
    Final public dataset revisions and HuggingFace repository links will be added before publication.
    See [Paper Reproduction](../paper-reproduction.md) for the datasets used in the Seahorse paper.

## Expected Format

Each listed dataset will follow the [Seahorse JSONL format](../data-format.md) and be usable with:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --dataset <hf_repo_id> \
  --dataset-revision <revision> \
  --out runs/fit
```

## Preparing Your Own Data

If you have STPP data in a different format, see:

- [Add Your Dataset](add-dataset.md) — checklist for preparing and registering a dataset.
- [Conversion Standard](conversion.md) — how to convert from common formats (pandas DataFrame, NumPy, HDF5).
