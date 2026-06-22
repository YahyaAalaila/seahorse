# Dataset Sources

Seahorse works with any dataset that follows the JSONL split contract. The
examples and Colab notebooks generate small synthetic datasets in that format so
users can run the complete training and benchmark workflow without downloading
external data.

## Expected Format

Any public Hugging Face dataset repository that exposes `train.jsonl`,
`val.jsonl`, and `test.jsonl` at the repository root or a named subdirectory is
usable with:

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision <revision> \
  --out runs/fit
```

Use pinned revisions for benchmark or paper runs so results remain reproducible.

## Preparing Your Own Data

If you have STPP data in a different format, see:

- [Add Your Dataset](add-dataset.md) — checklist for preparing and registering a dataset.
- [Conversion Standard](conversion.md) — how to convert from common formats (pandas DataFrame, NumPy, HDF5).
