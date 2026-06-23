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

## Validated Public Datasets

The datasets listed in the [Dataset Catalog](catalog.md) have been checked
against Seahorse's Hugging Face loader. They expose split files at the repository
root and use an accepted event-array JSONL layout that Seahorse canonicalizes
into `times` and `locations`.

Use them directly with `--dataset`:

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --dataset I5m41L/chicago_crime_stpp \
  --dataset-revision e6dc2c9edc427fac98b61ef181c750ae0b2bb818 \
  --out runs/chicago_crime_poisson
```

For benchmark runs, keep the revision pinned:

```bash
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm auto_stpp \
  --dataset I5m41L/gowalla_checkins_stpp \
  --dataset-revision 18b615fa840e6f92511c350522c762bcf351d0ec \
  --seeds 1 2 3 \
  --out runs/gowalla_benchmark
```

!!! warning "Project-controlled mirrors"
    The current validated repositories are hosted under `I5m41L`, not under the
    Seahorse project owner. Before a release or paper benchmark depends on them,
    either mirror the datasets to a project-controlled Hugging Face namespace or
    make the project owner an admin collaborator on the dataset repositories.
    The current cards also report `license:unknown`; update that metadata after
    confirming each upstream source's redistribution terms.

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

The Python API does not download Hugging Face datasets automatically. Download and cache the splits first, then load with `load_jsonl`:

```python
from seahorse import load_jsonl

# After downloading splits to a local directory:
train = load_jsonl("cache/my_dataset/train.jsonl")
val   = load_jsonl("cache/my_dataset/val.jsonl")
test  = load_jsonl("cache/my_dataset/test.jsonl")
```

## Hosting Your Own Dataset on Hugging Face

To make a dataset work with `--dataset`, the repository must:

1. Contain `train.jsonl`, `val.jsonl`, and `test.jsonl` at the repository root or a named subdirectory.
2. Use the Seahorse JSONL format: one JSON object per line, each with `times` and `locations` arrays of equal length. The loader also accepts records with an `events` array whose events expose `t`, `x`, and `y`.
3. Publish the source, license, and any preprocessing notes in the dataset card.

See [Conversion Standard](conversion.md) for format details and [Add Your Dataset](add-dataset.md) for the preparation checklist.

## Dataset Catalog

See [Dataset Catalog](catalog.md) for a list of known-working public datasets.
