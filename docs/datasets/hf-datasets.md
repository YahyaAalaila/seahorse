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

`load_dataset` resolves a HuggingFace repo id (or the short name of a released
`seahorse-stpp` dataset, at its pinned revision), downloads the splits, caches
them, and returns parsed sequences:

```python
from seahorse.data import load_dataset

splits = load_dataset("covid")
train, val, test = splits["train"], splits["val"], splits["test"]
```

For splits already on local disk, use `load_jsonl` directly:

```python
from seahorse import load_jsonl

train = load_jsonl("cache/my_dataset/train.jsonl")
```

## Released benchmark datasets (hash-verified)

The datasets of the Seahorse paper are published under the
[`seahorse-stpp`](https://huggingface.co/seahorse-stpp) organisation. `seahorse.data.hf`
pins every repository to its release commit and verifies each downloaded split against the
`split_sha256` block of the dataset card before parsing:

```python
from seahorse.data import list_released_datasets, load_released_dataset

list_released_datasets(in_paper=True)
splits = load_released_dataset("austin311")                      # default config: jittered_seed20260910
native = load_released_dataset("austin311", config="native")
synthetic = load_released_dataset("synthetic", config="gamma_renewal")
splits["__provenance__"]                                         # repo, revision, config, sha256, verified
```

| Short name | Repository | Configs | Licence (see card) |
| --- | --- | --- | --- |
| `comcat_california` | `comcat_california_stpp` | `native` | public domain (USGS) |
| `geonet_nz` | `geonet_nz_stpp` | `jittered_seed20260910`, `native` | CC BY 3.0 NZ |
| `austin311` | `austin_311_stpp` | `jittered_seed20260910`, `native` | public domain (City of Austin) |
| `nyc311` | `nyc311_stpp` | `jittered_seed20260910`, `native` | NYC Open Data terms |
| `nyc_collisions` | `nyc_collisions_stpp` | `jittered_seed20260910`, `native` | NYC Open Data terms |
| `chicago_crime` | `chicago_crime_stpp` | `jittered_seed20260910`, `native` | City of Chicago terms |
| `la_crime` | `la_crime_stpp` | `jittered_seed20260910`, `native` | CC0 1.0 |
| `uber_nyc` | `uber_pickups_nyc_stpp` | `jittered_seed20260910`, `native` | CC0 1.0 |
| `synthetic` | `seahorse_synthetic_stpp` | `homogeneous_poisson`, `moving_seasonal_poisson`, `gamma_renewal` | CC BY 4.0 |
| `covid` (legacy) | `covid-stpp` | `default` | CC BY-NC 4.0 |
| `earthquakes` (legacy) | `earthquakes-stpp` | `default` | public domain (USGS) |

**Script-only repositories.** `seattle_fire911` (`seattle_fire911_stpp`), `citibike_nyc`
(`nyc_bikeshare_2019_stpp`) and `citibike_legacy` (`nyc_bikeshare_legacy_stpp`) do not host data,
because the source licences or the content (address-level emergency calls) do not allow it. Each
repository ships `rebuild/rebuild.py`, `verify.py` and the expected split hashes; after rebuilding,
pass `local_root=<directory>` to `load_released_dataset`, which verifies the rebuilt splits against
the card:

```python
splits = load_released_dataset("seattle_fire911", config="native", local_root="data/seattle_fire911")
```

The legacy cards (`covid`, `earthquakes`) declare no split hashes; they load with `verified=False`
in `__provenance__` at the pinned revision.

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
