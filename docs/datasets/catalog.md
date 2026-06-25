# Dataset Sources

Seahorse works with any dataset that follows the JSONL split contract. The
examples and Colab notebooks generate small synthetic datasets in that format so
users can run the complete training and benchmark workflow without downloading
external data.

The public Hugging Face datasets below are hosted under the project-controlled
[`seahorse-stpp`](https://huggingface.co/seahorse-stpp) organization. Each
repository exposes `train.jsonl`, `val.jsonl`, and `test.jsonl` at the repository
root, and sample records validate through Seahorse's JSONL canonicalization
path.

Seahorse registers these datasets as built-in aliases. For example,
`--dataset chicago_crime_stpp` resolves to
`seahorse-stpp/chicago_crime_stpp` at the validated pinned revision.

!!! note "Dataset card status"
    The repositories are project-controlled, but some source/license metadata is
    still being normalized after transfer. Check each Hugging Face dataset card
    before citing or redistributing a benchmark run.

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

## Built-in Seahorse Hugging Face Datasets

| Dataset | Domain | Dataset alias | Hugging Face repo | Pinned revision | Train / Val / Test |
| --- | --- | --- | --- | --- | --- |
| Austin 311 | civic service requests | `austin_311_stpp` | [`seahorse-stpp/austin_311_stpp`](https://huggingface.co/datasets/seahorse-stpp/austin_311_stpp) | `806f06a83dd821fda1702f1aa71b1fbeadbff728` | 17,167 / 3,701 / 3,695 |
| Uber pickups NYC | urban mobility | `uber_pickups_nyc_stpp` | [`seahorse-stpp/uber_pickups_nyc_stpp`](https://huggingface.co/datasets/seahorse-stpp/uber_pickups_nyc_stpp) | `5de1bb62d750c7a225a736a3cb36674bb61e3f70` | 31,741 / 6,802 / 6,802 |
| US wildfires | natural hazards | `us_wildfires_stpp` | [`seahorse-stpp/us_wildfires_stpp`](https://huggingface.co/datasets/seahorse-stpp/us_wildfires_stpp) | `82e84d9b2d2d0197664e26b3e77d96084c75f368` | 13,164 / 2,821 / 2,821 |
| Global Terrorism Database | conflict events | `gtd_stpp` | [`seahorse-stpp/gtd_stpp`](https://huggingface.co/datasets/seahorse-stpp/gtd_stpp) | `379d8d3de49fdecf8dee797fc3133245d6d8eeec` | 1,230 / 272 / 271 |
| US accidents | traffic incidents | `us_accidents_stpp` | [`seahorse-stpp/us_accidents_stpp`](https://huggingface.co/datasets/seahorse-stpp/us_accidents_stpp) | `20ed5757a52f0b1d67c0ee9cf428a98fa8b1c886` | 48,363 / 11,441 / 10,050 |
| LA crime | public safety incidents | `la_crime_stpp` | [`seahorse-stpp/la_crime_stpp`](https://huggingface.co/datasets/seahorse-stpp/la_crime_stpp) | `1225c94f924839f2e6216d28e35fcd9ddccae370` | 7,035 / 1,508 / 1,508 |
| Brightkite check-ins | location check-ins | `brightkite_checkins_stpp` | [`seahorse-stpp/brightkite_checkins_stpp`](https://huggingface.co/datasets/seahorse-stpp/brightkite_checkins_stpp) | `d2e9079a672fd8ed762f2816e2f73a4788b3f95c` | 33,231 / 7,121 / 7,121 |
| Chicago crime | public safety incidents | `chicago_crime_stpp` | [`seahorse-stpp/chicago_crime_stpp`](https://huggingface.co/datasets/seahorse-stpp/chicago_crime_stpp) | `e6dc2c9edc427fac98b61ef181c750ae0b2bb818` | 59,407 / 12,684 / 12,634 |
| Gowalla check-ins | location check-ins | `gowalla_checkins_stpp` | [`seahorse-stpp/gowalla_checkins_stpp`](https://huggingface.co/datasets/seahorse-stpp/gowalla_checkins_stpp) | `18b615fa840e6f92511c350522c762bcf351d0ec` | 45,101 / 9,665 / 9,665 |
| Earthquakes | seismic events | `earthquakes-stpp` | [`seahorse-stpp/earthquakes-stpp`](https://huggingface.co/datasets/seahorse-stpp/earthquakes-stpp) | `b8a95e944c696bfe15a9684cc8532d6f0598d648` | 950 / 50 / 50 |
| Citibike | urban mobility | `citibike-stpp` | [`seahorse-stpp/citibike-stpp`](https://huggingface.co/datasets/seahorse-stpp/citibike-stpp) | `3e058400fd46eb0995e39d3acf038069d05d2122` | 2,440 / 300 / 320 |
| COVID-19 | public health events | `covid-stpp` | [`seahorse-stpp/covid-stpp`](https://huggingface.co/datasets/seahorse-stpp/covid-stpp) | `2c235058853d4920a92c42dafb49e1188ff08f86` | 1,450 / 100 / 100 |
| BOLD5000 | neural response embeddings | `bold5000-stpp` | [`seahorse-stpp/bold5000-stpp`](https://huggingface.co/datasets/seahorse-stpp/bold5000-stpp) | `928e9cbc6df930e7d895844d64d3f273eed05b69` | 70 / 10 / 20 |

Example:

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --dataset austin_311_stpp \
  --out runs/austin_311_poisson
```

Passing the alias uses the pinned revision in the table. To opt into the latest
remote commit instead, pass the full repository id with an explicit revision:

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --dataset seahorse-stpp/austin_311_stpp \
  --dataset-revision main \
  --out runs/austin_311_latest
```

The checked records use the accepted event-array layout:

```json
{"sequence_id": 0, "events": [{"t": 0.0, "x": -97.77, "y": 30.34}]}
```

At load time Seahorse canonicalizes that form to `times` and `locations`.

## Preparing Your Own Data

If you have STPP data in a different format, see:

- [Add Your Dataset](add-dataset.md) — checklist for preparing and registering a dataset.
- [Conversion Standard](conversion.md) — how to convert from common formats (pandas DataFrame, NumPy, HDF5).
