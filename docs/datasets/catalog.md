# Dataset Sources

Seahorse works with any dataset that follows the JSONL split contract. The
examples and Colab notebooks generate small synthetic datasets in that format so
users can run the complete training and benchmark workflow without downloading
external data.

The public Hugging Face datasets below have been checked against the Seahorse
loader. Each repository exposes `train.jsonl`, `val.jsonl`, and `test.jsonl` at
the repository root, and sample records validate through Seahorse's JSONL
canonicalization path.

!!! note "Ownership and license status"
    These datasets are currently hosted under the collaborator namespace
    `I5m41L`, and the Hugging Face cards currently report `license:unknown`.
    For long-term public benchmarks, mirror them under a project-controlled
    namespace or make the project owner an admin collaborator, then update the
    dataset cards with the upstream license/source terms.

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

## Validated Public Hugging Face Datasets

| Dataset | Domain | Hugging Face ref | Pinned revision | Train / Val / Test sequences | Status |
| --- | --- | --- | --- | --- | --- |
| Austin 311 | civic service requests | [`I5m41L/austin_311_stpp`](https://huggingface.co/datasets/I5m41L/austin_311_stpp) | `806f06a83dd821fda1702f1aa71b1fbeadbff728` | 17,167 / 3,701 / 3,695 | Loader-compatible external dataset |
| Uber pickups NYC | urban mobility | [`I5m41L/uber_pickups_nyc_stpp`](https://huggingface.co/datasets/I5m41L/uber_pickups_nyc_stpp) | `5de1bb62d750c7a225a736a3cb36674bb61e3f70` | 31,741 / 6,802 / 6,802 | Loader-compatible external dataset |
| US wildfires | natural hazards | [`I5m41L/us_wildfires_stpp`](https://huggingface.co/datasets/I5m41L/us_wildfires_stpp) | `82e84d9b2d2d0197664e26b3e77d96084c75f368` | 13,164 / 2,821 / 2,821 | Loader-compatible external dataset |
| Global Terrorism Database | conflict events | [`I5m41L/gtd_stpp`](https://huggingface.co/datasets/I5m41L/gtd_stpp) | `379d8d3de49fdecf8dee797fc3133245d6d8eeec` | 1,230 / 272 / 271 | Loader-compatible external dataset |
| US accidents | traffic incidents | [`I5m41L/us_accidents_stpp`](https://huggingface.co/datasets/I5m41L/us_accidents_stpp) | `20ed5757a52f0b1d67c0ee9cf428a98fa8b1c886` | 48,363 / 11,441 / 10,050 | Loader-compatible external dataset |
| LA crime | public safety incidents | [`I5m41L/la_crime_stpp`](https://huggingface.co/datasets/I5m41L/la_crime_stpp) | `1225c94f924839f2e6216d28e35fcd9ddccae370` | 7,035 / 1,508 / 1,508 | Loader-compatible external dataset |
| Brightkite check-ins | location check-ins | [`I5m41L/brightkite_checkins_stpp`](https://huggingface.co/datasets/I5m41L/brightkite_checkins_stpp) | `d2e9079a672fd8ed762f2816e2f73a4788b3f95c` | 33,231 / 7,121 / 7,121 | Loader-compatible external dataset |
| Chicago crime | public safety incidents | [`I5m41L/chicago_crime_stpp`](https://huggingface.co/datasets/I5m41L/chicago_crime_stpp) | `e6dc2c9edc427fac98b61ef181c750ae0b2bb818` | 59,407 / 12,684 / 12,634 | Loader-compatible external dataset |
| Gowalla check-ins | location check-ins | [`I5m41L/gowalla_checkins_stpp`](https://huggingface.co/datasets/I5m41L/gowalla_checkins_stpp) | `18b615fa840e6f92511c350522c762bcf351d0ec` | 45,101 / 9,665 / 9,665 | Loader-compatible external dataset |

Example:

```bash
python -m seahorse fit \
  --preset poisson_gmm \
  --dataset I5m41L/austin_311_stpp \
  --dataset-revision 806f06a83dd821fda1702f1aa71b1fbeadbff728 \
  --out runs/austin_311_poisson
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
