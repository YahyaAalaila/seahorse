# Data Format

The public data contract is JSONL split files. Each line is one event sequence.
The same format is used by the Python API and the CLI.

## Minimal Record

```json
{"times": [0.1, 0.4, 1.2], "locations": [[0.2, 0.4], [0.3, 0.8], [0.7, 0.1]]}
```

`times[i]` and `locations[i]` describe the same event. `locations` should be a
list of coordinate pairs. Optional per-event arrays such as `marks`,
`event_covariates`, and `field_covariates` must have the same length as
`times`.

## Local Data For One Run

Use this layout for Python API experiments, `fit`, `tune`, or one-dataset
`bench` runs:

```text
dataset_root/
  train.jsonl
  val.jsonl
  test.jsonl
```

`train.jsonl` and `val.jsonl` are required for fitting. `test.jsonl` is optional
for resolution but is normally needed for evaluation and examples.

Python code can load the files directly:

```python
from unified_stpp import load_jsonl

train = load_jsonl("dataset_root/train.jsonl")
val = load_jsonl("dataset_root/val.jsonl")
test = load_jsonl("dataset_root/test.jsonl")
```

CLI `fit` can use explicit paths:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train dataset_root/train.jsonl \
  --val dataset_root/val.jsonl \
  --test dataset_root/test.jsonl \
  --out runs/local_fit
```

## Local Data For Benchmarks

Use a split collection when benchmarking multiple datasets:

```text
splits_root/
  dataset_a/
    train.jsonl
    val.jsonl
    test.jsonl
  dataset_b/
    train.jsonl
    val.jsonl
    test.jsonl
```

Run all datasets found under the root:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --seeds 1 2 3 \
  --out runs/bench
```

Restrict a split collection to selected dataset names:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits_root \
  --datasets dataset_a dataset_b \
  --seeds 1 \
  --out runs/bench_subset
```

## Hugging Face Data

CLI workflows can resolve a Hugging Face dataset repository:

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --out runs/hf_fit
```

Use `--dataset-revision` to pin the source for reproducible runs. The resolved
dataset must expose `train.jsonl` and `val.jsonl`; `test.jsonl` is used when
present.

Benchmark campaigns can also use a Hugging Face source:

```bash
python -m unified_stpp bench \
  --preset poisson_gmm \
  --dataset owner/repo[/subdir] \
  --dataset-revision main \
  --seeds 1 \
  --out runs/bench_hf
```

## Command Support Matrix

| Command | Local split paths | Local dataset directory | Split collection | Hugging Face source |
| --- | --- | --- | --- | --- |
| Python API | yes, via `load_jsonl` | yes, by reading files | manual loop | not directly |
| `fit` | `--train --val --test` | `--dataset` | no | `--dataset` |
| `tune` | `--train --val` | `--dataset` | no | `--dataset` |
| `bench` | no | `--dataset` | `--splits_dir` | `--dataset` |
| `evaluate metrics` | `--data` | no | no | no |

## Practical Checks

- Keep train, validation, and test splits separate.
- Use the same coordinate system across all splits for a dataset.
- Use one JSON object per line, not a single JSON array file.
- Keep optional per-event arrays aligned with `times`.
- Pin Hugging Face revisions for benchmark or paper results.
