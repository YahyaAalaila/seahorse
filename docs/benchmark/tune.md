# Tune Hyperparameters

Seahorse uses Ray Tune for HPO. Install the HPO extras before running any tune command:

```bash
pip install "seahorse-stpp[hpo]"
```

## CLI: tune a single preset

```bash
python -m seahorse tune \
  --preset poisson_gmm \
  --train data/my_dataset/train.jsonl \
  --val data/my_dataset/val.jsonl \
  --n_trials 20 \
  --search-alg random \
  --scheduler asha \
  --out runs/hpo/poisson_gmm_best.yaml
```

`tune` writes the best hyperparameter config to `--out` as a YAML file. Pass that file back to `fit` or `bench` with `--config` or `--hpo_configs_dir`.

### Key Options

| Option | Default | Notes |
| --- | --- | --- |
| `--preset` / `--config` | required | Config source to tune |
| `--dataset` / `--train --val` | required | Data source |
| `--n_trials` | 10 | Maximum HPO trials |
| `--search-alg` | `random` | `random` or `bayesian` |
| `--scheduler` | `asha` | `asha` or `none` |
| `--seed` | — | HPO seed |
| `--max-concurrent-trials` | 1 | Concurrency cap |
| `--out` | required | Best-config YAML output path |

## Python API: tune

The Python API exposes a thin HPO wrapper that uses the same Ray Tune path:

```python
from seahorse import AutoSTPP, load_jsonl

train = load_jsonl("data/my_dataset/train.jsonl")
val   = load_jsonl("data/my_dataset/val.jsonl")

model = AutoSTPP(device="cpu")
best_config = model.tune(train, val, n_trials=10, max_epochs=20)
print(best_config)
```

`tune()` returns the best config dictionary. The fitted model still needs a subsequent `fit()` call with the best config applied.

## HPO Inside a Benchmark

Run HPO before a benchmark campaign using a designated tuning dataset:

```bash
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --tune \
  --tune-dataset dataset_a \
  --n_trials 20 \
  --seeds 1 2 3 \
  --out runs/bench_hpo
```

Or re-use previously tuned configs so you do not re-run HPO for every benchmark:

```bash
python -m seahorse bench \
  --presets poisson_gmm hawkes_gmm \
  --splits_dir splits \
  --hpo_configs_dir runs/hpo \
  --seeds 1 \
  --out runs/bench_reuse_hpo
```

`--hpo_configs_dir` expects one `{preset}_best.yaml` per preset inside the directory.

## Tips

- Start with `random` search and `asha` scheduler; they work well for most presets.
- Use a small `--n_trials` first to verify the HPO loop runs before committing to a full search.
- Tune on a representative dataset, not the benchmark evaluation datasets.
- Fix seeds in `--seeds` for downstream benchmark runs so HPO and evaluation are reproducible.
