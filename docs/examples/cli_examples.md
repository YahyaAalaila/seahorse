# Public CLI Examples

These examples use the stable module CLI:

```bash
python -m unified_stpp fit
python -m unified_stpp tune
python -m unified_stpp bench
python -m unified_stpp evaluate
```

The commands below use the tiny JSONL toy data in `examples/tiny_jsonl/`. They are intended to show the public command shape, not to produce meaningful scientific results.

Normal public usage should call the module CLI directly. Old root-level wrappers such as `train.py` and temporary visualization scripts are not part of the v1 public surface.

## Fit

```bash
python -m unified_stpp fit \
  --preset poisson_gmm \
  --train examples/tiny_jsonl/train.jsonl \
  --val examples/tiny_jsonl/val.jsonl \
  --test examples/tiny_jsonl/test.jsonl \
  --out runs/examples/tiny_fit \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

## Tune

```bash
python -m unified_stpp tune \
  --preset poisson_gmm \
  --train examples/tiny_jsonl/train.jsonl \
  --val examples/tiny_jsonl/val.jsonl \
  --n_trials 1 \
  --out runs/examples/tiny_tune/poisson_gmm_best.yaml
```

## Bench

```bash
python -m unified_stpp bench \
  --presets poisson_gmm \
  --dataset examples/tiny_jsonl \
  --seeds 1 \
  --out runs/examples/tiny_bench \
  --n_workers 1 \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0
```

## Evaluate

Use the run directory produced by `fit` or `bench`:

```bash
python -m unified_stpp evaluate metrics \
  --run runs/examples/tiny_fit/fit/poisson_gmm/<run_id> \
  --data examples/tiny_jsonl/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/examples/tiny_eval
```

Replace `<run_id>` with the timestamped run directory created by the fit command.
