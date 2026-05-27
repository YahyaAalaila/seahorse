# Testing Checklist

Run these tests before adding a new preset to benchmark examples or documentation. Keep the first tests narrow — a slow or brittle test suite is worse than a simple one.

## Minimum Required Tests

### 1. Preset registration

```python
from unified_stpp import list_available_models
from unified_stpp import STPPEstimator

def test_preset_registered():
    assert "my_preset" in list_available_models()

def test_preset_constructs():
    model = STPPEstimator("my_preset", device="cpu")
    assert model is not None
```

### 2. Config loading

```python
from unified_stpp import STPPConfig

def test_config_from_preset():
    cfg = STPPConfig.from_preset("my_preset")
    assert cfg.model.preset == "my_preset"

def test_config_roundtrip(tmp_path):
    cfg = STPPConfig.from_preset("my_preset")
    yaml_path = tmp_path / "config.yaml"
    cfg.to_yaml(str(yaml_path))
    cfg2 = STPPConfig.from_yaml(str(yaml_path))
    assert cfg2.model.preset == cfg.model.preset
```

### 3. One-epoch fit on tiny data

```python
from unified_stpp import STPPEstimator, load_jsonl

def test_fit_tiny(tiny_train, tiny_val, tiny_test):
    model = STPPEstimator("my_preset", device="cpu")
    model.fit(
        tiny_train, tiny_val, tiny_test,
        epochs=1, batch_size=2,
    )
    scores = model.evaluate(tiny_test)
    assert "test_nll" in scores
    assert scores["test_nll"] < 0 or scores["test_nll"] > 0  # not NaN
```

Use tiny data (2–5 sequences, 3–5 events each) so the test finishes in seconds.

### 4. Save and reload

```python
def test_save_load(tmp_path, tiny_train, tiny_val, tiny_test):
    model = STPPEstimator("my_preset", device="cpu")
    model.fit(tiny_train, tiny_val, tiny_test, epochs=1, batch_size=2)

    save_dir = model.save(str(tmp_path / "saved"))
    loaded = STPPEstimator.load(save_dir)

    scores_orig   = model.evaluate(tiny_test)
    scores_loaded = loaded.evaluate(tiny_test)
    assert abs(scores_orig["test_nll"] - scores_loaded["test_nll"]) < 1e-4
```

### 5. CLI core evaluation

```bash
python -m unified_stpp fit \
  --preset my_preset \
  --train data/tiny/train.jsonl \
  --val data/tiny/val.jsonl \
  --test data/tiny/test.jsonl \
  --out runs/test_my_preset \
  --override training.n_epochs=1 training.batch_size=2 data.num_workers=0

python -m unified_stpp evaluate metrics \
  --run runs/test_my_preset/fit/my_preset/<run_id> \
  --data data/tiny/test.jsonl \
  --split test \
  --metric-profile core \
  --out runs/test_my_preset/eval_core
```

Check that `metrics.json` is written and `test_nll` is `available: true`.

## Optional Tests (by Claimed Capability)

### Sampling

```python
def test_predict_next(tiny_train, tiny_val, tiny_test):
    model = STPPEstimator("my_preset", device="cpu")
    model.fit(tiny_train, tiny_val, tiny_test, epochs=1, batch_size=2)

    samples = model.predict_next(tiny_test[:1], n_samples=4)
    assert "next_times" in samples
    assert "next_locations" in samples
```

### Surface diagnostics

```bash
python -m unified_stpp evaluate surface \
  --run runs/test_my_preset/fit/my_preset/<run_id> \
  --history data/tiny/test.jsonl \
  --split test \
  --seq-idx 0 \
  --profile history_frame \
  --out runs/test_my_preset/surface
```

## What NOT to Test Here

- Training convergence on real data — that belongs in an experiment script, not in the test suite.
- Numerical equivalence with a reference implementation — add a separate regression test if needed.
- Speed or memory — benchmark these separately before claiming production-readiness.

## Conftest Helpers

Add a `conftest.py` that generates tiny synthetic sequences:

```python
import json
import pytest

@pytest.fixture
def tiny_train(tmp_path):
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for _ in range(5):
            seq = {
                "times": [0.1, 0.3, 0.6, 0.9],
                "locations": [[0.1, 0.2], [0.4, 0.5], [0.6, 0.3], [0.8, 0.7]],
            }
            f.write(json.dumps(seq) + "\n")
    from unified_stpp import load_jsonl
    return load_jsonl(str(path))
```

Mirror this for `tiny_val` and `tiny_test`.
