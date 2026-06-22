# Declare Capabilities

Seahorse evaluation profiles are gated by **capability flags**. Before adding a preset to benchmark examples, decide which capabilities it supports and verify them with explicit tests.

## Capability Flags

| Capability | What it enables | How to declare |
| --- | --- | --- |
| **Exact NLL** | `core` and `nll` metric profiles; benchmark NLL tables | `EventModel.log_prob()` returns exact log-likelihood |
| **Approximate NLL** | Included in tables but flagged as approximate | `log_prob()` returns a surrogate (score-matching, ELBO) |
| **Next-event sampling** | `predictive` metric profile; `predict_next` Python method | `EventModel.sample_next()` implemented |
| **Generative rollouts** | `generative` and `autoregressive` profiles | Sampling path supports multi-step rollout |
| **Intensity surface** | `surface` profile via `evaluate surface` | `EventModel` exposes an intensity or density grid query |
| **Save / load** | `STPPEstimator.load()`, re-evaluation from disk | Works automatically via runner checkpoint |

## Declaring in EventModel

Capabilities are declared by which methods are implemented (not raising `NotImplementedError`):

```python
class MyEventModel(nn.Module):

    def log_prob(self, times, locations, state, mask):
        """Exact log-likelihood — enables NLL metrics."""
        ...

    def sample_next(self, state, t_last, n_samples=1):
        """Implement for next-event sampling; raise NotImplementedError otherwise."""
        ...

    def intensity_grid(self, state, t_query, s_query):
        """Implement for surface diagnostics; raise NotImplementedError otherwise."""
        ...
```

When a method raises `NotImplementedError`, the corresponding metric is marked `available: false` in `metrics.json` with a clear reason — it is not treated as a failure.

## Exact vs Approximate NLL

| Families | NLL type | Comparable? |
| --- | --- | --- |
| `auto_stpp`, `deep_stpp`, `nsmpp`, `njsde`, `neural_*`, `poisson_*`, `hawkes_*`, `rmtpp_gmm`, `thp_gmm` | Exact | Yes — directly comparable in benchmark tables |
| `smash` | Score-matching surrogate | Note in tables; not directly comparable to exact families |
| `diffusion_stpp` | ELBO | Note in tables; not directly comparable to exact families |

## Capability Testing

Before adding your preset to benchmark examples, verify each claimed capability:

```python
from seahorse import STPPEstimator, load_jsonl

train = load_jsonl("data/tiny/train.jsonl")
val   = load_jsonl("data/tiny/val.jsonl")
test  = load_jsonl("data/tiny/test.jsonl")

model = STPPEstimator("my_preset", device="cpu")
model.fit(train, val, test, epochs=1, batch_size=4)

# Test NLL
scores = model.evaluate(test)
assert "test_nll" in scores

# Test sampling (if claimed)
samples = model.predict_next(test[:2], n_samples=4)
assert "next_times" in samples

# Test save/load
save_dir = model.save("runs/test_save")
loaded = STPPEstimator.load(save_dir)
scores2 = loaded.evaluate(test)
assert abs(scores["test_nll"] - scores2["test_nll"]) < 1e-4
```

See the [Testing Checklist](testing.md) for the full test suite.
