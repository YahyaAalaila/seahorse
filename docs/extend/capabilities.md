# Declare Capabilities

Seahorse evaluation profiles are gated by **capability flags**. Before adding a preset to benchmark examples, decide which capabilities it supports and verify them with explicit tests.

## What each capability unlocks

Each capability is a method you implement. Implement it → its metrics turn on.
Skip it → those metrics report a clean skip, never a wrong number.

<div class="sh-cap-grid" markdown="0">
  <div class="sh-cap sh-cap--exact">
    <div class="sh-cap-top"><span class="sh-cap-name">Exact NLL</span><span class="sh-cap-tag">exact</span></div>
    <span class="sh-cap-by">declared by <code>log_prob()</code> returning an exact log-likelihood</span>
    <span class="sh-cap-unlocks">unlocks</span>
    <div class="sh-cap-chips"><span class="sh-cap-chip">core</span><span class="sh-cap-chip">nll</span><span class="sh-cap-chip">benchmark NLL tables</span></div>
  </div>
  <div class="sh-cap sh-cap--approx">
    <div class="sh-cap-top"><span class="sh-cap-name">Bounded NLL</span><span class="sh-cap-tag">bound</span></div>
    <span class="sh-cap-by">declared by <code>log_prob()</code> returning a surrogate (score-matching, ELBO)</span>
    <span class="sh-cap-unlocks">unlocks</span>
    <div class="sh-cap-chips"><span class="sh-cap-chip">nll tables · non-exact</span></div>
  </div>
  <div class="sh-cap sh-cap--optional">
    <div class="sh-cap-top"><span class="sh-cap-name">Next-event sampling</span><span class="sh-cap-tag">optional</span></div>
    <span class="sh-cap-by">declared by implementing <code>sample_next()</code></span>
    <span class="sh-cap-unlocks">unlocks</span>
    <div class="sh-cap-chips"><span class="sh-cap-chip">predictive</span><span class="sh-cap-chip">predict_next</span></div>
  </div>
  <div class="sh-cap sh-cap--optional">
    <div class="sh-cap-top"><span class="sh-cap-name">Generative rollouts</span><span class="sh-cap-tag">optional</span></div>
    <span class="sh-cap-by">declared by a sampling path that supports multi-step rollout</span>
    <span class="sh-cap-unlocks">unlocks</span>
    <div class="sh-cap-chips"><span class="sh-cap-chip">generative</span><span class="sh-cap-chip">autoregressive</span></div>
  </div>
  <div class="sh-cap sh-cap--optional">
    <div class="sh-cap-top"><span class="sh-cap-name">Intensity surface</span><span class="sh-cap-tag">optional</span></div>
    <span class="sh-cap-by">declared by implementing <code>intensity_grid()</code></span>
    <span class="sh-cap-unlocks">unlocks</span>
    <div class="sh-cap-chips"><span class="sh-cap-chip">surface</span><span class="sh-cap-chip">evaluate surface</span></div>
  </div>
</div>

Save/load and re-evaluation from disk come for free through the runner — there is
no capability to declare.

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

## Exact vs approximate NLL

NLL is only comparable across families that compute it the same way. Seahorse
keeps the two tiers visibly separate in benchmark tables.

<div class="sh-tier" markdown="0">
  <div class="sh-tier-row sh-tier-row--exact">
    <div class="sh-tier-label"><span class="sh-tier-name">Exact</span><span class="sh-tier-sub">directly comparable</span></div>
    <div class="sh-tier-chips">
      <span class="sh-fam">auto_stpp</span><span class="sh-fam">deep_stpp</span><span class="sh-fam">nsmpp</span><span class="sh-fam">njsde</span><span class="sh-fam">neural_*</span><span class="sh-fam">poisson_*</span><span class="sh-fam">hawkes_*</span><span class="sh-fam">rmtpp_gmm</span><span class="sh-fam">thp_gmm</span>
    </div>
  </div>
  <div class="sh-tier-row sh-tier-row--approx">
    <div class="sh-tier-label"><span class="sh-tier-name">Approximate</span><span class="sh-tier-sub">a bound on the likelihood</span></div>
    <div class="sh-tier-chips">
      <span class="sh-fam">smash <em>· score-matching</em></span><span class="sh-fam">diffusion_stpp <em>· ELBO</em></span>
    </div>
  </div>
</div>

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
