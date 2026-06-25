# Wrap an Existing Model

Have a PyTorch STPP model already? You expose it to Seahorse by splitting it into
two `nn.Module` adapters — a **StateModel** and an **EventModel** — that
`UnifiedSTPP` drives. This page shows how a batch of events flows through them,
what each part owns, and where an optional **PresetDescriptor** fits.

## How a batch flows through your model

A padded batch of sequences enters at the top; each component's method transforms
it into the next shape, ending in the per-event likelihood Seahorse trains on.

<div class="sh-flow" markdown="0">
  <div class="sh-flow-node sh-flow-node--io">
    <span class="sh-flow-shape">(B,T) times · (B,T,2) locations · (B,T) mask</span>
    <span class="sh-flow-note">a padded batch of event sequences</span>
  </div>
  <div class="sh-flow-edge sh-flow-edge--state"><span class="sh-flow-call">StateModel.encode()</span></div>
  <div class="sh-flow-node">
    <span class="sh-flow-shape">(B, T, H) hidden</span>
    <span class="sh-flow-note">one hidden vector per observed event</span>
  </div>
  <div class="sh-flow-edge sh-flow-edge--state"><span class="sh-flow-call">StateModel.evolve(query_times)</span></div>
  <div class="sh-flow-node">
    <span class="sh-flow-shape">(B, T_q, H) state</span>
    <span class="sh-flow-note">state carried forward to each query time</span>
  </div>
  <div class="sh-flow-edge sh-flow-edge--event"><span class="sh-flow-call">EventModel.log_prob(times, locs, state, mask)</span></div>
  <div class="sh-flow-node sh-flow-node--io sh-flow-node--out">
    <span class="sh-flow-shape">(B, T) per-event log-likelihood</span>
    <span class="sh-flow-note">summed over real events → the NLL Seahorse trains and scores</span>
  </div>
</div>

## What each part owns

<div class="sh-comp-grid" markdown="0">
  <div class="sh-comp sh-comp--state">
    <span class="sh-comp-tag">required</span>
    <span class="sh-comp-name">StateModel</span>
    <span class="sh-comp-role">Turns raw event history into a state vector.</span>
    <div class="sh-comp-methods">
      <div class="sh-comp-m"><code>encode(times, locs, mask)</code><span>→ (B, T, H) hidden</span></div>
      <div class="sh-comp-m"><code>evolve(hidden, query_times)</code><span>→ (B, T_q, H) state</span></div>
    </div>
    <span class="sh-comp-where">your <code>StateModel</code> nn.Module</span>
  </div>
  <div class="sh-comp sh-comp--event">
    <span class="sh-comp-tag">required</span>
    <span class="sh-comp-name">EventModel</span>
    <span class="sh-comp-role">Scores events under the state — the likelihood, plus optional generation.</span>
    <div class="sh-comp-methods">
      <div class="sh-comp-m"><code>log_prob(times, locs, state, mask)</code><span>→ (B, T) · required</span></div>
      <div class="sh-comp-m"><code>sample_next(state, t_last)</code><span>→ next events · optional</span></div>
      <div class="sh-comp-m"><code>intensity_grid(state, t, s)</code><span>→ density · optional</span></div>
    </div>
    <span class="sh-comp-where">your <code>EventModel</code> nn.Module</span>
  </div>
  <div class="sh-comp sh-comp--descriptor">
    <span class="sh-comp-tag sh-comp-tag--opt">optional</span>
    <span class="sh-comp-name">PresetDescriptor</span>
    <span class="sh-comp-role">Injects data-derived setup before the model is built — bounding box, coordinate stats, device fallback.</span>
    <div class="sh-comp-methods">
      <div class="sh-comp-m"><code>data_init_overrides(dm)</code><span>→ dict merged into the build</span></div>
    </div>
    <span class="sh-comp-where">runs once before <code>build_model()</code>, given the fitted data module</span>
  </div>
</div>

`UnifiedSTPP(state_model, event_model, *, hidden_dim)` is the wiring that ties the
two modules together; the full API lives in `seahorse/models/unified_model.py`.

## Minimal StateModel Adapter

```python
import torch
import torch.nn as nn

class MyStateModel(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.encoder = MyExistingEncoder(hidden_dim)
        self.hidden_dim = hidden_dim

    def encode(self, times, locations, mask):
        """Encode event history.

        Args:
            times:     (B, T) event times
            locations: (B, T, 2) event locations
            mask:      (B, T) bool mask — True where events exist

        Returns:
            hidden: (B, T, hidden_dim) per-event hidden states
        """
        return self.encoder(times, locations, mask)

    def evolve(self, hidden, query_times):
        """Evolve state to query times (for piecewise-constant families: no-op).

        Args:
            hidden:      (B, T, hidden_dim)
            query_times: (B, T_q) query times

        Returns:
            state: (B, T_q, hidden_dim)
        """
        return hidden  # piecewise-constant: return last hidden state before each query
```

## Minimal EventModel Adapter

```python
class MyEventModel(nn.Module):
    def __init__(self, hidden_dim: int, spatial_dim: int = 2):
        super().__init__()
        self.decoder = MyExistingDecoder(hidden_dim, spatial_dim)

    def log_prob(self, times, locations, state, mask):
        """Compute per-event log-probability.

        Args:
            times:     (B, T) event times
            locations: (B, T, 2) event locations
            state:     (B, T, hidden_dim) evolved state at event times
            mask:      (B, T) bool mask

        Returns:
            log_prob: (B, T) per-event log-likelihood (masked positions can be 0)
        """
        return self.decoder.log_prob(times, locations, state, mask)

    def sample_next(self, state, t_last, n_samples: int = 1):
        """Optional: sample next event given state.

        Raise NotImplementedError if sampling is not supported.
        """
        raise NotImplementedError("MyEventModel does not support next-event sampling")
```

## Wire Into a Preset

Once you have `MyStateModel` and `MyEventModel`, create a `ModelFamilyConfig`:

```python
from dataclasses import dataclass
from seahorse.models.configs.base import BaseModelConfig, ConfigRegistry
from seahorse.models.unified_model import UnifiedSTPP

@ConfigRegistry.register("my_preset")
@dataclass
class MyPresetConfig(BaseModelConfig):
    hidden_dim: int = 64

    @classmethod
    def from_dict(cls, d, *, hidden_dim, **kwargs):
        return cls(hidden_dim=hidden_dim)

    def build_model(self) -> UnifiedSTPP:
        state = MyStateModel(self.hidden_dim)
        event = MyEventModel(self.hidden_dim)
        return UnifiedSTPP(state, event, hidden_dim=self.hidden_dim)
```

Then follow the [Register a Preset](preset.md) page to expose it through the CLI and Python API.

## Common Pitfalls

- **Shape mismatches**: Seahorse passes `(B, T, *)` tensors. Check your existing model's expected input shape.
- **`inference_mode` conflict**: if your model uses `torch.autograd.grad` internally, ensure the runner is configured with `inference_mode=False`.
- **Missing mask handling**: padding positions in a batch have `mask=False`. Sum or mean log-prob over `mask=True` positions only.
