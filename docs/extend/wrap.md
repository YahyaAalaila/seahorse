# Wrap an Existing Model

If you have an existing PyTorch STPP model and want to register it as a Seahorse preset, the adapter pattern requires implementing two classes: a `StateModel` and an `EventModel`.

## The UnifiedSTPP Contract

```
UnifiedSTPP(state_model, event_model, *, hidden_dim)
```

| Component | Responsibility |
| --- | --- |
| `StateModel` | Encodes event history into a hidden state; evolves state to a query time |
| `EventModel` | Computes `log_prob(event | state)` and optionally samples next events |

Both are `nn.Module` subclasses. The full API is defined in `unified_stpp/models/unified_model.py`.

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
from unified_stpp.models.configs.base import BaseModelConfig, ConfigRegistry
from unified_stpp.models.unified_model import UnifiedSTPP

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
