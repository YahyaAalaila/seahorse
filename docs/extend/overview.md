# Extension Overview

Seahorse is designed to be extended. The three main extension points are:

| What to extend | How |
| --- | --- |
| **New model family** | Add a `ModelFamilyConfig`, register it, optionally add a `PresetDescriptor` |
| **Existing PyTorch model** | Wrap it as a `StateModel` / `EventModel` pair |
| **New dataset** | Format as JSONL splits; optionally publish to HuggingFace |

## Architecture Recap

Every model in Seahorse is a `UnifiedSTPP(state_model, event_model, *, hidden_dim)`. The event history flows through:

```
history → StateModel.encode() → StateModel.evolve() → EventModel.log_prob() → NLL
```

You implement a `StateModel` (history encoder + state evolution) and an `EventModel` (spatial and temporal decoder). A `ModelFamilyConfig` dataclass wires them together and registers the preset.

## Extension Pages

| Page | What it covers |
| --- | --- |
| [Add a Model](../adding-a-model.md) | End-to-end checklist for a new preset |
| [Wrap an Existing Model](wrap.md) | Adapter pattern for importing a PyTorch model |
| [Register a Preset](preset.md) | Config registry, YAML defaults, and CLI wiring |
| [Declare Capabilities](capabilities.md) | Which evaluation paths your model supports |
| [Testing Checklist](testing.md) | Minimal test suite before adding to benchmarks |

## When to Use Each Page

- **New model from scratch**: read [Add a Model](../adding-a-model.md), then [Register a Preset](preset.md).
- **Wrapping an existing architecture**: start with [Wrap an Existing Model](wrap.md).
- **Adding sampling or surface support to an existing model**: read [Declare Capabilities](capabilities.md).
- **Checking your implementation before benchmarking**: run the [Testing Checklist](testing.md).

## See Also

- [Architecture](../architecture.md) — full model layer documentation.
- [Model Capability Matrix](../model-capability-matrix.md) — what capabilities each preset declares.
