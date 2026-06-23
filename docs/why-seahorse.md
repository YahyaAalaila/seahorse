# Why Seahorse

Spatio-temporal point processes (STPPs) describe systems where discrete events occur at observed times and locations: earthquake aftershock sequences, disease case reports, crime incidents, ride-hailing requests. Modelling them well requires fitting a joint intensity over time and space.

## The Problem

Existing STPP research code is scattered. Each paper ships its own training loop, its own metric definition, and its own data loader. Reproducing published results requires reconstructing experiment details from scratch, and comparing two models from different papers is rarely apples-to-apples: they may use different normalisation, different evaluation windows, or different likelihood definitions.

## What Seahorse Adds

| Without Seahorse | With Seahorse |
| --- | --- |
| Per-paper training loops | One shared training harness for all models |
| Incompatible data layouts | One JSONL format and Hugging Face integration |
| Ad-hoc evaluation scripts | Metric profiles gated by model capability |
| Results that can't be directly compared | Shared benchmark contract enforced for every run |
| Saved runs that are hard to reload | `RunResult` artifact + runner save/load |

Seahorse imposes a **benchmark contract**: all presets are evaluated on the same dataset, the same normalization policy, and the same metric definition. This makes comparisons trustworthy by construction rather than by convention.

## Scope

Seahorse is built for reproducible STPP research, benchmarking, and offline evaluation. It deliberately keeps online serving concerns outside the core framework:

- Model serving and API deployment belong in an application layer.
- Streaming ingestion and online inference belong in a production data plane.
- Dataset-specific preprocessing remains explicit so benchmark inputs stay auditable.

If you want to run one model in a script or notebook, the [Python API](python-api.md) is the starting point. If you want reproducible multi-model comparisons, the [Benchmark](benchmarks.md) path is the right tool.

## Design Choices

**Unified model interface.** Every model is a `UnifiedSTPP(state_model, event_model)`. This makes it possible to swap components, compare families, and evaluate with a single shared harness.

**Artifact-first evaluation.** Every `fit`, `tune`, and `bench` call writes a `RunResult`. Evaluation is always post-hoc from a saved artifact, not embedded in training. This separates concerns and makes re-evaluation cheap.

**Capability gating.** Metrics that require sampling, grid evaluation, or rollouts are gated behind explicit `--metric-profile` flags. A model that does not support a capability fails explicitly rather than silently returning wrong numbers.
