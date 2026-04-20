# Seahorse Benchmarking Section Draft

Internal draft for the paper section that explains Seahorse as a benchmarking system built on top of the unified STPP framework.

## Recommended Section Structure

1. **Overview:** Seahorse as a controlled benchmarking contract for heterogeneous STPPs, not a repetition of the model abstraction.  
2. **Benchmark construction and semantics:** raw-first inputs, preset/config resolution, family-owned transforms/support statistics, and explicit reporting metadata.  
3. **Execution pipeline:** shared `fit`, `tune`, `bench`, and `evaluate` paths that preserve benchmark semantics across single runs, HPO, benchmark sweeps, and post-fit analysis.

## Compact Paper-Facing Draft

### Seahorse as a Benchmarking System

Building on the unified model abstraction above, Seahorse operationalizes heterogeneous STPPs under a single benchmarking contract. The key goal is not to make all families look identical internally, but to make their comparison explicit and controlled. To this end, benchmark inputs are kept **raw-first**: the benchmark policy fixes the external data contract in original coordinates, while model construction remains **preset-driven**. Each preset resolves a concrete state model and family-specific event law from a shared configuration interface, then injects any family-owned data-dependent quantities required for faithful execution, such as reversible coordinate transforms or support bounds. This keeps the benchmark protocol model-agnostic while allowing individual families to retain their native spaces and likelihood semantics.

The central benchmarking contribution of Seahorse is that comparison semantics are recorded explicitly rather than inferred informally. In particular, the framework distinguishes (i) the quantity optimized during training, (ii) the benchmark-facing NLL reported at test time, and (iii) the fitted model’s evaluation exposure. These are first-class metadata attached to each preset and preserved in run artifacts. As a result, benchmark outputs can state whether a reported NLL is exact or approximate, whether it is expressed in native or raw space, and whether a model exposes calibrated intensity or density queries, a score field, or only a native sampler. This matters because the benchmark includes exact factorized models, paper-faithful likelihood models, direct joint-intensity models, and generative denoisers; without explicit reporting semantics, a single leaderboard would collapse scientifically different quantities into one uninterpretable metric.

Seahorse further enforces a shared execution path for training, HPO, sweeps, and post-fit analysis. A single-run `fit` path resolves the preset under the benchmark policy and writes semantically annotated run artifacts. `tune` searches hyperparameters without changing the benchmark-facing contract. `bench` executes multi-preset, multi-dataset, multi-seed sweeps under a frozen raw-first policy and aggregates runs into tables whose exact and approximate views are separated by recorded metadata rather than by ad hoc preset filtering. Finally, `evaluate` performs post-fit analysis through capability-aware metric profiles: models that expose calibrated intensity or density are queried directly, whereas sample-based families are evaluated through native sampling or thinning-backed artifacts as appropriate. The framework therefore turns the conceptual STPP abstraction into a reproducible benchmarking system in which heterogeneity is preserved internally but comparison is standardized externally.

From the paper’s perspective, Seahorse is not merely experiment infrastructure. Its contribution is a benchmark-aware execution system that allows heterogeneous STPP families to be instantiated from presets, run under a raw-first benchmark contract, and evaluated with explicit semantics for objective, likelihood quality, reporting space, and inference exposure. This is the mechanism that makes the empirical comparisons in the benchmark technically defensible.

## Workflow Figure Suggestion

Use a left-to-right workflow figure with the following nodes and arrows:

`Raw dataset / fixed splits`  
→ `Benchmark policy (raw-first protocol, reporting space, checkpoint policy)`  
→ `Preset + config resolution`  
→ `Family-owned data-dependent transforms / support statistics`  
→ `Model instantiation (state model + family-specific event law)`  
→ branching execution node with `fit`, `tune`, `bench`, `evaluate`  
→ `Run and benchmark artifacts`  
→ small metadata callout attached to the artifact node: `objective`, `nll_kind`, `nll_report_space`, `evaluation exposure`, `preset_status`.

The figure should emphasize that the benchmark policy is shared, while transforms and event laws are family-owned.

## Appendix Note

Push the exact preset inventory, preset-to-component tables, metric-profile details, artifact schemas, and implementation-specific command behavior to the appendix. The main text should retain only the benchmark contract and the semantic guarantees it provides.

## TL;DR

This section should present Seahorse as the system that turns the unified STPP abstraction into a controlled benchmark: raw-first inputs, preset-driven model instantiation with family-owned native spaces, explicit semantics for what is trained and reported, and shared execution paths that preserve those semantics across fitting, tuning, sweeping, and post-fit evaluation.
