# Seahorse Tool Section Draft

Internal paper draft for the framework/tool section. Intended for later condensation and insertion into the main paper.

## Proposed Paper Structure

1. **Unified model construction:** state model + event model as the minimal framework abstraction for heterogeneous STPP families.  
2. **Benchmark and evaluation contract:** raw-first benchmark inputs, family-owned native spaces, and explicit reporting semantics.  
3. **Execution contract:** a shared `fit` / `tune` / `bench` / `evaluate` pipeline that preserves these semantics across single runs, HPO, benchmark sweeps, and post-fit analysis.

## Main Paper-Facing Draft

### Seahorse: A Framework for Heterogeneous STPPs

Seahorse is designed as an execution contract for heterogeneous spatial-temporal point process (STPP) families rather than as a single-model library. The central abstraction is a decomposition of each preset into a **state model** and an **event model**. Given the canonical event batch \((t_{1:L}, s_{1:L}, \ell)\), the state model constructs a family-owned history representation, while the event model attaches the corresponding event law and training interface. This decomposition is broad enough to cover analytically factorized baselines, paper-faithful AutoSTPP and DeepSTPP ports, continuous-time neural variants such as `neural_attncnf`, `neural_jumpcnf`, and `neural_cond_gmm`, and sample-based families such as `smash` and `diffusion_stpp`. The framework therefore does not impose a universal decoder form; it standardizes only the minimal outer contract needed to train and compare models with genuinely different internal semantics.

A second design choice is that benchmarking is **raw-first**, while native model spaces remain family-owned. Benchmark inputs are presented in original coordinates under one shared external interface. Presets that require standardized or paper-specific spaces fit and serialize their own reversible transforms or support statistics at construction time, and the resulting state and event modules operate in that native space internally. This prevents model-specific preprocessing from leaking into the benchmark protocol while still allowing faithful implementations of families whose likelihoods are defined in different coordinates. In practice, this is what makes it possible to compare exact factorized models, paper-faithful windowed models, and direct joint-intensity models under one benchmark contract without rewriting them into an artificial common representation.

Seahorse also makes a distinction that is often implicit in STPP codebases: the **training objective**, the **benchmark-facing NLL**, and the model’s **evaluation exposure** are treated as separate contracts. Each event model declares what is optimized and checkpointed, whether test-time NLL is exact or approximate, and whether the fitted model exposes calibrated intensity or density queries, a score field, or only a native sampler. This separation is essential in our setting because the benchmark spans exact-intensity models, factorized likelihood models, and generative denoisers. A single scalar reported as “NLL” would otherwise conflate exact and approximate quantities, and a single inference interface would blur the difference between calibrated intensity evaluation and sample-only comparison.

These modeling contracts are matched by an explicit execution contract. A single-run `fit` path builds the model from a resolved preset and data-dependent family statistics; `tune` performs HPO without changing the benchmark-facing semantics; `bench` executes multi-preset, multi-dataset, multi-seed sweeps under a frozen benchmark policy; and `evaluate` performs post-fit analysis through capability-aware metric profiles. The key point is not the presence of separate commands, but that the same semantic metadata are preserved throughout the pipeline. In particular, benchmark artifacts keep track of whether a family reports exact or approximate NLL, whether values are in native or raw space, and whether a preset is canonical, provisional, or legacy. This makes cross-family comparison auditable rather than implicit.

From a paper perspective, the contribution of Seahorse is therefore not generic experiment orchestration. Its contribution is a compact framework in which heterogeneous STPPs can be executed under one raw-first benchmark interface while preserving their native history representations, event laws, objectives, and evaluation semantics. That explicit separation is what makes the benchmark comparisons in this work technically defensible.

## Appendix Cross-Reference Note

Push the exact preset-to-component mapping, inventory-style tables, metric inventories, and implementation-specific details into the appendix. The main text should retain only the framework contracts needed to justify heterogeneous yet comparable benchmarking.
