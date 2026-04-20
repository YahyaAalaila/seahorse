# 4  The Seahorse Benchmarking System

The unified abstraction described in the previous section provides a shared
instantiation interface for STPP families, but does not by itself guarantee
that comparisons between them are scientifically meaningful. STPP families
differ not only in architecture but in the *mathematical objects they expose*:
exact factorized densities, evidence lower bounds, score fields, and ancestral
samplers are qualitatively different quantities. A benchmark that ignores this
heterogeneity risks collapsing scientifically distinct quantities into a single
leaderboard column. Prior benchmarks for temporal point processes [CITE EasyTPP]
sidestep this issue by restricting the model scope to families sharing a common
likelihood structure; the spatio-temporal setting, which introduces spatial
support, reversible coordinate systems, and generative spatial models without
tractable densities, does not admit this simplification.

Seahorse addresses this by operationalizing the unified abstraction under an
explicit *benchmark contract* built around three principles: (i) a **raw-first
data policy** that anchors all external inputs in original coordinates; (ii)
**preset-driven model instantiation** that lets each family retain its native
space while exposing a common configuration interface to the benchmark; and
(iii) **first-class reporting metadata** that records what was trained, what
was reported, and which inference modalities the fitted model supports.
Together these three principles turn the unified abstraction into a
reproducible benchmarking system in which heterogeneity is preserved
internally but comparison is standardized externally.

---

## 4.1  The Benchmark Contract

The benchmark contract specifies the terms under which all models are
compared, independently of their internal representations.

**Raw-first data policy.**  The benchmark fixes input data in original,
unnormalized coordinates and enforces this as a hard constraint at
construction time, applied identically to every preset. Each family is then
responsible for injecting any required data-dependent transforms—reversible
spatiotemporal normalizations, spatial support bounds—*after* the data contract
is fixed but *before* model instantiation. This design eliminates a class of
comparison artifacts that arise when heterogeneous models apply incompatible
preprocessing pipelines or when normalization statistics are inadvertently
derived from held-out data during model construction.

**Separation of training objective, reported metric, and evaluation
exposure.**  The contract distinguishes three quantities that are conflated in
many existing evaluations:

1. *Training objective*: the scalar minimized during optimization, which may
   be an exact NLL, an evidence lower bound, a score-matching loss, or a
   flow-matching objective.
2. *Benchmark-facing NLL*: a standardized per-event log-likelihood computed on
   the held-out test partition in a fixed reporting coordinate system, used as
   the primary cross-model comparison metric.
3. *Evaluation exposure*: the inference modalities the fitted model supports—
   calibrated intensity queries, conditional density evaluation, ancestral
   sampling, or score-field access.

Separating these quantities is not cosmetic. The training objective and the
benchmark-facing NLL need not be the same quantity: a score-matching model
cannot report an exact likelihood, and an approximate model's lower bound may
differ substantially from its true NLL. Recording which quantity appears in
the leaderboard allows Seahorse to present exact and approximate views of
the benchmark table as distinct, annotated outputs rather than suppressing
this distinction.

---

## 4.2  Preset-Driven Model Instantiation

Model instantiation in Seahorse is **preset-driven**: each preset names a
family and a set of configuration defaults, and the benchmark policy resolves
each preset into a fully instantiated model without modifying the data
contract. The resolution process has three stages.

**Stage 1 — Config merge.**  User-provided YAML overrides are merged over
preset defaults, yielding a complete `ModelConfig`. The benchmark policy
enforces that the data-contract fields—normalization mode, split protocol,
reporting space—are not overridable by per-preset YAML.

**Stage 2 — Data-dependent initialization.**  A family-specific
`PresetDescriptor` receives the training partition and injects required
quantities as `build_overrides`. These may include spatial bounding boxes for
CNF-based spatial models, or the coordinate statistics required by reversible
spatiotemporal normalizations. Critically, these quantities are derived
*from the training partition only* and are injected as opaque overrides that
supplement but do not supersede the config. This stage is the only point at
which family internals are allowed to observe data; the benchmark policy
triggers it uniformly for all presets.

**Stage 3 — Model construction.**  `build_model()` is called on the resolved
config, instantiating a `UnifiedSTPP` with the family-specific state model
and event law fully assembled. The benchmark thereafter interacts only with
the shared interface: it knows nothing about the internals of individual
families.

This three-stage protocol keeps the benchmark policy model-agnostic while
allowing each family to retain its native parametrization and likelihood
semantics. A new family is added to the benchmark by implementing a
`PresetDescriptor` and registering a preset; the benchmark contract requires
no further changes.

---

## 4.3  Explicit Reporting Metadata

Each completed run produces a `RunResult` annotated with a fixed set of
semantic fields, summarized in Table 1. These fields are first-class citizens
of the run artifact, not external annotations, and are written atomically with
the model checkpoint.

| Field | Description |
|---|---|
| `objective` | Training loss identifier (`nll_exact`, `elbo`, `score_matching`, …) |
| `nll_kind` | Whether the reported NLL is `exact`, `approx_bound`, or `sample_based` |
| `nll_report_space` | Coordinate space of the reported NLL (`normalized` or `raw`) |
| `evaluation_exposure` | Supported inference modalities (e.g., `calibrated_intensity`, `density`, `sampler`) |
| `preset_status` | Reproducibility tier: `paper_port`, `tuned`, or `best_effort` |

**Table 1.** Per-run reporting metadata stored in every Seahorse `RunResult`.

`bench` aggregates these fields into a `BenchmarkTable` that exposes both a
unified view—all presets, for overview—and a filtered exact-NLL view
restricted to presets with `nll_kind=exact`, for principled cross-family
comparison. Downstream analyses condition on `evaluation_exposure` to select
the appropriate computation path (Section 4.5).

The `preset_status` field addresses a reproducibility gap common in
spatiotemporal point process evaluations: published results routinely mix
original reimplementations, architecture-matched but retuned variants, and
paper-faithful runs without making the distinction explicit. Seahorse makes
this distinction mandatory at the artifact level, so that a reproduction study
can reconstruct the benchmark table and immediately identify which entries are
directly comparable to published numbers.

---

## 4.4  The Execution Pipeline

Seahorse exposes four execution modes, all operating under the same benchmark
contract. Figure 1 illustrates the shared pipeline.

**`fit`.**  Trains a single preset under the benchmark policy and writes a
semantically annotated `RunResult`. The training loop, early stopping
criterion, and checkpoint policy are controlled by the benchmark config, not
by individual presets.

**`tune`.**  Runs hyperparameter search via distributed parallel workers
without modifying the benchmark-facing data contract. Explored configurations
are evaluated on the validation partition; the best configuration is written
back as a YAML override that can be committed alongside the preset. The
benchmark-facing contract—raw-first data, fixed splits, normalized reporting
space—remains frozen across all HPO trials, preventing the common failure mode
in which tuning choices implicitly encode held-out statistics.

**`bench`.**  Executes a multi-preset, multi-dataset, multi-seed sweep under
a frozen raw-first policy. `Benchmark._apply_data_contract()` forcibly
overrides any per-preset protocol or normalization setting before the sweep
begins, ensuring that all presets in a benchmark run receive identical input
data regardless of their standalone defaults.

**`evaluate`.**  Performs post-fit analysis on stored `RunResult` artifacts,
computing capability-specific metric profiles (Section 4.5). `evaluate` is
intentionally decoupled from `bench` so that new metrics can be added to
existing benchmark artifacts without re-running the sweep.

The separation between `tune` and `bench` reflects a principled stance on
what constitutes a benchmark: hyperparameter search is a model-development
activity and is explicitly labeled as such in the resulting artifact; the
benchmark sweep applies fixed policies to the outputs of that development
process.

---

## 4.5  Capability-Aware Metric Profiles

STPP evaluation is intrinsically multi-modal: log-likelihood measures
distributional fit, spatial calibration measures alignment between predicted
and empirical intensity maps, and next-event metrics assess predictive
accuracy. Not all families support all metrics. Seahorse handles this through
**capability-aware metric profiles** that condition evaluation dispatch on a
model's `evaluation_exposure`.

**Direct-query families** (`calibrated_intensity` or `density` exposure) are
evaluated by querying the model on a spatiotemporal grid or directly on
held-out events. This yields exact log-likelihood, intensity-RMSE on synthetic
benchmarks with known ground-truth intensity, and spatial calibration via
probability-integral-transform diagnostics.

**Sample-based families** (only `sampler` exposure) are evaluated through a
thinning-based artifact pipeline. A superposition intensity is constructed
from model samples; Lewis–Ogata thinning then recovers approximate
event-level log-densities for NLL estimation. The resulting metric is labeled
`nll_kind=sample_based` in the run artifact, clearly distinguishing it from
exact evaluations.

This design ensures that generative models—score-matching denoisers,
CNF-based samplers without analytic inverses—are not excluded from the
benchmark table for lacking a tractable likelihood, while also preventing
their sample-based estimates from being placed on equal footing with
exact-likelihood families without explicit annotation. The capability-aware
dispatch is implemented in `evaluate` and produces a per-preset metric profile
that surfaces direct and thinning-based estimates in separate columns.

---

## Figure 1 (suggested placement: after Section 4.1)

A left-to-right pipeline diagram with the following nodes:

```
[Raw dataset / fixed splits]
        ↓
[Benchmark policy]
  · raw-first protocol
  · reporting space
  · checkpoint policy
        ↓
[Preset + config resolution]  ←  YAML overrides
        ↓
[PresetDescriptor]
  · data-dependent transforms
  · support statistics
        ↓
[Model instantiation]
  state model + event law
        ↓  (branches)
  fit ──── tune ──── bench ──── evaluate
        ↓
[RunResult artifact]
  ┌─────────────────────────────────┐
  │ objective · nll_kind            │
  │ nll_report_space                │
  │ evaluation_exposure             │
  │ preset_status                   │
  └─────────────────────────────────┘
```

Emphasis: the benchmark policy box is shared across all presets; the
`PresetDescriptor` box is family-owned.
