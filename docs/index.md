---
hide:
  - toc
---

# Seahorse

<p class="hero-tagline">
Unified benchmarking for heterogeneous spatio-temporal point processes.<br>
One API. One CLI. Many models. Consistent metrics.
</p>

<div class="grid cards" markdown>

-   :material-play-circle:{ .lg .middle } **Run One Model**

    ---

    Fit, evaluate, and sample from any registered model in a few lines of Python.

    [:octicons-arrow-right-24: Python API](python-api.md)

-   :material-chart-bar-stacked:{ .lg .middle } **Benchmark Many Models**

    ---

    Compare presets across datasets and seeds with one CLI command. Reproducible run artifacts included.

    [:octicons-arrow-right-24: Run A Benchmark](examples/run-a-small-benchmark.md)

-   :material-magnify:{ .lg .middle } **Evaluate Results**

    ---

    Run metric profiles — likelihood, predictive, surface — on any saved run directory.

    [:octicons-arrow-right-24: Evaluation Guide](evaluation.md)

-   :material-plus-box-outline:{ .lg .middle } **Add Your Model**

    ---

    Register a preset and your model works automatically with `fit`, `bench`, and all metric profiles.

    [:octicons-arrow-right-24: Developer Guide](adding-a-model.md)

</div>

## Quick Start

=== "Python API"

    ```python
    from unified_stpp import AutoSTPP, load_jsonl

    train = load_jsonl("data/hawkesnest_hard_v2/topology_T5/train.jsonl")
    val   = load_jsonl("data/hawkesnest_hard_v2/topology_T5/val.jsonl")
    test  = load_jsonl("data/hawkesnest_hard_v2/topology_T5/test.jsonl")

    model = AutoSTPP(device="cpu", seed=42)
    model.fit(train, val, test, epochs=10, batch_size=64)
    print(model.evaluate(test))
    ```

=== "CLI"

    ```bash
    python -m unified_stpp bench \
      --presets poisson_gmm auto_stpp deep_stpp \
      --dataset hawkesnest_hard_v2/topology_T5 \
      --dataset-revision main \
      --seeds 1 2 3 \
      --out runs/bench
    ```

## What Seahorse Does

Start with a hosted Seahorse dataset such as `hawkesnest_hard_v2/topology_T5`,
then swap the same split layout over to your own JSONL files when you are ready
to iterate locally.

<div class="hero-figure">
  <img src="assets/cool_figure.png" alt="Seahorse overview: event data, model, and YAML config feed into the framework, which outputs reproducible metrics and tuned results.">
</div>

## Core Concepts

| Concept | What it is |
| --- | --- |
| **Data** | JSONL split files — one sequence per line with `times` and `locations` |
| **Preset** | Named model configuration, works with all CLI commands and the Python API |
| **Python API** | `STPPEstimator` subclasses for single-model scripts and notebooks |
| **CLI** | `fit`, `tune`, `bench`, `evaluate` — reproducible runs with saved artifacts |
| **Metric profiles** | `core`, `predictive`, `surface`, `full` — gated by model capability |

For a full walkthrough, open the [tutorial notebooks](examples/colabs.md).
