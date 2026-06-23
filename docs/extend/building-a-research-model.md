# Building a Research Model in Seahorse

This tutorial is for researchers who want to add a new STPP model without
rewriting training, evaluation, or benchmarking code.

The example model is intentionally small:

```text
event history → GRU encoder → exponential state evolution → temporal + spatial decoder
```

It is registered as:

```text
demo_gru_gaussian
```

## What You Will Build

The tutorial script writes everything under:

```text
runs/tutorials/building_a_research_model/
```

Expected outputs:

- generated tutorial train/val/test splits;
- a fitted `demo_gru_gaussian` run directory;
- a styled model-result table;
- a tiny benchmark directory comparing:
  - `poisson_gmm`;
  - `hawkes_gmm`;
  - `demo_gru_gaussian`.

## Run The Tutorial

Open in Colab:

<a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/04_building_a_research_model.ipynb">04 Building a Research Model</a>

Then choose **Runtime → Run all**. The notebook setup cell clones and installs
Seahorse automatically when running inside Colab.

Local script:

```bash
python examples/building_a_research_model.py
```

For a faster model-only smoke run:

```bash
python examples/building_a_research_model.py --skip-benchmark
```

## Architecture

The tutorial model uses the normal Seahorse extension layer:

| Layer | Implementation |
| --- | --- |
| State model | `DemoGRUDecayStateModel` |
| Event model | `DemoTemporalGaussianEventModel` |
| Config | `DemoGRUGaussianConfig` |
| Preset | `demo_gru_gaussian` |
| Training | Existing `STPPRunner` and Lightning wrapper |

The key design point is that the tutorial does **not** create a custom training
loop. Once the model is registered, it can use the same fit, evaluation, and
benchmark machinery as the bundled presets.

## Model Contract

The state model:

- reads padded `(time, x, y)` histories;
- encodes them with a GRU;
- shifts the hidden state by one step so event `i` is scored from history before
  event `i`;
- applies a learnable exponential decay over the inter-arrival interval.

The event model:

- predicts an exponential inter-arrival likelihood;
- predicts a diagonal Gaussian spatial likelihood;
- returns eventwise NLL terms so core held-out evaluation can use exact
  next-event scores.

## Tiny Benchmark

The script uses Seahorse's in-process `Benchmark` API. It is equivalent to the
following CLI workflow:

```bash
python -m unified_stpp bench \
  --presets poisson_gmm hawkes_gmm demo_gru_gaussian \
  --dataset runs/tutorials/building_a_research_model/data/tutorial_events \
  --seeds 123 \
  --out runs/tutorials/building_a_research_model/benchmark \
  --n_workers 1 \
  --override training.n_epochs=1 training.batch_size=4 training.device=cpu
```

This is a smoke benchmark. It proves the model enters the framework correctly;
it is not a scientific claim about model quality.

## Notebook

The notebook version is:

```text
docs/notebooks/04_building_a_research_model.ipynb
```

It runs the same end-to-end script from a notebook environment.
