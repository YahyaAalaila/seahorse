# Modeling Events in Space and Time

This tutorial is the end-to-end path for a first-time Seahorse user. It starts
with a spatiotemporal event dataset, inspects the data, trains two STPP models,
evaluates held-out likelihood, and produces visual artifacts that make the
space-time structure visible.

## What You Will Build

The tutorial script writes everything under:

```text
runs/tutorials/modeling_events_in_space_and_time/
```

Expected outputs:

- `data/tutorial_events/{train,val,test}.jsonl` — deterministic ST event splits.
- `figures/eda_panel.svg` — polished EDA panel.
- `figures/event_movie.html` — browser animation of events unfolding over time.
- `figures/model_comparison.svg` — held-out NLL comparison.
- `tables/model_comparison.html` — styled result table.
- `summary.json` — machine-readable artifact index.

## Run The Tutorial

Open in Colab:

<a href="https://colab.research.google.com/github/YahyaAalaila/seahorse/blob/main/docs/notebooks/03_modeling_events_in_space_and_time.ipynb">03 Modeling Events in Space and Time</a>

Then choose **Runtime → Run all**. The notebook setup cell clones and installs
Seahorse automatically when running inside Colab.

Local script:

```bash
python examples/modeling_events_in_space_and_time.py
```

For a faster smoke run that trains only the baseline:

```bash
python examples/modeling_events_in_space_and_time.py --skip-auto
```

## Workflow

### 1. Inspect the event data

The dataset follows the standard Seahorse JSONL contract:

```json
{"times": [0.12, 0.27, 0.51], "locations": [[0.25, 0.31], [0.29, 0.35], [0.64, 0.40]]}
```

The EDA panel checks the first things an STPP user should understand:

- how many events each sequence contains;
- how quickly events arrive;
- where events concentrate in space;
- how one held-out history moves through the spatial domain over time.

### 2. Watch the process evolve

Open:

```text
runs/tutorials/modeling_events_in_space_and_time/figures/event_movie.html
```

This is intentionally a movie-style artifact, not a static scatter plot. STPP
data are spatio-temporal, so the tutorial shows the history as it becomes
available to the model.

### 3. Fit two models

The tutorial trains:

| Model | Purpose |
| --- | --- |
| `poisson_gmm` | Simple baseline with exact likelihood |
| `auto_stpp` | More flexible history-aware neural STPP |

Both use CPU-safe tutorial settings. The goal is to demonstrate the workflow,
not to reproduce a benchmark paper.

### 4. Evaluate and conclude

The tutorial reports held-out NLL and mean sequence NLL, then writes a polished
comparison table. Lower NLL is better when comparing models on the same dataset,
split, and reporting convention.

The final conclusion should answer:

- Did the history-aware model improve over the baseline?
- Does the EDA suggest spatial clustering, temporal rhythm, or both?
- Are the results tutorial-scale or benchmark-scale?

## Notebook

The notebook version is:

```text
docs/notebooks/03_modeling_events_in_space_and_time.ipynb
```

It runs the same end-to-end script from a notebook environment.
