# Visualize Results

Seahorse provides two visualization paths: Python helper methods on fitted estimators, and CLI commands that operate on saved run artifacts.

## Python Helpers

Fitted estimators expose two plotting methods:

```python
from unified_stpp import AutoSTPP, load_jsonl

test = load_jsonl("data/my_dataset/test.jsonl")
model = AutoSTPP.load("runs/api/auto_stpp")

# Intensity surface over space for one test sequence
surface = model.plot_intensity(
    test[0],
    output_path="runs/plots/intensity",
)

# KDE of sampled next-event locations
kde = model.plot_kde_surface(
    test[0],
    n_samples=128,
    output_path="runs/plots/kde",
)
```

- `plot_intensity` requires a fitted or loaded runner with a run directory. It calls the model's intensity grid path.
- `plot_kde_surface` requires `plotly`. It draws a kernel-density estimate of sampled next-event locations over the spatial domain.

!!! note
    Python visualization helpers work on a single fitted estimator. For outputs that
    need to align with benchmark artifacts, use the CLI commands below.

## CLI Surface Diagnostics

Surface diagnostics render an intensity or density grid for one sequence from a saved run:

```bash
python -m unified_stpp evaluate surface \
  --run path/to/run_dir \
  --history data/my_dataset/test.jsonl \
  --split test \
  --seq-idx 0 \
  --profile history_frame \
  --out runs/evaluate/surface
```

Two surface profiles are available:

| Profile | Supported families | Notes |
| --- | --- | --- |
| `history_frame` | `auto_stpp`, `deep_stpp` | Exact intensity evaluated over a spatial grid given the observed history |
| `future_exact` | Neural exact families (`njsde`, `neural_jumpcnf`, `neural_attncnf`) | May prefer `--device cpu` for numerical stability |

Run `python -m unified_stpp evaluate surface --help` for the full option list.

## CLI Predictive Comparison

Overlay sampled next-event predictions against ground truth for one sequence:

```bash
python -m unified_stpp evaluate predictive-compare \
  --run path/to/run_a \
  --run path/to/run_b \
  --label model_a \
  --label model_b \
  --history data/my_dataset/test.jsonl \
  --split test \
  --seq-idx 0 \
  --horizon 1.0 \
  --out runs/evaluate/predictive_compare
```

Repeat `--run` and `--label` to compare two models side by side. `--horizon` is the prediction window duration (required).

## What Each Output Shows

| Output | Useful for |
| --- | --- |
| Intensity surface | Checking whether the model concentrates event mass near observed clusters |
| KDE of next-event samples | Inspecting predictive uncertainty over space |
| Predictive-compare overlay | Qualitative comparison of where two models place their predictions |

For quantitative evaluation see [Evaluate a Model](evaluate.md) and [Evaluation Profiles](../evaluation.md).
