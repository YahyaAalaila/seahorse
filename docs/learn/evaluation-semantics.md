# Evaluation Semantics

This page defines what each metric means, how it is computed, and when results are comparable.

## Per-Event NLL

The primary benchmark metric is **per-event negative log-likelihood**:

```
NLL = -1/N Σ log p(t_i, s_i | history up to t_i)
```

- N is the total number of events across all test sequences.
- `p(t, s | history)` is the joint intensity or density evaluated at each observed event.
- Lower NLL is better.

This is the value stored in `RunResult.test_nll` and reported in benchmark tables.

## Exact vs Approximate NLL

| NLL type | What it means | Families |
| --- | --- | --- |
| **Exact** | True log-likelihood of the point process | `auto_stpp`, `deep_stpp`, `nsmpp`, `njsde`, `neural_*`, factorized families |
| **Approximate (score-matching)** | Score-matching surrogate for the log-likelihood | `smash` |
| **Approximate (ELBO)** | Evidence lower bound on the log-likelihood | `diffusion_stpp` |

Exact and approximate NLL measure likelihood differently; the [Model Capability Matrix](../model-capability-matrix.md) lists which kind each preset reports.

## Normalization and Comparability

`RunResult.norm_stats` records whether normalization was applied and the scaling parameters:

```json
{
  "normalize": true,
  "time_mean": 5.2,
  "time_std": 3.1,
  "loc_mean": [0.5, 0.5],
  "loc_std": [0.3, 0.3]
}
```

NLL values are **comparable across presets only when all use the same normalization setting**. The `bench` command enforces this via the [execution contract](execution-contract.md).

To convert normalised NLL to original-coordinate NLL:

```
NLL_original = NLL_normalised − log(time_std × loc_std_x × loc_std_y)
```

## Metric Profiles

Seahorse gates metrics behind explicit profiles so expensive sampling or grid work is always opt-in:

| Profile | Metrics computed | Requires |
| --- | --- | --- |
| `core` | `test_nll`, `temporal_nll`, `spatial_nll`, `mean_seq_nll` | Exact or approximate NLL |
| `nll` | Extended NLL-family checks | Exact NLL |
| `predictive` | CRPS, energy score, MAE, RMSE, coverage | Next-event sampling |
| `generative` | Distribution metrics over full rollouts | Generative sampling |
| `autoregressive` | Fixed-prefix degradation metrics | Generative sampling |
| `surface` | Intensity/density grid diagnostics | Intensity surface query |
| `full` | All registered benchmark metrics | All of the above |

## Predictive Metrics

Predictive metrics use sampled next-event predictions. They measure how well a model predicts the next event given the observed history:

| Metric | Temporal | Spatial |
| --- | --- | --- |
| CRPS | Continuous ranked probability score | — |
| Energy score | — | Multivariate energy score |
| MAE | Mean absolute error in time | Mean absolute error in space |
| RMSE | Root mean squared error in time | Root mean squared error in space |
| Coverage | Marginal calibration (temporal) | — |

## Unavailable Metrics

When a metric is marked `available: false` in `metrics.json`, it means the model does not support the required capability — not that evaluation failed. Common reasons:

| Reason | What to do |
| --- | --- |
| `missing NLL capability` | Model uses a surrogate objective; switch to a profile the model supports |
| `missing sampling capability` | Model does not implement `sample_next`; check the capability matrix |
| `missing surface capability` | Model does not expose an intensity grid; try a different preset |
| `heavy artifacts not planned` | You requested a metric that needs sampling but used `--metric-profile core`; re-run with `--metric-profile predictive` |

## Temporal vs Spatial NLL

For **factorized families** (`poisson_gmm`, `hawkes_gmm`, etc.), the joint log-likelihood decomposes:

```
log p(t, s | history) = log p_temporal(t | history) + log p_spatial(s | t, history)
```

Seahorse reports `temporal_nll` and `spatial_nll` separately for these families. Non-factorized families report only the joint `test_nll`.
