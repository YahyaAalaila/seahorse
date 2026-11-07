# From Time Series to Point Processes

Event data are often aggregated into counts on a space–time grid, with cells
$A_k \subset \mathbb{R}^2$ and intervals of length $h$, and modelled with time-series or
spatiotemporal forecasting models. Aggregation fixes the spatial and temporal resolution before
modelling and discards the exact times and locations within a cell. It is the right choice when only
counts are needed, when the data arrive aggregated, or when the volume is so large that per-event
likelihoods are impractical.

A point process keeps the events, and any aggregation can be recovered from it afterwards, since the
count in a cell is a functional of the process. The table lists the correspondence.

| Time series of counts | Spatiotemporal point process |
| --- | --- |
| count $Y_{k,t}$ in cell $A_k$ over $(t, t+h]$ | $N(A_k \times (t, t+h])$ |
| conditional mean $\mathbb{E}[Y_{k,t} \mid \text{past}]$ | $\mathbb{E}\big[\int_t^{t+h}\!\int_{A_k} \lambda \mid \mathcal{H}_t\big]$; equal to $\int\!\int \lambda$ if $\lambda$ does not depend on events in $(t, t+h]$ (e.g. Poisson) |
| one-step-ahead forecast | law of the next event $(t_{i+1}, \mathbf{s}_{i+1})$ given $\mathcal{H}_{t_i}$ |
| count likelihood (e.g. Poisson or negative binomial per cell) | point-process likelihood $\sum_i \log \lambda(t_i, \mathbf{s}_i \mid \mathcal{H}_{t_i}) - \int_0^T\!\int_{\mathbb{R}^2} \lambda$ |
| resolution $(A_k, h)$ fixed in advance | none: continuous time and space |

One entry needs care. The conditional mean of a future count is the *expected* integrated
intensity,

$$
\mathbb{E}\big[N(A \times (t, t+h]) \mid \mathcal{H}_t\big]
= \mathbb{E}\Big[\int_t^{t+h}\!\int_A \lambda(u, \mathbf{s} \mid \mathcal{H}_u)\, d\mathbf{s}\, du \;\Big|\; \mathcal{H}_t\Big],
$$

where the expectation is over the events in $(t, t+h]$, which themselves change $\lambda$. It reduces
to the integral of the intensity computed from $\mathcal{H}_t$ when the intensity does not depend on
events inside the interval, as for a Poisson process. For a self-exciting process, count forecasts
require simulation or moment equations.

Here $\lambda(t, \mathbf{s} \mid \mathcal{H}_t)$ is the conditional intensity (expected events per
unit time and unit area given the history $\mathcal{H}_t$ of events before $t$), and $N(B)$ is the
number of events in a space–time set $B$.

## Worked example: aggregating a released dataset

The events of a released dataset can be turned into the count series of the left column in a few
lines. Here the NYC collisions of 2023 (one sequence per calendar day, times in days) go onto a
10 × 10 grid of cells and hourly intervals:

```python
import numpy as np
from seahorse.data import load_released_dataset

splits = load_released_dataset("nyc_collisions")
# one sequence per calendar day
days = splits["train"] + splits["val"] + splits["test"]

# The events: (day index, time of day in days, lon, lat)
t = np.concatenate([np.asarray(d["times"]) for d in days])
xy = np.concatenate([np.asarray(d["locations"]) for d in days])
day = np.concatenate([np.full(len(d["times"]), i) for i, d in enumerate(days)])

# Aggregate to a 10 x 10 grid of cells A_k and intervals of h = 1 hour: Y[k, t]
lon_edges = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 11)
lat_edges = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 11)
cell = (np.clip(np.digitize(xy[:, 0], lon_edges) - 1, 0, 9) * 10
        + np.clip(np.digitize(xy[:, 1], lat_edges) - 1, 0, 9))
hour = day * 24 + np.floor(t * 24).astype(int)
Y = np.zeros((100, len(days) * 24), dtype=int)
np.add.at(Y, (cell, hour), 1)

# every event lands in exactly one cell-hour
print(Y.shape, Y.sum() == len(t))
print("share of empty cell-hours:", round((Y == 0).mean(), 3))
print("largest cell-hour count:", Y.max())
```

```text
(100, 8760) True
share of empty cell-hours: 0.916
largest cell-hour count: 8
```

The map from events to $Y$ is many-to-one: $Y$ follows from the events, but the events cannot be
recovered from $Y$, and a different grid or $h$ means re-aggregating. At this resolution more than
nine in ten cell-hours are empty, so a count model works with a sparse, low-count series, while a
point-process model scores each of the events at its exact time and place. Some of the 100 cells lie
outside the city; a real count model would drop them or use administrative areas.

To compare the two views on the same events, fit a point-process preset on the same splits
([Quickstart](../getting-started.md)) and integrate its intensity over each cell-hour. For a
self-exciting model, the expected count needs simulation, as noted above.
