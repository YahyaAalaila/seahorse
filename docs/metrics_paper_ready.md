# Evaluation Metrics — Paper-Ready Descriptions

Concise descriptions suitable for methods sections and supplementary material.
All metrics operate on a held-out test set of $N$ events $\{(t_i, s_i)\}$ with conditioning histories $\mathcal{H}_{t_i}$.
Predictive samples are drawn at $K=32$ per event (thinning or native sampler);
generative rollouts at $K_\text{gen}=20$ per test sequence.
Superscripts on NLL-family metrics indicate computation method: $^\text{exact}$, $^\text{VB}$, $^\text{KDE}(K)$.

---

## Fit quality (M1–M5)

**M1. NLL.**
$\text{NLL} = -\tfrac{1}{N}\sum_i \log f^*(t_i, s_i \mid \mathcal{H}_{t_i})$
in raw (physical) coordinates, with Jacobian correction applied to normalised model outputs.
For models without an explicit density (SMASH), NLL is estimated via 2D KDE from $K=2{,}000$ samples.
Lower is better.

**M2. Temporal NLL.**
$\text{NLL}_T = -\tfrac{1}{N}\sum_i \log f^*_T(t_i \mid \mathcal{H}_{t_i})$
where $f^*_T(t) = \int_\mathcal{S} f^*(t,s)\,ds$.
For factorised models, evaluated directly.
For joint-intensity models (AutoSTPP), marginalised by Monte Carlo over the spatial domain.
For sample-only models (SMASH), estimated via 1D KDE over sampled inter-event times.
Jacobian correction: $\text{NLL}_T^\text{orig} = \text{NLL}_T^\text{norm} + \log \sigma_t$,
where $\sigma_t$ is the training-set time standard deviation.

**M3. Spatial NLL.**
$\text{NLL}_S = -\tfrac{1}{N}\sum_i \log f^*_S(s_i \mid t_i, \mathcal{H}_{t_i})$.
For factorised models: direct.
For joint models: $\text{NLL}_S = \text{NLL} - \text{NLL}_T$.
For sample-only models: 2D KDE over spatial samples evaluated at the true location,
with scipy `gaussian_kde` called with input shape $(2, 1)$ to obtain a scalar log-density.

**M4. NLL train/test gap.**
$\Delta\text{NLL} = \text{NLL}_\text{test} - \text{NLL}_\text{train}$.
A single additional forward pass over training data after fitting; stored in `RunResult.extra_metrics`.
Positive values indicate overfitting; negative values indicate underfitting or distribution shift.

**M5. NLL gap to ground truth** (synthetic datasets only).
$\Delta^* = \text{NLL}_\text{model} - \text{NLL}_\text{true}$,
where $\text{NLL}_\text{true}$ is computed from the known generating intensity via the HawkesNest metadata.
Separates approximation error from irreducible process stochasticity.

---

## Predictive quality (M6–M14)

All metrics in this group share a single pool of $K$ next-event samples per test event,
realised as $\{(\hat{\tau}_k, \hat{s}_k)\}_{k=1}^K$ where $\hat{\tau}_k = \hat{t}_k - t_{i-1}$
is the sampled inter-event time.

**M6. Temporal CRPS.**
Energy-form continuous ranked probability score for the inter-event time:
$$\text{CRPS}_i = \tfrac{2}{K}\sum_k |\hat\tau_k - \tau_i| - \tfrac{1}{K^2}\sum_{k,l}|\hat\tau_k - \hat\tau_l|$$
where $\tau_i = t_i - t_{i-1}$.
The second term uses the sorted-sample estimator ($O(K\log K)$) for numerical stability.
Report mean over $N$ events. Proper scoring rule; lower is better.

**M7. Spatial energy score.**
$$\text{ES}_i = \tfrac{2}{K}\sum_k \|\hat{s}_k - s_i\| - \tfrac{1}{K^2}\sum_{k,l}\|\hat{s}_k - \hat{s}_l\|$$
Multivariate proper scoring rule for the spatial location.
Report mean over events. Lower is better.

**M8. Temporal PIT.**
Probability integral transform: $u_i = \hat{F}^*_T(\tau_i \mid \mathcal{H}_{t_{i-1}})$,
estimated as the empirical CDF of the $K$ samples evaluated at $\tau_i$.
Under a calibrated model, $\{u_i\}_{i=1}^N \sim \text{Uniform}(0,1)$.
Report the Kolmogorov–Smirnov statistic $D_N = \sup_u |F_N(u) - u|$ against the uniform.
Lower is better; $D_N \approx 0$ indicates temporal calibration.

**M9. Spatial PIT (random projections).**
For 10 independent unit directions $v_j \in \mathbb{R}^2$, project sampled locations and the true location
onto $v_j$ to obtain a 1D sample set; compute the 1D PIT and KS statistic.
Report the maximum KS statistic across directions.
Detects directional miscalibration without requiring a sequential density decomposition.

**M10. Top-$\alpha$ hotspot recall.**
On a $32\times32$ spatial grid at the conditioning time $t_{i-1}$, compute predicted density
(or KDE from samples).
Hotspot recall at level $\alpha$: fraction of test events that fall in the top-$\alpha$-fraction cells.
Report curves over $\alpha \in \{0.01, 0.05, 0.10, 0.20, 0.50\}$.

**M11. Coverage at distance $r$.**
For each test event, generate $M=200$ location samples.
$\text{Cov}(r) = \tfrac{1}{N}\sum_i \mathbf{1}[\min_m \|\hat{s}_i^{(m)} - s_i\| < r]$.
Report curves over 10 log-spaced $r$ values spanning the observation domain.

**M12. Temporal MAE.**
$\text{MAE}_T = \tfrac{1}{N}\sum_i |\tau_i - \hat\tau_i|$
where $\hat\tau_i = \text{median}(\hat\tau_{i,1},\ldots,\hat\tau_{i,K})$.

**M13. Spatial MAE / RMSE.**
$\text{MAE}_S = \tfrac{1}{N}\sum_i \|\hat{s}_i - s_i\|$,
$\text{RMSE}_S = \bigl(\tfrac{1}{N}\sum_i \|\hat{s}_i - s_i\|^2\bigr)^{1/2}$,
where $\hat{s}_i = \tfrac{1}{K}\sum_k \hat{s}_{i,k}$ (sample mean).

**M14. Joint event distance.**
Scale factor $\alpha = \text{median}(\|\Delta s_j\|) / \text{median}(\Delta t_j)$ computed from ground-truth
test-set inter-event statistics (not model outputs).
$\text{JD}_i = \bigl\|(\alpha(\tau_i - \hat\tau_i),\; s_i - \hat{s}_i)\bigr\|_2$.
Report mean. Treats events as points in normalised spacetime.

---

## Generative quality (M15–M20, M27)

All metrics share $K_\text{gen}=20$ free-running rollout sequences per test sequence.

**M15. Wasserstein $W_1$.**
For each test sequence of length $n$, generate $K_\text{gen}$ rollout sequences of the same length.
Normalise $(t, s_x, s_y)$ by test-set range.
$W_1(\text{real}, \text{generated})$ with Euclidean ground metric, computed via linear programming (POT).
Report median over test sequences. Computed on a random 30% subsample of test sequences
to control cost; fall back to Sinkhorn ($\varepsilon=0.05$) on LP failure.

**M16. Maximum Mean Discrepancy (MMD²).**
Unbiased estimator with Gaussian kernel, bandwidth from the median heuristic:
$$\widehat{\text{MMD}}^2 = \tfrac{1}{n(n-1)}\sum_{i\neq j} k(x_i, x_j)
  + \tfrac{1}{m(m-1)}\sum_{i\neq j} k(y_i, y_j)
  - \tfrac{2}{nm}\sum_{i,j} k(x_i, y_j)$$
Higher values indicate worse distributional match. Near-zero values for identical distributions.

**M17. Temporal count $\chi^2$.**
Bin $[0,T]$ into $\lceil T/\Delta t\rceil$ intervals ($\Delta t = 10\%$ of observation window).
$\chi^2 = \sum_k (N_k^\text{real} - N_k^\text{gen})^2 / N_k^\text{gen}$.
Adjacent bins merged until expected count $\geq 5$ (to validate $\chi^2$ approximation).
Also report KL divergence between normalised count distributions.

**M18. Spatial count $\chi^2$.**
Identical procedure to M17 over a $10\times10$ spatial grid.

**M19. Spatial Ripley's $K$.**
Edge-corrected $K(r)$ estimator on the real and generated event sets.
Report $L^2$ distance between $K$-function curves over $r$.
Sequences subsampled to $n_\text{max}=500$ events before computation.
Tests whether second-order spatial clustering structure is reproduced.

**M20. Temporal Ripley's $K$.**
One-dimensional analogue of M19 for inter-event time clustering.

**M27. Rollout coherence.**
Starting from each test history, generate $K_\text{gen}=20$ rollouts autoregressively.
Compute $W_1$ between generated and true continuation events at horizon $H \in \{1, 5, 20, 50\}$.
Report mean $W_1$ as a function of $H$.
Reveals whether compounding errors grow faster for some model families.

---

## Ground-truth surface comparison (M21–M26, M29) — synthetic datasets only

**M21. Intensity RMSE.**
$\text{RMSE}_\lambda = \bigl(|\mathcal{G}|^{-1}\sum_{(t,s)\in\mathcal{G}}(\hat\lambda^*(t,s) - \lambda^*_\text{true}(t,s))^2\bigr)^{1/2}$
on a $50\times50\times100$ spatiotemporal grid $\mathcal{G}$.
For intensity-queryable models: direct evaluation.
For sample-only models: 3D Gaussian KDE from $K_\text{gen}$ rollout events,
bandwidth $\max(\sigma_\text{Scott}, 0.02 \cdot \text{domain width})$ per dimension.

**M22. Intensity relative error.**
$\text{RE}_\lambda = \text{RMSE}_\lambda / \bar\lambda^*_\text{true}$.
Scale-invariant version of M21; enables cross-dataset comparison.

**M23. Intensity correlation.**
Pearson correlation between $\hat\lambda^*$ and $\lambda^*_\text{true}$ on the grid.
Tests whether the model gets the spatial *shape* right independent of overall magnitude.

**M24. Log-intensity RMSE.**
RMSE after applying $\log$, with both fields clipped to $\max(\lambda, 10^{-8})$ before taking logs.
Assesses multiplicative accuracy, which matters in regions of low background intensity.

**M25. Mass placement fraction.**
For threshold $\alpha$, identify top-$\alpha$ fraction of grid cells by $\lambda^*_\text{true}$.
Compute the fraction of $\hat\lambda^*$ mass placed in those cells.
Report over $\alpha \in \{0.10, 0.20, 0.30, 0.50\}$.
Higher is better (unlike the other grid metrics).

**M26. Background/triggering decomposition.**
Evaluate the model with an empty history to obtain the background estimate $\hat\mu(s)$.
Compare to the true background $\mu(s)$ from HawkesNest metadata.
Report separate RMSE for background and triggering (residual) components.
*Note:* for models without an explicit background/triggering structure
(e.g., DSTPP, NeuralSTPP), the empty-history baseline is a proxy, not a true decomposition;
results are annotated accordingly.

**M29. Support leakage** (topology suite only).
$\text{Leak} = \int_{\mathcal{S}^c}\hat\lambda^*(t,s)\,ds \,/\, \int_{\mathbb{R}^2}\hat\lambda^*(t,s)\,ds$
where $\mathcal{S}^c$ is the forbidden region defined in the dataset geometry mask.
Computed from the intensity grid for queryable models; from the fraction of sample
events outside the domain for sample-only models.

---

## Diagnostics (M28)

**M28. Context sensitivity.**
Re-evaluate NLL conditioning on only the $k$ most recent events, for $k \in \{0, 1, 5, 20, 100, \text{all}\}$.
Six forward passes over the test set (six pass-throughs of the history encoder with truncated inputs).
Report NLL as a function of $k$.
A model that is insensitive to $k$ uses only local history; steep improvement as $k$ grows
indicates long-range temporal dependence.

---

## Implementation notes

All metrics are registered via `@register_metric` and skipped automatically when their
`requires` set is not satisfied by the model's capabilities or the provided context.
The `EvalContext` computes shared artifacts (`samples_predictive`, `samples_generative`,
`intensity_grid`) lazily via `@cached_property` — artifacts are materialised at most once
per evaluation run regardless of how many metrics request them.

**Coordinate convention.**
Model outputs are in normalised coordinates.
Jacobian corrections are applied when reporting NLL in physical units:
$$\text{NLL}^\text{orig}_T = \text{NLL}^\text{norm}_T + \log\sigma_t, \quad
  \text{NLL}^\text{orig}_S = \text{NLL}^\text{norm}_S + \log(\sigma_{s_x}\sigma_{s_y})$$
where $\sigma$ are training-set standard deviations.

**Numerical stability.**
CRPS and energy score use sorted-sample estimators.
KDE bandwidths are floored at $0.02 \times \text{domain width}$.
Log quantities are clipped to $\max(\cdot, 10^{-8})$.
Spatial $\chi^2$ bins with expected count $<5$ are merged before computing the statistic.
