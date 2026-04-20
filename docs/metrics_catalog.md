# STPP Evaluation Metrics Catalog

This document lists every metric we may compute for a trained STPP model on a test set. Metrics are grouped by what aspect of model behavior they probe. Compute all that are feasible for each model family; analysis will select the informative ones.

**Notation.** Test set has $N$ events $\{(t_i, s_i)\}_{i=1}^N$ with histories $\mathcal{H}_{t_i}$. Model produces some subset of: $\lambda^*(t,s \mid \mathcal{H})$, $f^*(t,s \mid \mathcal{H})$, samples $\{(\hat{t}_k, \hat{s}_k)\}$. Raw physical coordinates throughout (apply Jacobian correction to native NLL).

---

## Tier 1: Fit quality — how well does the model explain observed data

### M1. Raw-space NLL (primary metric)

**Math:** $\text{NLL} = -\frac{1}{N}\sum_{i=1}^N \log f^*(t_i, s_i \mid \mathcal{H}_{t_i})$ with Jacobian correction to raw coordinates.

**Assesses:** Overall probabilistic fit. The standard metric in the literature.

**Available for:** All models, but computation differs. Exact for AutoSTPP/DeepSTPP/NeuralSTPP; variational bound for DSTPP; KDE-from-samples for SMASH.

**Annotation:** Superscript indicates method ($^\text{exact}$, $^\text{VB}(T)$, $^\text{KDE}(K)$, $^\text{quad}(n)$).

### M2. Temporal NLL

**Math:** $\text{NLL}_T = -\frac{1}{N}\sum_i \log f^*_T(t_i \mid \mathcal{H}_{t_i})$ where $f^*_T(t) = \int_\mathcal{S} f^*(t, s) \, ds$.

**Assesses:** Isolates temporal modeling quality from spatial. A model can be good temporally but poor spatially (or vice versa) and M1 alone won't distinguish them.

**Available for:** Factorized models directly. Joint models need marginalization (MC or quadrature).

### M3. Spatial NLL

**Math:** $\text{NLL}_S = -\frac{1}{N}\sum_i \log f^*_S(s_i \mid t_i, \mathcal{H}_{t_i})$ where $f^*_S(s \mid t) = f^*(t,s) / f^*_T(t)$.

**Assesses:** Isolates spatial modeling quality. Pair with M2 to diagnose where NLL failures come from.

### M4. NLL train/test gap

**Math:** $\Delta\text{NLL} = \text{NLL}_{\text{test}} - \text{NLL}_{\text{train}}$.

**Assesses:** Overfitting. Flexible models (score-based) should show larger gaps than structured models under small $N$.

### M5. NLL gap to ground truth (synthetic only)

**Math:** $\Delta^* = \text{NLL}_{\text{model}} - \text{NLL}_{\text{true process}}$ where the true process NLL is computed from the known generating intensity.

**Assesses:** Approximation error separated from irreducible stochasticity. Two models with the same raw NLL can have different $\Delta^*$ if one is closer to the Bayes-optimal predictor.

---

## Tier 2: Predictive quality — how well does the model forecast the next event

### M6. Temporal CRPS (continuous ranked probability score)

**Math:** For each event $i$, with $\tau_i = t_i - t_{i-1}$ the true inter-event time:
$$\text{CRPS}_i = \int_0^\infty \left(\hat{F}^*_T(\tau \mid \mathcal{H}_{t_{i-1}}) - \mathbb{1}[\tau \geq \tau_i]\right)^2 d\tau$$
Estimated from $S \geq 200$ samples via the empirical CDF.

**Assesses:** Proper scoring rule for temporal prediction. Rewards both calibration and sharpness. Unlike NLL, robust to heavy tails.

**Available for:** Any model that can sample temporal component.

### M7. Spatial energy score

**Math:** $\text{ES}_i = 2\,\hat{\mathbb{E}}\|X - s_i\| - \hat{\mathbb{E}}\|X - X'\|$ where $X, X' \sim \hat{f}^*_S(\cdot \mid t_i)$ independently, estimated from $S \geq 200$ samples.

**Assesses:** Spatial analog of CRPS. Proper scoring rule for multivariate distributions.

### M8. Temporal PIT (probability integral transform)

**Math:** $u_i = \hat{F}^*_T(\tau_i \mid \mathcal{H}_{t_{i-1}})$. Under a perfectly calibrated model, $\{u_i\} \sim \text{Uniform}(0,1)$.

**Assesses:** Temporal calibration. Report: Kolmogorov-Smirnov statistic $D_n = \sup_u |F_n(u) - u|$ and histogram.

### M9. Spatial PIT via random projections

**Math:** For 10 random unit directions $\{v_k\}$: project $s_i$ and sampled spatial distribution onto $v_k$, compute 1D PIT. Report max KS statistic across directions.

**Assesses:** Spatial calibration without requiring conditional density sequential decomposition.

### M10. Top-$k$ hotspot recall

**Math:** Compute predicted spatial density on a grid at time $t_i$. Take the top-$\alpha$ fraction of grid cells by predicted density (e.g., $\alpha = 0.1$). Hotspot recall $= \frac{1}{N}\sum_i \mathbb{1}[s_i \in \text{top-}\alpha\text{ cells}]$.

**Assesses:** Operational forecasting utility. A model with good NLL can have poor top-$k$ recall if it spreads mass thinly. Report curves over $\alpha \in [0.01, 0.5]$.

### M11. Coverage at distance $r$

**Math:** Generate $M$ predictions per test event. $\text{Cov}(r) = \frac{1}{N}\sum_i \mathbb{1}[\min_m \|\hat{s}_i^{(m)} - s_i\| < r]$.

**Assesses:** "If I deployed resources to model-predicted hotspots, what fraction of events land within $r$?" Directly operational.

---

## Tier 3: Point-estimate prediction — distance-based errors

### M12. Temporal MAE

**Math:** $\text{MAE}_T = \frac{1}{N}\sum_i |t_i - \hat{t}_i|$ where $\hat{t}_i$ is the predicted expected or median next-event time.

**Assesses:** Point-estimate temporal accuracy. Coarse but standard.

### M13. Spatial MAE / RMSE

**Math:** $\text{MAE}_S = \frac{1}{N}\sum_i \|s_i - \hat{s}_i\|_2$, $\text{RMSE}_S = \sqrt{\frac{1}{N}\sum_i \|s_i - \hat{s}_i\|_2^2}$.

**Assesses:** Point-estimate spatial accuracy. Penalizes models that hedge with broad distributions.

### M14. Joint event distance

**Math:** $\text{JD}_i = \|(\alpha(t_i - \hat{t}_i), s_i - \hat{s}_i)\|_2$ with $\alpha$ a temporal rescaling factor (median spatial / median temporal scale). Report mean.

**Assesses:** Treats events as points in normalized spacetime. Prevents temporal or spatial dominance artifacts from different units.

---

## Tier 4: Generative quality — does the model produce realistic sequences

### M15. Wasserstein distance $W_1$

**Math:** Generate $K = 20$ sample sequences of matching length to each test sequence. $W_1$(real, generated) with Euclidean ground metric on $(t, s_x, s_y)$ normalized to unit scales. Report median over sequences.

**Assesses:** Distributional similarity, sparsity-robust (unlike binned metrics).

### M16. Maximum Mean Discrepancy (MMD)

**Math:** Gaussian kernel with bandwidth from median heuristic. Estimator: $\text{MMD}^2 = \frac{1}{n(n-1)}\sum_{i\neq j} k(x_i, x_j) + \frac{1}{m(m-1)}\sum_{i\neq j} k(y_i, y_j) - \frac{2}{nm}\sum_{i,j}k(x_i, y_j)$.

**Assesses:** Distributional similarity at all scales determined by the kernel. Complements $W_1$.

### M17. Event count statistics — temporal

**Math:** Partition $[0, T]$ into bins of width $\Delta t$. Let $N_k^{\text{real}}, N_k^{\text{gen}}$ be counts. Report $\chi^2 = \sum_k (N_k^{\text{real}} - N_k^{\text{gen}})^2 / N_k^{\text{gen}}$ and KL divergence between normalized count distributions.

**Assesses:** Whether the model reproduces first-order temporal statistics.

### M18. Event count statistics — spatial

**Math:** Same as M17 but spatial bins over $\mathcal{S}$.

**Assesses:** Whether the model reproduces first-order spatial statistics.

### M19. Ripley's K-function

**Math:** $K(r) = \frac{1}{\hat{\lambda}} \mathbb{E}[\text{number of other events within } r]$. Compute for real and generated point sets; report $L^2$ distance between curves over $r$.

**Assesses:** Second-order spatial structure (clustering vs. regularity). A model can match first-order statistics (M18) while missing clustering.

### M20. Temporal Ripley's K (or pair correlation)

**Math:** Analog of M19 for temporal clustering.

**Assesses:** Whether the model reproduces temporal clustering structure (e.g., self-excitation).

---

## Tier 5: Ground-truth comparison (synthetic only)

### M21. Intensity RMSE

**Math:** On a spatiotemporal grid $\mathcal{G}$ of size $100 \times 100 \times 200$:
$$\text{RMSE}_\lambda = \sqrt{\frac{1}{|\mathcal{G}|}\sum_{(t,s) \in \mathcal{G}} \left(\hat{\lambda}^*(t,s) - \lambda^*_{\text{true}}(t,s)\right)^2}$$

**Assesses:** Geometric fidelity of the intensity surface. A model can have good NLL while placing mass in wrong locations if the overall magnitude is right.

**Available for:** Intensity-queryable models. For sample-only models, compute via KDE from samples.

### M22. Intensity relative error

**Math:** $\text{RE}_\lambda = \text{RMSE}_\lambda / \bar{\lambda}^*_{\text{true}}$.

**Assesses:** Scale-invariant version of M21 for cross-dataset comparison.

### M23. Intensity correlation

**Math:** Pearson correlation between $\hat{\lambda}^*$ and $\lambda^*_{\text{true}}$ on the grid.

**Assesses:** Whether the model gets the *shape* right even if magnitude is off. Decouples calibration from structure.

### M24. Log-intensity RMSE

**Math:** Same as M21 but on $\log \lambda^*$.

**Assesses:** Relative (multiplicative) error, which is what matters for regions of low intensity.

### M25. Spatial mass placement error

**Math:** On ground-truth intensity, identify top-$\alpha$ mass regions. Compute fraction of predicted mass placed in those regions. Report over $\alpha \in [0.1, 0.5]$.

**Assesses:** Whether mass is in the right places, ignoring magnitude.

---

## Tier 6: Failure mode diagnostics

### M26. Decomposed error — background vs triggering (synthetic only)

**Math:** If the generating process has an explicit background + triggering decomposition $\lambda^* = \mu(s) + \sum_j g(t-t_j, s-s_j)$, compute intensity RMSE separately for the two components by ablating history. Requires running the model with empty history (gives $\mu$) and comparing to the true background.

**Assesses:** Whether errors come from misspecified background, misspecified triggering, or both. Directly tests the additive inductive bias of models like AutoSTPP.

### M27. Sequential coherence over rollout horizon $H$

**Math:** Given a history, autoregressively generate $H$ events. Compute $W_1$ between generated and real continuation. Report curves over $H \in \{1, 5, 20, 50\}$.

**Assesses:** Compounding errors over multi-step prediction. Models with weak autoregressive structure should degrade faster.

### M28. Context sensitivity curve

**Math:** At test time, condition each prediction on only the $k$ most recent events. Report NLL as function of $k \in \{0, 1, 5, 20, 100, \text{all}\}$.

**Assesses:** How much local history each model needs. Reveals whether long-range dependencies are being used or ignored.

### M29. Support leakage (for bounded domains)

**Math:** $\text{Leak} = \int_{\mathcal{S}^c} \hat{\lambda}^*(t, s) \, ds / \int_{\mathbb{R}^2} \hat{\lambda}^*(t, s) \, ds$ where $\mathcal{S}^c$ is outside the observation region (or a forbidden subregion).

**Assesses:** Whether the model respects geometric constraints of the domain. Gaussian-mixture methods leak most.

### M30. Near-boundary calibration

**Math:** PIT values (M8, M9) computed separately for events near the domain boundary vs. interior.

**Assesses:** Whether models fail systematically at the edges where their parametric assumptions may break down.

---

## Tier 7: Computational profile (recorded per experiment, not a "metric" but essential)

### C1. Training wall-clock time
### C2. Training GPU memory peak
### C3. Number of parameters
### C4. Inference time per test event (NLL evaluation)
### C5. Inference time per generated event (sampling)
### C6. Inference memory per test batch

**Rationale:** A model that is "better" by 0.01 nats but 100× more expensive is a different proposition than one that is equally expensive. Computational cost changes the story.

---

## What to compute per experiment

Aim to compute as many metrics as are feasible for each model, even if they aren't all analyzed. Storage is cheap; re-running experiments is expensive.

**Always compute:** M1, M2, M3, M4, M12, M13, M14, C1–C6.

**Compute when sampler is available:** M6, M7, M8, M9, M10, M11, M15, M16, M17, M18, M19, M20, M27.

**Compute when ground truth is available (synthetic):** M5, M21, M22, M23, M24, M25, M26, M29.

**Always compute for all models on all datasets:** M28 (context sensitivity is cheap and reveals a lot).

**Store:** Raw per-event NLL values (not just the mean), raw samples, computed PIT values. These enable post-hoc statistical tests and re-analysis without re-training.
