# Paper Results

<p class="sh-ds-eyebrow">NeurIPS 2026 · 19 presets · 13 real datasets · HawkesNest entanglement suite</p>

The NeurIPS 2026 paper runs 19 presets (6 classical, 5 factorized, 6 neural STPPs and 2
generative models) on 13 real datasets with three seeds each, under one protocol: fixed
chronological splits (or the original splits of the three Neural STPP datasets), every model scored
on the same target events with the same 20 preceding events, and test NLL reported in raw units
(time in days, space in degrees). The short answer: **no model dominates.** This page collects the
headline findings and the aggregate figures; the commands that reproduce them are in
[Reproduce Paper Results](paper-reproduction.md), and every dataset has its own
[card](datasets/index.md).

<div class="sh-stat-row" markdown="0">
  <div class="sh-stat"><span class="sh-stat-num">6 / 4 / 2 / 1</span><span class="sh-stat-label">datasets won by Attn-CNF / AutoSTPP / DeepSTPP / Jump-CNF (test NLL)</span></div>
  <div class="sh-stat"><span class="sh-stat-num">W = 0.86</span><span class="sh-stat-label">Kendall's W of the within-dataset ranks: fairly consistent, no preset best everywhere</span></div>
  <div class="sh-stat"><span class="sh-stat-num">0.17 nats</span><span class="sh-stat-label">median gap of DeepSTPP to the best model, at about seven minutes of training per seed</span></div>
  <div class="sh-stat"><span class="sh-stat-num">≥ 7 of 10</span><span class="sh-stat-label">benchmark datasets on which the best-calibrated model is not the likelihood winner</span></div>
</div>

## Real-data likelihood

<figure class="sh-figure" markdown="0">
  <img src="../assets/figures/main_results.png" alt="Two panels. Left: gap in test NLL to the best preset on each of the 13 real datasets, per preset, on a log scale. Right: median training wall-clock per seed against mean normalized NLL rank; DeepSTPP, AutoSTPP, Attn-CNF and Jump-CNF have the best ranks, DeepSTPP at minutes, the CNF models at hours." loading="lazy">
  <figcaption>(a) Gap in test NLL to the best preset on each of the 13 real datasets (nats/event, log scale); dot: one dataset, bar: median over datasets. (b) Median training time per seed (wall-clock on heterogeneous GPUs; bar: interquartile range) against the mean normalized within-dataset NLL rank (0 = best). Generative models (hollow) are placed by approximate likelihoods, for orientation only. Figure from the NeurIPS 2026 paper (Fig. 4).</figcaption>
</figure>

- **The ranking depends on the domain.** Attn-CNF is best on COVID, Citi Bike (daily) and four of
  the urban datasets (Seattle, NYC collisions, Chicago, LA); DeepSTPP on Earthquakes and Austin;
  AutoSTPP on ComCat, GeoNet, Citi Bike (hourly) and NYC 311; Jump-CNF on Uber.
- **Classical self-excitation still competes.** The classical S-Hawkes beats NSMPP on every
  dataset. Poisson and self-correcting variants are consistently weaker.
- **Cost changes the picture.** DeepSTPP is within a median 0.17 nats of the best model at about
  seven minutes of training per seed; Attn-CNF and Jump-CNF reach 0.15 and 0.09 at six to seven
  hours, and Jump-CNF diverged on two datasets.

<figure class="sh-figure" markdown="0">
  <img src="../assets/figures/rank_overview.png" alt="Heatmap of the rank of each exact or numerical-likelihood preset within each of the 13 real datasets, with a side panel of mean normalized rank and bootstrap intervals." loading="lazy">
  <figcaption>Within-dataset NLL ranks: (a) rank per dataset (1 = best; dark: diverged), (b) mean normalized rank with the median and 90% interval of a bootstrap over datasets and seeds. Kendall's W = 0.86 over the 16 presets ranked on all 13 datasets; Jump-CNF is ranked on 11. Figure from the NeurIPS 2026 paper (appendix, Fig. 8).</figcaption>
</figure>

Generative models have no exact likelihood. Their table entries are a variational bound (DSTPP)
or an approximation (SMASH, temporal term plus a grid estimate of the spatial density), are shown
for orientation and are not ranked. The scripts that compute them are in `scripts/approx_nll/`.

## Calibration and next-event prediction

- On the ten chronological datasets, the best-calibrated model (lowest temporal PIT statistic)
  differs from the likelihood winner on at least seven.
- Sample-based energy scores of the next event mostly preserve the likelihood ranking (median
  Spearman correlation 0.88), with exceptions: DeepSTPP has the best mean rank (2.1 of 19),
  AutoSTPP, among the top three under NLL, is mid-field (8.3), and SMASH, which has no exact
  likelihood, ranks ahead of it (7.2).

The metrics are defined in [Exact Evaluation Metrics](reference/exact-evaluation.md); they come
from the same compensator as the NLL, so no sampler is needed for the intensity-based presets.

## Controlled entanglement (HawkesNest)

The [HawkesNest entanglement suite](datasets/cards/hawkesnest_entanglement.md) is a
spatiotemporal Hawkes process on the unit square whose triggering kernel drifts in space with the
age of the parent event (σ = 0.03, β = 0.3, drift speed v = 0, 0.1, 0.2, 0.3 for levels L0–L3),
while the background and the branching ratio stay fixed. Because the true process is known, the
table reports the NLL **in excess of the true process**:

| Model | L0 | L1 | L2 | L3 |
| --- | ---: | ---: | ---: | ---: |
| Separable Hawkes fit (reference) | 0.00 | 0.08 | 0.17 | 0.25 |
| **AutoSTPP** | **0.04** | **0.06** | **0.07** | **0.10** |
| DeepSTPP | 0.21 | 0.35 | 0.50 | 0.63 |
| NSMPP | 1.45 | 1.46 | 1.50 | 1.56 |
| NJSDE | 0.96 | 0.97 | 0.93 | 0.95 |
| RMTPP | 2.25 | 2.29 | 2.32 | 2.40 |
| THP | 2.14 | 2.18 | 2.21 | 2.29 |
| DSTPP (variational bound) | 3.29 | 3.32 | 3.33 | 3.35 |

Test NLL in excess of the true process (nats/event, mean over three seeds; paper Table 3). SMASH
has no likelihood.

<figure class="sh-figure" markdown="0">
  <div class="sh-fig-grid">
    <img src="../assets/figures/hn_budget_auto_stpp.png" alt="AutoSTPP validation excess NLL against training budget, one curve per level L0 to L3." loading="lazy">
    <img src="../assets/figures/hn_budget_deep_stpp.png" alt="DeepSTPP validation excess NLL against training budget, one curve per level." loading="lazy">
    <img src="../assets/figures/hn_budget_nsmpp.png" alt="NSMPP validation excess NLL against training budget, curves tightly packed." loading="lazy">
    <img src="../assets/figures/hn_budget_njsde.png" alt="NJSDE validation excess NLL against training budget, curves tightly packed." loading="lazy">
  </div>
  <figcaption>Validation NLL in excess of the true process against training budget for AutoSTPP, DeepSTPP, NSMPP and NJSDE (left to right, top to bottom), one curve per level L0–L3; dotted lines for AutoSTPP and DeepSTPP mark the separable reference of each level. Figure from the NeurIPS 2026 paper (Fig. 5).</figcaption>
</figure>

- **AutoSTPP is the only model below the separable reference at L1–L3**: it parameterizes the
  joint intensity over time and space with an exact compensator, so it can absorb part of the
  coupling.
- DeepSTPP is second at every level, but its excess grows from 0.21 to 0.63, faster than the
  separable reference.
- NSMPP, NJSDE and the factorized baselines stay 0.9–2.4 nats from the true process at every
  level, and their excess rises by at most 0.15 nats from L0 to L3.
- Intensity recovery: DeepSTPP's correlation with the true intensity falls from 0.94 to 0.65 as the
  coupling grows, while AutoSTPP's stays above 0.95.

<figure class="sh-figure sh-figure--narrow" markdown="0">
  <img src="../assets/figures/hawkesnest_intensity.png" alt="Line chart of the correlation between model and true intensity across entanglement levels L0 to L3; AutoSTPP stays near 1, DeepSTPP falls, RMTPP and THP stay at the background-only level." loading="lazy">
  <figcaption>Ground-truth intensity recovery on the HawkesNest entanglement suite: correlation between the model and the true intensity across levels (mean over 3 seeds). Dashed: a model that knows only the true background. Figure from the NeurIPS 2026 paper (Fig. 6).</figcaption>
</figure>

A Hawkes generator favours models whose inductive bias matches it, so the absolute ranking on this
suite is not generator-independent; the suite diagnoses how models degrade as the coupling grows.

## Cost and reliability

<figure class="sh-figure" markdown="0">
  <img src="../assets/figures/runtime_reliability.png" alt="Per-seed final state of every real-data fit, by preset and dataset: completed, completed with caveat, excluded, or diverged." loading="lazy">
  <figcaption>Final state of every seed of the real-data campaign. Figure from the NeurIPS 2026 paper (appendix, Fig. 9a).</figcaption>
</figure>

<figure class="sh-figure" markdown="0">
  <img src="../assets/figures/runtime_compute.png" alt="Bar chart of device-hours per preset, largest for Attn-CNF and Jump-CNF, and a dot plot of peak GPU memory per seed." loading="lazy">
  <figcaption>Device-hours per preset summed over the 13 datasets (final fits and tuning trials) and peak allocated GPU memory per seed. Figure from the NeurIPS 2026 paper (appendix, Fig. 9b).</figcaption>
</figure>

Of the **741 real-data fits** (19 presets × 13 datasets × 3 seeds), 715 completed without caveat,
17 are reported with a caveat (fewer seeds, the 72-hour budget limit, or a failed numerical check),
2 Attn-CNF seeds were excluded after the 72-hour limit without a usable checkpoint, and 7 Jump-CNF
seeds diverged (six in every tuning trial on Citi Bike (hourly) and NYC 311, one during the refit
on ComCat). Every gradient-trained preset had the same budget: three tuning trials of 5,000 steps
selected on validation NLL, then a refit of at most 20,000 steps with early stopping.

## Where to go next

- [Reproduce Paper Results](paper-reproduction.md): campaign specifications and commands.
- [Add your model and benchmark it](benchmark/your-model.md): put a new model on the same footing.
- [Dataset cards](datasets/index.md): the 13 real datasets and the HawkesNest suite.
