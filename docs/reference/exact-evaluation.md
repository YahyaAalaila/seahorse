# Exact evaluation metrics (`seahorse.evaluation.v1`)

This page defines every metric produced by `seahorse.evaluation.v1`, the compensator-based evaluation layer used for the paper's camera-ready results. Implementation: `seahorse/evaluation/v1/` (scoring primitives in `scoring.py`, model access in `families/*.py`, orchestration in `run_eval.py`). Run it with

```bash
python -m seahorse.evaluation.v1.run_eval --run <run_dir> --test <test.jsonl> --train <train.jsonl> --out <out_dir>
```

The legacy metrics in `seahorse/evaluation/metrics/` are kept unchanged and marked deprecated where superseded (see the [changelog](../changelog.md)).

## 0. Notation and conventions

- A test sequence is $\{(t_i, s_i)\}_{i=0}^{n-1}$, with $t_i\in\mathbb R_{\ge 0}$ and $s_i\in\mathbb R^2$, in **raw data units**. Every number we report is in raw units, whatever coordinate transform a preset was trained in. Densities carry the corresponding Jacobians.
- **Context** $k$ is the teacher-forced history $\mathcal H_k=\{(t_j,s_j)\}_{j\le k}$. Its **target** is event $k+1$, and the observed waiting time is $y_k=t_{k+1}-t_k$.
- $\lambda(t,s\mid\mathcal H)$ is the model's conditional intensity. $\lambda_g(t\mid\mathcal H)=\int_S\lambda(t,s\mid\mathcal H)\,ds$ is the ground intensity. $p(s\mid t,\mathcal H)=\lambda(t,s\mid\mathcal H)/\lambda_g(t\mid\mathcal H)$ is the spatial conditional.
- **Spatial domain $S$.** $S$ is the domain the preset's own likelihood integrates over:
  - $S=\mathbb R^2$ for density-factorised presets (Neural-STPP family, RMTPP/THP-GMM, DeepSTPP);
  - $S$ = the preset's training **support box** for the joint-intensity presets (AutoSTPP: the min–max box of the training locations; NSMPP: `support_space_min/max`).
- **Target sets.**
  - *Defined* targets are those for which the preset's likelihood defines a conditional. That is $k+1\ge 1$ for most presets, and $k+1\ge 20$ for AutoSTPP and DeepSTPP, which need a full 20-event window.
  - The **common set** $\mathcal C_{20}$ is the set of targets with index $\ge 20$, i.e. at least 20 history events.
  - **Every headline number is reported on $\mathcal C_{20}$**, so that all presets are scored on the same events. Defined-set values are stored as well.
- **Aggregation.**
  1. Per-context values are averaged over the contexts of a run (pooled over its test sequences).
  2. A *cell* (preset × dataset/level) reports the mean ± standard deviation over the completed seeds, together with the number of seeds $n$.
  3. Paired comparisons use a cluster bootstrap over test sequences (Section 9).

## 1. Negative log-likelihood (NLL)

Per target,

$$\mathrm{NLL}_k = -\log\lambda(t_{k+1},s_{k+1}\mid\mathcal H_k) + \Delta\Lambda_{k+1},\qquad \Delta\Lambda_{k+1}=\int_{t_k}^{t_{k+1}}\!\!\int_S\lambda(u,s\mid\mathcal H_k)\,ds\,du .$$

| Field | Definition |
|---|---|
| `nll_common_mean` | **Headline.** Mean over $\mathcal C_{20}$, in nats/event, raw units. Lower is better. |
| `nll_defined_mean` | Mean over the preset's defined targets. |
| `legacy_protocol_mean` | The packaged benchmark path (`compute_next_event_test_nll`) with the NLL fixes applied. It reproduces the *protocol* of the paper's `run_result.test_nll`: in 0.1.0 it averaged over all targets $1..n-1$, **including undefined targets, which it scored as 0**; since 1.0.0 undefined targets are unscored. Reported only for continuity. |

**NLL kind**, one per preset:

| Preset | `nll_kind` | Compensator |
|---|---|---|
| AutoSTPP, DeepSTPP, RMTPP-GMM | `exact` | closed form |
| Attn-CNF, Jump-CNF, NJSDE | `exact_ode` | integrated as part of the ODE state; exact up to solver tolerance ($10^{-4}$) |
| NSMPP | `numerical_quadrature` | space: 20×20 node rule on the support box; time: 2-node midpoint per inter-event interval |
| THP-GMM | `numerical_quadrature` | NLL path: 30-node midpoint. v1 PIT uses 64-node Gauss–Legendre; the difference is reported |
| SMASH, DSTPP | `framework_approx` | sample-KDE approximation: Gaussian KDE in $(\log\tau, s)$ fitted to $K$ native samples. *Not* a likelihood bound |

Consistency checks, written to `summary.json`:
- `max_abs_diff_pkg_vs_v1_defined`: on defined targets, the per-target NLL of the v1 adapters must equal the packaged per-context NLL (observed ≤ $10^{-5}$ for exact presets).
- The repository test `tests/test_nll_reporting_consistency.py` checks that the eventwise and prefix-difference routes agree.

## 2. Temporal calibration: time-rescaling PIT

$$u_k = 1-\exp(-\Delta\Lambda_{k+1}).$$

- **Theory.** Under the model, the $\Delta\Lambda$ are i.i.d. Exp(1) (time-rescaling theorem), so the $u_k$ are i.i.d. Uniform(0,1).
- **Source of $\Delta\Lambda$.** It is taken from the NLL pass, on the NLL's own domain $S$. For THP we use the 64-node Gauss–Legendre compensator and report the KS distance for the 30-node NLL rule alongside it.
- **No sampling and no K.** The PIT is a deterministic function of the model and the data.

Reported:
- `pit_ks`: the Kolmogorov–Smirnov distance $\sup_u|\hat F_n(u)-u|$ to Uniform(0,1) on $\mathcal C_{20}$, with the asymptotic p-value. Lower is better. The 95% critical value is about $1.36/\sqrt n$, i.e. ≈0.021 for $n\approx 4{,}100$ per run.
- `pit_hist`: a 20-bin histogram.
- `pit_tails`: the fractions of $u<1/33$ and $u>32/33$. These are directly comparable to the legacy "truth outside all K=32 samples" fractions. Under calibration each fraction is 0.0303.

Reading: mass near 0 means the event came *earlier* than the model predicts (intensity too low). Mass near 1 means it came *later* (intensity too high).

**Sample-based temporal PIT** (SMASH and DSTPP, and a validation for all presets) is the randomized rank PIT

$$u_k = \frac{r_k + V_k(e_k+1)}{K+1},$$

where $r_k$ = #samples $<y_k$, $e_k$ = #samples $=y_k$, and $V_k\sim U(0,1)$. It is exactly Uniform(0,1) under calibration for any $K$. Invalid samples (non-finite, or $\tau\le 0$) are excluded, and each row uses its own effective $K$.

## 3. Temporal CRPS (threshold-weighted)

$$\mathrm{CRPS}_c(F_k,y_k)=\int_0^{c}\big(F_k(\tau)-\mathbf 1\{\tau\ge y_k\}\big)^2\,d\tau,\qquad F_k(\tau)=1-\exp\!\big(-\Lambda_k(\tau)\big),\quad \Lambda_k(\tau)=\int_{t_k}^{t_k+\tau}\lambda_g(u\mid\mathcal H_k)\,du .$$

- **Propriety.** This is the CRPS with weight $w(\tau)=\mathbf 1\{\tau\le c\}$ (Gneiting & Ranjan 2011), which is a proper scoring rule. It equals the ordinary CRPS of the censored variables $\min(X,c)$ vs $\min(y,c)$.
- **Why truncate.** The truncation keeps the score finite and comparable for *defective* waiting-time laws: RMTPP's $\lambda(t)=a e^{-b(t-t_k)}$ puts mass $e^{-a/b}$ on "no further event". It also needs no tail extrapolation.
- **Threshold $c$.** $c$ is fixed per dataset, independent of every model. It is the maximum training inter-event time, rounded up to 0.1. HawkesNest Suite 3: $c=1.0$ at every level (max train IET 0.925 / 0.82). The residual survival $1-F_k(c)$ is reported (`surv_at_c_mean`).
- **Exact estimator** (intensity presets, `method = exact_compensator`):
  1. $\Lambda_k$ is computed on the grid $\tau_j=4c\,(j/1023)^2$, $j=0..1023$, with one batched solve or closed-form evaluation per sequence.
     - Neural-STPP: one ODE solve of $(\Lambda,h)$ from the post-jump state at $t_k$.
     - AutoSTPP, DeepSTPP, RMTPP: closed form.
     - THP: Gauss–Legendre.
     - NSMPP: cumulative trapezoid of the NLL's spatial node rule.
  2. $F$ is linear between grid nodes, and the piecewise-quadratic integral is evaluated exactly.
  3. Grid error is below $2\cdot10^{-6}$ for the exponential reference (test `test_crps_grid_matches_exponential_closed_form`).
  4. Built-in check: $\Lambda_k(y_k)$ interpolated on the grid must equal the NLL-path $\Delta\Lambda_{k+1}$ (`check_grid_vs_nll_compensator`).
- **Sample estimator** (SMASH, DSTPP; `method = native_samples`): the fair estimator

  $$\widehat{\mathrm{CRPS}}=\frac1K\sum_i|x_i-y|-\frac{1}{2K(K-1)}\sum_{i\ne j}|x_i-x_j|,\qquad x_i=\min(\tau_i,c),\ y\leftarrow\min(y,c),$$

  which is unbiased for $\mathrm{CRPS}_c$ (Ferro 2014).
- **Validation.** For every intensity preset we also draw $K$ exact samples by inverse transform from the same $\Lambda_k$ grid and report the fair sample CRPS on the same contexts. It must agree with the exact value up to Monte Carlo error.

Units are raw time. Lower is better.

## 4. Spatial calibration

### 4a. Conditional on the true event time: HPD-PIT

$$u_k = \Pr_{S\sim p(\cdot\mid t_{k+1},\mathcal H_k)}\big[p(S)\ge p(s_{k+1})\big].$$

The mass of the highest-density region that just excludes the observation. Under calibration $u_k\sim U(0,1)$. Reading: $u\to 1$ means observations fall in low-density regions (mass misplaced or too concentrated). $u\to 0$ means they fall in the mode more often than expected (too dispersed).

- **Estimator.** A randomized rank of $g=-\log p$ at the observation among $K$ exact draws $S_1..S_K$ with their exact log-densities, $u=(r+V(e+1))/(K+1)$. It is exactly uniform under calibration for every $K$ (exchangeability), and it avoids grid quadrature for densities on $\mathbb R^2$.
- **Draws and densities by family:**
  - **Neural-STPP.**
    - NJSDE: exact per-dimension GMM sampling.
    - Jump-CNF, Attn-CNF: exact **inverse-flow** sampling (base draw → reverse ODE/radial maps; `families/neural_flows.py`), with the log-density tracked along the inverse path. Invertibility is checked numerically: the tracked log-density must equal the model's own forward `conditional_logprob_fn` at the draw. Error ≤ $10^{-4}$ in tests; reported per run.
  - **RMTPP/THP-GMM, DeepSTPP.** Exact Gaussian-mixture sampling and density.
  - **AutoSTPP, NSMPP** (bounded $S$). Piecewise-constant sampling on a 128×128 cell grid of $S$, with cell values $\lambda(t,\text{centre})$. The rank uses the same piecewise-constant density for the draws and for the observation (`spatial_sampler = fine_grid`).
- `s_hpd_cond_ks` is the KS distance to uniform. Not available for SMASH/DSTPP (no density).
- `s_es_cond` is the fair energy score of the $K$ conditional draws vs $s_{k+1}$ (raw units; lower is better).

### 4b. Marginal over the next time: comparable across all presets

The draws $(\tau_i,S_i)$ come from the model's *joint* next-event law:
- **Intensity presets:** $\tau_i$ by inverse transform of $\Lambda_k$, then $S_i\sim p(\cdot\mid t_k+\tau_i,\mathcal H_k)$ by the exact sampler above.
- **SMASH and DSTPP:** their native samplers.

Reported:
- `s_pit_x_ks`, `s_pit_y_ks`, and their mean `s_pit_marg_ks_mean`: KS distance of the randomized rank PIT of each coordinate.
- `s_es_marg_mean`: fair energy score of $S_i$ vs $s_{k+1}$.

**Comparability caveat.** SMASH's Langevin sampler clamps to the training min–max box, and the grid samplers are restricted to $S$. Mass outside the data box is therefore truncated for these presets but not for the $\mathbb R^2$ presets. On HawkesNest all events lie in $[0,1]^2$, so truncation can only *help* the box-restricted presets.

## 5. Intensity recovery (synthetic only)

This uses the ground-truth conditional intensity $\lambda^*$ of the generator (`gt_hawkesnest.py`). It reproduces the stored reference grids to $10^{-6}$, conditioned on the test-window history.

**Evaluation grid.**
- For **all** test sequences: 50 equispaced time slices in $(t_{m-1},t_{n-1})$, with $m$ the preset's minimum history.
- Spatial points: the 32×32 cell centres of $[0,1]^2$.
- Model: $\lambda(t,s\mid\mathcal H_{t^-})$ in raw units (events per unit time per unit area). The context at slice $t$ is the last event strictly before $t$.

| Metric | Definition | Direction |
|---|---|---|
| `slice_pearson_mean` | **Headline.** Mean over slices of the spatial Pearson correlation at fixed $t$ (shape recovery, unaffected by temporal-rate mismatch) | higher is better |
| `pooled_pearson` | Pearson over all (slice, point) pairs; the legacy definition, now over all sequences | higher is better |
| `spearman` | Pooled rank correlation | higher is better |
| `log_rmse` | $\sqrt{\mathrm{mean}(\log\lambda-\log\lambda^*)^2}$, raw units (same units for every preset) | lower is better |
| `mass_ratio_model_over_true` | $\sum\lambda\,\Delta A\,\Delta t\,/\sum\lambda^*\,\Delta A\,\Delta t$ | 1 is ideal |

`mass_ratio_true_over_events` is a sanity check of the ground truth itself, against the observed event count.

SMASH and DSTPP are excluded (no intensity).

## 6. Autoregressive rollouts (proper scores)

**Protocol.**
- **Starts.** Start contexts are $k_0 = 49, 89, 129,\dots$ (stride 40, so $k_0+1\ge 50$ history events) while $k_0+H<n-1$.
- **Trajectories.** For each start, $M=50$ independent trajectories of $H=10$ events are generated with the *exact* samplers of Sections 3–4: each step samples the next event from $\mathcal H_{k_0}\cup\{\text{generated so far}\}$ and appends it.
- **SMASH, DSTPP:** native samplers.

**Scores.** For $h\in\{1,5,10\}$, with time expressed as offsets from the conditioning end $t_{k_0}$:
- `es_time_h{h}`: fair energy score of the $h$-vectors of cumulative offsets $(t_{k_0+1}-t_{k_0},\dots,t_{k_0+h}-t_{k_0})$. Raw time units; lower is better.
- `es_space_h{h}`: fair energy score of the $2h$-vectors of locations $(s_{k_0+1},\dots,s_{k_0+h})$. Raw space units; lower is better.

Order is preserved: the energy score compares vectors, not point sets. Both are proper scores of the joint law of the next $h$ events. Time and space are kept separate to avoid an arbitrary unit weighting.

**Failures.** A trajectory is *invalid* if a time draw falls beyond the grid ($\tau>4c$) or a native draw has $\tau\le0$. `frac_invalid` is reported, and invalid trajectories are excluded.

The legacy `rollout_coherence` (median $W_1$ between pooled generated points and one real continuation) is **deprecated**. It is a distance, so **lower is better**. It is not a proper score, treats the $H$ events as an unordered set, and compares against a single real continuation. Legacy values are kept only for continuity.

## 7. Sample sizes and randomness

- $K=200$ draws per context for all non-flow presets.
- Flow presets (Jump-CNF, Attn-CNF): $K=64$ on every 4th context.
  - Reason: the exact inverse flow costs a joint ODE solve over the full history per draw.
  - Every rank PIT above is exactly uniform for any $K$, so a smaller $K$ costs power, not validity.
- All randomness is seeded via `derive_seed(0, preset, section)`.

## 8. Not computed / unavailable

- Intensity recovery for SMASH and DSTPP.
- HPD-PIT for SMASH and DSTPP.
- Exact temporal PIT/CRPS for SMASH and DSTPP: they are sample-based (Section 2, 3).

## 9. Uncertainty and comparisons

- **Cells.** Mean ± sd over seeds, always with $n$ seeds.
- **Paired comparisons of two presets A, B on one level.** Per-target NLL (or CRPS) is first averaged over each preset's seeds. The difference $d=\bar\ell_A-\bar\ell_B$ is then averaged over $\mathcal C_{20}$. A 95% interval comes from a cluster bootstrap over the 10 test sequences (5,000 resamples).
- A comparison is called **resolved** only if the interval excludes 0.

## 10. Calibration from the NLL compensator (`seahorse campaign calibrate`)

For the frozen fixed-history campaign cells, `seahorse.evaluation.v1.calibration` computes the
temporal PIT (Section 2), the threshold-weighted CRPS and predictive quantiles (Section 3) and the
spatial HPD-PIT (Section 4a) **without any sampler**, from the same forward pass that produces the
reported NLL: $\Delta\Lambda_i$ is read from the model's own likelihood module (AutoSTPP exposes it
directly; DeepSTPP, the Neural-STPP ODE family, the factorized GMM baselines, NSMPP, the parametric
Hawkes/Poisson baselines and the statistical presets recover it from their compensators), and the
cumulative hazard $\Lambda_i(\tau)$ on a raw-unit waiting-time grid gives $F_i(\tau)=1-\exp(-\Lambda_i(\tau))$.
Per-cell identity checks against the NLL matrices and a reproduction of the stored per-target NLL are
written to `summary.json`; `per_target.npz` holds the per-target arrays. CNF spatial decoders are not
covered by the spatial route. The threshold $c$ is the maximum training inter-event time, rounded up.

```bash
seahorse campaign calibrate --manifest <campaign>/manifest.json --job <job_id> --out <dir>
```
