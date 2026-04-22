# Model Parameters Lock

**Status:** Pre-experiment audit — working document for locking HPO and training setup.
**Date:** 2026-04-15
**Scope:** All benchmark-eligible presets plus provisional neural variants.
**Benchmark-eligible (canonical):** `auto_stpp`, `deep_stpp`, `smash`
**Provisional (excluded from headline):** `neural_attncnf`, `neural_jumpcnf`, `neural_cond_gmm`
**Experimental:** `nsmpp`

---

## 1. Purpose and Locking Policy

This document is the single source of truth for:
- What hyperparameters are set for each model, and why
- What is locked globally across models for fairness
- What is model-specific and intentionally different
- What still needs resolution before experiments are launched

### Locking levels
| Level | Meaning |
|---|---|
| **GLOBAL-LOCKED** | Same value◊ enforced across all models; changing it would make cross-model NLL comparisons unfair |
| **MODEL-LOCKED** | Fixed per model, not tuned; deviation from upstream requires justification |
| **HPO-TUNED** | Searched over per model per dataset; best config persisted in run artifact |
| **UNRESOLVED** | Decision pending; experiments should not depend on it being stable |

---

## 2. Global Benchmark Training Policy

These apply to every model in every benchmark run. They are enforced by `Benchmark._apply_data_contract()` for data, and by convention in all YAML configs for training.

| Aspect | Value | Justification |
|---|---|---|
| Data protocol | `unified`, `normalize: true` | Enforced by `Benchmark._apply_data_contract()`. Per-event NLL is comparable only when all models see identically scaled inputs. |
| NLL reporting space | Normalized z-score space | Locked by design. See memory note on NLL metric definition. |
| Gradient clipping | `grad_clip: 1.0` | Recommended global lock. Paper defaults vary: deep_stpp=1.0, neural=0.0, auto_stpp=1.0. A common moderate clip prevents outlier instability without changing gradient direction under normal training. **Currently inconsistent in YAMLs — see per-model notes.** |
| Checkpoint selection | `best` (lowest `val/nll`) | Locked. All models use best-val-NLL checkpoint. |
| Seed | `data.seed: 42` | Locked for data splits. Model init seed not currently controlled globally. |
| Test NLL space | `test_nll_space: raw` | Locked for all models that support it. Ensures test numbers reflect original-space quality. |

**What is NOT globally locked (deliberately):**
- Optimizer family (Adam vs AdamW vs Adadelta — model-specific, justified below)
- LR value and schedule (HPO-tuned or model-specific; same LR across architecturally different models is not meaningful)
- batch_size (model-specific; auto_stpp and deep_stpp need 128 for paper-faithful behavior; NeuralSTPP was designed with event-budget batching)
- Number of epochs (model-specific; NeuralSTPP needs 200, AutoSTPP/DeepSTPP use 50)

---

## 3. Per-Model Sections

---

### 3.1 AutoSTPP (`auto_stpp`)

**Paper:** Zhou & Yu, "AutoSTPP: Efficient Spatiotemporal Point Process with Automatic Integration", NeurIPS 2023.
**Preset status:** canonical

#### A. Upstream defaults

From paper and upstream repo (`Rose-STL-Lab/AutoSTPP`):

| Parameter | Paper/upstream | Source |
|---|---|---|
| Optimizer | Adam | paper |
| lr | 4.0e-3 | paper (Table in appendix) |
| adam_beta1 | 0.9 | paper |
| weight_decay | 0.0 (not specified) | paper |
| grad_clip | not specified | paper |
| batch_size | 128 | paper |
| n_epochs | 50 | paper |
| lr_schedule | StepLR (step_size=20, gamma=0.5) | upstream code |
| n_prodnet | 10 | upstream code default |
| hidden_size | 128 | upstream code |
| num_layers | 2 | upstream code |
| lookback / max_history | 20 | upstream code |
| lookahead | 1 | upstream code |
| activation | tanh | upstream code |

#### B. Repo current state

File: `unified_stpp/configs/auto_stpp.yaml`

| Parameter | Repo value | Status |
|---|---|---|
| lr | **0.008149663** | STALE HPO RESULT — not paper default |
| n_prodnet | 2 | Undermatches code default (10) and HPO max (10) |
| hidden_size | 128 | Matches upstream |
| num_layers | 2 | Matches upstream |
| lr_step_size | 30 | Paper: 20 |
| lr_step_gamma | 0.3 | Paper: 0.5 |
| weight_decay | 1.0e-5 | Paper: 0.0 (stale HPO result) |
| grad_clip | 1.0 | Not in paper; added for stability |
| batch_size | 128 | Matches |
| n_epochs | 50 | Matches |

**Critical mismatch:** The YAML `lr: 0.008149663367631796` is a stale HPO result from a prior run on a specific dataset. It is NOT a paper default and NOT a benchmark-agnostic default. Using it as the fixed config would bias this model in favor of whatever dataset it was tuned on.

**Code-level mismatch:** `AutoSTPPConfig.n_prodnet` code default is `10`, but the canonical YAML overrides this to `2`. The HPO range is `[2, 4, 6, 10]`. The final config should reflect an HPO result, not the YAML default.

#### C. Recommended HPO subset

**Tune:** lr, n_prodnet, hidden_size, num_layers, weight_decay, grad_clip, lr_step_size, lr_step_gamma

HPO config (`auto_stpp_hpo.yaml`) is already present and covers these. **It is correct as written.**

| Parameter | Search primitive | Range |
|---|---|---|
| lr | `loguniform` | 5e-4 to 1e-2 |
| n_prodnet | `choice` | [2, 4, 6, 10] |
| hidden_size | `choice` | [64, 128, 192, 256] |
| num_layers | `choice` | [1, 2, 3] |
| weight_decay | `choice` | [0.0, 1e-6, 1e-5] |
| grad_clip | `choice` | [0.0, 1.0, 5.0] |
| lr_step_size | `choice` | [10, 20, 30] |
| lr_step_gamma | `choice` | [0.3, 0.5, 0.7] |

**Do NOT tune:** activation (tanh is architectural), batch_size (128 matches paper), n_epochs (50 matches paper), lookback/lookahead (paper window size, changing breaks architecture assumptions).

#### D. Training-specific notes

- **Optimizer: Adam.** Keep model-specific. Adam (not AdamW) is paper-faithful. The weight_decay range covers W-ish regularization if needed.
- **LR schedule: StepLR.** Keep model-specific. The step schedule is the paper-faithful choice. Cosine is not paper-faithful for AutoSTPP.
- **inference_mode=False required** on Trainer. AutoIntDecoder uses `torch.autograd.grad` internally. Do not change this.
- **`test_nll_space: raw`** required. The model applies an internal paper-space affine transform and reports both native (normalized) and raw-space NLL. The raw number is what should go in benchmark tables.

#### E. What is locked

| Aspect | Status | Value |
|---|---|---|
| optimizer | MODEL-LOCKED | adam |
| batch_size | MODEL-LOCKED | 128 |
| n_epochs | MODEL-LOCKED | 50 |
| adam_beta1 | MODEL-LOCKED | 0.9 |
| adam_beta2 | MODEL-LOCKED | 0.999 |
| lookback, max_history | MODEL-LOCKED | 20 |
| lookahead | MODEL-LOCKED | 1 |
| activation | MODEL-LOCKED | tanh |
| lr, weight_decay, grad_clip, n_prodnet, hidden_size, num_layers, lr step params | HPO-TUNED | see above |

---

### 3.2 DeepSTPP (`deep_stpp`)

**Paper:** Lin et al., "DeepSTPP: Deep Space-Time-Point Process for Learning Event Dynamics", ICLR 2022.
**Preset status:** canonical

#### A. Upstream defaults

Confirmed from YAML comment annotations against paper (Lin et al. 2021/2022):

| Parameter | Paper/upstream | Source |
|---|---|---|
| Optimizer | Adam | paper |
| lr | 3.0e-4 | paper |
| adam_beta1 | 0.9 (momentum) | paper |
| weight_decay | 0.0 | paper |
| grad_clip | 1.0 | paper |
| batch_size | 128 | paper |
| n_epochs | 50 | paper |
| lr_step_size | 50 | paper |
| lr_step_gamma | 0.2 | paper |
| vae_beta | 1.0e-3 | paper (KL weight for VAE path) |
| enc.num_heads | 2 | paper |
| enc.num_layers | 3 | paper |
| enc.dropout | 0.0 | paper |
| dec.seq_len | 20 | paper |
| dec.lookahead | 1 | paper |
| dec.num_points | 20 | paper |
| dec.sigma_min | 1e-4 | paper |
| dec.n_layers | 3 | paper |
| dec.b_max | 20.0 | paper |

#### B. Repo current state

File: `unified_stpp/configs/deep_stpp.yaml`

The YAML is **well-aligned with paper defaults.** All values noted above are present and correct.

**One inconsistency:** `vae: false` but `vae_beta: 1.0e-3`. The vae_beta field is harmless when vae=false because the KL term is only added when `result.kl is not None` (see `lightning_module.py:127`). But the comment says "beta=1e-3 (KL weight in ELBO)" which implies VAE mode. **Decision: benchmark runs use `vae: false` (deterministic encoder, no ELBO), which is the faithful STHP notebook path. The `vae_beta` line is dead code for the benchmark and should be noted.**

**No HPO config exists for deep_stpp.** This is a gap.

#### C. Recommended HPO subset

A deep_stpp_hpo.yaml does not exist and should be created. Recommended search:

| Parameter | Search primitive | Range | Rationale |
|---|---|---|---|
| lr | `loguniform` | 1e-4 to 1e-3 | Paper uses 3e-4; narrow range around paper value |
| num_points | `choice` | [10, 20, 50] | Directly controls model capacity (# Hawkes kernels); YAML comment confirms these as HPO choices |
| enc.num_heads | `choice` | [2, 4] | YAML comment marks these as HPO candidates |
| enc.num_layers | `choice` | [2, 3, 4] | YAML comment marks these as HPO candidates |
| grad_clip | `choice` | [0.5, 1.0, 5.0] | Paper: 1.0; small range exploration |
| weight_decay | `choice` | [0.0, 1e-5] | Paper: 0.0; light regularization option |

**Do NOT tune:** seq_len, lookahead, sigma_min, b_max (paper architectural choices), n_epochs, batch_size, lr schedule (StepLR with step_size=50 and gamma=0.2 is paper-faithful).

#### D. Training-specific notes

- **Optimizer: Adam.** Keep model-specific (paper-faithful).
- **LR schedule: StepLR.** Keep model-specific. Paper uses step_size=50 and gamma=0.2.
- **vae=false for benchmark.** The VAE path (vae=true) produces stochastic outputs at training time. For fair NLL comparison across seeds and runs, use the deterministic encoder path. The paper's STHP notebook uses sample=False.
- **`test_nll_space: raw` required.** The model performs an internal paper-affine back-transform (denormalize input, recompute paper-space MinMax stats from training data) and exposes both native and raw-space NLL. The raw number is the comparable one.
- **Paper-space transform is data-dependent:** `DeepSTPPConfig.data_init_overrides()` computes `paper_dt_min`, `paper_dt_range`, `paper_loc_min`, `paper_loc_range` from training data at fit time. These are saved in the model state. This is correct behavior but means NLL is only comparable when the same training split is used.

#### E. What is locked

| Aspect | Status | Value |
|---|---|---|
| optimizer | MODEL-LOCKED | adam |
| batch_size | MODEL-LOCKED | 128 |
| n_epochs | MODEL-LOCKED | 50 |
| lr_step_size | MODEL-LOCKED | 50 |
| lr_step_gamma | MODEL-LOCKED | 0.2 |
| adam_beta1 | MODEL-LOCKED | 0.9 |
| vae | MODEL-LOCKED | false (benchmark path) |
| seq_len, lookahead | MODEL-LOCKED | 20, 1 |
| dec.sigma_min, b_max | MODEL-LOCKED | 1e-4, 20.0 |
| lr, num_points, enc params, grad_clip | HPO-TUNED | see above |
| vae_beta | DEAD CODE for benchmark | 1e-3 (harmless, vae=false) |

---

### 3.3 NeuralSTPP — SelfAttentiveCNF (`neural_attncnf`)

**Paper:** Chen et al., "Neural Spatio-Temporal Point Processes", ICLR 2021.
**Preset status:** provisional — excluded from headline experiments until stabilized.

#### A. Upstream defaults

From upstream NeuralSTPP repo and `original_models_manifesto.md`:

| Parameter | Paper/upstream | Source |
|---|---|---|
| Optimizer | AdamW | upstream code |
| lr | 1.0e-3 | upstream `--lr` |
| adam_beta2 | 0.98 | upstream `betas=(0.9, 0.98)` |
| weight_decay | 1.0e-6 | upstream `--weight_decay` |
| grad_clip | 0.0 (= 1e10 in code) | upstream `--gradclip 0` |
| batch_size | event-budget (max_events=4000), test_bsz=32 | upstream code |
| n_iterations | 10000 (not epoch-based) | upstream code |
| lr_schedule | warmup + cosine decay | upstream code |
| spatial.hidden_dims | 64-64-64 | upstream `--hdims` |
| spatial.layer_type | concat (ConcatSquash) | upstream `--layer_type` |
| spatial.actfn | swish | upstream `--actfn` |
| spatial.l2_attn | False (default), optional | upstream `--l2_attn` |
| spatial.tol | 1e-4 | upstream `--tol` |
| spatial.otreg_strength | 1e-4 | upstream `--otreg_strength` |
| backbone.tpp_hidden_dims | 32-32 | upstream `--tpp_hdims` |
| backbone.tpp_actfn | softplus | upstream `--tpp_actfn` |
| backbone.tpp_style | gru | upstream `--tpp_style` |
| backbone.energy_reg | 1e-4 | upstream `--tpp_otreg_strength` |
| EMA | used at training time | upstream code (EMA commented-out at test) |
| param groups | separate group for `self_attns` | upstream code |

#### A.1 Singleton-safe ActNorm initialization note

The upstream temporal `ActNorm` implementation computes `torch.var(x_, dim=0)` at
initialization time. Under modern PyTorch defaults this is undefined for a
singleton init batch and can yield `NaN`, which then propagates into the neural
ODE hidden state and crashes training. The repo therefore includes a **narrow
guard only for the singleton init case** in the temporal ActNorm path:

- if the flattened init batch has more than one row, keep the upstream code path
- if it has exactly one row, use the same numerical floor (`0.2`) directly for
  the variance term instead of calling `torch.var(...)`

This is an implementation-stability guard, not a modeling change. It preserves
upstream behavior for non-singleton batches and only patches the undefined edge
case so campaign runs do not fail on particular batch realizations.

#### A.2 Temporal ODE interval diagnostics note

The temporal NeuralSTPP ODE path now includes a narrow diagnostic guard around
the upstream `odeint` call:

- active event times passed into `integrate_lambda(...)` must be strictly
  increasing relative to the previous accepted event time; if not, the code now
  raises an explicit runtime error before the solver call with the offending
  event index and interval stats
- solver endpoints that are theoretically positive but too small to survive
  float32 rounding are repaired with the next representable float above `t0`
  instead of a fixed `t0 + 1e-6` offset
- if torchdiffeq still raises `"underflow in dt"`, the repo now re-raises with
  temporal interval diagnostics (`min/max/mean_raw_dt`,
  `non_increasing_count`, `tiny_interval_count`, solver settings, `nfe`)

This is a debugging and numerical-stability guard for pathological or
effectively zero temporal intervals. It does not alter the model path for
ordinary strictly increasing event sequences.

#### B. Repo current state

File: `unified_stpp/configs/neural_stpp_attn_sc.yaml`

The YAML comments explicitly document deviations:

| Parameter | Repo value | Paper value | Gap | Status |
|---|---|---|---|---|
| lr | 5.0e-4 | 1.0e-3 | 2× lower | Intentional tweak |
| weight_decay | 1.0e-5 | 1.0e-6 | 10× higher | Intentional tweak |
| grad_clip | 1.0 | 0.0 | Added clipping | Intentional tweak |
| batch_size | 64 | event-budget (~32 avg) | Fixed vs dynamic | Simplification |
| atol/rtol | 1.0e-5 | 1.0e-4 | 10× tighter | Intentional (slower!) |
| lr_schedule | none (early stopping) | warmup + cosine | Missing | Not implemented |
| actfn | softplus (default) | swish | Missing | Marked in YAML comment |
| l2_attn | not set | optional | Missing | Marked in YAML comment |
| EMA | not implemented | used | Missing | Not implemented |
| param groups | not implemented | `self_attns` separate | Missing | Not implemented |

**Bottom line:** This preset has 8 known gaps or deliberate deviations from upstream. It will likely not reproduce paper numbers. It is provisional for a reason. Do not benchmark this as "NeuralSTPP (paper)" in published results.

**Critically: atol/rtol=1e-5 (10× tighter than paper) substantially increases ODE solve time with no documented benefit. Consider reverting to 1e-4 before any benchmark run.**

No HPO config exists for this model.

#### C. Recommended HPO subset (if this model is ever promoted to canonical)

| Parameter | Search primitive | Range |
|---|---|---|
| lr | `loguniform` | 5e-4 to 2e-3 |
| weight_decay | `loguniform` | 1e-6 to 1e-4 |
| atol/rtol | `choice` | [1e-5, 1e-4] |
| grad_clip | `choice` | [0.0, 1.0] |

**Do NOT tune:** spatial.hidden_dims (paper-fixed 64-64-64), backbone.tpp_hidden_dims (32-32), layer_type (concat is paper-fixed), otreg_strength (1e-4 from paper), solver (dopri5).

#### D. Training-specific notes

- **Optimizer: AdamW with beta2=0.98.** This is paper-faithful and kept (repo uses adamw, beta2=0.98 set in canonical configs). Do not change to Adam.
- **LR schedule: cosine with warmup** is not implemented. This is a fairness concern if this model runs fewer effective update steps than others. Partially mitigated by early stopping (patience=30).
- **ODE solver on MPS:** `_neural_stpp_resolve_accelerator()` falls back from MPS to CPU automatically. Float64 requirement from torchdiffeq. This is correct behavior and must not be removed.
- **Event-budget batching** is architecturally motivated: sequences of very different lengths cause severe padding waste and ODE instability. The adapter supports `max_events` kwarg; consider using it for any benchmark run. This is different from simple fixed batch_size.

---

### 3.4 NeuralSTPP — JumpCNF (`neural_jumpcnf`)

**Paper:** Same paper as 3.3. **Preset status:** provisional.

Same upstream defaults as AttentiveCNF. Key JumpCNF-specific notes:

- **solve_reverse=True** (in SPATIAL_DEFAULTS) is required for paper-faithful JumpCNF. This wires `aux_odefunc = state_model.temporal_core.hidden_state_dynamics` via `NeuralSTPPConfig.build_model()`. Do not remove this.
- No HPO config.
- Same deviations from upstream as AttentiveCNF.
- `n_flows=4` in spatial defaults — paper-faithful.
- Same recommendation: atol/rtol should be 1e-4 not 1e-5.

---

### 3.5 SMASH (`smash`)

**Paper:** Not explicitly cited in code. Score-matching approach for STPP. Appears to follow the SMASH/DDPM-style denoising framework.
**Preset status:** canonical

#### A. Upstream defaults

No upstream paper citation found in the codebase. Key parameters reconstructed from YAML and config class:

| Parameter | Value | Source |
|---|---|---|
| Optimizer | AdamW | smash.yaml |
| lr | 1.0e-3 | smash.yaml |
| adam_beta2 | 0.99 | smash.yaml |
| weight_decay | 1.0e-2 | smash.yaml |
| grad_clip | 1.0 | smash.yaml |
| n_epochs | 200 | smash.yaml |
| batch_size | 64 | smash.yaml |
| lr_schedule | linear_decay with warmup | smash.yaml |
| lr_warmup_epochs | 5 | smash.yaml |
| lr_final | 5.0e-5 | smash.yaml |
| d_model | 64 | smash.yaml |
| d_rnn | 256 | smash.yaml |
| d_inner | 128 | smash.yaml |
| enc.n_layers | 4 | smash.yaml |
| sigma_time | 0.05 | smash.yaml |
| sigma_loc | 0.05 | smash.yaml |
| num_noise | 50 | smash.yaml |
| sampling_timesteps | 500 | smash.yaml |
| loss_lambda | 0.5 | smash.yaml |
| loss_lambda2 | 1.0 | smash.yaml |

#### B. Critical constraint: SMASH cannot compute exact NLL

`nll_kind = "none"` for SMASH's EventCapabilities. This means:
- SMASH cannot produce exact NLL values at test time
- The `val/nll` metric tracked during training is the score-matching loss (not NLL)
- SMASH **cannot be compared directly against auto_stpp or deep_stpp on NLL metrics**
- SMASH is valid for sampling-based metrics: predictive RMSE, Wasserstein, spatial distribution metrics

**This is not a bug — it is an architectural property of score-matching models.** The benchmark should separate SMASH into a "sampling-based comparison" lane only.

#### C. Recommended HPO subset

No HPO config exists. If HPO is desired:

| Parameter | Search primitive | Range | Rationale |
|---|---|---|---|
| lr | `loguniform` | 5e-4 to 5e-3 | 1e-3 is the current fixed value |
| weight_decay | `loguniform` | 1e-3 to 1e-1 | Currently 1e-2; relatively aggressive |
| sigma_time | `choice` | [0.01, 0.05, 0.1] | Noise schedule parameter |
| sigma_loc | `choice` | [0.01, 0.05, 0.1] | Noise schedule parameter |
| loss_lambda | `uniform` | 0.2 to 0.8 | Trade-off between time and space score |

**Do NOT tune:** n_epochs (200 is required for score-matching convergence), d_model/d_rnn/d_inner (encoder capacity, leave at upstream values), num_noise/sampling_timesteps (architecture fixed), lr_schedule (linear_decay is appropriate for this style of training).

**Recommendation: Do not run HPO on SMASH for the first benchmark round.** Use fixed config and report only sampling metrics. HPO adds complexity for a model that cannot contribute to the primary NLL comparison.

#### D. Training-specific notes

- **Optimizer: AdamW with beta2=0.99.** Keep model-specific. The higher beta2 provides smoother gradient estimates for the noisy score-matching objective.
- **LR schedule: linear_decay with warmup.** Keep model-specific. Score-matching benefits from gradual LR decay.
- **200 epochs.** Model-specific and justified: score-matching converges more slowly than exact-likelihood methods.
- **Token normalization is computed from train+val+test data** (`data_init_overrides` iterates all three splits). This is by design — SMASH's MinMax token scaling needs to cover the full observed range. This is paper-faithful behavior.
- **Benchmark NLL comparison: EXCLUDE.** Report SMASH only in sampling-metric tables.

#### E. What is locked

| Aspect | Status | Value |
|---|---|---|
| optimizer | MODEL-LOCKED | adamw |
| adam_beta2 | MODEL-LOCKED | 0.99 |
| n_epochs | MODEL-LOCKED | 200 |
| batch_size | MODEL-LOCKED | 64 |
| lr_schedule | MODEL-LOCKED | linear_decay |
| lr_warmup_epochs | MODEL-LOCKED | 5 |
| lr_final | MODEL-LOCKED | 5e-5 |
| d_model, d_rnn, d_inner, n_layers | MODEL-LOCKED | 64, 256, 128, 4 |
| num_noise, sampling_timesteps | MODEL-LOCKED | 50, 500 |
| NLL comparison | EXCLUDED | use sampling metrics only |

---

### 3.6 NSMPP (`nsmpp`)

**Paper:** NSMPP deep-basis parametric self-exciting point process. No upstream paper citation in code.
**Preset status:** experimental

#### A. Key characteristics

- **Adadelta optimizer with lr=1.0.** This is the standard Adadelta convention: lr=1.0 is the global scale, not a learning rate in the Adam sense. Adadelta adapts per-parameter scale.
- **int_res=20 in HPO config.** Explicitly flagged as "faster trials; use 30-50 in final config." This means: HPO runs at lower accuracy, final run should use int_res=30+.
- **Exact NLL** available (`nll_kind="exact"`). NSMPP can participate in the primary NLL comparison.
- HPO config exists (`nsmpp_hpo.yaml`).

#### B. HPO config assessment

The existing `nsmpp_hpo.yaml` covers the right parameters. Notable:
- `n_basis`, `basis_dim`, `nn_width` — kernel capacity
- `mu` — background rate
- `init_std`, `init_weight_mean` — initialization (important for stability)
- `lr` — Adadelta global scale
- `grad_clip` — not in upstream but added for 2D data stability

**Issue:** `int_res=20` in HPO config must be bumped to `int_res: 30` (minimum) for the final config used in benchmark. The HPO tuning is valid at int_res=20 to select architecture, but the final eval should use higher resolution.

#### C. Training-specific notes

- **Do not change optimizer to Adam.** Adadelta is architectural for this model (per-parameter adaptive rates without a global LR to tune in the Adam sense).
- `val_objective` (not `val_nll`) is the HPO metric because the model uses a custom objective. Verify this maps to comparable NLL for benchmark reporting.
- `test_nll_space: raw` is set — correct.

---

## 4. Cross-Model Fairness Notes

### 4.1 Gradient clipping

Current state (from YAMLs):
- `auto_stpp`: 1.0 (or HPO choice [0.0, 1.0, 5.0])
- `deep_stpp`: 1.0
- `neural_attncnf` / `neural_jumpcnf`: 1.0 (paper: 0.0)
- `smash`: 1.0
- `nsmpp`: HPO choice [0.0, 0.5, 1.0, 5.0]

**Recommendation: Lock all non-HPO models to grad_clip=1.0 globally.** The neural presets deviate from paper here, but adding clipping at 1.0 is fairer than 0.0 for the benchmark (prevents single-model divergence inflating variance). Document the deviation.

### 4.2 Batch size and effective gradient noise

| Model | batch_size | Events/batch (approx) |
|---|---|---|
| auto_stpp | 128 | 128 × 20 lookback events |
| deep_stpp | 128 | 128 × 20 seq_len events |
| neural_attncnf | 64 | 64 × (up to 4000 events, event-budget) |
| smash | 64 | variable |
| nsmpp | 25 | 25 sequences |

This is not standardizable across architecturally different models. **Accept as model-specific.** Document it when reporting results.

### 4.3 Early stopping vs fixed epoch budget

| Model | Epochs | Early stopping |
|---|---|---|
| auto_stpp | 50 | None |
| deep_stpp | 50 | None |
| neural_attncnf | 200 | patience=30 |
| smash | 200 | None |
| nsmpp | 50 | None |

**Concern:** neural variants with early stopping get variable compute budgets across seeds and datasets. This can inflate variance in reported metrics. For provisional neural presets this is acceptable. For canonical models, fixed epoch budget is preferred.

**Recommendation:** All canonical models (`auto_stpp`, `deep_stpp`, `smash`) should use fixed epoch budgets (no early stopping). Provisional neural presets may keep early stopping.

### 4.4 LR schedule diversity

| Model | Schedule |
|---|---|
| auto_stpp | StepLR (HPO-tuned step_size, gamma) |
| deep_stpp | StepLR (step_size=50, gamma=0.2) |
| neural_attncnf | Constant (early stopping acts as budget) |
| smash | linear_decay + warmup |
| nsmpp | Constant |

This is intentionally model-specific — each schedule matches what the upstream paper or optimizer choice motivates. Do not standardize this.

### 4.5 NLL comparability matrix

| Model | val_nll monitored during training | test_nll in benchmark table |
|---|---|---|
| auto_stpp | normalized z-score space | raw (via test_nll_space=raw) |
| deep_stpp | normalized z-score space | raw (via test_nll_space=raw) |
| neural_attncnf | normalized z-score space | raw (if supported; provisional) |
| smash | score-matching loss | **EXCLUDED from NLL table** |
| nsmpp | val_objective | raw (via test_nll_space=raw) |

**Important:** `val_nll` used for early stopping / checkpoint selection is the normalized-space NLL for all models except SMASH. The test-time `test_nll_space: raw` only affects the reported test number, not training dynamics. This is consistent and correct.

---

## 5. Final Locked Recommendations

### 5.1 Globally locked (all models)

| Aspect | Value |
|---|---|
| data.protocol | unified |
| data.normalize | true |
| data.seed | 42 |
| checkpoint_select | best |
| test_nll_space | raw |
| grad_clip (recommendation) | 1.0 (lock this — currently inconsistent) |

### 5.2 Model-specific and fixed (not HPO-tuned)

| Model | Aspect | Fixed value |
|---|---|---|
| auto_stpp | batch_size, n_epochs, lookback, optimizer | 128, 50, 20, adam |
| deep_stpp | batch_size, n_epochs, lr_step_size/gamma, vae, optimizer | 128, 50, 50/0.2, false, adam |
| smash | n_epochs, lr_schedule, warmup, batch_size, optimizer | 200, linear_decay, 5, 64, adamw |
| nsmpp | optimizer, batch_size, int_res (final) | adadelta, 25, ≥30 |

### 5.3 HPO-tuned (per model, per dataset)

| Model | HPO config | Status |
|---|---|---|
| auto_stpp | `auto_stpp_hpo.yaml` | exists, correct |
| deep_stpp | **missing** | needs `deep_stpp_hpo.yaml` |
| neural_attncnf | **missing** | model is provisional; defer |
| neural_jumpcnf | **missing** | model is provisional; defer |
| smash | **missing** | recommend no HPO for first round |
| nsmpp | `nsmpp_hpo.yaml` | exists, correct; bump int_res for final run |

### 5.4 Still unresolved

| Issue | Impact | Recommended action |
|---|---|---|
| `auto_stpp.yaml` has stale lr (0.00815) | Misleading default if used without HPO | Always run HPO for auto_stpp; never use YAML lr as benchmark default |
| `neural_attncnf` atol/rtol=1e-5 (10× paper) | Slower training for no documented benefit | Revert to 1e-4 before any benchmark run |
| SMASH paper citation missing | Cannot fully verify upstream defaults | Document as "unverifiable against paper" in any publication |
| `nsmpp` val_objective vs val_nll | May not align with val/nll used by other models | Verify that HPO metric (val_objective) and checkpoint monitor are equivalent for this model |
| `deep_stpp_hpo.yaml` does not exist | DeepSTPP results will be based on paper defaults, not tuned | Create HPO config before final benchmark |
| Neural preset activation: softplus vs paper swish | Minor architecture difference; lower expressiveness | Acceptable for provisional; not paper-faithful |

---

## 6. Open Questions

1. **Should SMASH use `lr_schedule: cosine` instead of `linear_decay`?** Score-matching literature sometimes favors cosine. No paper citation to validate current choice.

2. **Should all models use the same number of HPO trials (40)?** Currently auto_stpp uses 40 and nsmpp uses 60. For fair comparison, either standardize or document the difference.

3. **What is the correct `int_res` for NSMPP final runs?** The HPO config says "use 30-50 in final config" but does not commit to a value. Recommend: lock to `int_res: 40` for final benchmark configs.

4. **Should neural presets use event-budget batching (`max_events: 4000`) in the benchmark?** The original NeuralSTPP uses it. The current YAML does not set it for benchmark runs. This may affect both memory use and gradient signal quality.

5. **Is there a target dataset on which to validate the HPO results?** The stale `auto_stpp.yaml` lr was tuned on an unspecified Hawkes dataset. New HPO should target the canonical benchmark split (HawkesNest hard or similar) to avoid dataset-specific overfitting.
