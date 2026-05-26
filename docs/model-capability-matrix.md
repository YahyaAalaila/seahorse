# Model Capability Matrix

This matrix summarizes the public model names a user is expected to reach for
from the Python API and CLI. Capability flags describe the implemented public
paths, not every internal method a component may contain.

Use `python -m unified_stpp evaluate metrics --help` and
`python -m unified_stpp evaluate surface --help` for the exact controls in the
installed version.

| Model family | Python class | CLI preset | Likelihood/NLL support | Sampling support | Surface/intensity query support | Typical use | Notes/caveats |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AutoSTPP paper family | `AutoSTPP` | `auto_stpp` | Exact NLL with raw-space reporting support | Predictive sampling through exact/intensity path when supported by the fitted runner | CLI `evaluate surface --profile history_frame`; Python `plot_intensity()` | Paper-style AutoSTPP runs and surface diagnostics | Uses paper transform semantics; surface workflow is diagnostic-oriented. |
| DeepSTPP paper family | `DeepSTPP` | `deep_stpp` | Exact NLL with raw-space reporting support | Predictive sampling through exact/intensity path when supported by the fitted runner | CLI `evaluate surface --profile history_frame`; Python `plot_intensity()` | Paper-style DeepSTPP baseline and diagnostics | Uses paper window semantics; surface workflow is diagnostic-oriented. |
| NSMPP DeepBasis family | `NSMPP` | `nsmpp` | Exact NLL with raw-space reporting support | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile listed for this preset | Direct conditional-intensity baseline | Optimizes a sequence-mean objective and reports test NLL/event separately. |
| SMASH family | `SMASH` | `smash` | Approximate framework-added NLL | Native sampler | No dedicated public CLI surface profile | Score-matching model and sampling-based comparison | NLL is approximate; use metric footnotes in reports when comparing to exact-NLL models. |
| Diffusion STPP family | `DiffusionSTPP` | `diffusion_stpp` | Approximate variational-bound NLL | Native sampler | No dedicated public CLI surface profile | Diffusion-based generative STPP experiments | Benchmark-facing test NLL is approximate. |
| NJSDE exact neural family | `STPPEstimator("njsde")` | `njsde` | Exact NLL with raw-space reporting support | Predictive sampling through exact/intensity path | CLI `evaluate surface --profile future_exact`; Python `plot_intensity()` when loaded through estimator | Canonical conditional-GMM neural exact model | `NJSDE` is the public name; see compatibility note for older aliases. |
| Neural JumpCNF exact family | `NeuralJumpCNF` | `neural_jumpcnf` | Exact NLL with raw-space reporting support | Predictive sampling through exact/intensity path | CLI `evaluate surface --profile future_exact`; Python `plot_intensity()` | Neural exact-density model with JumpCNF spatial decoder | Future-exact surface diagnostics may prefer CPU for numerical stability. |
| Neural AttnCNF exact family | `NeuralAttnCNF` | `neural_attncnf` | Exact NLL with raw-space reporting support | Predictive sampling through exact/intensity path | CLI `evaluate surface --profile future_exact`; Python `plot_intensity()` | Neural exact-density model with attentive CNF spatial decoder | Future-exact surface diagnostics may prefer CPU for numerical stability. |
| RMTPP + GMM family | `RMTPPGMM` | `rmtpp_gmm` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Neural temporal baseline with GMM spatial output | Useful baseline before heavier neural spatial decoders. |
| THP + GMM family | `THPGMM` | `thp_gmm` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Transformer Hawkes-style temporal baseline | Useful for sequence-history baselines with a simple spatial head. |
| Poisson + GMM factorized baseline | `PoissonGMM` | `poisson_gmm` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Fast baseline and smoke tests | Good first preset for checking data and CLI wiring. |
| Hawkes + GMM factorized baseline | `HawkesGMM` | `hawkes_gmm` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Classical self-exciting baseline | Use when event triggering is expected. |
| Self-correcting + GMM factorized baseline | `SelfCorrectingGMM` | `selfcorrecting_gmm` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Classical self-correcting baseline | Use when inhibition or regularization over time is expected. |
| Poisson + CNF factorized baseline | `PoissonCNF` | `poisson_cnf` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Baseline with more flexible spatial density than GMM | More expensive than GMM spatial baselines. |
| Hawkes + CNF factorized baseline | `HawkesCNF` | `hawkes_cnf` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Self-exciting baseline with flexible spatial density | More expensive than GMM spatial baselines. |
| Self-correcting + CNF factorized baseline | `SelfCorrectingCNF` | `selfcorrecting_cnf` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Self-correcting baseline with flexible spatial density | More expensive than GMM spatial baselines. |
| Poisson + TVCNF factorized baseline | `PoissonTVCNF` | `poisson_tvcnf` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Baseline with time-varying spatial flow | Use when spatial density changes over time. |
| Hawkes + TVCNF factorized baseline | `HawkesTVCNF` | `hawkes_tvcnf` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Self-exciting model with time-varying spatial flow | Heavier than GMM and static-CNF baselines. |
| Self-correcting + TVCNF factorized baseline | `SelfCorrectingTVCNF` | `selfcorrecting_tvcnf` | Exact NLL via factorized event model | Predictive sampling through exact/intensity path when supported by the fitted runner | Intensity-capable internally; no dedicated public CLI surface profile | Self-correcting model with time-varying spatial flow | Heavier than GMM and static-CNF baselines. |

## Compatibility Notes

- `njsde` is the canonical CLI preset for the conditional-GMM neural exact
  family. Older names such as `neural_cond_gmm` are compatibility aliases and
  should not be used in new documentation or commands.
- Some concrete Python classes are thin wrappers over `STPPEstimator`. If a
  preset has no dedicated concrete class, use `STPPEstimator("<preset>")`.
- Python `evaluate()` exposes a small likelihood-oriented surface. Use
  `python -m unified_stpp evaluate metrics ...` for benchmark-aligned metric
  profiles and output artifacts.
