# Model Capability Matrix

Use this table to choose a metric profile and evaluation path for each preset.

!!! tip "Source of truth"
    Run `python -m seahorse evaluate metrics --help` and
    `python -m seahorse evaluate surface --help` for the exact controls in
    your installed version.

## Capability Flags

| Flag | Meaning |
| --- | --- |
| **NLL** | Exact or approximate per-event log-likelihood |
| **Sampling** | Next-event predictive sampling (`predict_next`) |
| **Surface** | Intensity/density grid for `evaluate surface` |

## Matrix

| Family | Python class | CLI preset | NLL | Sampling | Surface profile |
| --- | --- | --- | --- | --- | --- |
| AutoSTPP | `AutoSTPP` | `auto_stpp` | Exact | Yes | `history_frame` |
| DeepSTPP | `DeepSTPP` | `deep_stpp` | Exact | Yes | `history_frame` |
| NSMPP DeepBasis | `NSMPP` | `nsmpp` | Exact | Yes | — |
| SMASH | `SMASH` | `smash` | Approximate | Native | — |
| Diffusion STPP | `DiffusionSTPP` | `diffusion_stpp` | Approx (ELBO) | Native | — |
| NJSDE | `STPPEstimator("njsde")` | `njsde` | Exact | Yes | `future_exact` |
| Neural JumpCNF | `NeuralJumpCNF` | `neural_jumpcnf` | Exact | Yes | `future_exact` |
| Neural AttnCNF | `NeuralAttnCNF` | `neural_attncnf` | Exact | Yes | `future_exact` |
| RMTPP + GMM | `RMTPPGMM` | `rmtpp_gmm` | Exact (factorized) | Yes | — |
| THP + GMM | `THPGMM` | `thp_gmm` | Exact (factorized) | Yes | — |
| Poisson + GMM | `PoissonGMM` | `poisson_gmm` | Exact (factorized) | Yes | — |
| Hawkes + GMM | `HawkesGMM` | `hawkes_gmm` | Exact (factorized) | Yes | — |
| Self-correcting + GMM | `SelfCorrectingGMM` | `selfcorrecting_gmm` | Exact (factorized) | Yes | — |
| Poisson + CNF | `PoissonCNF` | `poisson_cnf` | Exact (factorized) | Yes | — |
| Hawkes + CNF | `HawkesCNF` | `hawkes_cnf` | Exact (factorized) | Yes | — |
| Self-correcting + CNF | `SelfCorrectingCNF` | `selfcorrecting_cnf` | Exact (factorized) | Yes | — |
| Poisson + TVCNF | `PoissonTVCNF` | `poisson_tvcnf` | Exact (factorized) | Yes | — |
| Hawkes + TVCNF | `HawkesTVCNF` | `hawkes_tvcnf` | Exact (factorized) | Yes | — |
| Self-correcting + TVCNF | `SelfCorrectingTVCNF` | `selfcorrecting_tvcnf` | Exact (factorized) | Yes | — |

## Choose This If…

| Goal | Recommended preset(s) |
| --- | --- |
| Smoke-test data and CLI wiring | `poisson_gmm` |
| Classical self-exciting baseline | `hawkes_gmm` |
| Flexible spatial density | `*_cnf` variants |
| Paper-style AutoSTPP reproduction | `auto_stpp` |
| Neural exact-density + surface diagnostics | `njsde`, `neural_jumpcnf`, `neural_attncnf` |
| Score-matching / generative experiments | `smash`, `diffusion_stpp` |

## Compatibility Notes

- `njsde` is the canonical preset for the conditional-GMM neural exact family.
  Older aliases such as `neural_cond_gmm` are compatibility names — do not use them in new work.
- SMASH and Diffusion STPP report approximate NLL; note this in benchmark comparisons.
- `future_exact` surface profiles may prefer `--device cpu` for numerical stability on some hardware.
- `python evaluate()` exposes likelihood metrics only. Use the CLI `evaluate metrics` command for full benchmark-aligned metric profiles.
