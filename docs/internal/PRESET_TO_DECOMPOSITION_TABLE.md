# Preset-to-Decomposition Summary

This table maps the public benchmark presets in the current repo to the unified decomposition used in the paper text.

| Preset | History Encoder | State Evolution | Event Head | Objective | Evaluation Exposure |
|---|---|---|---|---|---|
| `poisson_gmm` | None; raw-history passthrough | Analytic homogeneous Poisson clock | Time-decay Gaussian-mixture spatial density over past events | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `hawkes_gmm` | None; raw-history passthrough | Analytic Hawkes self-exciting clock | Time-decay Gaussian-mixture spatial density over past events | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `selfcorrecting_gmm` | None; raw-history passthrough | Analytic self-correcting clock | Time-decay Gaussian-mixture spatial density over past events | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `poisson_cnf` | None; raw-history passthrough | Analytic homogeneous Poisson clock | Independent time-conditioned spatial CNF | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `hawkes_cnf` | None; raw-history passthrough | Analytic Hawkes self-exciting clock | Independent time-conditioned spatial CNF | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `selfcorrecting_cnf` | None; raw-history passthrough | Analytic self-correcting clock | Independent time-conditioned spatial CNF | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `poisson_tvcnf` | None; raw-history passthrough | Analytic homogeneous Poisson clock | Independent time-varying spatial CNF | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `hawkes_tvcnf` | None; raw-history passthrough | Analytic Hawkes self-exciting clock | Independent time-varying spatial CNF | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `selfcorrecting_tvcnf` | None; raw-history passthrough | Analytic self-correcting clock | Independent time-varying spatial CNF | Exact factorized temporal+spatial NLL | Exact intensity and density |
| `deep_stpp` | Transformer over fixed paper sliding windows | Window latent code, optionally sampled as a VAE latent; no explicit continuous-time state beyond windows | Coupled Hawkes-Gaussian decoder over window latent | Exact paper-window NLL; optional outer \(\beta\)-KL | Exact intensity; exact NLL |
| `auto_stpp` | Fixed paper lookback window; no learned encoder | Explicit finite-history triggering state in paper MinMax space; no learned latent dynamics | Exact AutoSTPP Cuboid joint-intensity head | Exact paper-window NLL | Exact intensity; exact NLL |
| `auto_stpp_legacy` | Transformer over full event history | Discrete encoder states; no explicit continuous-time latent dynamics | AutoInt monotone-integral joint-intensity decoder | Exact joint NLL | Exact intensity; exact NLL |
| `nsmpp` | None; event-vector history passthrough | No latent state; direct causal history set | DeepBasis direct joint conditional-intensity kernel with numerical compensator | Exact NLL, optimized as sequence mean and reported per event at test time | Exact intensity; exact NLL |
| `neural_cond_gmm` | Neural point-process backbone over full history | Continuous-time neural hidden state with eventwise temporal NLL and energy regularization | Conditional GMM spatial decoder on shared temporal hidden state | Exact joint NLL; training adds state regularization | Exact intensity; exact NLL |
| `neural_jumpcnf` | Neural point-process backbone over full history | Continuous-time neural hidden state with eventwise temporal NLL and energy regularization | Neural jump-CNF spatial decoder conditioned on shared temporal hidden state | Exact joint NLL; training adds state regularization | Exact intensity; exact NLL |
| `neural_attncnf` | Neural point-process backbone over full history | Continuous-time neural hidden state with eventwise temporal NLL and energy regularization | Attention-conditioned neural CNF spatial decoder on shared temporal hidden state | Exact joint NLL; training adds state regularization | Exact intensity; exact NLL |
| `smash` | Upstream TransformerST over reconstructed raw-time / min-max event tokens | Flattened next-event conditioning tokens; no calibrated intensity state | Score network for denoising time, space, and optional marks | Denoising score matching | Native sampler, score field, approximate test NLL, proxy-KDE surfaces |
| `diffusion_stpp` | DSTPP Transformer_ST over reconstructed raw-time / min-max diffusion tokens | Flattened next-event conditioning tokens; no calibrated intensity state | Gaussian diffusion denoiser over next-event \([\Delta t, s]\) token | Variational ELBO / diffusion denoising loss | Native sampler, approximate test NLL, proxy-KDE surfaces |

Notes:

- Deprecated aliases are omitted; rows use the canonical public preset IDs from the benchmark contract.
- `auto_stpp` denotes the canonical faithful AutoSTPP implementation; `auto_stpp_legacy` is the retained legacy encoder-decoder path.
- “Exact intensity and density” applies only to the factorized baselines, whose event head exposes both interfaces; other exact families expose calibrated intensity but not a separate conditional density API.
