# Approximate next-event NLL for DSTPP and SMASH

Evaluation-only scripts for the generative rows of the NeurIPS 2026 camera-ready NLL table. They
re-score the final-seed checkpoints of a finished campaign (`python -m seahorse campaign run`) on
the same test targets as the exact presets (event index ≥ H of each length-H context window) and
write per-target values in raw units (nats per target). Neither model defines a likelihood that
can be read off directly; the numbers are labelled accordingly.

Both scripts import the installed package; they need the campaign `manifest.json`, a completed
job directory (`model/`, `status.json`, `input_contract.json`) and the test split recorded in the
manifest (`--path-map OLD=NEW` rewrites its location; the split hash is still checked).

```bash
python scripts/approx_nll/dstpp_bound.py \
  --manifest campaigns/v1/manifest.json --job-dir campaigns/v1/jobs/<dataset>__diffusion_stpp__seed11 \
  --out runs/approx_nll/<dataset>__diffusion_stpp__seed11 --device cuda \
  [--ref campaigns/v1/jobs/<dataset>__poisson_gmm__seed11/target_nll.npz]

python scripts/approx_nll/smash_grid_density.py \
  --manifest campaigns/v1/manifest.json --job-dir campaigns/v1/jobs/<dataset>__smash__seed11 \
  --out runs/approx_nll/<dataset>__smash__seed11 --device cuda --n-sub 2000 --grids 128
python scripts/approx_nll/smash_grid_density.py ... --n-sub 0 --no-grid   # temporal term, all targets
```

`--ref` checks that the target order equals that of an exact preset's `target_nll.npz`.

## Coordinates

Raw `r = (τ_raw, s_raw)` → campaign coordinates `x` (`τ_c = τ_raw / time_scale`,
`s_c = (s_raw − offset) / scale`) → model token `u ∈ [0, 1]` (train-split min/max per dimension,
range `R_d`) → `y = 2u − 1`, the coordinates both networks are trained in. With
`p_x = p_y |dy/dx|`, `NLL_x = NLL_y − log|dy/dx|`, and the campaign constant
`log(time_scale · scale_lon · scale_lat)` converts to raw units as for the exact presets.

## DSTPP: full variational bound (`dstpp_bound.py`)

`GaussianDiffusionST.NLL_cal` with `clip_denoised=True`: all T steps (continuous Gaussian decoder
at t = 0, KL terms) plus the prior KL, one forward-noise draw per step and target. `NLL_cal`
returns means over the D = 3 token dimensions, so

    NLL_raw = D · NLL_y,mean − Σ_d log(2 / R_d) + log(time_scale · scale_lon · scale_lat)

The temporal (1 dimension) and spatial (2 dimensions) parts are converted with their own
Jacobians. This is an upper bound on the model's NLL, not a density value; at T = 100 it is loose
by about 3 nats per event on the homogeneous Poisson control, about 2.1 of which a Bayes-optimal
denoiser would also lose. The target token is rebuilt without the state model's clamp to [0, 1];
targets outside the training range get very large values (`n_clamped_targets` in `summary.json`).

## SMASH: temporal term and grid density (`smash_grid_density.py`)

SMASH defines no likelihood. **Temporal**: `ScoreMatchingProcess.nll_terms` evaluates the
intensity in the training coordinates, `log f_y(y_t) = log λ(y_t) − ∫_{−1}^{y_t} λ(y) dy`
(trapezoid rule, 20 nodes), and returns `NLL_u = −log f_y − log 2` on the `[0, 1]` time token.
With log-normalised time `u_t = (log max(τ_c, 10⁻⁴) − m_t) / R_t`,

    NLL_t,raw = NLL_u + log R_t + log max(τ_c, 10⁻⁴) + log time_scale

(the `log τ_c` term is absent without log-normalisation). Score matching identifies λ only up to
a Riccati family and `f` may have mass below one, so this is the NLL of a sub-density, an upper
bound on that of its normalised version.

**Spatial (grid density)**: the location score `s(y_s | y_t, H)` is evaluated on a G × G node
grid over `[−1, 1]²` with `y_t` at the target's value. The potential
`φ = argmin ‖Dφ / h − s_mid‖²` (forward differences, scores averaged to edge midpoints) solves the
Neumann Poisson problem, exactly by DCT-II; `log p = φ − log ∫ exp φ` (trapezoid rule), bilinear
interpolation at the true location, and

    NLL_s,raw = NLL_s,y + Σ_d log(R_d / 2) + log(scale_lon · scale_lat)

`resid_ratio = ‖∇φ − s‖ / ‖s‖` is the non-conservative fraction of the score field (0.02–0.27 on
the paper datasets); the density is restricted to the token box. Joint = temporal + grid spatial.
`--n-sub N` scores one uniform draw from each of N equal strata of the target list (seed
`--seed`); the paper uses N = 2000 and G = 128 (G = 64/128/256 agree to < 0.01 nats on the
controls).

The framework value `SMASHEventModel.eval_nll` adds the Tweedie term
`σ²|s(y)|²/2 + (d/2) log(2πσ²)` instead of the grid density. That term is
`−log N(y; y + σ²s(y), σ²I)`, not a density of the location, and is dominated by the σ constant;
it is reported as a surrogate only.

Check on the homogeneous Poisson control (true NLL −3.380 per event, spatial part 0): temporal
−3.375, grid spatial 0.115, joint −3.239.
