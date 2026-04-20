# Unified Framework Draft

Drafted from the current implementation. Claims are limited to abstractions that are realized in the live model-construction path.

## NeurIPS-Style Framework Subsection Draft

### Unified STPP Construction

Our framework instantiates each preset as a pair of modules, a state model \(S\) and an event model \(E\), wrapped by a thin coordinator. The preset layer does more than name a family: it resolves nested overrides, binds protocol and loader constraints, fits any family-owned reversible coordinate transform or support statistics from the training data, and then builds the concrete \(S\) and \(E\) objects. The shared construction contract is therefore
\[
\texttt{preset} \;\rightarrow\; \texttt{native space / family stats} \;\rightarrow\; S \;\rightarrow\; E \;\rightarrow\; \texttt{objective, NLL, query/sampling interfaces}.
\]

The common input to \(S\) is the repository-wide batch representation \((t_{1:L}, s_{1:L}, \ell)\), optionally with marks and covariates. The output is an opaque `StateContext`, not a fixed tensor interface. This is the key unifying choice: families are free to define their own native history representation, while the outer framework only assumes that the resulting context can be consumed by the paired event model. In the shipped presets this context ranges from raw history passthrough (factorized baselines and NSMPP), to continuous-time hidden trajectories with eventwise temporal terms (Neural STPP), to paper-style sliding windows and latent codes (DeepSTPP and AutoSTPP), to flattened next-event conditioning tokens for generative models (SMASH and Diffusion).

The event model is the locus where family-specific mechanics enter. Exact factorized baselines instantiate an analytic temporal process together with a separate spatial model, and compute
\[
\log p(t_{1:L}, s_{1:L}) = \log p_t(t_{1:L}) + \sum_i \log p_s(s_i \mid t_i, H_i).
\]
Neural STPP splits this more asymmetrically: the state model owns the temporal point-process backbone and emits both per-event temporal NLL terms and a shared hidden sequence, while the event model attaches a family-specific spatial decoder and adds the spatial contribution to form the joint NLL. DeepSTPP and faithful AutoSTPP instead use the state model to reconstruct the paper-native window representation and the event model to evaluate the exact window likelihood of the corresponding decoder. The legacy AutoSTPP path remains in the same interface but replaces paper-window likelihoods with a learned monotone-integral decoder over encoder states. NSMPP is even more direct: the state model only packages event vectors, and the event model defines a joint conditional intensity over \((t,s)\) with a numerical compensator over a preset-specific support.

This yields an honest minimal decomposition. Across exact families, the invariant is not a single decoder formula but a two-stage factorization into history representation and event law. What varies is where temporal structure lives. In factorized baselines it is an explicit temporal module inside \(E\); in Neural STPP it is a continuous-time backbone inside \(S\); in DeepSTPP and AutoSTPP it is embedded in a coupled paper-faithful decoder; in NSMPP it is part of a direct joint-intensity kernel. The framework is unified at the level of assembly and interfaces, not at the level of a universal temporal-spatial parameterization.

The objective layer is likewise explicit. Each event model declares a training objective, a logging key, and a benchmark-facing NLL contract separately. Exact models set `training_loss = eval_nll` up to aggregation or regularization. Neural STPP optimizes exact NLL plus energy regularization while still reporting pure NLL. DeepSTPP can emit a latent KL term on the state side, and the trainer attaches the final \(\beta\,\mathrm{KL}\) weight outside the model core. NSMPP uses the same exact likelihood for training and testing but optimizes the sequence mean during training and reports per-event NLL at evaluation. Diffusion and SMASH fit the same outer contract without pretending to be exact-intensity models: their event models define ELBO or score-matching objectives, expose approximate test NLL paths, and provide native samplers instead of calibrated intensities.

This separation is what allows exact-intensity, factorized, and generative families to coexist in one framework. Exact models expose `intensity` and, in the factorized case, `density`, so surfaces and thinning-based predictive rollouts are evaluated from calibrated quantities. Generative families instead expose `sample_native`; their surface interface is explicitly downgraded to a proxy KDE over samples, and their test NLL is marked approximate. The same outer wrapper therefore supports three scientifically distinct regimes: exact likelihood models, exact but differently aggregated likelihood models, and sample-based models with approximate likelihood reporting.

Operationally, the shared execution path is minimal: `UnifiedSTPP` asks \(S\) to encode history, forwards the resulting context to \(E\), standardizes the returned loss dictionary, and leaves optimizer, checkpoint selection, and optional KL weighting to the training wrapper. The framework is thus unified by a small number of stable contracts, while the mathematics of the event law remain family-owned.

## Design Axes

- History representation: passthrough history, continuous-time backbone state, paper-window latent state, or flattened next-event conditioning tokens.
- Event law: factorized temporal-plus-spatial likelihood, coupled exact decoder, direct joint intensity, or generative denoiser / score model.
- Evaluation exposure: exact intensity or density, exact but family-specific NLL aggregation, or native sampling with approximate NLL.

## Scope Note

The current interface contains explicit `query_state` and `sequence_states` hooks, but most active presets reuse the encoded history directly rather than learning a distinct query-time state transformation. The clean unification in the codebase is therefore the state-context/event-law split, not a deeper claim that every family shares the same internal state dynamics or the same likelihood decomposition.
