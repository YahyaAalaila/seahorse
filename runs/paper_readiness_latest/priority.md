# Synthetic Suite 3/4 Priority Status

Audit time: 2026-04-29 22:12 CEST.

Scope: synthetic suite 3 `suite3_entanglement` and suite 4 `suite4_heterogeneity` only. A cell is counted as evaluated only when `evaluate/metrics/<profile>_test/metrics.json` exists and contains numeric metric values.

Relevant evaluation bundles:

- `surface_test`: ground-truth intensity metrics for synthetic data.
- `predictive_test`: next-event predictive metrics.
- Bench/test-NLL results remain in the existing suite 3/4 test-NLL tables.

## Closed

| suite | preset | training cells | surface metrics | predictive metrics | status |
| --- | ---: | ---: | ---: | ---: | --- |
| suite3 | `auto_stpp` | 12/12 | 12/12 | 12/12 | closed |
| suite3 | `deep_stpp` | 12/12 | 12/12 | 12/12 | closed |
| suite4 | `auto_stpp` | 12/12 | 12/12 | 12/12 | closed |
| suite4 | `deep_stpp` | 12/12 | 12/12 | 12/12 | closed |

## Needs Evaluation Only

These have completed training cells, but no `surface_test` or `predictive_test` metrics yet.

| suite | preset | training cells | surface metrics | predictive metrics | next action |
| --- | ---: | ---: | ---: | ---: | --- |
| suite3 | `nsmpp` | 12/12 | 0/12 | 0/12 | launch CPU metrics eval |
| suite3 | `rmtpp_gmm` | 12/12 | 0/12 | 0/12 | launch CPU metrics eval |
| suite3 | `thp_gmm` | 12/12 | 0/12 | 0/12 | launch CPU metrics eval |
| suite4 | `nsmpp` | 12/12 | 0/12 | 0/12 | launch CPU metrics eval |
| suite4 | `rmtpp_gmm` | 12/12 | 0/12 | 0/12 | launch CPU metrics eval |
| suite4 | `thp_gmm` | 12/12 | 0/12 | 0/12 | launch CPU metrics eval |

Launch after pulling the latest branch:

```bash
cd ~/projects/uni-stpp

export UNIFIED_STPP_PYTHON=/netscratch/aalaila/venvs/unified_stpp-py312/bin/python
export INSTALL_PROJECT=0
export HF_HOME=/netscratch/aalaila/.cache/huggingface
export PIP_CACHE_DIR=/netscratch/aalaila/.cache/pip

METRIC_PROFILES=predictive,surface \
CPU_CPUS_PER_TASK=4 CPU_MEM=24G CPU_TIME_LIMIT=08:00:00 \
K_PRED=32 EXACT_TIME_BINS=8 EXACT_SPATIAL_BINS=8 \
scripts/submit_hawkesnest_campaign_eval.sh \
  runs/hawkesnest_campaigns/suite3_entanglement/s3ent_v2__nsmpp__04251144 \
  runs/hawkesnest_campaigns/suite3_entanglement/s3ent_v2__rmtpp_gmm__04251144 \
  runs/hawkesnest_campaigns/suite3_entanglement/s3ent_v2__thp_gmm__04251144 \
  runs/hawkesnest_campaigns/suite4_heterogeneity/s4het_v2__nsmpp__04251145 \
  runs/hawkesnest_campaigns/suite4_heterogeneity/s4het_v2__rmtpp_gmm__04251145 \
  runs/hawkesnest_campaigns/suite4_heterogeneity/s4het_v2__thp_gmm__04251145
```

## Not Yet Trained

The full factorized family campaign directories currently contain manifests only, not run outputs:

- `poisson_gmm`
- `hawkes_gmm`
- `selfcorrecting_gmm`
- `poisson_cnf`
- `hawkes_cnf`
- `selfcorrecting_cnf`
- `poisson_tvcnf`
- `hawkes_tvcnf`
- `selfcorrecting_tvcnf`

Launch training first if these are still required:

```bash
cd ~/projects/uni-stpp

export UNIFIED_STPP_PYTHON=/netscratch/aalaila/venvs/unified_stpp-py312/bin/python
export INSTALL_PROJECT=0
export HF_HOME=/netscratch/aalaila/.cache/huggingface
export PIP_CACHE_DIR=/netscratch/aalaila/.cache/pip

GPUS=0 DEVICE=cpu CPUS_PER_TASK=8 MEM=32G TIME_LIMIT=24:00:00 \
scripts/submit_hawkesnest_suite_family.sh suite3_entanglement factorized

GPUS=0 DEVICE=cpu CPUS_PER_TASK=8 MEM=32G TIME_LIMIT=24:00:00 \
scripts/submit_hawkesnest_suite_family.sh suite4_heterogeneity factorized
```

After training outputs exist, evaluate the resulting `s3ent_v2__fact__...` and `s4het_v2__fact__...` campaign roots with `METRIC_PROFILES=predictive,surface`.

## Excluded Or Blocked

- **EXCLUDED: `neural_attncnf` surface/predictive evaluation for suite 3/4.** The failure mode is CUDA OOM inside attention/CNF divergence-style evaluation, not missing GT files.
- **EXCLUDED: `neural_jumpcnf` suite 3/4 evaluation for now.** The current suite3 JumpCNF eval job had no metric artifacts in the last audit and JumpCNF real/synthetic runs are not on pace for the deadline under the current recipe.
- `neural_cond_gmm` predictive eval was blocked by multi-time proposal-cache calls into `NeuralSTPPEventModel.intensity()`. The local fix changes the eval caller to query one fixed time-slice per batch; relaunch predictive eval for this preset only after pulling the fix.
