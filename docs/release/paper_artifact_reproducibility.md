# Paper Artifact Reproducibility

This is the v1/paper release map for paper-facing tables and figures. It records
the current source files, scripts, commands, output paths, and release status.
Long campaign runs are not part of release validation; this document records how
paper artifacts are regenerated from completed runs and uploaded datasets.

## Data Inputs

| Artifact group | Input files | Status |
|---|---|---|
| Real-data benchmark tables | Hugging Face real dataset repos, final repo IDs/revisions pending in README/release docs; completed run artifacts under `runs/exp1/*/bench/**/run_result.json` when available locally. | Pending HF path documentation |
| Suite 3 synthetic tables/figures | Processed suite 3 dataset to be uploaded to Hugging Face; local completed run artifacts under `runs/hawkesnest_campaigns/suite3_entanglement/**`; post-NLL metrics under `runs/local_eval_analysis/suite34_metrics/table_metrics_long.csv`. | Pending HF upload |
| Suite 4 synthetic tables/figures | Processed suite 4 dataset to be uploaded to Hugging Face; local completed run artifacts under `runs/hawkesnest_campaigns/suite4_heterogeneity/**`; post-NLL metrics under `runs/local_eval_analysis/suite34_metrics/table_metrics_long.csv`. | Pending HF upload |
| HawkesNest synthetic generation | Curated generation notebook to be added/linked from release docs; generator scripts include `scripts/generate_hawkesnest_suite_ground_truth.py` and related HawkesNest suite utilities. | Manual notebook documentation pending |

## Paper Tables And Figures

| Table/figure | Input files | Script/notebook | Command or section | Output path | Current status |
|---|---|---|---|---|---|
| Paper readiness status tables | Local run artifacts under `runs/exp1/**`, `runs/hawkesnest_campaigns/**`, and optional cluster status snapshot JSON. | `scripts/paper_readiness_report.py` | `python scripts/paper_readiness_report.py render --snapshot <snapshot.json> --out-dir runs2/paper_readiness_latest` | `runs2/paper_readiness_latest/*.md`, `*.csv` | Reproducible from snapshot; snapshot generation is environment-specific |
| Suite 3 training budget/recovery figures | `runs/hawkesnest_campaigns/suite3_entanglement/**/test_nll_curve.csv`; `runs/local_eval_analysis/suite34_metrics/table_metrics_long.csv`. | `scripts/make_suite3_paper_artifacts.py` | `python scripts/make_suite3_paper_artifacts.py` | `runs/local_eval_analysis/suite3_paper_artifacts/` | Reproducible after suite 3 runs/metrics are present |
| Suite 3 post-NLL diagnostics | `runs/local_eval_analysis/suite34_metrics/table_metrics_long.csv`; optional W1/rollout artifacts under campaign eval outputs. | `scripts/make_suite3_paper_artifacts.py` | `python scripts/make_suite3_paper_artifacts.py` | `runs/local_eval_analysis/suite3_paper_artifacts/` | Reproducible after metrics aggregation |
| Suite 3/4 post-NLL multi-metric figures | `runs/local_eval_analysis/suite34_metrics/table_metrics_long.csv` and `table_metrics_summary.csv`. | `scripts/make_post_nll_hawkesnest_figures.py` | `python scripts/make_post_nll_hawkesnest_figures.py --metrics-dir runs/local_eval_analysis/suite34_metrics --out-dir runs/local_eval_analysis/post_nll_figures` | `runs/local_eval_analysis/post_nll_figures/{main,appendix,tables,logs}/` | Reproducible after suite 3/4 metrics are present |
| Suite 3/4 training-time diagnostics | Completed run artifacts under `runs/hawkesnest_campaigns/suite3_entanglement/**` and `runs/hawkesnest_campaigns/suite4_heterogeneity/**`. | `scripts/make_training_time_diagnostics.py` | `python scripts/make_training_time_diagnostics.py` | `runs/local_eval_analysis/training_time_diagnostics/` | Reproducible after completed run artifacts are present |
| Benchmark repository pipeline figure | No experiment data input; diagram source is code. | `make_fig_benchmark_repo_pipeline.py` | `python make_fig_benchmark_repo_pipeline.py` | `fig_benchmark_repo_pipeline.svg` plus optional PDF/PNG if converters are installed | Reproducible locally |
| Main HawkesNest post-NLL paper artifacts | Aggregated campaign/evaluation outputs under `runs/local_eval_analysis/` and `runs/hawkesnest_campaigns/`. | `scripts/make_post_nll_hawkesnest_main_v2.py` | `python scripts/make_post_nll_hawkesnest_main_v2.py` | Script-defined outputs under `runs/local_eval_analysis/` | Pending final input inventory |

## Release Notes

- Commands above assume the repository root as working directory.
- HF repo IDs and revisions must be filled in before tagging v1.
- Suite 3 and suite 4 processed datasets are pending manual Hugging Face upload.
- The HawkesNest synthetic generation notebook is pending and should become the public generation reference.
- Cluster submission scripts are optional operational helpers, not required public reproduction commands.
