# Benchmark Contract

This doc defines the public benchmark contract for `unified_stpp`.

For the cluster-side execution flow that uses this contract in production, see
[PEGASUS_CAMPAIGN.md](PEGASUS_CAMPAIGN.md).

## Scope

Benchmark tables include **all models**. The table itself is inclusive; result
interpretation comes from the metadata stored with each run and benchmark
report, not from hiding approximate or provisional families.

The key metadata fields are:

- `nll_kind`: `exact`, `approx`, or `none`
- `nll_report_space`: `raw` or `native`
- `preset_status`: `canonical`, `provisional`, `deprecated`, or `legacy`

Those fields are preserved in `run_result.json`, carried into benchmark
aggregation, and should stay available for downstream reporting and LaTeX
footnote logic.

## Defaults

The frozen benchmark contract is raw-first:

- `BenchmarkConfig.protocol = "raw"`
- `BenchmarkConfig.normalize = false`
- `BenchmarkConfig.checkpoint_select = "best"`
- `BenchmarkConfig.test_nll_space = "raw"`

That means benchmark inputs remain in original dataset coordinates. Some model
families may still fit and apply model-owned transform artifacts internally,
but benchmark inputs and benchmark-facing reporting semantics are tracked
explicitly in artifacts.

## Status Groups

Current public status groups are:

- `canonical` exact families:
  `poisson_gmm`, `hawkes_gmm`, `selfcorrecting_gmm`, `poisson_cnf`,
  `hawkes_cnf`, `selfcorrecting_cnf`, `poisson_tvcnf`, `hawkes_tvcnf`,
  `selfcorrecting_tvcnf`, `deep_stpp`, `auto_stpp`
- `canonical` approximate-reporting families:
  `smash`, `diffusion_stpp`
- `provisional` exact families:
  `nsmpp`, `neural_cond_gmm`, `neural_jumpcnf`, `neural_attncnf`
- `legacy` public preset:
  `auto_stpp_legacy`

Deprecated aliases remain accepted for compatibility, but new configs and run
artifacts canonicalize them to the public preset IDs.

## Outputs

A benchmark run writes benchmark-level outputs such as:

- `bench_meta.json`
- `results.json`
- `report.html`
- all-model and derived table exports such as `table_*_all.csv` and
  `table_*_exact.csv` when those views are available

Per-run directories continue to carry:

- `config.yaml`
- `resolved_config.yaml`
- `run_result.json`
- `artifacts.json`
- `metrics.csv`
- checkpoints

Benchmark-aligned predictive artifacts come from:

- `python -m unified_stpp evaluate metrics --metric-profile predictive ...`

That path evaluates held-out next-event prediction on teacher-forced test
prefixes and saves the context-level and per-sequence predictive summary
artifacts.

`evaluate predictive-compare` remains a separate qualitative future-window
visualization workflow and should not be described as the primary benchmark
artifact path.

## Compatibility

Some historical preset IDs remain load-compatible for backward compatibility,
but they are not part of the public benchmark surface and are intentionally
omitted from public preset lists and examples.

## Minimal Example

```bash
python -m unified_stpp bench \
    --presets auto_stpp deep_stpp smash diffusion_stpp \
    --splits_dir data/bench_splits \
    --no-normalize
```

This preserves the frozen raw-first benchmark contract while keeping all models
visible in the resulting benchmark tables.
