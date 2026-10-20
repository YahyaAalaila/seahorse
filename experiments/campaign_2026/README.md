# Frozen campaign specifications

Specifications of the frozen fixed-context (H = 20) campaign behind the Seahorse paper's
expanded real-data tables. Campaign identifiers (`iclr2027_*`) are the names the frozen
manifests and the Hugging Face dataset cards carry; they are identifiers, not venues.

| File | Purpose |
|---|---|
| `spec_v1.json` | neural-model campaign (conditional next-event NLL, HPO budget, seeds) |
| `spec_nnstat_v2.json`, `spec_nnstat_pegasus_v1.json` | neural + statistical rosters as run on the two worker clusters |
| `etas_spec_v1.json` | author-ETAS adapter contract (execution gated; see `seahorse.baselines.provenance`) |
| `torch-constraint.txt` | worker torch pin |

Paths inside the specs (`campaign_inputs/...`, `campaign_outputs/...`, `third_party/...`) are
relative to the campaign working directory; the prepared-data root comes from
`SEAHORSE_PREPARED_DATA_ROOT`. Compile and verify a campaign with
`seahorse campaign prepare --spec <spec> --out <dir>`; see `seahorse campaign --help`.
The byte-exact specs used by the worker snapshot are preserved on the `paper/neurips2026` branch
(tag `paper/neurips2026-v9`).
