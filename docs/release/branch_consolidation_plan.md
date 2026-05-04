# Release Baseline State

This file intentionally does not provide a broad branch consolidation plan. For v1, the current repository is the source of truth and release cleanup should start from the latest integrated pre-cleanup state.

## Baseline To Use

Use `release/v1-integration` at commit `a3a954a` as the v1 cleanup baseline.

Baseline source:

- Branch: `codex/predictive-test-nll-bench`
- Commit: `a3a954a`
- Subject: `Add real-data curve bench launcher`

`release/v1-integration` was created from that state. Do not merge active branches for v1 cleanup unless a specific missing change is identified later.

## Branch Policy For v1

- Ignore legacy/history concerns.
- Do not produce branch-by-branch consolidation work for v1.
- Do not fetch, pull, push, merge, or cherry-pick without an explicit request.
- Implement release cleanup directly on `release/v1-integration`.
- Validate the release candidate from the current repository state, not from historical branch labels.
