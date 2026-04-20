# REPO_CLEANUP_PLAN.md

## Purpose

This file is the cleanup contract for the repo.

It exists to prevent random local cleanup, architectural drift, and “small fixes” that silently change benchmark semantics.

The cleanup is split into two explicit phases:

- **Phase A — benchmark-ready cleanup**
- **Phase B — release/publication cleanup**

Every cleanup proposal and implementation should be checked against this file.

---

## Non-negotiable rules

### 1. Do not change benchmark semantics silently
Cleanup may rename, move, delete dead code, and narrow public interfaces, but it must not silently change:
- model mechanics
- training objective
- reporting semantics
- evaluation semantics
- benchmark artifact semantics

If a cleanup would change any of the above, it is **not cleanup**. It is a method/evaluation change and must be treated separately.

### 2. Respect repo layering
The repo is modular by design. Cleanup must preserve that.

The intended separation of concerns is:

- **config / config factory**
  - preset definition
  - model family knobs
  - defaults
  - configuration validation
  - construction-time wiring
  - no runtime benchmark logic here

- **runner**
  - orchestration of fit/test/load
  - thin execution boundary
  - no model-family-specific hacks unless absolutely unavoidable
  - no evaluation-specific logic unless explicitly designated and minimal

- **models**
  - actual modeling semantics
  - state/history handling
  - event likelihood / intensity / density logic
  - family-specific internal preprocessing
  - kernel / decoder / temporal / spatial modules

- **data**
  - loading
  - validation
  - canonical dataset contract
  - batching/collation
  - transform artifacts where explicitly shared
  - no model-specific parsing hidden in data code

- **evaluation**
  - post-fit scientific objects
  - predictive comparison
  - diagnostic surfaces
  - metric computation over saved artifacts
  - artifact schemas/manifests
  - no training logic here

- **viz**
  - rendering only
  - consume saved bundles/artifacts/results
  - should not own model execution logic

- **CLI**
  - thin argument mapping to the correct layer
  - should not become a second orchestration framework

### 3. Public names matter
Preset names, CLI names, and doc-facing terms should be:
- publication-ready
- stable
- short
- honest

Avoid names that expose implementation history, temporary phases, or internal scaffolding.

### 4. Prefer deletion over indefinite legacy clutter
If a file/class/path is dead, duplicated, or superseded, remove it once parity is proven and compatibility risk is understood.

Do not keep zombie code “just in case” unless it is explicitly marked as:
- compatibility shim
- legacy
- reference-only

### 5. Prefer bounded cleanup
Do not rewrite broad areas of the repo at once.
Use bounded passes with explicit goals and acceptance criteria.

---

## Repo ethos

The repo should remain:

- **modular**
- **separated by responsibility**
- **benchmark-defensible**
- **reproducible**
- **honest about exact vs approximate methods**
- **honest about canonical vs provisional vs legacy presets**

Good cleanup reduces:
- ambiguity
- overlap
- stale aliases
- dead files
- awkward naming
- confusing public interfaces

Good cleanup does **not** chase aesthetic purity at the cost of stability.

---

## Cleanup phases

# Phase A — Benchmark-ready cleanup

## Goal
Make the repo clean enough to launch benchmark experiments confidently, without changing benchmark mechanics.

## Allowed in Phase A
- rename presets to final public names
- add deprecated aliases where needed
- remove dead files
- remove dead classes
- remove stale references
- tighten public exports
- retire superseded temp/reference paths once parity is proven
- clean runtime path mismatches
- clean benchmark-facing docs/config names
- mark provisional vs canonical vs legacy clearly

## Not allowed in Phase A
- changing model semantics
- changing training objectives
- changing benchmark definitions
- changing evaluation semantics without explicit signoff
- broad “nice architecture” refactors that risk behavior
- release-polish churn that does not help benchmark execution

## Phase A focus areas

### A1. Preset naming cleanup
For every preset:
- check whether the name is publication-ready
- check whether the runtime path matches the intended final preset name
- identify stale aliases
- identify dead historical variants
- identify whether it should be:
  - `canonical`
  - `provisional`
  - `deprecated`
  - `legacy`

Examples:
- transitional migration names should not survive as public benchmark names
- short public names should carry publication-facing identity, with status metadata carrying the provisional/canonical distinction where needed (for example, `nsmpp`)

### A2. Runtime path cleanup
Check whether:
- config names
- preset registry names
- run directory names
- docs/examples
- CLI-facing names

are aligned.

The public-facing identifier should be stable and clean. Old names may remain only as explicit deprecated aliases.

### A3. Dead code / dead file cleanup
Flag and remove:
- unused files
- stale helpers
- old temporary wrappers
- superseded modules
- code paths that are no longer reachable
- unused low-level exports

### A4. Layer boundary cleanup
Flag areas where responsibilities drifted:
- evaluation logic hidden in runner
- model-family hacks in CLI
- config logic leaking into runtime code
- viz functions sitting inside evaluation/model code
- old helpers that no longer belong where they currently live

When such a problem is found, propose the fix using the repo layering rules above.

### A5. Public interface cleanup
Check:
- public preset names
- public evaluation subcommands
- public data-loading entrypoints
- what is exported from `__init__.py` modules
- whether old/internal APIs are being exposed as if they are stable

### A6. Benchmark-facing doc/config cleanup
Before experiments:
- docs and config examples must use canonical names
- benchmark-facing docs must match the actual contract
- stale examples should be removed or updated

## Phase A acceptance criteria
Before Phase A is considered done:

- [ ] public preset names are benchmark/publication-ready
- [ ] preset statuses are explicit and correct
- [ ] deprecated aliases are bounded and intentional
- [ ] dead files/classes are removed or explicitly marked legacy/reference-only
- [ ] runtime path naming is aligned with public preset naming
- [ ] no obvious layer-boundary violations remain in benchmark-critical paths
- [ ] benchmark-facing docs/examples use the real current interfaces
- [ ] cleanup did not change benchmark mechanics

---

# Phase B — Release/publication cleanup

## Goal
Polish the repo for public release and paper-facing presentation while experiments are running, without invalidating reproducibility.

## Allowed in Phase B
- README cleanup
- docs polishing
- notebook cleanup
- examples
- public API niceties
- package polish
- architecture explanation
- release-facing cleanup
- better tutorial flow

## Caution in Phase B
Any cleanup that could alter benchmark results/mechanics must still be treated as risky and require explicit signoff.

## Phase B focus areas
- documentation clarity
- released public API polish
- notebook UX
- examples/tutorials
- repo layout clarity
- architecture description for paper/release
- remaining non-critical cleanup

## Phase B acceptance criteria
- [ ] public release/docs are coherent
- [ ] getting-started path is clean
- [ ] benchmark architecture is documented honestly
- [ ] repo no longer looks like an active migration project
- [ ] no release polish changed experiment semantics

---

## How cleanup proposals should be evaluated

For every flagged issue, answer these:

1. **What is the issue?**
2. **Why is it a problem?**
   - naming
   - dead code
   - runtime mismatch
   - layer violation
   - public-interface confusion
   - docs mismatch
3. **Which phase does it belong to?**
   - Phase A
   - Phase B
   - not cleanup / separate architectural change
4. **What is the minimal safe fix?**
5. **What must not be broken by that fix?**
6. **What tests or smoke checks should confirm safety?**

---

## Cleanup issue template

Use this template when identifying cleanup items:

### Item ID
`CLEANUP-XXX`

### Title
Short descriptive title

### Category
- naming
- path/runtime mismatch
- dead file
- dead class
- stale alias
- layer violation
- export surface
- docs/examples
- release polish

### Phase
- Phase A
- Phase B

### Current state
What exists now

### Desired state
What should be true after cleanup

### Why it matters
Why this improves the repo

### Minimal safe change
The bounded change to make

### Must not break
List what must remain unchanged

### Validation
Tests / smoke checks / manual checks

### Status
- todo
- in_progress
- done
- deferred
- rejected

---

## Current known cleanup targets

### Preset names
- [ ] audit every preset name for publication-readiness
- [ ] remove or deprecate transitional names
- [ ] ensure runtime path and public preset name match
- [x] canonicalize `nsmpp_deepbasis_provisional` to public preset `nsmpp` and keep the old name as a hidden deprecated alias

### Runtime path / registry consistency
- [ ] check preset registry vs docs vs run dirs vs CLI references
- [ ] remove stale aliases where no longer needed
- [x] keep `unified_stpp.registry` explicitly compatibility-only during Phase A
- [ ] stop stale docs/comments from implying `unified_stpp.registry` is the live build path
- [x] before shipping, migrate remaining internal test usage off `unified_stpp.registry`
- [ ] before shipping, decide whether the shim can be removed entirely or must remain as an explicit compatibility surface

### Evaluation layer
- [ ] continue tightening file/module overlap where clearly justified
- [ ] preserve the split between:
  - predictive comparison
  - surface diagnostics
  - metrics/artifacts
  - viz
- [x] consolidate the surface API onto `unified_stpp.evaluation.surface` and remove the old `surface_query.py` split
- [ ] flag any remaining line-level/local cleanup opportunities without re-opening semantics

### Data layer
- [ ] keep benchmark-safety fixes separate from architecture overbuilding
- [ ] support thin official HF-backed dataset API with local fallback
- [ ] avoid premature file explosion

### Dead/stale code
- [ ] identify superseded temp/reference files
- [ ] identify unused old helper modules
- [ ] identify stale exports and wrappers
- [ ] identify dead tests tied to retired paths

### Docs/examples
- [ ] ensure benchmark-facing docs use canonical names
- [ ] ensure public evaluation/data API shown in docs matches real package APIs
- [ ] delay nonessential polish until Phase B

### Compatibility shims
- [ ] keep compatibility shims narrow and explicitly labeled
- [ ] do not let compatibility shims appear as primary public APIs
- [ ] re-audit `unified_stpp.registry` in Phase B for removal vs retention

---

## Things to explicitly avoid

Do not:
- rename everything at once without a manifest/alias plan
- turn cleanup into a large architecture rewrite
- add abstract frameworks for their own sake
- move files only because the tree “looks nicer”
- keep multiple names alive forever
- let cleanup silently alter training/evaluation semantics
- confuse benchmark-cleanup work with release-polish work

---

## Working method for the agent

For any cleanup work, use this loop:

1. read this file
2. audit current repo against the relevant phase
3. propose one bounded cleanup chunk
4. implement that chunk only
5. report what changed
6. update this file’s status items if appropriate

The agent should not freewheel beyond the current phase goals.

---

## Status tracker

### Phase A
- [ ] preset naming audit complete
- [ ] runtime path cleanup complete
- [ ] dead file/class cleanup complete
- [ ] layer-boundary cleanup complete for benchmark-critical paths
- [ ] benchmark-facing docs/config cleanup complete
- [ ] compatibility shims are bounded and clearly labeled
- [ ] benchmark-safe cleanup validated

### Phase B
- [ ] release-facing docs cleanup complete
- [ ] notebook/example cleanup complete
- [ ] public API polish complete
- [ ] internal tests no longer depend on removable compatibility shims
- [ ] `unified_stpp.registry` removal-vs-retention decision recorded
- [ ] release polish validated

---

## Final rule

When in doubt:
- prefer the smallest change that improves clarity
- prefer preserving reproducibility over elegance
- prefer explicit status labels over silent ambiguity
