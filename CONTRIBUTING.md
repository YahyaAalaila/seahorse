# Contributing to Seahorse

Thanks for helping improve Seahorse. Contributions are most useful when they are
small, reproducible, and easy to review.

## Ways To Contribute

- Improve documentation, tutorials, examples, and Colab notebooks.
- Report bugs with a minimal reproduction.
- Add tests for existing behavior.
- Add model presets, dataset adapters, or evaluation utilities.
- Improve benchmark reliability, logging, and artifact inspection.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
```

On Windows, activate the environment with:

```powershell
.\.venv\Scripts\activate
```

## Before Opening A Pull Request

Run the focused checks that match your change:

```bash
python -m pytest
mkdocs build --strict
```

For documentation-only changes, `mkdocs build --strict` is the minimum expected
check. For model, runner, benchmark, or data-loading changes, include tests or a
small reproducible command that exercises the changed path.

## Pull Request Guidelines

- Keep the PR focused on one topic.
- Explain what changed and why.
- Include screenshots for visual documentation changes.
- Link related issues when applicable.
- Avoid committing generated outputs such as `site/`, temporary HTML previews,
  caches, checkpoints, or local run directories.
- Do not commit secrets, private datasets, local paths, cluster names, or
  institution-private infrastructure identifiers.

## Adding Models Or Datasets

New model presets should follow the existing registry and config pattern, expose
capabilities explicitly, and include at least a small construction or smoke-fit
test. New dataset documentation should describe the data source, license or
access constraints, expected JSONL schema, and reproducible conversion steps.

## Documentation Style

Use confident but precise language. Seahorse is a framework for reproducible STPP
research, benchmarking, and offline evaluation. Avoid overclaiming production
serving, streaming deployment, or results that are not demonstrated by the code.
