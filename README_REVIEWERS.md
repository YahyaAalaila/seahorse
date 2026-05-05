# Reviewer Instructions

This repository contains the source package, documentation, tests, and
reproduction entry points for the anonymous submission.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Basic Checks

```bash
python -m pytest
python -m unified_stpp --help
python -m unified_stpp fit --help
python -m unified_stpp bench --help
python -m unified_stpp evaluate --help
```

## Documentation

The documentation source starts at [docs/index.md](docs/index.md). Additional
review and reproduction notes are in [docs/paper-reproduction.md](docs/paper-reproduction.md).
