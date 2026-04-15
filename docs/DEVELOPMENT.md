# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
pre-commit install
```

## Common Commands

```bash
make lint
make test
make format
```
