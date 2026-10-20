.PHONY: lint test format

lint:
	ruff check seahorse tests scripts

test:
	pytest

format:
	ruff format seahorse tests scripts

