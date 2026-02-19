.PHONY: lint test format

lint:
	ruff check tests scripts

test:
	pytest

format:
	ruff format tests scripts

