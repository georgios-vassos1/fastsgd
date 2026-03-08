.PHONY: install build clear-cache test

install:
	uv sync --extra dev

build:
	uv build

clear-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache dist build *.egg-info

test:
	uv run pytest tests/ -q --cov=fastsgd --cov-report=term-missing
