.PHONY: help install install-dev test lint format check type-check security-check clean pre-commit

help:  ## Display this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv sync --no-dev

install-dev:  ## Install development dependencies
	uv sync --group dev --group lint --group types

test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov=src/imkb --cov-report=term-missing --cov-report=html

lint:  ## Run all linting tools
	uv run ruff check .
	uv run black --check .
	uv run isort --check-only .

format:  ## Format code
	uv run black .
	uv run isort .
	uv run ruff check --fix .

type-check:  ## Run type checking
	uv run mypy src/

security-check:  ## Run security checks
	uv run bandit -r src/

check: lint type-check security-check  ## Run all checks (lint, type, security)

pre-commit-install:  ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	uv run pre-commit run --all-files

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	uv build

publish:  ## Publish to PyPI (requires authentication)
	uv publish