.PHONY: help install test clean lint format type-check dev-install build

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

dev-install:  ## Install with development dependencies
	uv sync --dev

test:  ## Run tests
	uv run pytest tests/ -v

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

lint:  ## Run linting (flake8)
	uv run flake8 flowplan_g2p/ tests/ main.py

format:  ## Format code with Black
	uv run black flowplan_g2p/ tests/ main.py

type-check:  ## Run type checking with mypy
	uv run mypy flowplan_g2p/

check: lint type-check  ## Run all checks (lint + type-check)

build:  ## Build the package
	uv build

cli-test:  ## Test CLI with validation only (requires provider API key)
	uv run flowplan-g2p --validate-only --provider openai

dev-setup:  ## Complete development setup
	@echo "Setting up development environment..."
	uv sync --dev
	@echo "Development environment ready!"
	@echo "Run 'make help' to see available commands"

quality: format lint type-check  ## Run all quality checks and formatting

# Documentation
docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

# Release
release-check: clean quality test build  ## Check if ready for release
	@echo "✅ Release check completed successfully"
