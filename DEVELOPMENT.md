# Development Guide

This guide covers development setup and best practices for FlowPlan-G2P using `uv`.

## Development Setup

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/repoofideas/flowplan_g2p.git
   cd flowplan_g2p
   ```

3. **Set up development environment**:
   ```bash
   # Create and sync virtual environment with dev dependencies
   uv sync --dev

   # Activate the virtual environment (optional, uv run will use it automatically)
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

## Development Workflow

### Running Tests

All tests run without API keys or network access — they use mocks for LLM calls.

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=flowplan_g2p --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_graph.py -v

# Run a specific test
uv run pytest tests/test_llm_client.py::test_factory_returns_openai_client -v
```

### Code Quality

**Format code with Black:**
```bash
uv run black flowplan_g2p/ tests/ main.py
```

**Lint with flake8:**
```bash
uv run flake8 flowplan_g2p/ tests/ main.py
```

**Type checking with mypy:**
```bash
uv run mypy flowplan_g2p/
```

### Running the CLI

**Development mode:**
```bash
# Run directly from source
uv run flowplan-g2p input.md output.md --provider openai

# Or use the main.py script
uv run python main.py input.md output.md --provider anthropic
```

**Test installation:**
```bash
# Install in editable mode
uv pip install -e .

# Now you can use the command globally
flowplan-g2p input.md output.md
```

## Project Structure

```
flowplan_g2p/
├── flowplan_g2p/           # Main package
│   ├── __init__.py         # Package public API
│   ├── cli.py              # Command-line interface
│   ├── graph.py            # Graph data structures
│   ├── llm_client.py       # Multi-provider LLM client abstraction
│   ├── pipeline.py         # Main pipeline orchestration
│   └── stages/             # Pipeline stages
│       ├── __init__.py
│       ├── concept_induction.py
│       ├── paragraph_planning.py
│       └── graph_generation.py
├── tests/                  # Test suite (runs without API keys)
│   ├── __init__.py
│   ├── test_graph.py       # ConceptGraph unit tests
│   ├── test_llm_client.py  # LLM client and factory tests
│   ├── test_cli.py         # CLI argument parsing tests
│   └── test_pipeline.py    # Pipeline integration tests
├── main.py                 # Alternative CLI entry point
├── .env.example            # Environment variable template
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
├── Makefile                # Development commands
└── .python-version         # Python version specification
```

## Adding Dependencies

**Add runtime dependency:**
```bash
uv add package-name
```

**Add development dependency:**
```bash
uv add --dev package-name
```

**Update dependencies:**
```bash
uv sync
```

## Configuration Files

### pyproject.toml
Main project configuration including:
- Package metadata
- Dependencies (runtime, dev, test, bedrock optional)
- Tool configurations (Black, mypy, pytest)
- Build system configuration

### uv.lock
Automatically generated lock file that pins exact versions of all dependencies for reproducible builds.

### .python-version
Specifies the Python version for the project (used by uv and other tools).

### .env.example
Template for environment variables. Copy to `.env` and fill in your API keys.

## Building and Publishing

### Build package:
```bash
uv build
```

### Install locally:
```bash
uv pip install dist/*.whl
```

## IDE Setup

### VS Code
Recommended extensions:
- Python
- Pylance
- Black Formatter
- flake8

Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true
}
```

### PyCharm
- Set interpreter to `.venv/bin/python`
- Enable Black as external tool
- Configure flake8 for linting

## Common Development Tasks

**Add new stage:**
1. Create module in `flowplan_g2p/stages/`
2. Import in `flowplan_g2p/stages/__init__.py`
3. Integrate in `flowplan_g2p/pipeline.py`
4. Add tests in `tests/`

**Update CLI:**
1. Modify `flowplan_g2p/cli.py`
2. Update `main.py` if needed
3. Test with `uv run flowplan-g2p --help`

**Add new LLM provider:**
1. Create a new subclass of `BaseLLMClient` in `flowplan_g2p/llm_client.py`
2. Add the provider to `create_llm_client` factory function
3. Update CLI `--provider` choices in `cli.py` and `main.py`
4. Add tests in `tests/test_llm_client.py`

**Add new dependencies:**
1. Run `uv add package-name`
2. Update code to use new package
3. Commit both `pyproject.toml` and `uv.lock`

## Troubleshooting

**Virtual environment issues:**
```bash
# Remove and recreate
rm -rf .venv
uv sync
```

**Dependency conflicts:**
```bash
# Update lock file
uv lock --upgrade

# Sync with updated dependencies
uv sync
```

**CLI not working:**
```bash
# Check if properly installed
uv run which flowplan-g2p

# Reinstall in development mode
uv pip install -e .
```
