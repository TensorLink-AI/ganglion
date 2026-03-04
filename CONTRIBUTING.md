# Contributing to Ganglion

## Prerequisites

- Python 3.11+
- Git

## Setup

```bash
git clone https://github.com/TensorLink-AI/ganglion.git
cd ganglion
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=ganglion --cov-report=term-missing
```

## Code Style

- Python: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- Boolean fields use `is_`, `has_`, `should_`, `can_` prefixes
- All public functions require docstrings and type hints
- Format with `ruff format`, lint with `ruff check`

## Making Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests for new functionality
4. Ensure all tests pass and coverage stays above 80%
5. Submit a pull request

## Architecture

See the [README](README.md) for the 5-layer architecture overview. Key design principles:

- **Extract from two, not one** — abstractions validated against 2+ subnets
- **Hooks over hardcodes** — recovery and detection are overridable
- **Contracts over conventions** — typed interfaces, not string keys
- **Thin orchestration** — orchestrator only sequences and routes
