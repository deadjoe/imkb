# imkb

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub](https://img.shields.io/github/stars/deadjoe/imkb?style=social)](https://github.com/deadjoe/imkb)

**AI-powered incident knowledge base and root cause analysis SDK for operations**

imkb is a Python SDK that transforms incident/alert events into AI-inferrable context and helps local or remote LLMs generate root cause analysis (RCA) and remediation suggestions.

## Quick Start

```bash
# Install with basic dependencies
uv add imkb

# Or install with full features
uv add "imkb[all]"

# Initialize configuration
cp examples/imkb.example.yml imkb.yml

# Analyze an incident
uv run imkb get-rca --event-file examples/event.example.json
```

## Features

- **Hybrid Knowledge Retrieval**: Combines vector similarity and graph relationships using Mem0
- **Pluggable Architecture**: Support for multiple knowledge sources (MySQL KB, Solr, web scraping)
- **Multi-Model Support**: Works with local LLMs (llama.cpp) and cloud APIs (OpenAI, etc.)
- **Production Ready**: P95 < 800ms performance target with full observability
- **Multi-Tenant**: Secure namespace isolation for enterprise environments

## Architecture

```
Event → Extractor.match() → Mem0 Hybrid Recall → LLM Inference → RCA Result
                         ↗ Vector Search
                         ↗ Graph Traversal  
                         ↗ External KB Query
```

## Installation Options

```bash
# Minimal installation
uv add imkb

# With Mem0 hybrid storage
uv add "imkb[mem0,graph]"

# With knowledge base connectors
uv add "imkb[kb]"

# With local LLM support  
uv add "imkb[llm-gpu]"

# With cloud LLM support
uv add "imkb[cloud-llm]"

# With observability
uv add "imkb[telemetry]"

# Everything
uv add "imkb[all]"
```

## Usage

### Python API

```python
import asyncio
from imkb import get_rca, gen_playbook

async def main():
    # Analyze an incident
    event = {
        "id": "alert-001",
        "signature": "mysql_connection_pool_exhausted", 
        "severity": "P1",
        "source": "prometheus",
        "labels": {"service": "mysql-primary"},
        "message": "Connection pool exhausted: 150/150 active"
    }
    
    rca = await get_rca(event, namespace="prod-team1")
    print(f"Root cause: {rca['root_cause']}")
    print(f"Confidence: {rca['confidence']}")
    
    # Generate remediation playbook
    playbook = await gen_playbook(rca, namespace="prod-team1")
    print(f"Remediation steps: {playbook['actions']}")

asyncio.run(main())
```

### CLI Usage

```bash
# Analyze incident from file
imkb get-rca --event-file incident.json --namespace prod-team1

# Generate playbook from RCA results  
imkb gen-playbook --rca-file rca.json --namespace prod-team1

# Show configuration
imkb config --show
```

## Configuration

Create `imkb.yml` in your project root:

```yaml
llm:
  default: "deepseek_local"
  routers:
    deepseek_local:
      provider: "llama_cpp"
      model: "deepseek-33b.awq"
      gpu_layers: 20

mem0:
  vector_store:
    provider: "qdrant"
    host: "localhost"
    port: 6333
  graph_store:
    provider: "neo4j"
    url: "neo4j://localhost:7687"

extractors:
  enabled:
    - mysqlkb
    - rhokp
```

See [examples/imkb.example.yml](examples/imkb.example.yml) for full configuration options.

## Documentation

- [Design Document](DESIGN.md) - Detailed architecture and design decisions
- [Development Guide](CLAUDE.md) - Development environment and patterns

## Contributing

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and modern Python tooling:

```bash
# Clone and setup
git clone https://github.com/deadjoe/imkb.git
cd imkb
uv sync --group dev --group lint --group types

# Run tests
uv run pytest

# Code formatting and linting
make format        # Format code with black and isort
make lint          # Run all linting tools
make type-check    # Run mypy type checking
make security-check # Run bandit security checks
make check         # Run all checks (lint, type, security)

# Or run tools individually
uv run black .
uv run isort .
uv run ruff check --fix .
uv run mypy src/
uv run bandit -r src/

# Pre-commit hooks (optional but recommended)
make pre-commit-install
make pre-commit-run
```

### Code Quality Tools

This project uses modern Python tooling for code quality:

- **[Black](https://github.com/psf/black)**: Code formatting
- **[isort](https://github.com/pycqa/isort)**: Import sorting
- **[Ruff](https://github.com/astral-sh/ruff)**: Fast Python linter (replaces flake8, pylint)
- **[MyPy](https://github.com/python/mypy)**: Static type checking
- **[Bandit](https://github.com/PyCQA/bandit)**: Security vulnerability scanner
- **[Pre-commit](https://github.com/pre-commit/pre-commit)**: Git hooks for code quality

### Testing

```bash
# Run tests with coverage
make test-cov

# Run specific test files
uv run pytest tests/test_rca_pipeline.py -v

# Run tests with markers
uv run pytest -m "not slow"
```

### Pre-commit Hooks Usage

Pre-commit is a powerful automation tool that runs code quality checks before each commit. Here's how to use it effectively:

#### 🔧 Initial Setup (One-time)

```bash
# Install pre-commit hooks to git repository
uv run pre-commit install

# Optional: Install hooks for push and commit messages
uv run pre-commit install --hook-type pre-push
uv run pre-commit install --hook-type commit-msg
```

#### 📝 Daily Workflow

**Normal Development and Commits**
```bash
# After modifying code
git add .
git commit -m "your commit message"
# ↑ pre-commit runs automatically, checking your code
```

**When Checks Fail**
```bash
# pre-commit will:
# - Auto-fix fixable issues (formatting)
# - Show manual fixes needed
# - Block the commit

# Review auto-fixes
git diff

# If satisfied with fixes, re-add and commit
git add .
git commit -m "your commit message"
```

#### 🛠️ Common Commands

**Manual Checks (Recommended before committing)**
```bash
# Check all files
uv run pre-commit run --all-files

# Check only staged files
uv run pre-commit run

# Run specific hooks
uv run pre-commit run black
uv run pre-commit run ruff
uv run pre-commit run mypy
```

**Managing Hooks**
```bash
# Update hooks to latest versions
uv run pre-commit autoupdate

# Clean cache (useful for troubleshooting)
uv run pre-commit clean

# Uninstall hooks
uv run pre-commit uninstall
```

#### 📋 Configured Checks

Our pre-commit setup automatically runs:

1. **Basic Checks**
   - YAML syntax validation
   - Trailing whitespace removal
   - End-of-file newline enforcement
   - Large file detection

2. **Code Formatting**
   - **Black**: Automatic Python code formatting
   - **isort**: Import statement sorting

3. **Code Quality**
   - **Ruff**: Fast linting checks
   - **MyPy**: Static type checking
   - **Bandit**: Security vulnerability scanning

4. **Testing**
   - **pytest**: Run test suite to ensure functionality

#### 💡 Best Practices

**Pre-commit Checks**
```bash
# Run checks before committing (recommended)
uv run pre-commit run --all-files
git add .
git commit -m "feature: add new functionality"
```

**Emergency Bypass**
```bash
# Skip pre-commit in emergencies only
git commit -m "urgent fix" --no-verify
```

**Troubleshooting Failures**
```bash
# If MyPy type checking fails
uv run mypy src/  # View specific errors
# Fix type issues, then recommit

# If tests fail  
uv run pytest -v  # Run tests to see issues
# Fix failing tests, then recommit
```

**Team Setup**
```bash
# After cloning the project
git clone <repo>
cd <repo>
uv sync  # Install dependencies
uv run pre-commit install  # Install hooks
```

#### ⚡ Quick Fixes

```bash
# Format code only
uv run black .
uv run isort .

# Quick ruff fixes
uv run ruff check . --fix

# Run specific tests
uv run pytest tests/test_specific.py
```

This setup ensures consistent code quality and smooth team collaboration!

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Related Projects

- [Mem0](https://github.com/mem0ai/mem0) - Hybrid memory storage for AI applications (core dependency)
- [Qdrant](https://github.com/qdrant/qdrant) - Vector similarity search engine (vector storage backend)
- [Neo4j](https://github.com/neo4j/neo4j) - Graph database for relationship-based knowledge retrieval
- [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-python) - Observability framework for distributed tracing and metrics
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient inference of Large Language Models