# imkb

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

## Development Status

**Current**: Project initialization and design phase
**Next**: MVP implementation with Mem0 hybrid recall

### Roadmap

- **Phase 1**: Core recall + inference (Q1 2024)
  - Mem0 hybrid vector+graph storage
  - Single LLM client (local llama.cpp)
  - MySQL KB extractor
  - Basic error handling

- **Phase 2**: Production features (Q2 2024)  
  - Multi-tenant isolation
  - OpenTelemetry observability
  - Multi-LLM routing
  - Action pipeline

- **Phase 3**: Extensions (Q3 2024)
  - Additional extractors
  - Web UI
  - Performance optimizations

## Documentation

- [Design Document](DESIGN.md) - Detailed architecture and design decisions
- [Development Guide](CLAUDE.md) - Development environment and patterns

## Contributing

This project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
# Clone and setup
git clone <repo>
cd imkb
uv sync

# Install development dependencies
uv add --dev pytest pytest-asyncio black isort mypy

# Run tests
uv run pytest

# Format code
uv run black src/ tests/
uv run isort src/ tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [Mem0](https://github.com/mem0ai/mem0) - Hybrid memory storage for AI applications
- [LangChain](https://github.com/langchain-ai/langchain) - Framework for developing LLM applications  
- [Qdrant](https://github.com/qdrant/qdrant) - Vector similarity search engine