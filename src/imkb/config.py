"""
Configuration management for imkb

Provides pydantic-based configuration with environment variable support
and YAML file loading capabilities.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import observability config if available
try:
    from .observability.config import TelemetryConfig

    TELEMETRY_AVAILABLE = True
except ImportError:
    TelemetryConfig = None
    TELEMETRY_AVAILABLE = False


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""

    provider: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "imkb_vectors"
    embedding_model_dims: int = 1536


class GraphStoreConfig(BaseModel):
    """Graph store configuration"""

    provider: str = "neo4j"
    url: str = "neo4j://localhost:7687"
    username: str = "neo4j"
    password: str = "password123"
    database: str = "neo4j"


class Mem0Config(BaseModel):
    """Mem0 hybrid storage configuration"""

    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    history_db_path: Optional[str] = None
    version: str = "v1.1"


class LLMRouterConfig(BaseModel):
    """Single LLM router configuration"""

    provider: str = "openai"  # "openai", "llama_cpp", etc.
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = "sk-placeholder-test-key"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout: float = Field(default=30.0, gt=0)
    # For llama.cpp
    gpu_layers: Optional[int] = None
    model_path: Optional[str] = None


class LLMConfig(BaseModel):
    """LLM configuration"""

    default: str = "openai_default"
    routers: dict[str, LLMRouterConfig] = Field(
        default_factory=lambda: {"openai_default": LLMRouterConfig()}
    )


class ExtractorConfig(BaseModel):
    """Individual extractor configuration"""

    timeout: float = 5.0
    max_results: int = 10
    enabled: bool = True
    # Additional extractor-specific config
    config: dict[str, Any] = Field(default_factory=dict)


class ExtractorsConfig(BaseModel):
    """All extractors configuration"""

    enabled: list[str] = Field(default_factory=lambda: ["test"])
    test: ExtractorConfig = Field(default_factory=ExtractorConfig)
    mysqlkb: ExtractorConfig = Field(default_factory=ExtractorConfig)
    # rhokp: ExtractorConfig = Field(default_factory=ExtractorConfig)


class FeaturesConfig(BaseModel):
    """Feature flags configuration"""

    mem0_graph: bool = True
    solr_kb: bool = False
    playwright_kb: bool = False
    local_llm: bool = False
    cloud_llm: bool = True


class TelemetryConfig(BaseModel):
    """Telemetry and observability configuration"""

    otlp_endpoint: Optional[str] = None
    enable_metrics: bool = False
    enable_tracing: bool = False
    service_name: str = "imkb"
    environment: str = "development"


class PerformanceConfig(BaseModel):
    """Performance settings"""

    recall_timeout: float = 2.0
    llm_timeout: float = 10.0
    max_concurrent: int = 5


class SecurityConfig(BaseModel):
    """Security settings"""

    enable_audit_log: bool = True
    max_prompt_length: int = 4096
    sanitize_inputs: bool = True


class ImkbConfig(BaseSettings):
    """Main imkb configuration"""

    model_config = SettingsConfigDict(
        env_prefix="IMKB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Core configuration sections
    mem0: Mem0Config = Field(default_factory=Mem0Config)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    extractors: ExtractorsConfig = Field(default_factory=ExtractorsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Global settings
    namespace: str = "default"
    log_level: str = "INFO"

    @classmethod
    def load_from_file(cls, config_path: str = "imkb.yml") -> "ImkbConfig":
        """Load configuration from YAML file with environment variable override"""
        import yaml

        config_file = Path(config_path)
        config_data = {}

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        # Create instance with file data, env vars will override automatically
        return cls(**config_data)

    def get_mem0_config(self) -> dict[str, Any]:
        """Get Mem0-compatible configuration dict"""
        return {
            "vector_store": {
                "provider": self.mem0.vector_store.provider,
                "config": {
                    "host": self.mem0.vector_store.host,
                    "port": self.mem0.vector_store.port,
                    "collection_name": self.mem0.vector_store.collection_name,
                    "embedding_model_dims": self.mem0.vector_store.embedding_model_dims,
                },
            },
            "graph_store": {
                "provider": self.mem0.graph_store.provider,
                "config": {
                    "url": self.mem0.graph_store.url,
                    "username": self.mem0.graph_store.username,
                    "password": self.mem0.graph_store.password,
                    "database": self.mem0.graph_store.database,
                },
            },
            "version": self.mem0.version,
        }

    def get_llm_router_config(
        self, router_name: Optional[str] = None
    ) -> LLMRouterConfig:
        """Get LLM router configuration"""
        router_name = router_name or self.llm.default
        if router_name not in self.llm.routers:
            raise ValueError(f"LLM router '{router_name}' not found in configuration")
        return self.llm.routers[router_name]


# Global configuration instance
_config: Optional[ImkbConfig] = None


def get_config() -> ImkbConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = ImkbConfig.load_from_file()
    return _config


def set_config(config: ImkbConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config
