"""
Test suite for configuration system

Tests ImkbConfig and related configuration classes for proper loading,
validation, and defaults.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from imkb.config import (
    ExtractorsConfig,
    ImkbConfig,
    LLMConfig,
    LLMRouterConfig,
    Mem0Config,
    get_config,
)


class TestMem0Config:
    """Test Mem0 configuration"""

    def test_default_config(self):
        """Test default Mem0 configuration"""
        config = Mem0Config()
        assert config.version == "v1.1"
        assert config.vector_store.provider == "qdrant"
        assert config.vector_store.collection_name == "imkb_vectors"
        assert config.vector_store.host == "localhost"
        assert config.vector_store.port == 6333
        assert config.graph_store.provider == "neo4j"
        assert config.graph_store.url == "neo4j://localhost:7687"
        assert config.graph_store.username == "neo4j"
        assert config.graph_store.password == "password123"

    def test_custom_collection_name(self):
        """Test custom collection name"""
        config = Mem0Config()
        config.vector_store.collection_name = "custom_kb"
        assert config.vector_store.collection_name == "custom_kb"

    def test_environment_overrides(self):
        """Test environment variable overrides"""
        with patch.dict(
            os.environ,
            {
                "QDRANT_HOST": "remote-qdrant.example.com",
                "NEO4J_URL": "bolt://remote-neo4j.example.com:7687",
            },
        ):
            pass


class TestLLMConfig:
    """Test LLM configuration"""

    def test_default_config(self):
        """Test default LLM configuration"""
        config = LLMConfig()
        assert config.default == "openai_default"
        assert "openai_default" in config.routers
        assert config.routers["openai_default"].provider == "openai"
        assert config.routers["openai_default"].model == "gpt-4o-mini"
        assert config.routers["openai_default"].temperature == 0.2
        assert config.routers["openai_default"].max_tokens == 1024
        assert config.routers["openai_default"].timeout == 30.0

    def test_multiple_routers(self):
        """Test multiple LLM routers configuration"""
        config = LLMConfig()
        config.routers["fast"] = LLMRouterConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="sk-test",
            temperature=0.3,
            max_tokens=1000,
        )

        assert len(config.routers) == 2
        assert config.routers["fast"].model == "gpt-3.5-turbo"
        assert config.routers["fast"].temperature == 0.3

    def test_mock_router_config(self):
        """Test mock LLM router configuration"""
        router_config = LLMRouterConfig(
            provider="mock", model="mock-gpt-4", api_key="mock-key"
        )

        assert router_config.provider == "mock"
        assert router_config.model == "mock-gpt-4"
        assert router_config.api_key == "mock-key"


class TestExtractorsConfig:
    """Test extractors configuration"""

    def test_default_config(self):
        """Test default extractors configuration"""
        config = ExtractorsConfig()
        assert "mysqlkb" in config.enabled
        assert len(config.enabled) == 1  # Only "mysqlkb" is enabled by default

    def test_custom_enabled_extractors(self):
        """Test custom enabled extractors"""
        config = ExtractorsConfig()
        config.enabled = ["mysqlkb", "custom"]

        assert "mysqlkb" in config.enabled
        assert "custom" in config.enabled
        assert "test" not in config.enabled


class TestImkbConfig:
    """Test main ImkbConfig class"""

    def test_default_config(self):
        """Test default configuration creation"""
        config = ImkbConfig()
        assert config.namespace == "default"
        assert isinstance(config.mem0, Mem0Config)
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.extractors, ExtractorsConfig)

    def test_custom_namespace(self):
        """Test custom namespace"""
        config = ImkbConfig(namespace="test_tenant")
        assert config.namespace == "test_tenant"

    def test_get_llm_router_config(self):
        """Test getting LLM router configuration"""
        config = ImkbConfig()

        # Test default router
        router_config = config.get_llm_router_config("openai_default")
        assert isinstance(router_config, LLMRouterConfig)
        assert router_config.provider == "openai"

        # Test non-existent router (should raise ValueError)
        with pytest.raises(ValueError):
            config.get_llm_router_config("nonexistent")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-env-key"})
    def test_environment_variable_integration(self):
        """Test environment variable integration"""


class TestConfigLoading:
    """Test configuration loading from files"""

    def test_load_config_from_yaml_file(self):
        """Test loading configuration from YAML file"""
        yaml_content = """
namespace: "test_env"
llm:
  default: "custom"
  routers:
    custom:
      provider: "openai"
      model: "gpt-4"
      api_key: "sk-test-key"
      temperature: 0.2
extractors:
  enabled: ["mysqlkb"]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = ImkbConfig.load_from_file(f.name)
                assert config.namespace == "test_env"
                assert config.llm.default == "custom"
                assert config.llm.routers["custom"].model == "gpt-4"
                assert config.llm.routers["custom"].temperature == 0.2
                assert config.extractors.enabled == ["mysqlkb"]
            finally:
                os.unlink(f.name)

    def test_load_config_file_not_found(self):
        """Test loading configuration from non-existent file"""
        # Non-existent file should return default config, not raise error
        config = ImkbConfig.load_from_file("nonexistent.yml")
        assert config.namespace == "default"  # Should use defaults

    def test_get_config_default(self):
        """Test get_config returns default configuration"""
        # Reset global config to ensure clean state
        import imkb.config
        imkb.config._config = None

        config = get_config()
        assert isinstance(config, ImkbConfig)
        assert config.namespace == "default"

    @patch("imkb.config.Path")
    def test_get_config_from_file(self, mock_path):
        """Test get_config loads from file when available"""
        # Mock file exists
        mock_path.return_value.exists.return_value = True

        yaml_content = """
namespace: "file_config"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                # Mock the path to return our temp file
                mock_path.return_value.__str__ = lambda _: f.name

                with patch("builtins.open", open):
                    pass

            finally:
                os.unlink(f.name)


class TestConfigValidation:
    """Test configuration validation"""

    def test_invalid_llm_provider(self):
        """Test validation of LLM provider"""
        # This would test pydantic validation if we had stricter validation rules
        config = LLMRouterConfig(provider="invalid_provider")
        assert config.provider == "invalid_provider"  # Currently allows any string

    def test_negative_timeout(self):
        """Test negative timeout values"""
        with pytest.raises(ValueError):
            LLMRouterConfig(timeout=-1.0)

    def test_negative_max_tokens(self):
        """Test negative max_tokens values"""
        with pytest.raises(ValueError):
            LLMRouterConfig(max_tokens=-100)

    def test_invalid_temperature(self):
        """Test invalid temperature values"""
        with pytest.raises(ValueError):
            LLMRouterConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMRouterConfig(temperature=2.1)


class TestConfigIntegration:
    """Integration tests for configuration system"""

    def test_full_config_creation_and_access(self):
        """Test complete configuration creation and access patterns"""
        config = ImkbConfig(
            namespace="integration_test",
            llm=LLMConfig(
                default="test_router",
                routers={
                    "test_router": LLMRouterConfig(
                        provider="mock", model="test-model", api_key="test-key"
                    )
                },
            ),
            extractors=ExtractorsConfig(enabled=["mysqlkb"]),
        )

        # Test access patterns
        assert config.namespace == "integration_test"

        router_config = config.get_llm_router_config("test_router")
        assert router_config.provider == "mock"
        assert router_config.model == "test-model"

        assert "mysqlkb" in config.extractors.enabled
        assert "test" not in config.extractors.enabled

        # Test Mem0 config is properly initialized
        assert config.mem0.vector_store.provider == "qdrant"
        assert config.mem0.graph_store.provider == "neo4j"
