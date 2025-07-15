"""
Tests for LocalLLMClient

Tests the local LLM client that works with OpenAI-compatible inference services
like Ollama, LMStudio, vLLM, and Text Generation WebUI.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imkb.config import LLMRouterConfig
from imkb.llm_client import LLMResponse, LocalLLMClient


class TestLocalLLMClient:
    """Test LocalLLMClient functionality"""

    def test_local_llm_client_initialization(self):
        """Test LocalLLMClient initialization"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="not-needed"
        )
        client = LocalLLMClient(config)
        assert client.config == config
        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_client_success(self):
        """Test successful client initialization"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="not-needed"
        )
        client = LocalLLMClient(config)

        with patch("openai.AsyncOpenAI") as mock_openai:
            mock_openai_instance = MagicMock()
            mock_openai.return_value = mock_openai_instance

            result = await client._get_client()

            assert result == mock_openai_instance
            mock_openai.assert_called_once_with(
                api_key="not-needed",
                base_url="http://localhost:11434/v1",
                timeout=30.0
            )

    @pytest.mark.asyncio
    async def test_get_client_missing_base_url(self):
        """Test client initialization with missing base_url"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            api_key="not-needed"
        )
        client = LocalLLMClient(config)

        with pytest.raises(ValueError, match="base_url is required for local LLM provider"):
            await client._get_client()

    @pytest.mark.asyncio
    async def test_get_client_missing_openai_package(self):
        """Test client initialization when OpenAI package is missing"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="not-needed"
        )
        client = LocalLLMClient(config)

        with patch("openai.AsyncOpenAI", side_effect=ImportError("No module named 'openai'")):
            with pytest.raises(ImportError, match="OpenAI package not installed"):
                await client._get_client()

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful response generation"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="not-needed",
            temperature=0.7,
            max_tokens=1024
        )
        client = LocalLLMClient(config)

        # Mock OpenAI response
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response from local LLM"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.total_tokens = 50
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "llama3.1:8b"
        mock_response.usage = mock_usage

        with patch.object(client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await client.generate(
                prompt="Test prompt",
                system_prompt="Test system prompt"
            )

            assert isinstance(result, LLMResponse)
            assert result.content == "Test response from local LLM"
            assert result.model == "llama3.1:8b"
            assert result.tokens_used == 50
            assert result.finish_reason == "stop"
            assert result.metadata["base_url"] == "http://localhost:11434/v1"
            assert result.metadata["prompt_tokens"] == 20
            assert result.metadata["completion_tokens"] == 30

            # Verify API call
            mock_client.chat.completions.create.assert_called_once_with(
                model="llama3.1:8b",
                messages=[
                    {"role": "system", "content": "Test system prompt"},
                    {"role": "user", "content": "Test prompt"}
                ],
                temperature=0.7,
                max_tokens=1024
            )

    @pytest.mark.asyncio
    async def test_generate_without_system_prompt(self):
        """Test generation without system prompt"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="not-needed"
        )
        client = LocalLLMClient(config)

        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "llama3.1:8b"
        mock_response.usage = None

        with patch.object(client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await client.generate(prompt="Test prompt")

            assert result.content == "Test response"
            assert result.tokens_used is None

            # Verify API call without system message
            mock_client.chat.completions.create.assert_called_once_with(
                model="llama3.1:8b",
                messages=[
                    {"role": "user", "content": "Test prompt"}
                ],
                temperature=0.2,
                max_tokens=1024
            )

    @pytest.mark.asyncio
    async def test_generate_with_custom_kwargs(self):
        """Test generation with custom parameters"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1"
        )
        client = LocalLLMClient(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "llama3.1:8b"
        mock_response.usage = None

        with patch.object(client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            await client.generate(
                prompt="Test",
                temperature=0.9,
                max_tokens=2048,
                top_p=0.95
            )

            # Verify custom parameters override config defaults
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args["temperature"] == 0.9
            assert call_args["max_tokens"] == 2048
            assert call_args["top_p"] == 0.95

    @pytest.mark.asyncio
    async def test_generate_error_handling(self):
        """Test error handling in generation"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1"
        )
        client = LocalLLMClient(config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception("Connection failed")
            mock_get_client.return_value = mock_client

            with pytest.raises(Exception, match="Connection failed"):
                await client.generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1"
        )
        client = LocalLLMClient(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]

        with patch.object(client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await client.health_check()

            assert result is True
            mock_client.chat.completions.create.assert_called_once_with(
                model="llama3.1:8b",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1"
        )
        client = LocalLLMClient(config)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception("Service unavailable")
            mock_get_client.return_value = mock_client

            result = await client.health_check()

            assert result is False


class TestLocalLLMClientIntegration:
    """Integration tests for LocalLLMClient with different service configurations"""

    def test_ollama_configuration(self):
        """Test configuration for Ollama service"""
        config = LLMRouterConfig(
            provider="local",
            model="llama3.1:8b",
            base_url="http://localhost:11434/v1",
            api_key="not-needed"
        )
        client = LocalLLMClient(config)
        assert client.config.base_url == "http://localhost:11434/v1"

    def test_lmstudio_configuration(self):
        """Test configuration for LMStudio service"""
        config = LLMRouterConfig(
            provider="local",
            model="llama-3.1-8b-instruct",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )
        client = LocalLLMClient(config)
        assert client.config.base_url == "http://localhost:1234/v1"

    def test_vllm_configuration(self):
        """Test configuration for vLLM service"""
        config = LLMRouterConfig(
            provider="local",
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://vllm-server:8000/v1",
            api_key="your-vllm-key"
        )
        client = LocalLLMClient(config)
        assert client.config.base_url == "http://vllm-server:8000/v1"

    def test_text_generation_webui_configuration(self):
        """Test configuration for Text Generation WebUI service"""
        config = LLMRouterConfig(
            provider="local",
            model="llama-3.1-8b",
            base_url="http://localhost:5000/v1",
            api_key="not-needed"
        )
        client = LocalLLMClient(config)
        assert client.config.base_url == "http://localhost:5000/v1"
