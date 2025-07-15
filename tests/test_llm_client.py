"""
Test suite for LLM client system

Tests LLMResponse, OpenAIClient, MockLLMClient, and LLMRouter functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imkb.config import ImkbConfig, LLMRouterConfig
from imkb.llm_client import (
    LLMResponse,
    LLMRouter,
    LocalLLMClient,
    MockLLMClient,
    OpenAIClient,
)


class TestLLMResponse:
    """Test LLMResponse class"""

    def test_basic_response(self):
        """Test basic LLMResponse creation"""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            tokens_used=100,
            finish_reason="stop",
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.tokens_used == 100
        assert response.finish_reason == "stop"
        assert response.metadata == {}

    def test_response_with_metadata(self):
        """Test LLMResponse with metadata"""
        metadata = {"prompt_tokens": 50, "completion_tokens": 50}
        response = LLMResponse(
            content="Test response", model="gpt-4", metadata=metadata
        )

        assert response.metadata == metadata

    def test_model_dump(self):
        """Test LLMResponse model_dump (Pydantic serialization)"""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            tokens_used=100,
            finish_reason="stop",
            metadata={"test": "value"},
        )

        result = response.model_dump()
        expected = {
            "content": "Test response",
            "model": "gpt-4",
            "tokens_used": 100,
            "finish_reason": "stop",
            "metadata": {"test": "value"},
        }

        assert result == expected


class TestMockLLMClient:
    """Test MockLLMClient functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = LLMRouterConfig(provider="mock", model="mock-gpt-4")
        self.client = MockLLMClient(self.config)

    @pytest.mark.asyncio
    async def test_basic_generation(self):
        """Test basic mock response generation"""
        response = await self.client.generate("Test prompt")

        assert isinstance(response, LLMResponse)
        assert "mock" in response.content.lower()
        assert response.model == "mock-mock-gpt-4"
        assert response.metadata.get("mock") is True
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_rca_template(self):
        """Test RCA template response"""
        response = await self.client.generate(
            "Analyze this incident", template_type="rca"
        )

        assert "root_cause" in response.content
        assert "confidence" in response.content
        assert "contributing_factors" in response.content
        assert response.metadata.get("template_type") == "rca"

    @pytest.mark.asyncio
    async def test_action_generation_template(self):
        """Test action generation template response"""
        response = await self.client.generate(
            "Generate actions for this issue", template_type="action_generation"
        )

        assert "actions" in response.content
        assert "playbook" in response.content
        assert "priority" in response.content
        assert response.metadata.get("template_type") == "action_generation"

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test mock health check always returns True"""
        assert await self.client.health_check() is True

    @pytest.mark.asyncio
    async def test_network_delay_simulation(self):
        """Test that mock client simulates network delay"""
        import time

        start_time = time.time()
        await self.client.generate("test")
        elapsed = time.time() - start_time

        # Should take at least 0.1 seconds due to simulated delay
        assert elapsed >= 0.1


class TestOpenAIClient:
    """Test OpenAIClient functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = LLMRouterConfig(
            provider="openai", model="gpt-4o-mini", api_key="sk-test-key", timeout=30.0
        )

    def test_client_initialization(self):
        """Test OpenAI client initialization"""
        client = OpenAIClient(self.config)
        assert client.config == self.config
        assert client._client is None  # Lazy loading

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_lazy_client_creation(self, mock_openai):
        """Test lazy OpenAI client creation"""
        mock_instance = AsyncMock()
        mock_openai.return_value = mock_instance

        client = OpenAIClient(self.config)
        openai_client = await client._get_client()

        assert openai_client == mock_instance
        mock_openai.assert_called_once_with(api_key="sk-test-key", timeout=30.0)

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_generate_response(self, mock_openai):
        """Test OpenAI response generation"""
        # Mock OpenAI response
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated response"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.total_tokens = 100
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = mock_usage

        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance

        client = OpenAIClient(self.config)
        response = await client.generate("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "Generated response"
        assert response.model == "gpt-4o-mini"
        assert response.tokens_used == 100
        assert response.finish_reason == "stop"
        assert response.metadata["prompt_tokens"] == 50
        assert response.metadata["completion_tokens"] == 50

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_generate_with_system_prompt(self, mock_openai):
        """Test generation with system prompt"""
        # Mock OpenAI response
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated response"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.total_tokens = 100
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 50

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = mock_usage

        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_instance

        client = OpenAIClient(self.config)
        await client.generate("User prompt", system_prompt="System instructions")

        # Verify system prompt was included in messages
        call_args = mock_instance.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System instructions"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User prompt"

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_health_check_success(self, mock_openai):
        """Test successful health check"""
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.return_value = MagicMock()
        mock_openai.return_value = mock_instance

        client = OpenAIClient(self.config)
        assert await client.health_check() is True

    @pytest.mark.asyncio
    @patch("openai.AsyncOpenAI")
    async def test_health_check_failure(self, mock_openai):
        """Test failed health check"""
        mock_instance = AsyncMock()
        mock_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_instance

        client = OpenAIClient(self.config)
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_import_error_handling(self):
        """Test handling when OpenAI package is not installed"""
        with patch.dict("sys.modules", {"openai": None}):
            client = OpenAIClient(self.config)

            with pytest.raises(ImportError, match="OpenAI package not installed"):
                await client._get_client()


class TestLLMRouter:
    """Test LLMRouter functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = ImkbConfig()
        self.router = LLMRouter(self.config)

    def test_router_initialization(self):
        """Test router initialization"""
        assert self.router.config == self.config
        assert self.router._clients == {}
        assert self.router._health_status == {}

    @pytest.mark.asyncio
    async def test_get_default_client(self):
        """Test getting default client"""
        client = await self.router.get_client()
        assert client is not None
        assert "openai_default" in self.router._clients

    @pytest.mark.asyncio
    async def test_get_specific_client(self):
        """Test getting specific client by name"""
        # Add a mock router to config
        self.config.llm.routers["mock"] = LLMRouterConfig(provider="mock")

        client = await self.router.get_client("mock")
        assert isinstance(client, MockLLMClient)
        assert "mock" in self.router._clients

    @pytest.mark.asyncio
    async def test_client_caching(self):
        """Test that clients are cached"""
        client1 = await self.router.get_client("openai_default")
        client2 = await self.router.get_client("openai_default")

        assert client1 is client2  # Same instance

    def test_create_openai_client_with_placeholder_key(self):
        """Test that placeholder API keys result in mock client"""
        config = LLMRouterConfig(provider="openai", api_key="sk-placeholder-test-key")

        client = self.router._create_client("test", config)
        assert isinstance(client, MockLLMClient)

    def test_create_mock_client(self):
        """Test creating mock client"""
        config = LLMRouterConfig(provider="mock")
        client = self.router._create_client("test", config)
        assert isinstance(client, MockLLMClient)

    def test_create_local_client(self):
        """Test creating LocalLLMClient for local provider"""
        config = LLMRouterConfig(
            provider="local",
            base_url="http://localhost:11434/v1"
        )
        client = self.router._create_client("local_router", config)

        assert isinstance(client, LocalLLMClient)
        assert client.config.base_url == "http://localhost:11434/v1"

    def test_create_unknown_provider_client(self):
        """Test creating client for unknown provider defaults to mock"""
        config = LLMRouterConfig(provider="unknown_provider")
        client = self.router._create_client("test", config)
        assert isinstance(client, MockLLMClient)

    @pytest.mark.asyncio
    async def test_generate_with_default_router(self):
        """Test generating with default router"""
        response = await self.router.generate("Test prompt")
        assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_generate_with_specific_router(self):
        """Test generating with specific router"""
        # Add mock router
        self.config.llm.routers["mock"] = LLMRouterConfig(provider="mock")

        response = await self.router.generate("Test prompt", router_name="mock")
        assert isinstance(response, LLMResponse)

    @pytest.mark.asyncio
    async def test_health_check_single_router(self):
        """Test health check for single router"""
        result = await self.router.health_check("openai_default")
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_health_check_all_routers(self):
        """Test health check for all routers"""
        # Add additional router
        self.config.llm.routers["mock"] = LLMRouterConfig(provider="mock")

        results = await self.router.health_check_all()
        assert isinstance(results, dict)
        assert "openai_default" in results
        assert "mock" in results

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client context manager"""
        async with self.router.get_client_context("openai_default") as client:
            assert client is not None
            response = await client.generate("Test")
            assert isinstance(response, LLMResponse)


class TestLLMRouterIntegration:
    """Integration tests for LLM router system"""

    @pytest.mark.asyncio
    async def test_end_to_end_mock_workflow(self):
        """Test complete mock LLM workflow"""
        config = ImkbConfig()
        router = LLMRouter(config)

        # Generate RCA response
        rca_response = await router.generate(
            "Analyze database connection issue", template_type="rca"
        )

        assert "root_cause" in rca_response.content
        assert rca_response.metadata.get("mock") is True

        # Generate action response
        action_response = await router.generate(
            "Generate remediation actions", template_type="action_generation"
        )

        assert "actions" in action_response.content
        assert "playbook" in action_response.content

    @pytest.mark.asyncio
    async def test_router_failover_behavior(self):
        """Test router behavior when generation fails"""
        config = ImkbConfig()
        router = LLMRouter(config)

        # Mock a client that will fail
        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("Network error")
        router._clients["openai_default"] = mock_client

        with pytest.raises(Exception, match="Network error"):
            await router.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        config = ImkbConfig()
        router = LLMRouter(config)

        # Generate multiple concurrent requests
        tasks = []
        for i in range(5):
            task = router.generate(f"Test prompt {i}")
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        for response in responses:
            assert isinstance(response, LLMResponse)
