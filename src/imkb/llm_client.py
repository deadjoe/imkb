"""
LLM client for routing requests to different language models

Supports both local and cloud-based LLMs with unified interface.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import ImkbConfig, LLMRouterConfig

# Import observability if available
try:
    from .observability.metrics import get_metrics
    from .observability.tracer import (
        add_event,
        set_attribute,
        trace_async,
        trace_operation,
    )

    OBSERVABILITY_AVAILABLE = True
except ImportError:

    def trace_async(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def trace_operation(*args, **kwargs):
        from contextlib import nullcontext

        return nullcontext()

    def add_event(*args, **kwargs):
        pass

    def set_attribute(*args, **kwargs):
        pass

    def get_metrics():
        return None

    OBSERVABILITY_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMResponse:
    """Standardized LLM response format"""

    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: Optional[int] = None,
        finish_reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.finish_reason = finish_reason
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata,
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""

    def __init__(self, config: LLMRouterConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Generate response from LLM"""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if LLM is available"""


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self, config: LLMRouterConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.config.api_key, timeout=self.config.timeout
                )
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: uv add openai"
                ) from e
        return self._client

    @trace_async("llm.openai.generate")
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            client = await self._get_client()

            set_attribute("llm.provider", "openai")
            set_attribute("llm.model", self.config.model)
            set_attribute("prompt.length", len(prompt))
            if system_prompt:
                set_attribute("system_prompt.length", len(system_prompt))

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Merge kwargs with config defaults
            generation_params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs,
            }

            set_attribute("llm.temperature", self.config.temperature)
            set_attribute("llm.max_tokens", self.config.max_tokens)

            response = await client.chat.completions.create(**generation_params)

            choice = response.choices[0]
            result = LLMResponse(
                content=choice.message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=choice.finish_reason,
                metadata={
                    "prompt_tokens": (
                        response.usage.prompt_tokens if response.usage else None
                    ),
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else None
                    ),
                },
            )

            # Record metrics
            metrics = get_metrics()
            if metrics:
                metrics.record_llm_request(
                    "openai",
                    self.config.model,
                    "default",
                    kwargs.get("template_type", ""),
                )
                if response.usage:
                    metrics.record_llm_tokens(
                        "openai",
                        self.config.model,
                        "default",
                        response.usage.prompt_tokens or 0,
                        response.usage.completion_tokens or 0,
                    )

            set_attribute("response.tokens_used", result.tokens_used)
            set_attribute("response.finish_reason", result.finish_reason)
            add_event("llm_generation_complete")

            return result

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")

            # Record error metrics
            metrics = get_metrics()
            if metrics:
                metrics.record_llm_error(
                    "openai", self.config.model, "default", type(e).__name__
                )

            set_attribute("error.type", type(e).__name__)
            add_event("llm_generation_error", {"error": str(e)})
            raise

    async def health_check(self) -> bool:
        """Check OpenAI API availability"""
        try:
            client = await self._get_client()
            # Simple test request
            await client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


class LocalLLMClient(BaseLLMClient):
    """
    Local LLM client for OpenAI-compatible inference services
    
    Supports popular local inference services like:
    - Ollama (http://localhost:11434/v1)
    - LMStudio (http://localhost:1234/v1)
    - vLLM (http://your-server:8000/v1)
    - Text Generation WebUI (http://localhost:5000/v1)
    """

    def __init__(self, config: LLMRouterConfig):
        super().__init__(config)
        self._client = None

    async def _get_client(self):
        """Lazy initialization of local OpenAI-compatible client"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                # Use custom base_url for local inference services
                base_url = self.config.base_url
                if not base_url:
                    raise ValueError(
                        "base_url is required for local LLM provider. "
                        "Examples: http://localhost:11434/v1 (Ollama), "
                        "http://localhost:1234/v1 (LMStudio)"
                    )

                self._client = AsyncOpenAI(
                    api_key=self.config.api_key or "not-needed",
                    base_url=base_url,
                    timeout=self.config.timeout
                )
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: uv add openai"
                ) from e
        return self._client

    @trace_async("llm.local.generate")
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """Generate response using local OpenAI-compatible API"""
        try:
            client = await self._get_client()

            set_attribute("llm.provider", "local")
            set_attribute("llm.base_url", self.config.base_url)
            set_attribute("llm.model", self.config.model)
            set_attribute("prompt.length", len(prompt))
            if system_prompt:
                set_attribute("system_prompt.length", len(system_prompt))

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Merge kwargs with config defaults
            generation_params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs,
            }

            set_attribute("llm.temperature", self.config.temperature)
            set_attribute("llm.max_tokens", self.config.max_tokens)

            response = await client.chat.completions.create(**generation_params)

            choice = response.choices[0]
            result = LLMResponse(
                content=choice.message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=choice.finish_reason,
                metadata={
                    "prompt_tokens": (
                        response.usage.prompt_tokens if response.usage else None
                    ),
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else None
                    ),
                    "base_url": self.config.base_url,
                },
            )

            # Record metrics
            metrics = get_metrics()
            if metrics:
                metrics.record_llm_request(
                    "local",
                    self.config.model,
                    "default",
                    kwargs.get("template_type", ""),
                )
                if response.usage:
                    metrics.record_llm_tokens(
                        "local",
                        self.config.model,
                        "default",
                        response.usage.prompt_tokens or 0,
                        response.usage.completion_tokens or 0,
                    )

            set_attribute("response.tokens_used", result.tokens_used)
            set_attribute("response.finish_reason", result.finish_reason)
            add_event("llm_generation_complete")

            return result

        except Exception as e:
            logger.error(f"Local LLM generation failed: {e}")

            # Record error metrics
            metrics = get_metrics()
            if metrics:
                metrics.record_llm_error(
                    "local", self.config.model, "default", type(e).__name__
                )

            set_attribute("error.type", type(e).__name__)
            add_event("llm_generation_error", {"error": str(e)})
            raise

    async def health_check(self) -> bool:
        """Check local LLM service availability"""
        try:
            client = await self._get_client()
            # Simple test request
            await client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            logger.error(f"Local LLM health check failed: {e}")
            return False


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing and development"""

    def __init__(self, config: LLMRouterConfig):
        super().__init__(config)
        self._load_mock_responses()

    def _load_mock_responses(self):
        """Load mock responses from external YAML file"""
        try:
            # Use configured path or default
            if self.config.mock_responses_path:
                mock_responses_path = Path(self.config.mock_responses_path)
            else:
                mock_responses_path = (
                    Path(__file__).parent / "prompts" / "mock_responses.yaml"
                )

            with open(mock_responses_path) as f:
                self.mock_responses = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load mock responses: {e}")
            # Fallback to simple default responses
            self.mock_responses = {
                "rca": {
                    "root_cause": "Mock RCA analysis for testing",
                    "confidence": 0.7,
                    "contributing_factors": ["Mock factor 1", "Mock factor 2"],
                    "evidence": ["Mock evidence"],
                    "immediate_actions": ["Mock action"],
                    "preventive_measures": ["Mock prevention"],
                    "additional_investigation": ["Mock investigation"],
                    "confidence_reasoning": "Mock reasoning",
                    "knowledge_gaps": ["Mock gap"]
                },
                "action_generation": {
                    "actions": ["Mock action 1", "Mock action 2"],
                    "playbook": "Mock playbook for testing",
                    "priority": "medium",
                    "estimated_time": "10 minutes",
                    "risk_level": "low",
                    "prerequisites": ["Mock prerequisite"],
                    "validation_steps": ["Mock validation"],
                    "rollback_plan": "Mock rollback plan",
                    "automation_potential": "manual"
                },
                "default": {
                    "content": "Mock LLM response for testing purposes"
                }
            }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        template_type: str = "default",
        **kwargs,
    ) -> LLMResponse:
        """Generate mock response"""
        await asyncio.sleep(0.1)  # Simulate network delay

        # Get response template
        response_data = self.mock_responses.get(
            template_type, self.mock_responses["default"]
        )

        # Generate appropriate response based on template type
        if template_type in ["rca", "action_generation"]:
            content = json.dumps(response_data, indent=2)
        else:
            content = response_data.get("content", f"Mock response for {template_type}")

        return LLMResponse(
            content=content,
            model=f"mock-{self.config.model}",
            tokens_used=len(content.split()),
            finish_reason="stop",
            metadata={"mock": True, "template_type": template_type},
        )

    async def health_check(self) -> bool:
        """Mock health check always returns True"""
        return True


class LLMRouter:
    """
    Routes LLM requests to appropriate clients

    Handles client instantiation, health checks, and failover.
    """

    def __init__(self, config: ImkbConfig):
        self.config = config
        self._clients: dict[str, BaseLLMClient] = {}
        self._health_status: dict[str, bool] = {}

    def _create_client(
        self, router_name: str, router_config: LLMRouterConfig
    ) -> BaseLLMClient:
        """Create LLM client based on provider"""
        provider = router_config.provider.lower()

        if provider == "openai":
            # Check if API key is placeholder, use mock instead
            if router_config.api_key and router_config.api_key.startswith(
                "sk-placeholder"
            ):
                logger.info(
                    f"Using mock client for placeholder API key in router {router_name}"
                )
                return MockLLMClient(router_config)
            return OpenAIClient(router_config)
        if provider == "local":
            return LocalLLMClient(router_config)
        if provider == "mock":
            return MockLLMClient(router_config)
        logger.warning(f"Unknown LLM provider '{provider}', using mock client")
        return MockLLMClient(router_config)

    async def get_client(self, router_name: Optional[str] = None) -> BaseLLMClient:
        """Get LLM client by router name"""
        router_name = router_name or self.config.llm.default

        if router_name not in self._clients:
            router_config = self.config.get_llm_router_config(router_name)
            self._clients[router_name] = self._create_client(router_name, router_config)

        return self._clients[router_name]

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        router_name: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using specified router"""
        client = await self.get_client(router_name)

        try:
            return await client.generate(prompt, system_prompt, **kwargs)
        except Exception as e:
            logger.error(f"LLM generation failed with router {router_name}: {e}")
            # TODO: Implement fallback logic for production
            raise

    async def health_check(self, router_name: Optional[str] = None) -> bool:
        """Check health of specified router"""
        try:
            client = await self.get_client(router_name)
            is_healthy = await client.health_check()
            self._health_status[router_name or self.config.llm.default] = is_healthy
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed for router {router_name}: {e}")
            return False

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all configured routers"""
        results = {}
        for router_name in self.config.llm.routers:
            results[router_name] = await self.health_check(router_name)
        return results

    @asynccontextmanager
    async def get_client_context(self, router_name: Optional[str] = None):
        """Context manager for client access"""
        client = await self.get_client(router_name)
        try:
            yield client
        finally:
            # Cleanup logic if needed
            pass
