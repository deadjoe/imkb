"""
LLM client for routing requests to different language models

Supports both local and cloud-based LLMs with unified interface.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Optional

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


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing and development"""

    def __init__(self, config: LLMRouterConfig):
        super().__init__(config)
        self.response_templates = {
            "rca": """Based on the incident analysis, I can identify the following:

## Root Cause Analysis
The primary cause appears to be {primary_cause}. This is typically associated with {associated_factors}.

## Confidence Assessment
Confidence: {confidence}% - {confidence_reason}

## Recommended Actions
1. {action_1}
2. {action_2}
3. {action_3}

## Additional Context
{additional_context}

*Note: This is a mock response for development purposes.*""",
            "default": "This is a mock LLM response for testing purposes. Original prompt: {prompt_preview}...",
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

        template = self.response_templates.get(
            template_type, self.response_templates["default"]
        )

        # Generate contextual mock data
        if template_type == "rca":
            # Return a properly formatted JSON response for RCA
            content = """{
  "root_cause": "Resource contention or configuration issue based on the incident signature",
  "confidence": 0.75,
  "contributing_factors": [
    "High concurrent connection load",
    "Insufficient connection pool sizing",
    "Potential connection leaks in application code"
  ],
  "evidence": [
    "Connection pool exhaustion signature indicates resource limits reached",
    "MySQL-specific symptoms match known capacity issues"
  ],
  "immediate_actions": [
    "Investigate current connection usage patterns",
    "Review application connection handling",
    "Consider temporary pool size increase"
  ],
  "preventive_measures": [
    "Implement connection pooling best practices",
    "Add monitoring for connection pool metrics",
    "Review and optimize application connection lifecycle"
  ],
  "additional_investigation": [
    "Analyze connection growth patterns over time",
    "Review application deployment history"
  ],
  "confidence_reasoning": "Based on common patterns in similar database connectivity incidents",
  "knowledge_gaps": [
    "Actual connection usage metrics at time of incident",
    "Application-specific connection handling details"
  ]
}"""
        elif template_type == "action_generation":
            # Return properly formatted JSON for action generation
            content = """{
  "actions": [
    "Execute SHOW PROCESSLIST to identify active connections and blocking queries",
    "Increase max_connections parameter from 150 to 300 temporarily",
    "Identify and terminate long-running or idle connections using KILL command",
    "Review application connection pool configuration and increase pool size",
    "Implement connection monitoring and alerting for proactive management"
  ],
  "playbook": "1. Immediate Assessment: Run 'SHOW PROCESSLIST' and 'SHOW STATUS LIKE \\"Threads_connected\\"' to assess current state. 2. Emergency Relief: Increase max_connections to 300 with 'SET GLOBAL max_connections = 300'. 3. Connection Cleanup: Identify problematic connections and terminate using 'KILL <connection_id>'. 4. Application Review: Check application connection pools and increase timeout settings. 5. Monitoring Setup: Implement alerts for connection usage above 80% threshold. 6. Validation: Verify new connections can be established and application functionality restored.",
  "priority": "high",
  "estimated_time": "30 minutes",
  "risk_level": "medium",
  "prerequisites": [
    "MySQL administrative access",
    "Application deployment pipeline access",
    "Monitoring system configuration access"
  ],
  "validation_steps": [
    "Verify new database connections can be established",
    "Check application error logs for connection failures",
    "Monitor connection count remains below new threshold",
    "Confirm all application services are operational"
  ],
  "rollback_plan": "If issues arise, revert max_connections to original value (150) and restart MySQL service if necessary",
  "automation_potential": "semi-automated"
}"""
        else:
            content = template.format(prompt_preview=prompt[:100])

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
