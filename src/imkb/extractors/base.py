"""
Base extractor interface and utilities

Defines the protocol and common functionality for knowledge source extractors.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from ..config import ImkbConfig
from ..models import Event, KBItem

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseExtractor(Protocol):
    """
    Protocol for knowledge source extractors

    Extractors are responsible for:
    1. Determining if they can handle a given event (match)
    2. Retrieving relevant knowledge from their source (recall)
    3. Providing context for LLM prompt generation
    """

    name: str
    prompt_template: str

    async def match(self, event: Event) -> bool:
        """
        Determine if this extractor can handle the given event

        Args:
            event: The incident event to analyze

        Returns:
            True if this extractor should be used for this event
        """
        ...

    async def recall(self, event: Event, k: int = 10) -> list[KBItem]:
        """
        Retrieve relevant knowledge items for the event

        Args:
            event: The incident event
            k: Maximum number of items to return

        Returns:
            List of relevant knowledge base items
        """
        ...

    def get_prompt_context(
        self, event: Event, snippets: list[KBItem]
    ) -> dict[str, Any]:
        """
        Generate context dictionary for prompt template

        Args:
            event: The incident event
            snippets: Retrieved knowledge items

        Returns:
            Context dictionary for Jinja2 template rendering
        """
        ...


class ExtractorBase(ABC):
    """
    Abstract base class for extractors with common functionality

    Provides shared implementation and utilities for concrete extractors.
    """

    def __init__(self, config: ImkbConfig):
        self.config = config
        self.extractor_config = config.extractors.get_extractor_config(self.name)
        if self.name not in config.extractors.extractors:
            logger.warning(
                f"No configuration found for extractor '{self.name}', using defaults"
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Extractor name (must match config key)"""

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """Path to prompt template (e.g., 'mysql_rca:v1')"""

    @abstractmethod
    async def match(self, event: Event) -> bool:
        """Implementation-specific matching logic"""

    @abstractmethod
    async def recall(self, event: Event, k: int = 10) -> list[KBItem]:
        """Implementation-specific knowledge retrieval"""

    def get_prompt_context(
        self, event: Event, snippets: list[KBItem]
    ) -> dict[str, Any]:
        """
        Default prompt context generation

        Can be overridden by subclasses for custom context formatting.
        """
        return {
            "event": event.to_dict(),
            "snippets": [snippet.to_dict() for snippet in snippets],
            "extractor_name": self.name,
            "timestamp": event.timestamp,
            "severity": event.severity,
            "labels": event.labels,
            "message": event.message,
        }

    def is_enabled(self) -> bool:
        """Check if this extractor is enabled in configuration"""
        return (
            self.name in self.config.extractors.enabled
            and self.config.extractors.get_extractor_config(self.name).enabled
        )

    def get_timeout(self) -> float:
        """Get timeout setting for this extractor"""
        if self.extractor_config:
            return self.extractor_config.timeout
        return 5.0  # Default timeout

    def get_max_results(self) -> int:
        """Get max results setting for this extractor"""
        if self.extractor_config:
            return self.extractor_config.max_results
        return 10  # Default max results


class ExtractorRegistry:
    """
    Registry for managing available extractors

    Handles registration, discovery, and instantiation of extractors.
    """

    def __init__(self):
        self._extractors: dict[str, type] = {}

    def register(self, extractor_class: type) -> None:
        """Register an extractor class"""
        # Get name from class attribute
        name = getattr(extractor_class, "name", None)
        if not name:
            raise ValueError(
                f"Extractor class {extractor_class.__name__} must have a 'name' attribute"
            )

        self._extractors[name] = extractor_class
        logger.debug(f"Registered extractor: {name}")

    def get_extractor_class(self, name: str) -> Optional[type]:
        """Get extractor class by name"""
        return self._extractors.get(name)

    def get_available_extractors(self) -> list[str]:
        """Get list of available extractor names"""
        return list(self._extractors.keys())

    def create_extractor(
        self, name: str, config: ImkbConfig
    ) -> Optional[BaseExtractor]:
        """Create extractor instance by name"""
        extractor_class = self.get_extractor_class(name)
        if not extractor_class:
            logger.error(f"Unknown extractor: {name}")
            return None

        try:
            return extractor_class(config)
        except Exception as e:
            logger.error(f"Failed to create extractor {name}: {e}")
            return None

    def create_enabled_extractors(self, config: ImkbConfig) -> list[BaseExtractor]:
        """Create instances of all enabled extractors"""
        extractors = []

        for name in config.extractors.enabled:
            extractor = self.create_extractor(name, config)
            if extractor and extractor.is_enabled():
                extractors.append(extractor)
                logger.info(f"Enabled extractor: {name}")
            else:
                logger.warning(f"Failed to enable extractor: {name}")

        return extractors


# Global registry instance
registry = ExtractorRegistry()


def register_extractor(extractor_class: type) -> type:
    """Decorator for registering extractor classes"""
    registry.register(extractor_class)
    return extractor_class
