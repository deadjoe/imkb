"""
Knowledge base extractors

This module contains pluggable extractors for different knowledge sources:
- BaseExtractor: Abstract interface for all extractors
- Registry: Automatic discovery and registration of extractors
- Built-in extractors for common knowledge bases
"""

# Import extractors to trigger registration
from .base import (
    BaseExtractor,
    Event,
    ExtractorBase,
    ExtractorRegistry,
    KBItem,
    register_extractor,
    registry,
)
from .mysql_extractor import MySQLKBExtractor
from .test_extractor import TestExtractor

__all__ = [
    "registry",
    "ExtractorRegistry",
    "BaseExtractor",
    "ExtractorBase",
    "Event",
    "KBItem",
    "register_extractor",
    "TestExtractor",
    "MySQLKBExtractor",
]
