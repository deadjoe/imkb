"""
Caching system for imkb

Provides intelligent caching for LLM responses, knowledge items, and RCA results
to improve performance and reduce external API calls.
"""

from .base import CacheBackend, CacheKey
from .manager import CacheManager, get_cache_manager
from .memory import MemoryCache
from .redis_cache import RedisCache

__all__ = [
    "CacheBackend",
    "CacheKey",
    "MemoryCache",
    "RedisCache",
    "CacheManager",
    "get_cache_manager",
]
