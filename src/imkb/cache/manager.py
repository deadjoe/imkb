"""
Cache manager and high-level caching utilities

Provides intelligent caching strategies, cache warming, and automatic
cache management for imkb operations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from .base import CacheBackend, CacheEntry, CacheKey, CacheType
from .memory import MemoryCache
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global cache manager instance
_cache_manager: Optional["CacheManager"] = None


class CacheManager:
    """
    High-level cache manager with intelligent caching strategies

    Features:
    - Automatic cache backend selection
    - TTL management based on content type
    - Cache warming and preloading
    - Hit/miss statistics
    - Fallback mechanisms
    """

    def __init__(
        self,
        primary_backend: CacheBackend,
        fallback_backend: Optional[CacheBackend] = None,
        default_ttl: dict[CacheType, int] = None,
    ):
        """
        Initialize cache manager

        Args:
            primary_backend: Primary cache backend (e.g., Redis)
            fallback_backend: Fallback backend (e.g., MemoryCache)
            default_ttl: Default TTL values for different cache types
        """
        self.primary = primary_backend
        self.fallback = fallback_backend

        # Default TTL settings (in seconds)
        self.default_ttl = default_ttl or {
            CacheType.LLM_RESPONSE: 3600,  # 1 hour
            CacheType.KNOWLEDGE_ITEMS: 1800,  # 30 minutes
            CacheType.RCA_RESULT: 7200,  # 2 hours
            CacheType.ACTION_RESULT: 7200,  # 2 hours
            CacheType.EXTRACTOR_MATCH: 300,  # 5 minutes
        }

        # Statistics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_errors": 0,
            "fallback_uses": 0,
        }

    async def get(self, key: CacheKey) -> Optional[Any]:
        """
        Get value from cache with fallback support

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            # Try primary cache first
            entry = await self.primary.get(key)
            if entry is not None:
                self.stats["cache_hits"] += 1
                return entry.value

            # Try fallback cache if available
            if self.fallback:
                entry = await self.fallback.get(key)
                if entry is not None:
                    self.stats["cache_hits"] += 1
                    self.stats["fallback_uses"] += 1

                    # Promote to primary cache
                    await self._promote_to_primary(key, entry)
                    return entry.value

            self.stats["cache_misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error for key {key.to_string()}: {e}")
            self.stats["cache_errors"] += 1
            return None

    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live (uses default if None)
            metadata: Additional metadata
        """
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl.get(key.cache_type, 3600)

        try:
            # Set in primary cache
            await self.primary.set(key, value, ttl_seconds, metadata)

            # Also set in fallback if available
            if self.fallback:
                await self.fallback.set(key, value, ttl_seconds, metadata)

        except Exception as e:
            logger.error(f"Cache set error for key {key.to_string()}: {e}")
            self.stats["cache_errors"] += 1

    async def delete(self, key: CacheKey) -> bool:
        """Delete value from all cache backends"""
        deleted = False

        try:
            deleted = await self.primary.delete(key)

            if self.fallback:
                fallback_deleted = await self.fallback.delete(key)
                deleted = deleted or fallback_deleted

        except Exception as e:
            logger.error(f"Cache delete error for key {key.to_string()}: {e}")
            self.stats["cache_errors"] += 1

        return deleted

    async def clear(self, cache_type: Optional[CacheType] = None) -> int:
        """Clear cache entries by type"""
        pattern = f":{cache_type.value}:" if cache_type else None
        total_deleted = 0

        try:
            deleted = await self.primary.clear(pattern)
            total_deleted += deleted

            if self.fallback:
                deleted = await self.fallback.clear(pattern)
                total_deleted += deleted

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.stats["cache_errors"] += 1

        return total_deleted

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "manager_stats": self.stats.copy(),
            "primary_backend": "unknown",
            "fallback_backend": "unknown",
        }

        try:
            # Get primary backend stats
            primary_stats = await self.primary.get_stats()
            stats["primary_backend"] = type(self.primary).__name__
            stats["primary_stats"] = primary_stats

            # Get fallback backend stats
            if self.fallback:
                fallback_stats = await self.fallback.get_stats()
                stats["fallback_backend"] = type(self.fallback).__name__
                stats["fallback_stats"] = fallback_stats

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            stats["error"] = str(e)

        return stats

    async def warm_cache(
        self, warming_functions: list[Callable[[], dict[CacheKey, Any]]]
    ) -> int:
        """
        Warm cache with precomputed values

        Args:
            warming_functions: Functions that return key-value pairs to cache

        Returns:
            Number of items cached
        """
        total_cached = 0

        for warming_func in warming_functions:
            try:
                items = (
                    await warming_func()
                    if asyncio.iscoroutinefunction(warming_func)
                    else warming_func()
                )

                if items:
                    await self.set_many(items)
                    total_cached += len(items)

            except Exception as e:
                logger.error(f"Cache warming error: {e}")

        return total_cached

    async def set_many(self, items: dict[CacheKey, Any]) -> None:
        """Set multiple items in cache"""
        try:
            # Group by cache type for optimal TTL
            grouped_items = {}
            for key, value in items.items():
                cache_type = key.cache_type
                if cache_type not in grouped_items:
                    grouped_items[cache_type] = {}
                grouped_items[cache_type][key] = value

            # Set items by type with appropriate TTL
            for cache_type, type_items in grouped_items.items():
                ttl = self.default_ttl.get(cache_type, 3600)
                await self.primary.set_many(type_items, ttl)

                if self.fallback:
                    await self.fallback.set_many(type_items, ttl)

        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            self.stats["cache_errors"] += 1

    async def get_many(self, keys: list[CacheKey]) -> dict[CacheKey, Optional[Any]]:
        """Get multiple items from cache"""
        results = {}

        try:
            # Try primary cache first
            primary_results = await self.primary.get_many(keys)

            missing_keys = []
            for key, entry in primary_results.items():
                if entry is not None:
                    results[key] = entry.value
                    self.stats["cache_hits"] += 1
                else:
                    missing_keys.append(key)

            # Try fallback for missing keys
            if missing_keys and self.fallback:
                fallback_results = await self.fallback.get_many(missing_keys)

                promote_items = {}
                for key, entry in fallback_results.items():
                    if entry is not None:
                        results[key] = entry.value
                        promote_items[key] = entry
                        self.stats["cache_hits"] += 1
                        self.stats["fallback_uses"] += 1
                    else:
                        results[key] = None
                        self.stats["cache_misses"] += 1

                # Promote fallback hits to primary cache
                if promote_items:
                    await self._promote_many_to_primary(promote_items)
            else:
                # No fallback, mark missing as misses
                for key in missing_keys:
                    results[key] = None
                    self.stats["cache_misses"] += 1

        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            self.stats["cache_errors"] += 1
            # Return None for all keys on error
            results = dict.fromkeys(keys)

        return results

    async def _promote_to_primary(self, key: CacheKey, entry: CacheEntry) -> None:
        """Promote cache entry from fallback to primary"""
        try:
            ttl = self.default_ttl.get(key.cache_type, 3600)
            await self.primary.set(key, entry.value, ttl, entry.metadata)
        except Exception as e:
            logger.debug(f"Failed to promote cache entry: {e}")

    async def _promote_many_to_primary(
        self, entries: dict[CacheKey, CacheEntry]
    ) -> None:
        """Promote multiple cache entries from fallback to primary"""
        try:
            items = {key: entry.value for key, entry in entries.items()}
            await self.set_many(items)
        except Exception as e:
            logger.debug(f"Failed to promote cache entries: {e}")


def cache_result(
    cache_type: CacheType,
    ttl_seconds: Optional[int] = None,
    key_generator: Optional[Callable[..., CacheKey]] = None,
):
    """
    Decorator for caching function results

    Args:
        cache_type: Type of content being cached
        ttl_seconds: Time to live for cached result
        key_generator: Function to generate cache key from arguments
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            manager = get_cache_manager()
            if not manager:
                # No cache manager, execute function directly
                return (
                    await func(*args, **kwargs)
                    if asyncio.iscoroutinefunction(func)
                    else func(*args, **kwargs)
                )

            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                # Default key generation
                namespace = kwargs.get("namespace", "default")
                content = {"args": args, "kwargs": kwargs}
                cache_key = CacheKey.from_content(cache_type, namespace, content)

            # Try to get from cache
            cached_result = await manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache the result
            await manager.set(cache_key, result, ttl_seconds)

            return result

        return wrapper

    return decorator


@asynccontextmanager
async def cache_context(cache_manager: CacheManager):
    """Context manager for cache operations"""
    global _cache_manager
    old_manager = _cache_manager
    _cache_manager = cache_manager

    try:
        yield cache_manager
    finally:
        _cache_manager = old_manager


def initialize_cache_manager(
    backend_type: str = "memory",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    memory_max_size: int = 1000,
    **kwargs,
) -> CacheManager:
    """
    Initialize global cache manager

    Args:
        backend_type: Type of cache backend ("memory", "redis", "hybrid")
        redis_host: Redis host (for redis/hybrid backends)
        redis_port: Redis port (for redis/hybrid backends)
        redis_db: Redis database number
        redis_password: Redis password
        memory_max_size: Memory cache size limit
        **kwargs: Additional backend options

    Returns:
        Configured CacheManager instance
    """
    global _cache_manager

    if backend_type == "memory":
        primary = MemoryCache(max_size=memory_max_size)
        manager = CacheManager(primary)

    elif backend_type == "redis":
        primary = RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            **kwargs,
        )
        # Use memory cache as fallback
        fallback = MemoryCache(max_size=memory_max_size // 2)
        manager = CacheManager(primary, fallback)

    elif backend_type == "hybrid":
        # Redis primary with memory fallback
        primary = RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            **kwargs,
        )
        fallback = MemoryCache(max_size=memory_max_size)
        manager = CacheManager(primary, fallback)

    else:
        raise ValueError(f"Unknown cache backend type: {backend_type}")

    _cache_manager = manager
    logger.info(f"Cache manager initialized with {backend_type} backend")

    return manager


def get_cache_manager() -> Optional[CacheManager]:
    """Get the global cache manager instance"""
    return _cache_manager
