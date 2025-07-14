"""
Redis cache implementation

Provides a Redis-based cache backend for production use with
persistence, clustering support, and high performance.
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

from .base import (
    CacheBackend,
    CacheConnectionError,
    CacheEntry,
    CacheKey,
    CacheSerializationError,
)
import contextlib

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


class RedisCache(CacheBackend):
    """
    Redis-based cache backend

    Features:
    - Persistent storage
    - High performance
    - Clustering support
    - Atomic operations
    - Built-in expiration
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        connection_pool_size: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        **kwargs,
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            connection_pool_size: Size of connection pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            **kwargs: Additional Redis client options
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: uv add redis")

        self.host = host
        self.port = port
        self.db = db
        self.password = password

        # Create connection pool
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=connection_pool_size,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False,  # We handle encoding ourselves
            **kwargs,
        )

        self.client = redis.Redis(connection_pool=self.pool)

        # Key prefix for namespacing
        self.key_prefix = "imkb:cache:"

        # Statistics (stored in Redis for persistence)
        self.stats_key = f"{self.key_prefix}stats"

    def _make_redis_key(self, key: CacheKey) -> str:
        """Convert cache key to Redis key"""
        return f"{self.key_prefix}{key.to_string()}"

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry for Redis storage"""
        try:
            data = {
                "value": entry.value,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at,
                "hit_count": entry.hit_count,
                "metadata": entry.metadata,
            }
            return json.dumps(data, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise CacheSerializationError(f"Failed to serialize cache entry: {e}") from e

    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry from Redis storage"""
        try:
            obj = json.loads(data.decode("utf-8"))
            return CacheEntry(
                value=obj["value"],
                created_at=obj["created_at"],
                expires_at=obj.get("expires_at"),
                hit_count=obj.get("hit_count", 0),
                metadata=obj.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise CacheSerializationError(f"Failed to deserialize cache entry: {e}") from e

    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Retrieve value from cache"""
        try:
            redis_key = self._make_redis_key(key)
            data = await self.client.get(redis_key)

            if data is None:
                await self._increment_stat("misses")
                return None

            entry = self._deserialize_entry(data)
            current_time = time.time()

            # Check if expired (Redis TTL might not have cleaned it up yet)
            if entry.is_expired(current_time):
                await self.client.delete(redis_key)
                await self._increment_stat("misses")
                await self._increment_stat("expires")
                return None

            # Update hit count and statistics
            entry.increment_hit_count()
            await self.client.set(redis_key, self._serialize_entry(entry), keepttl=True)
            await self._increment_stat("hits")

            return entry

        except redis.RedisError as e:
            logger.error(f"Redis error in get: {e}")
            raise CacheConnectionError(f"Redis connection error: {e}") from e

    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store value in cache"""
        try:
            current_time = time.time()
            expires_at = None
            if ttl_seconds is not None:
                expires_at = current_time + ttl_seconds

            entry = CacheEntry(
                value=value,
                created_at=current_time,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            redis_key = self._make_redis_key(key)
            data = self._serialize_entry(entry)

            if ttl_seconds is not None:
                await self.client.setex(redis_key, ttl_seconds, data)
            else:
                await self.client.set(redis_key, data)

            await self._increment_stat("sets")

        except redis.RedisError as e:
            logger.error(f"Redis error in set: {e}")
            raise CacheConnectionError(f"Redis connection error: {e}") from e

    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache"""
        try:
            redis_key = self._make_redis_key(key)
            deleted = await self.client.delete(redis_key)

            if deleted > 0:
                await self._increment_stat("deletes")
                return True

            return False

        except redis.RedisError as e:
            logger.error(f"Redis error in delete: {e}")
            raise CacheConnectionError(f"Redis connection error: {e}") from e

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        try:
            if pattern is None:
                # Clear all cache entries
                scan_pattern = f"{self.key_prefix}*"
            else:
                # Clear entries matching pattern
                scan_pattern = f"{self.key_prefix}*{pattern}*"

            deleted_count = 0
            async for key in self.client.scan_iter(match=scan_pattern, count=100):
                await self.client.delete(key)
                deleted_count += 1

            return deleted_count

        except redis.RedisError as e:
            logger.error(f"Redis error in clear: {e}")
            raise CacheConnectionError(f"Redis connection error: {e}") from e

    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists in cache"""
        try:
            redis_key = self._make_redis_key(key)
            exists = await self.client.exists(redis_key)
            return bool(exists)

        except redis.RedisError as e:
            logger.error(f"Redis error in exists: {e}")
            raise CacheConnectionError(f"Redis connection error: {e}") from e

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        try:
            stats_data = await self.client.hgetall(self.stats_key)

            # Convert bytes to integers
            stats = {}
            for key, value in stats_data.items():
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                value_int = int(
                    value.decode("utf-8") if isinstance(value, bytes) else value
                )
                stats[key_str] = value_int

            # Calculate derived statistics
            total_requests = stats.get("hits", 0) + stats.get("misses", 0)
            hit_rate = (
                stats.get("hits", 0) / total_requests if total_requests > 0 else 0.0
            )

            # Get cache size (approximate)
            cache_size = 0
            async for _ in self.client.scan_iter(
                match=f"{self.key_prefix}*", count=1000
            ):
                cache_size += 1

            return {
                **stats,
                "size": cache_size,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "redis_info": await self._get_redis_info(),
            }

        except redis.RedisError as e:
            logger.error(f"Redis error in get_stats: {e}")
            return {"error": str(e)}

    async def get_many(
        self, keys: list[CacheKey]
    ) -> dict[CacheKey, Optional[CacheEntry]]:
        """Retrieve multiple values from cache (optimized with pipeline)"""
        if not keys:
            return {}

        try:
            redis_keys = [self._make_redis_key(key) for key in keys]

            # Use pipeline for better performance
            pipe = self.client.pipeline()
            for redis_key in redis_keys:
                pipe.get(redis_key)

            results_data = await pipe.execute()

            results = {}
            current_time = time.time()
            hit_count = 0
            miss_count = 0
            expire_count = 0

            for i, (key, data) in enumerate(zip(keys, results_data)):
                if data is None:
                    results[key] = None
                    miss_count += 1
                else:
                    try:
                        entry = self._deserialize_entry(data)

                        if entry.is_expired(current_time):
                            # Schedule for deletion
                            await self.client.delete(redis_keys[i])
                            results[key] = None
                            miss_count += 1
                            expire_count += 1
                        else:
                            entry.increment_hit_count()
                            results[key] = entry
                            hit_count += 1

                            # Update hit count in Redis (async, don't wait)
                            asyncio.create_task(
                                self.client.set(
                                    redis_keys[i],
                                    self._serialize_entry(entry),
                                    keepttl=True,
                                )
                            )
                    except CacheSerializationError:
                        results[key] = None
                        miss_count += 1

            # Update statistics
            if hit_count > 0:
                await self._increment_stat("hits", hit_count)
            if miss_count > 0:
                await self._increment_stat("misses", miss_count)
            if expire_count > 0:
                await self._increment_stat("expires", expire_count)

            return results

        except redis.RedisError as e:
            logger.error(f"Redis error in get_many: {e}")
            raise CacheConnectionError(f"Redis connection error: {e}") from e

    async def set_many(
        self,
        items: dict[CacheKey, Any],
        ttl_seconds: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store multiple values in cache (optimized with pipeline)"""
        if not items:
            return

        try:
            current_time = time.time()
            expires_at = None
            if ttl_seconds is not None:
                expires_at = current_time + ttl_seconds

            # Use pipeline for better performance
            pipe = self.client.pipeline()

            for key, value in items.items():
                entry = CacheEntry(
                    value=value,
                    created_at=current_time,
                    expires_at=expires_at,
                    metadata=metadata or {},
                )

                redis_key = self._make_redis_key(key)
                data = self._serialize_entry(entry)

                if ttl_seconds is not None:
                    pipe.setex(redis_key, ttl_seconds, data)
                else:
                    pipe.set(redis_key, data)

            await pipe.execute()
            await self._increment_stat("sets", len(items))

        except redis.RedisError as e:
            logger.error(f"Redis error in set_many: {e}")
            raise CacheConnectionError(f"Redis connection error: {e}") from e

    async def _increment_stat(self, stat_name: str, count: int = 1) -> None:
        """Increment cache statistic"""
        with contextlib.suppress(redis.RedisError):
            # Don't fail operations due to stats errors
            await self.client.hincrby(self.stats_key, stat_name, count)

    async def _get_redis_info(self) -> dict[str, Any]:
        """Get Redis server information"""
        try:
            info = await self.client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
            }
        except redis.RedisError:
            return {}

    async def ping(self) -> bool:
        """Test Redis connection"""
        try:
            result = await self.client.ping()
            return result is True
        except redis.RedisError:
            return False

    async def close(self) -> None:
        """Close Redis connection"""
        with contextlib.suppress(redis.RedisError):
            await self.client.close()
