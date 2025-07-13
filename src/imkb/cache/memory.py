"""
In-memory cache implementation

Provides a simple in-memory cache backend for development and testing.
Includes LRU eviction and automatic cleanup of expired entries.
"""

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional
from collections import OrderedDict

from .base import CacheBackend, CacheKey, CacheEntry, CacheError


class MemoryCache(CacheBackend):
    """
    Thread-safe in-memory cache with LRU eviction
    
    Features:
    - LRU (Least Recently Used) eviction policy
    - Automatic cleanup of expired entries
    - Thread-safe operations
    - Configurable maximum size
    """
    
    def __init__(self, max_size: int = 1000, cleanup_interval: int = 300):
        """
        Initialize memory cache
        
        Args:
            max_size: Maximum number of entries to store
            cleanup_interval: Interval in seconds for cleanup task
        """
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe storage using OrderedDict for LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "expires": 0
        }
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception:
                    # Continue cleanup even if individual cleanup fails
                    pass
        
        try:
            loop = asyncio.get_event_loop()
            self._cleanup_task = loop.create_task(cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will happen on access
            pass
    
    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """Retrieve value from cache"""
        key_str = key.to_string()
        current_time = time.time()
        
        with self._lock:
            entry = self._cache.get(key_str)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            # Check if expired
            if entry.is_expired(current_time):
                del self._cache[key_str]
                self._stats["misses"] += 1
                self._stats["expires"] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key_str)
            entry.increment_hit_count()
            self._stats["hits"] += 1
            
            return entry
    
    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store value in cache"""
        key_str = key.to_string()
        current_time = time.time()
        
        expires_at = None
        if ttl_seconds is not None:
            expires_at = current_time + ttl_seconds
        
        entry = CacheEntry(
            value=value,
            created_at=current_time,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Remove existing entry if present
            if key_str in self._cache:
                del self._cache[key_str]
            
            # Add new entry
            self._cache[key_str] = entry
            self._cache.move_to_end(key_str)
            
            # Evict oldest entries if over limit
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
            
            self._stats["sets"] += 1
    
    async def delete(self, key: CacheKey) -> bool:
        """Delete value from cache"""
        key_str = key.to_string()
        
        with self._lock:
            if key_str in self._cache:
                del self._cache[key_str]
                self._stats["deletes"] += 1
                return True
            
            return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        with self._lock:
            if pattern is None:
                # Clear all entries
                count = len(self._cache)
                self._cache.clear()
                return count
            else:
                # Clear entries matching pattern
                keys_to_delete = [
                    key for key in self._cache.keys()
                    if pattern in key
                ]
                
                for key in keys_to_delete:
                    del self._cache[key]
                
                return len(keys_to_delete)
    
    async def exists(self, key: CacheKey) -> bool:
        """Check if key exists in cache"""
        entry = await self.get(key)
        return entry is not None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    async def get_many(self, keys: List[CacheKey]) -> Dict[CacheKey, Optional[CacheEntry]]:
        """Retrieve multiple values from cache (optimized)"""
        results = {}
        current_time = time.time()
        
        with self._lock:
            for key in keys:
                key_str = key.to_string()
                entry = self._cache.get(key_str)
                
                if entry is None:
                    self._stats["misses"] += 1
                    results[key] = None
                elif entry.is_expired(current_time):
                    del self._cache[key_str]
                    self._stats["misses"] += 1
                    self._stats["expires"] += 1
                    results[key] = None
                else:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key_str)
                    entry.increment_hit_count()
                    self._stats["hits"] += 1
                    results[key] = entry
        
        return results
    
    async def set_many(
        self,
        items: Dict[CacheKey, Any],
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store multiple values in cache (optimized)"""
        current_time = time.time()
        
        expires_at = None
        if ttl_seconds is not None:
            expires_at = current_time + ttl_seconds
        
        with self._lock:
            for key, value in items.items():
                key_str = key.to_string()
                
                # Remove existing entry if present
                if key_str in self._cache:
                    del self._cache[key_str]
                
                # Add new entry
                entry = CacheEntry(
                    value=value,
                    created_at=current_time,
                    expires_at=expires_at,
                    metadata=metadata or {}
                )
                
                self._cache[key_str] = entry
                self._cache.move_to_end(key_str)
                self._stats["sets"] += 1
            
            # Evict oldest entries if over limit
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
    
    async def _cleanup_expired(self) -> int:
        """Clean up expired entries"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key_str, entry in self._cache.items():
                if entry.is_expired(current_time):
                    expired_keys.append(key_str)
            
            for key_str in expired_keys:
                del self._cache[key_str]
                self._stats["expires"] += 1
        
        return len(expired_keys)
    
    def shutdown(self):
        """Shutdown cache and cleanup resources"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        with self._lock:
            self._cache.clear()
    
    def __del__(self):
        """Cleanup when cache is garbage collected"""
        self.shutdown()