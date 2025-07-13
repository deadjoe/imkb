"""
Base caching interfaces and utilities

Defines the abstract interface for cache backends and common utilities
for cache key generation and serialization.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class CacheType(Enum):
    """Types of cached content"""
    LLM_RESPONSE = "llm_response"
    KNOWLEDGE_ITEMS = "knowledge_items"
    RCA_RESULT = "rca_result"
    ACTION_RESULT = "action_result"
    EXTRACTOR_MATCH = "extractor_match"


@dataclass
class CacheKey:
    """
    Structured cache key with type and version information
    
    Provides consistent key generation across different cache types
    and supports cache invalidation strategies.
    """
    
    cache_type: CacheType
    namespace: str
    identifier: str
    version: str = "v1"
    
    def __post_init__(self):
        """Validate cache key components"""
        if not self.namespace:
            raise ValueError("Namespace cannot be empty")
        if not self.identifier:
            raise ValueError("Identifier cannot be empty")
    
    def to_string(self) -> str:
        """Generate string representation for cache storage"""
        return f"imkb:{self.cache_type.value}:{self.namespace}:{self.version}:{self.identifier}"
    
    @classmethod
    def from_content(
        cls,
        cache_type: CacheType,
        namespace: str,
        content: Union[str, Dict[str, Any], List[Any]],
        version: str = "v1"
    ) -> "CacheKey":
        """
        Generate cache key from content using consistent hashing
        
        Args:
            cache_type: Type of content being cached
            namespace: Namespace for multi-tenant isolation
            content: Content to generate hash from
            version: Cache version for invalidation
            
        Returns:
            CacheKey instance with content-based identifier
        """
        if isinstance(content, str):
            content_str = content
        else:
            content_str = json.dumps(content, sort_keys=True, ensure_ascii=False)
        
        # Generate SHA-256 hash of content
        content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()[:16]
        
        return cls(
            cache_type=cache_type,
            namespace=namespace,
            identifier=content_hash,
            version=version
        )
    
    @classmethod
    def for_llm_request(
        cls,
        namespace: str,
        prompt: str,
        model: str,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> "CacheKey":
        """Generate cache key for LLM requests"""
        content = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "system_prompt": system_prompt,
            **{k: v for k, v in kwargs.items() if k not in ['template_type']}
        }
        
        return cls.from_content(CacheType.LLM_RESPONSE, namespace, content)
    
    @classmethod
    def for_knowledge_items(
        cls,
        namespace: str,
        extractor: str,
        event_signature: str,
        max_results: int = 5
    ) -> "CacheKey":
        """Generate cache key for knowledge item queries"""
        content = {
            "extractor": extractor,
            "event_signature": event_signature,
            "max_results": max_results
        }
        
        return cls.from_content(CacheType.KNOWLEDGE_ITEMS, namespace, content)
    
    @classmethod
    def for_rca_result(
        cls,
        namespace: str,
        event_id: str,
        extractor: str,
        knowledge_count: int
    ) -> "CacheKey":
        """Generate cache key for RCA results"""
        content = {
            "event_id": event_id,
            "extractor": extractor,
            "knowledge_count": knowledge_count
        }
        
        return cls.from_content(CacheType.RCA_RESULT, namespace, content)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    hit_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    def is_expired(self, current_time: float) -> bool:
        """Check if cache entry has expired"""
        return self.expires_at is not None and current_time >= self.expires_at
    
    def increment_hit_count(self) -> None:
        """Increment hit count for cache statistics"""
        self.hit_count += 1


class CacheBackend(ABC):
    """
    Abstract base class for cache backends
    
    Defines the interface that all cache implementations must follow.
    """
    
    @abstractmethod
    async def get(self, key: CacheKey) -> Optional[CacheEntry]:
        """
        Retrieve value from cache
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cache entry if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def set(
        self,
        key: CacheKey,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store value in cache
        
        Args:
            key: Cache key
            value: Value to store
            ttl_seconds: Time to live in seconds
            metadata: Additional metadata
        """
        pass
    
    @abstractmethod
    async def delete(self, key: CacheKey) -> bool:
        """
        Delete value from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries
        
        Args:
            pattern: Optional pattern to match keys for deletion
            
        Returns:
            Number of keys deleted
        """
        pass
    
    @abstractmethod
    async def exists(self, key: CacheKey) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        pass
    
    async def get_many(self, keys: List[CacheKey]) -> Dict[CacheKey, Optional[CacheEntry]]:
        """
        Retrieve multiple values from cache
        
        Default implementation calls get() for each key.
        Backends can override for better performance.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to cache entries
        """
        results = {}
        for key in keys:
            results[key] = await self.get(key)
        return results
    
    async def set_many(
        self,
        items: Dict[CacheKey, Any],
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store multiple values in cache
        
        Default implementation calls set() for each item.
        Backends can override for better performance.
        
        Args:
            items: Dictionary mapping keys to values
            ttl_seconds: Time to live in seconds
            metadata: Additional metadata
        """
        for key, value in items.items():
            await self.set(key, value, ttl_seconds, metadata)


class CacheError(Exception):
    """Base exception for cache-related errors"""
    pass


class CacheConnectionError(CacheError):
    """Exception for cache connection failures"""
    pass


class CacheSerializationError(CacheError):
    """Exception for cache serialization/deserialization failures"""
    pass