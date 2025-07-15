"""
Mem0 adapter for hybrid vector + graph storage

Provides a wrapper around the Mem0 SDK for imkb-specific usage patterns.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from mem0 import Memory

from ..config import ImkbConfig
from ..models import KBItem

logger = logging.getLogger(__name__)


class Mem0Adapter:
    """
    Adapter for Mem0 hybrid vector + graph storage

    Provides imkb-specific interface for memory operations while leveraging
    Mem0's hybrid storage capabilities.
    """

    def __init__(self, config: ImkbConfig):
        self.config = config
        self._memory: Optional[Memory] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Mem0 memory instance"""
        if self._initialized:
            return

        try:
            # Check if we have valid API keys for real Mem0 usage
            llm_router = self.config.get_llm_router_config()
            has_valid_api_key = (
                llm_router.api_key
                and not llm_router.api_key.startswith("sk-placeholder")
                and llm_router.api_key != "your-api-key-here"
            )

            if not has_valid_api_key:
                logger.info("No valid API key found, using local storage fallback")
                self._memory = "local_fallback"
                self._initialized = True
                return

            # Get Mem0-compatible configuration
            mem0_config = self.config.get_mem0_config()

            # Add LLM configuration for Mem0
            mem0_config["llm"] = {
                "provider": llm_router.provider,
                "config": {
                    "model": llm_router.model,
                    "temperature": llm_router.temperature,
                    "max_tokens": llm_router.max_tokens,
                    "api_key": llm_router.api_key,
                },
            }

            # Add embedder configuration
            mem0_config["embedder"] = {
                "provider": llm_router.provider,
                "config": {
                    "model": (
                        "text-embedding-3-small"
                        if llm_router.provider == "openai"
                        else llm_router.model
                    ),
                    "api_key": llm_router.api_key,
                },
            }

            # Initialize Mem0 Memory
            self._memory = Memory.from_config(mem0_config)
            self._initialized = True

            logger.info("Mem0 adapter initialized successfully with real API")

        except Exception as e:
            logger.warning(f"Mem0 initialization failed, using local fallback: {e}")
            self._memory = "local_fallback"
            self._initialized = True

    async def search(self, query: str, user_id: str, limit: int = 10) -> list[KBItem]:
        """
        Search for relevant memories using hybrid vector + graph retrieval

        Args:
            query: Search query (event signature or description)
            user_id: User/namespace identifier for isolation
            limit: Maximum number of results to return

        Returns:
            List of KBItem objects with relevance scores
        """
        await self.initialize()

        # Use local fallback if Mem0 is not available
        if self._memory == "local_fallback":
            from .local_storage import get_local_store

            local_store = get_local_store()
            return await local_store.search(query, user_id, limit)

        try:
            # Use Mem0's hybrid search
            results = self._memory.search(query=query, user_id=user_id, limit=limit)

            # Convert Mem0 results to KBItem format
            kb_items = []
            for i, result in enumerate(results):
                # Mem0 results structure: {"memory": str, "metadata": dict, ...}
                kb_item = KBItem(
                    doc_id=result.get("id", f"mem0_{i}"),
                    excerpt=result.get("memory", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                )
                kb_items.append(kb_item)

            logger.debug(f"Found {len(kb_items)} memories for query: {query[:50]}...")
            return kb_items

        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []

    async def add_memory(
        self, content: str, user_id: str, metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Add new memory to the hybrid storage

        Args:
            content: Memory content (e.g., incident description, KB article)
            user_id: User/namespace identifier
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        await self.initialize()

        # Use local fallback if Mem0 is not available
        if self._memory == "local_fallback":
            from .local_storage import get_local_store

            local_store = get_local_store()
            return await local_store.add_memory(content, user_id, metadata)

        try:
            result = self._memory.add(
                messages=[{"role": "user", "content": content}],
                user_id=user_id,
                metadata=metadata or {},
            )

            # Extract memory ID from result
            memory_id = result.get("results", [{}])[0].get("id", "unknown")

            logger.debug(f"Added memory {memory_id} for user {user_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

    async def add_memories_batch(
        self, memories: list[dict[str, Any]], user_id: str
    ) -> list[str]:
        """
        Add multiple memories in batch

        Args:
            memories: List of memory dicts with 'content' and optional 'metadata'
            user_id: User/namespace identifier

        Returns:
            List of memory IDs
        """
        memory_ids = []
        for memory in memories:
            memory_id = await self.add_memory(
                content=memory["content"],
                user_id=user_id,
                metadata=memory.get("metadata"),
            )
            memory_ids.append(memory_id)

        return memory_ids

    async def get_all_memories(self, user_id: str) -> list[KBItem]:
        """Get all memories for a user"""
        await self.initialize()

        try:
            results = self._memory.get_all(user_id=user_id)

            kb_items = []
            for i, result in enumerate(results):
                kb_item = KBItem(
                    doc_id=result.get("id", f"mem0_{i}"),
                    excerpt=result.get("memory", ""),
                    score=1.0,  # No relevance score for get_all
                    metadata=result.get("metadata", {}),
                )
                kb_items.append(kb_item)

            return kb_items

        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a specific memory"""
        await self.initialize()

        try:
            self._memory.delete(memory_id=memory_id, user_id=user_id)
            logger.debug(f"Deleted memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def update_memory(self, memory_id: str, content: str, user_id: str) -> bool:
        """Update existing memory content"""
        await self.initialize()

        try:
            self._memory.update(memory_id=memory_id, data=content, user_id=user_id)
            logger.debug(f"Updated memory {memory_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False

    @asynccontextmanager
    async def get_memory_client(self):
        """Context manager for direct Mem0 Memory access"""
        await self.initialize()
        try:
            yield self._memory
        finally:
            pass  # Mem0 handles cleanup internally

    async def health_check(self) -> bool:
        """Check if Mem0 and underlying stores are healthy"""
        try:
            await self.initialize()
            # Try a simple operation to verify connectivity
            test_user = f"{self.config.get_current_namespace()}_health_check"
            await self.search("health_check", test_user, limit=1)
            return True
        except Exception as e:
            logger.error(f"Mem0 health check failed: {e}")
            return False
