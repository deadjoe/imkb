"""
Local storage fallback for development without external dependencies

Provides in-memory and file-based storage for knowledge items when
external services (Mem0, Qdrant, Neo4j) are not available or configured.
"""

from typing import Dict, List, Optional, Any
import json
import logging
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import hashlib

from .local_embeddings import get_local_embeddings
from .mem0 import KBItem

logger = logging.getLogger(__name__)


class LocalKnowledgeStore:
    """
    Local knowledge store for development
    
    Provides basic vector similarity search and knowledge storage
    without requiring external databases.
    """
    
    def __init__(self, storage_dir: str = ".imkb_local_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.embeddings_adapter = get_local_embeddings()
        self._memories: Dict[str, List[Dict[str, Any]]] = {}
        self._loaded_users = set()
        
        # In-memory index for fast similarity search
        self._embeddings_cache: Dict[str, List[float]] = {}
    
    def _get_user_file(self, user_id: str) -> Path:
        """Get storage file path for user"""
        user_hash = hashlib.md5(user_id.encode()).hexdigest()[:16]
        return self.storage_dir / f"user_{user_hash}.json"
    
    async def _load_user_memories(self, user_id: str) -> None:
        """Load memories for user from disk"""
        if user_id in self._loaded_users:
            return
        
        user_file = self._get_user_file(user_id)
        self._memories[user_id] = []
        
        if user_file.exists():
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._memories[user_id] = data.get("memories", [])
                    logger.debug(f"Loaded {len(self._memories[user_id])} memories for user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to load memories for {user_id}: {e}")
        
        self._loaded_users.add(user_id)
    
    async def _save_user_memories(self, user_id: str) -> None:
        """Save memories for user to disk"""
        user_file = self._get_user_file(user_id)
        
        try:
            data = {
                "user_id": user_id,
                "last_updated": datetime.now().isoformat(),
                "memories": self._memories.get(user_id, [])
            }
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(data['memories'])} memories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save memories for {user_id}: {e}")
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            # Cosine similarity calculation
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            magnitude1 = sum(a * a for a in embedding1) ** 0.5
            magnitude2 = sum(a * a for a in embedding2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0
    
    async def search(self, query: str, user_id: str, limit: int = 10) -> List[KBItem]:
        """Search for relevant memories"""
        await self._load_user_memories(user_id)
        
        memories = self._memories.get(user_id, [])
        if not memories:
            return []
        
        # Generate query embedding
        query_embedding = await self.embeddings_adapter.embed_text(query)
        
        # Calculate similarities
        results = []
        for memory in memories:
            content = memory.get("content", "")
            if not content:
                continue
            
            # Get or generate embedding for memory
            memory_id = memory.get("id", "")
            if memory_id in self._embeddings_cache:
                memory_embedding = self._embeddings_cache[memory_id]
            else:
                memory_embedding = await self.embeddings_adapter.embed_text(content)
                if memory_id:
                    self._embeddings_cache[memory_id] = memory_embedding
            
            # Calculate similarity
            similarity = self._calculate_similarity(query_embedding, memory_embedding)
            
            if similarity > 0.1:  # Minimum threshold
                results.append({
                    "memory": memory,
                    "similarity": similarity
                })
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:limit]
        
        # Convert to KBItem format
        kb_items = []
        for result in results:
            memory = result["memory"]
            kb_item = KBItem(
                doc_id=memory.get("id", f"local_{len(kb_items)}"),
                excerpt=memory.get("content", ""),
                score=result["similarity"],
                metadata={
                    "source": "local_storage",
                    "created_at": memory.get("created_at"),
                    "user_id": user_id,
                    **memory.get("metadata", {})
                }
            )
            kb_items.append(kb_item)
        
        logger.debug(f"Local search found {len(kb_items)} results for query: {query[:50]}...")
        return kb_items
    
    async def add_memory(self, content: str, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add new memory to local storage"""
        await self._load_user_memories(user_id)
        
        # Generate memory ID
        memory_id = f"local_{user_id}_{len(self._memories.get(user_id, []))}"
        
        # Create memory object
        memory = {
            "id": memory_id,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add to in-memory storage
        if user_id not in self._memories:
            self._memories[user_id] = []
        self._memories[user_id].append(memory)
        
        # Save to disk
        await self._save_user_memories(user_id)
        
        logger.debug(f"Added memory {memory_id} for user {user_id}")
        return memory_id
    
    async def add_memories_batch(self, memories: List[Dict[str, Any]], user_id: str) -> List[str]:
        """Add multiple memories in batch"""
        memory_ids = []
        for memory in memories:
            memory_id = await self.add_memory(
                content=memory["content"],
                user_id=user_id,
                metadata=memory.get("metadata")
            )
            memory_ids.append(memory_id)
        
        return memory_ids
    
    async def get_all_memories(self, user_id: str) -> List[KBItem]:
        """Get all memories for a user"""
        await self._load_user_memories(user_id)
        
        memories = self._memories.get(user_id, [])
        kb_items = []
        
        for memory in memories:
            kb_item = KBItem(
                doc_id=memory.get("id", f"local_{len(kb_items)}"),
                excerpt=memory.get("content", ""),
                score=1.0,  # No relevance score for get_all
                metadata={
                    "source": "local_storage",
                    "created_at": memory.get("created_at"),
                    "user_id": user_id,
                    **memory.get("metadata", {})
                }
            )
            kb_items.append(kb_item)
        
        return kb_items
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a specific memory"""
        await self._load_user_memories(user_id)
        
        memories = self._memories.get(user_id, [])
        for i, memory in enumerate(memories):
            if memory.get("id") == memory_id:
                del memories[i]
                await self._save_user_memories(user_id)
                
                # Remove from embeddings cache
                if memory_id in self._embeddings_cache:
                    del self._embeddings_cache[memory_id]
                
                logger.debug(f"Deleted memory {memory_id} for user {user_id}")
                return True
        
        return False
    
    async def health_check(self) -> bool:
        """Check if local storage is working"""
        try:
            # Test write and read
            test_file = self.storage_dir / "health_check.json"
            test_data = {"timestamp": datetime.now().isoformat()}
            
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            with open(test_file, 'r') as f:
                loaded_data = json.load(f)
            
            test_file.unlink()  # Clean up
            
            return loaded_data["timestamp"] == test_data["timestamp"]
            
        except Exception as e:
            logger.error(f"Local storage health check failed: {e}")
            return False


# Global instance
_local_store: Optional[LocalKnowledgeStore] = None


def get_local_store() -> LocalKnowledgeStore:
    """Get global local storage instance"""
    global _local_store
    if _local_store is None:
        _local_store = LocalKnowledgeStore()
    return _local_store