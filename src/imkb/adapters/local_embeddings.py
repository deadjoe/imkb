"""
Local embeddings adapter for offline development

Provides embedding capabilities without requiring external API keys.
Uses sentence-transformers for local embedding generation.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LocalEmbeddingsAdapter:
    """
    Local embeddings using sentence-transformers

    Provides a fallback for development when API keys are not available.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".embeddings_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._model = None
        self._dimension = 384  # Default for all-MiniLM-L6-v2

    async def initialize(self) -> None:
        """Initialize the local embedding model"""
        if self._model is not None:
            return

        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Local embeddings initialized, dimension: {self._dimension}")

        except ImportError:
            logger.warning(
                "sentence-transformers not installed, using random embeddings"
            )
            self._model = "mock"
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = "mock"

    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for text"""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
        return self.cache_dir / f"{text_hash}.json"

    def _load_from_cache(self, text: str) -> Optional[list[float]]:
        """Load embedding from cache"""
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                    if data.get("model") == self.model_name:
                        return data["embedding"]
            except Exception as e:
                logger.debug(f"Cache read failed: {e}")
        return None

    def _save_to_cache(self, text: str, embedding: list[float]) -> None:
        """Save embedding to cache"""
        cache_path = self._get_cache_path(text)
        try:
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "text": text[:100],  # Store truncated text for reference
                        "model": self.model_name,
                        "embedding": embedding,
                    },
                    f,
                )
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def _generate_mock_embedding(self, text: str) -> list[float]:
        """Generate deterministic mock embedding for development"""
        # Use text hash to generate deterministic but pseudo-random embedding
        text_hash = hashlib.sha256(text.encode()).digest()

        # Convert hash bytes to floats in range [-1, 1]
        embedding = []
        for i in range(min(len(text_hash), self._dimension // 8)):
            # Take groups of 4 bytes and convert to float
            chunk = (
                text_hash[i * 4 : (i + 1) * 4]
                if i * 4 + 4 <= len(text_hash)
                else text_hash[i * 4 :] + b"\x00" * (4 - (len(text_hash) - i * 4))
            )
            value = int.from_bytes(chunk, "big") / (2**31) - 1.0
            embedding.append(value)

        # Pad with zeros if needed
        while len(embedding) < self._dimension:
            embedding.append(0.0)

        return embedding[: self._dimension]

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text"""
        await self.initialize()

        # Check cache first
        cached = self._load_from_cache(text)
        if cached:
            return cached

        # Generate embedding
        if self._model == "mock":
            embedding = self._generate_mock_embedding(text)
        else:
            try:
                embedding_array = self._model.encode([text])[0]
                embedding = embedding_array.tolist()
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                embedding = self._generate_mock_embedding(text)

        # Cache the result
        self._save_to_cache(text, embedding)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts"""
        await self.initialize()

        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension


# Global instance
_local_embeddings: Optional[LocalEmbeddingsAdapter] = None


def get_local_embeddings() -> LocalEmbeddingsAdapter:
    """Get global local embeddings instance"""
    global _local_embeddings
    if _local_embeddings is None:
        _local_embeddings = LocalEmbeddingsAdapter()
    return _local_embeddings
