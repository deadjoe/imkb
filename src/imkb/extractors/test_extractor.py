"""
Test extractor for development and testing

A simple extractor that demonstrates the interface and provides
test data for development purposes.
"""

import logging

from ..adapters.mem0 import Mem0Adapter
from ..config import ImkbConfig
from .base import Event, ExtractorBase, KBItem, register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class TestExtractor(ExtractorBase):
    __test__ = False  # Prevent pytest from collecting this as a test class
    """
    Test extractor for development and validation
    
    This extractor:
    - Matches all events (for testing purposes)
    - Returns mock knowledge items
    - Integrates with Mem0 for hybrid storage testing
    """

    name = "test"
    prompt_template = "test_rca:v1"

    def __init__(self, config: ImkbConfig):
        super().__init__(config)
        self.mem0_adapter = Mem0Adapter(config)
        self._test_knowledge = [
            {
                "id": "test_kb_001",
                "content": "High CPU usage can be caused by inefficient queries, insufficient indexing, or resource contention.",
                "metadata": {"category": "performance", "source": "test_kb"},
            },
            {
                "id": "test_kb_002",
                "content": "Connection pool exhaustion typically indicates too many concurrent connections or connection leaks in the application.",
                "metadata": {"category": "database", "source": "test_kb"},
            },
            {
                "id": "test_kb_003",
                "content": "Memory leaks in applications can cause gradual performance degradation and eventual system instability.",
                "metadata": {"category": "memory", "source": "test_kb"},
            },
            {
                "id": "test_kb_004",
                "content": "Network timeouts may be caused by firewall rules, network congestion, or DNS resolution issues.",
                "metadata": {"category": "network", "source": "test_kb"},
            },
            {
                "id": "test_kb_005",
                "content": "Disk I/O bottlenecks can be resolved by optimizing queries, adding indexes, or upgrading storage.",
                "metadata": {"category": "storage", "source": "test_kb"},
            },
        ]

    async def match(self, event: Event) -> bool:
        """
        Match logic for test extractor

        For testing, we match events based on simple keywords or always match.
        """
        if "test" in event.source.lower():
            return True

        keywords = ["cpu", "memory", "database", "connection", "timeout", "error"]
        event_text = f"{event.message} {event.signature}".lower()

        return any(keyword in event_text for keyword in keywords)

    async def recall(self, event: Event, k: int = 10) -> list[KBItem]:
        """
        Recall relevant knowledge using Mem0 hybrid storage + test data
        """
        try:
            user_id = f"{self.config.namespace}_{event.source}_{self.name}"

            memories = []
            try:
                memories = await self.mem0_adapter.search(
                    query=event.signature,
                    user_id=user_id,
                    limit=k // 2,
                )

                if len(memories) < 2:
                    await self._seed_test_memories(user_id)
                    memories = await self.mem0_adapter.search(
                        query=event.signature, user_id=user_id, limit=k // 2
                    )
            except Exception as mem_error:
                logger.warning(f"Mem0 search failed, using mock data only: {mem_error}")
                memories = []

            mock_items = self._generate_mock_items(event, k - len(memories))

            all_items = memories + mock_items

            logger.info(
                f"Test extractor recalled {len(all_items)} items for event {event.id}"
            )
            return all_items[:k]

        except Exception as e:
            logger.error(f"Test extractor recall failed: {e}")
            return self._generate_mock_items(event, k)

    async def _seed_test_memories(self, user_id: str) -> None:
        """Seed Mem0 with test knowledge for development"""
        try:
            for knowledge in self._test_knowledge:
                await self.mem0_adapter.add_memory(
                    content=knowledge["content"],
                    user_id=user_id,
                    metadata=knowledge["metadata"],
                )
            logger.info(
                f"Seeded {len(self._test_knowledge)} test memories for user {user_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to seed test memories: {e}")

    def _generate_mock_items(self, event: Event, count: int) -> list[KBItem]:
        """Generate mock knowledge items for testing"""
        mock_items = []

        event_text = f"{event.message} {event.signature}".lower()

        relevant_knowledge = []
        for knowledge in self._test_knowledge:
            score = 0.0
            content_lower = knowledge["content"].lower()

            if "cpu" in event_text and "cpu" in content_lower:
                score = 0.9
            elif "connection" in event_text and "connection" in content_lower:
                score = 0.85
            elif "memory" in event_text and "memory" in content_lower:
                score = 0.8
            elif "network" in event_text or "timeout" in event_text:
                if "network" in content_lower or "timeout" in content_lower:
                    score = 0.75
            elif "disk" in event_text or "storage" in event_text:
                if "disk" in content_lower or "storage" in content_lower:
                    score = 0.7
            else:
                score = 0.5

            relevant_knowledge.append((knowledge, score))

        relevant_knowledge.sort(key=lambda x: x[1], reverse=True)

        for _i, (knowledge, score) in enumerate(relevant_knowledge[:count]):
            kb_item = KBItem(
                doc_id=knowledge["id"],
                excerpt=knowledge["content"],
                score=score,
                metadata=knowledge["metadata"],
            )
            mock_items.append(kb_item)

        return mock_items

    def get_prompt_context(self, event: Event, snippets: list[KBItem]) -> dict:
        """
        Custom prompt context for test extractor
        """
        base_context = super().get_prompt_context(event, snippets)

        base_context.update(
            {
                "extractor_type": "test",
                "test_mode": True,
                "knowledge_sources": list(
                    set(
                        snippet.metadata.get("source", "unknown")
                        for snippet in snippets
                    )
                ),
                "confidence_note": "This is a test analysis. Results may not reflect real production knowledge.",
            }
        )

        return base_context
