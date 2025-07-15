"""
MySQL knowledge base extractor

Extracts relevant knowledge from MySQL-based knowledge repositories.
Supports both direct database connections and REST API endpoints.
"""

import logging
import re
from typing import Any

from ..adapters.mem0 import Mem0Adapter
from ..config import ImkbConfig
from ..models import Event, KBItem
from .base import ExtractorBase, register_extractor

logger = logging.getLogger(__name__)


@register_extractor
class MySQLKBExtractor(ExtractorBase):
    """
    MySQL Knowledge Base Extractor

    This extractor:
    - Matches events related to MySQL, database, or connection issues
    - Searches MySQL-based knowledge repositories
    - Integrates with Mem0 for caching and relationship learning
    """

    name = "mysqlkb"
    prompt_template = "mysql_rca:v1"

    def __init__(self, config: ImkbConfig):
        super().__init__(config)
        self.mem0_adapter = Mem0Adapter(config)
        self._mysql_knowledge = self._load_mysql_knowledge()

    def _load_mysql_knowledge(self) -> list[dict[str, Any]]:
        """Load MySQL knowledge base from external YAML file"""
        try:
            from pathlib import Path

            import yaml

            # Get the knowledge file path
            knowledge_file = Path(__file__).parent / "data" / "mysql_knowledge.yaml"

            if not knowledge_file.exists():
                logger.warning(f"MySQL knowledge file not found: {knowledge_file}")
                return []

            with open(knowledge_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            knowledge_base = data.get("knowledge_base", [])
            logger.info(f"Loaded {len(knowledge_base)} MySQL knowledge entries from {knowledge_file}")

            return knowledge_base

        except Exception as e:
            logger.error(f"Failed to load MySQL knowledge base: {e}")
            # Return empty list as fallback
            return []

    async def match(self, event: Event) -> bool:
        """
        Match events related to MySQL or database issues
        """
        # Check for MySQL-specific keywords
        mysql_keywords = [
            "mysql",
            "database",
            "db",
            "connection",
            "pool",
            "query",
            "innodb",
            "replication",
            "deadlock",
            "mariadb",
            "sql",
        ]

        # Check event signature and message
        event_text = f"{event.signature} {event.message}".lower()

        # Direct MySQL keyword match
        if any(keyword in event_text for keyword in mysql_keywords):
            return True

        # Check labels for database-related services
        if event.labels:
            service = event.labels.get("service", "").lower()
            component = event.labels.get("component", "").lower()

            if any(
                keyword in service or keyword in component for keyword in mysql_keywords
            ):
                return True

        # Check for database port numbers
        port_patterns = [":3306", ":3307", "port.*3306"]
        if any(re.search(pattern, event_text) for pattern in port_patterns):
            return True

        return False

    async def recall(self, event: Event, k: int = 10) -> list[KBItem]:
        """
        Recall relevant MySQL knowledge using Mem0 + local KB
        """
        try:
            # Generate user_id for namespace isolation
            user_id = f"{self.config.get_current_namespace()}_{event.source}_{self.name}"

            # First, try to search existing memories in Mem0
            memories = []
            try:
                memories = await self.mem0_adapter.search(
                    query=f"MySQL {event.signature} {event.message}",
                    user_id=user_id,
                    limit=k // 2,
                )

                # If we don't have enough memories, seed with MySQL KB data
                if len(memories) < 3:
                    await self._seed_mysql_knowledge(user_id)
                    # Search again after seeding
                    memories = await self.mem0_adapter.search(
                        query=f"MySQL {event.signature} {event.message}",
                        user_id=user_id,
                        limit=k // 2,
                    )
            except Exception as mem_error:
                logger.warning(
                    f"Mem0 search failed, using local MySQL KB only: {mem_error}"
                )
                memories = []

            # Generate relevant MySQL KB items based on event content
            mysql_items = self._search_mysql_kb(event, k - len(memories))

            # Combine results
            all_items = memories + mysql_items

            logger.info(
                f"MySQL extractor recalled {len(all_items)} items for event {event.id}"
            )
            return all_items[:k]

        except Exception as e:
            logger.error(f"MySQL extractor recall failed: {e}")
            # Fallback to local MySQL KB only
            return self._search_mysql_kb(event, k)

    async def _seed_mysql_knowledge(self, user_id: str) -> None:
        """Seed Mem0 with MySQL knowledge base"""
        try:
            for kb_item in self._mysql_knowledge:
                content = f"{kb_item['title']}: {kb_item['content']}"
                await self.mem0_adapter.add_memory(
                    content=content,
                    user_id=user_id,
                    metadata={
                        "source": "mysql_kb",
                        "category": kb_item["category"],
                        "severity": kb_item["severity"],
                        "tags": kb_item["tags"],
                        "kb_id": kb_item["id"],
                    },
                )
            logger.info(
                f"Seeded {len(self._mysql_knowledge)} MySQL KB items for user {user_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to seed MySQL knowledge: {e}")

    def _search_mysql_kb(self, event: Event, count: int) -> list[KBItem]:
        """Search local MySQL knowledge base"""
        event_text = f"{event.signature} {event.message}".lower()

        # Score and rank MySQL KB items
        scored_items = []

        for kb_item in self._mysql_knowledge:
            score = self._calculate_relevance_score(event_text, kb_item)
            if score > 0.3:  # Minimum relevance threshold
                scored_items.append((kb_item, score))

        # Sort by relevance score
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Convert to KBItem format
        kb_items = []
        for kb_item, score in scored_items[:count]:
            kb_item_obj = KBItem(
                doc_id=kb_item["id"],
                excerpt=f"{kb_item['title']}: {kb_item['content'][:300]}...",
                score=score,
                metadata={
                    "source": "mysql_kb",
                    "category": kb_item["category"],
                    "severity": kb_item["severity"],
                    "tags": kb_item["tags"],
                    "solution_steps": kb_item["solution_steps"],
                    "title": kb_item["title"],
                },
            )
            kb_items.append(kb_item_obj)

        return kb_items

    def _calculate_relevance_score(
        self, event_text: str, kb_item: dict[str, Any]
    ) -> float:
        """Calculate relevance score between event and KB item"""
        score = 0.0

        # Check title keywords
        title_words = kb_item["title"].lower().split()
        title_matches = sum(1 for word in title_words if word in event_text)
        score += (title_matches / len(title_words)) * 0.4

        # Check content keywords
        content_words = kb_item["content"].lower().split()[:50]  # First 50 words
        content_matches = sum(1 for word in content_words if word in event_text)
        score += (content_matches / len(content_words)) * 0.3

        # Check tags
        tag_matches = sum(1 for tag in kb_item["tags"] if tag in event_text)
        if kb_item["tags"]:
            score += (tag_matches / len(kb_item["tags"])) * 0.3

        # Boost score for exact keyword matches
        high_value_keywords = [
            "connection",
            "pool",
            "exhaustion",
            "deadlock",
            "replication",
        ]
        exact_matches = sum(
            1
            for keyword in high_value_keywords
            if keyword in event_text and keyword in kb_item["content"].lower()
        )
        score += exact_matches * 0.2

        return min(score, 1.0)  # Cap at 1.0

    def get_prompt_context(
        self, event: Event, snippets: list[KBItem]
    ) -> dict[str, Any]:
        """
        Generate MySQL-specific prompt context
        """
        base_context = super().get_prompt_context(event, snippets)

        # Extract MySQL-specific information
        mysql_context = {
            "database_type": "MySQL/MariaDB",
            "common_issues": self._identify_common_issues(event, snippets),
            "diagnostic_commands": self._suggest_diagnostic_commands(event, snippets),
            "solution_patterns": self._extract_solution_patterns(snippets),
        }

        base_context.update(mysql_context)
        return base_context

    def _identify_common_issues(
        self, event: Event, snippets: list[KBItem]
    ) -> list[str]:
        """Identify common MySQL issue patterns"""
        issues = []
        event_text = f"{event.signature} {event.message}".lower()

        if "connection" in event_text and ("pool" in event_text or "max" in event_text):
            issues.append("Connection pool exhaustion")

        if "cpu" in event_text or "high" in event_text:
            issues.append("High CPU usage")

        if "memory" in event_text or "oom" in event_text:
            issues.append("Memory management issues")

        if "deadlock" in event_text:
            issues.append("Transaction deadlocks")

        if "replication" in event_text or "lag" in event_text:
            issues.append("Replication lag")

        return issues

    def _suggest_diagnostic_commands(
        self, event: Event, snippets: list[KBItem]
    ) -> list[str]:
        """Suggest relevant MySQL diagnostic commands"""
        commands = ["SHOW PROCESSLIST;", "SHOW STATUS;"]

        event_text = f"{event.signature} {event.message}".lower()

        if "connection" in event_text:
            commands.extend(
                [
                    "SHOW VARIABLES LIKE 'max_connections';",
                    "SHOW STATUS LIKE 'Connections';",
                    "SHOW STATUS LIKE 'Threads_connected';",
                ]
            )

        if "deadlock" in event_text:
            commands.append("SHOW ENGINE INNODB STATUS;")

        if "replication" in event_text:
            commands.append("SHOW SLAVE STATUS;")

        return commands

    def _extract_solution_patterns(self, snippets: list[KBItem]) -> list[str]:
        """Extract solution patterns from KB snippets"""
        solutions = []

        for snippet in snippets:
            if "solution_steps" in snippet.metadata:
                solutions.extend(snippet.metadata["solution_steps"][:3])  # Top 3 steps

        return list(set(solutions))  # Remove duplicates
