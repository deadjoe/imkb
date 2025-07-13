"""
MySQL knowledge base extractor

Extracts relevant knowledge from MySQL-based knowledge repositories.
Supports both direct database connections and REST API endpoints.
"""

from typing import List, Optional, Dict, Any
import logging
import re
import asyncio

from .base import ExtractorBase, register_extractor
from ..adapters.mem0 import Mem0Adapter
from ..config import ImkbConfig
from ..models import Event, KBItem

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
        
        # Mock MySQL KB data for development
        self._mysql_knowledge = [
            {
                "id": "mysql_kb_001",
                "title": "Connection Pool Exhaustion Troubleshooting",
                "content": "MySQL connection pool exhaustion occurs when all available connections are in use. Common causes include connection leaks, insufficient pool sizing, long-running transactions, and application connection mismanagement. Solutions involve reviewing pool configuration, implementing connection monitoring, fixing connection leaks, and optimizing query performance.",
                "category": "connection_management",
                "severity": "P1",
                "tags": ["mysql", "connection", "pool", "exhaustion"],
                "solution_steps": [
                    "Check current connection count: SHOW PROCESSLIST",
                    "Review connection pool configuration",
                    "Identify and terminate blocking queries",
                    "Increase max_connections if needed",
                    "Audit application connection handling"
                ]
            },
            {
                "id": "mysql_kb_002",
                "title": "MySQL High CPU Usage Investigation",
                "content": "High CPU usage in MySQL can result from inefficient queries, missing indexes, table locks, or excessive concurrent connections. Investigation should focus on query analysis, index optimization, and connection monitoring. Use SHOW PROCESSLIST, EXPLAIN PLAN, and performance_schema for diagnosis.",
                "category": "performance",
                "severity": "P2",
                "tags": ["mysql", "cpu", "performance", "optimization"],
                "solution_steps": [
                    "Identify slow queries using slow query log",
                    "Run EXPLAIN on problematic queries",
                    "Check for missing indexes",
                    "Monitor concurrent connections",
                    "Optimize query performance"
                ]
            },
            {
                "id": "mysql_kb_003", 
                "title": "MySQL Memory Usage and OOM Prevention",
                "content": "MySQL memory issues often stem from incorrect buffer pool sizing, memory leaks in queries, or insufficient system memory allocation. Key areas include innodb_buffer_pool_size, query_cache_size, and connection memory usage. Monitor memory consumption patterns and adjust configuration accordingly.",
                "category": "memory_management",
                "severity": "P1",
                "tags": ["mysql", "memory", "oom", "innodb", "buffer_pool"],
                "solution_steps": [
                    "Monitor memory usage: SHOW STATUS LIKE 'memory%'",
                    "Review innodb_buffer_pool_size setting",
                    "Check for memory-intensive queries",
                    "Adjust memory allocation parameters",
                    "Implement memory monitoring alerts"
                ]
            },
            {
                "id": "mysql_kb_004",
                "title": "MySQL Replication Lag Resolution",
                "content": "Replication lag in MySQL master-slave setups can be caused by network issues, large transactions, insufficient slave resources, or binary log configuration problems. Resolution involves identifying bottlenecks, optimizing replication settings, and ensuring adequate slave capacity.",
                "category": "replication",
                "severity": "P2", 
                "tags": ["mysql", "replication", "lag", "master", "slave"],
                "solution_steps": [
                    "Check replication status: SHOW SLAVE STATUS",
                    "Monitor Seconds_Behind_Master metric",
                    "Identify large transactions in binary log",
                    "Optimize slave hardware resources",
                    "Review replication configuration"
                ]
            },
            {
                "id": "mysql_kb_005",
                "title": "MySQL Deadlock Detection and Resolution",
                "content": "MySQL deadlocks occur when two or more transactions wait for each other to release locks. Common scenarios include conflicting transaction orders, long-running transactions, and insufficient indexes. Detection involves analyzing InnoDB status and implementing proper transaction design patterns.",
                "category": "locking",
                "severity": "P2",
                "tags": ["mysql", "deadlock", "transaction", "innodb", "locks"],
                "solution_steps": [
                    "Review SHOW ENGINE INNODB STATUS output",
                    "Identify deadlock patterns in error logs",
                    "Implement consistent transaction ordering",
                    "Reduce transaction duration",
                    "Add appropriate indexes to reduce lock scope"
                ]
            }
        ]
    
    async def match(self, event: Event) -> bool:
        """
        Match events related to MySQL or database issues
        """
        # Check for MySQL-specific keywords
        mysql_keywords = [
            "mysql", "database", "db", "connection", "pool", 
            "query", "innodb", "replication", "deadlock",
            "mariadb", "sql"
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
            
            if any(keyword in service or keyword in component for keyword in mysql_keywords):
                return True
        
        # Check for database port numbers
        port_patterns = [":3306", ":3307", "port.*3306"]
        if any(re.search(pattern, event_text) for pattern in port_patterns):
            return True
        
        return False
    
    async def recall(self, event: Event, k: int = 10) -> List[KBItem]:
        """
        Recall relevant MySQL knowledge using Mem0 + local KB
        """
        try:
            # Generate user_id for namespace isolation
            user_id = f"{self.config.namespace}_{event.source}_{self.name}"
            
            # First, try to search existing memories in Mem0
            memories = []
            try:
                memories = await self.mem0_adapter.search(
                    query=f"MySQL {event.signature} {event.message}",
                    user_id=user_id,
                    limit=k//2
                )
                
                # If we don't have enough memories, seed with MySQL KB data
                if len(memories) < 3:
                    await self._seed_mysql_knowledge(user_id)
                    # Search again after seeding
                    memories = await self.mem0_adapter.search(
                        query=f"MySQL {event.signature} {event.message}",
                        user_id=user_id,
                        limit=k//2
                    )
            except Exception as mem_error:
                logger.warning(f"Mem0 search failed, using local MySQL KB only: {mem_error}")
                memories = []
            
            # Generate relevant MySQL KB items based on event content
            mysql_items = self._search_mysql_kb(event, k - len(memories))
            
            # Combine results
            all_items = memories + mysql_items
            
            logger.info(f"MySQL extractor recalled {len(all_items)} items for event {event.id}")
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
                        "kb_id": kb_item["id"]
                    }
                )
            logger.info(f"Seeded {len(self._mysql_knowledge)} MySQL KB items for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to seed MySQL knowledge: {e}")
    
    def _search_mysql_kb(self, event: Event, count: int) -> List[KBItem]:
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
                    "title": kb_item["title"]
                }
            )
            kb_items.append(kb_item_obj)
        
        return kb_items
    
    def _calculate_relevance_score(self, event_text: str, kb_item: Dict[str, Any]) -> float:
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
        high_value_keywords = ["connection", "pool", "exhaustion", "deadlock", "replication"]
        exact_matches = sum(1 for keyword in high_value_keywords if keyword in event_text and keyword in kb_item["content"].lower())
        score += exact_matches * 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def get_prompt_context(self, event: Event, snippets: List[KBItem]) -> Dict[str, Any]:
        """
        Generate MySQL-specific prompt context
        """
        base_context = super().get_prompt_context(event, snippets)
        
        # Extract MySQL-specific information
        mysql_context = {
            "database_type": "MySQL/MariaDB",
            "common_issues": self._identify_common_issues(event, snippets),
            "diagnostic_commands": self._suggest_diagnostic_commands(event, snippets),
            "solution_patterns": self._extract_solution_patterns(snippets)
        }
        
        base_context.update(mysql_context)
        return base_context
    
    def _identify_common_issues(self, event: Event, snippets: List[KBItem]) -> List[str]:
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
    
    def _suggest_diagnostic_commands(self, event: Event, snippets: List[KBItem]) -> List[str]:
        """Suggest relevant MySQL diagnostic commands"""
        commands = ["SHOW PROCESSLIST;", "SHOW STATUS;"]
        
        event_text = f"{event.signature} {event.message}".lower()
        
        if "connection" in event_text:
            commands.extend([
                "SHOW VARIABLES LIKE 'max_connections';",
                "SHOW STATUS LIKE 'Connections';",
                "SHOW STATUS LIKE 'Threads_connected';"
            ])
        
        if "deadlock" in event_text:
            commands.append("SHOW ENGINE INNODB STATUS;")
        
        if "replication" in event_text:
            commands.append("SHOW SLAVE STATUS;")
        
        return commands
    
    def _extract_solution_patterns(self, snippets: List[KBItem]) -> List[str]:
        """Extract solution patterns from KB snippets"""
        solutions = []
        
        for snippet in snippets:
            if "solution_steps" in snippet.metadata:
                solutions.extend(snippet.metadata["solution_steps"][:3])  # Top 3 steps
        
        return list(set(solutions))  # Remove duplicates