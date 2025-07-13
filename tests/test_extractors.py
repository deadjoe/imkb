"""
Test suite for extractors system

Tests base extractor functionality, specific extractors, and the registry system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from imkb.extractors import (
    Event, 
    KBItem, 
    BaseExtractor,
    ExtractorBase,
    registry
)
from imkb.extractors.test_extractor import TestExtractor
from imkb.extractors.mysql_extractor import MySQLKBExtractor
from imkb.config import ImkbConfig


class TestEvent:
    """Test Event class"""
    
    def test_basic_event_creation(self):
        """Test basic event creation"""
        event_data = {
            "id": "test-123",
            "signature": "db.connection.error",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "critical",
            "source": "monitoring",
            "labels": {"service": "database"},
            "message": "Cannot connect to MySQL"
        }
        
        event = Event.from_dict(event_data)
        
        assert event.id == "test-123"
        assert event.signature == "db.connection.error"
        assert event.timestamp == "2024-01-01T12:00:00Z"
        assert event.severity == "critical"
        assert event.source == "monitoring"
        assert event.labels == {"service": "database"}
        assert event.message == "Cannot connect to MySQL"
    
    def test_event_with_raw_data(self):
        """Test event creation with raw data"""
        event_data = {
            "id": "test-456",
            "signature": "api.timeout",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "warning",
            "source": "api-gateway",
            "labels": {"service": "user-api"},
            "message": "API timeout after 30s",
            "raw": {
                "endpoint": "/users",
                "timeout_duration": "30s",
                "response_code": 504
            }
        }
        
        event = Event.from_dict(event_data)
        
        assert event.raw["endpoint"] == "/users"
        assert event.raw["timeout_duration"] == "30s"
        assert event.raw["response_code"] == 504
    
    def test_event_to_dict(self):
        """Test event to_dict conversion"""
        event = Event(
            id="test-789",
            signature="memory.leak",
            timestamp="2024-01-01T12:00:00Z",
            severity="warning",
            source="metrics",
            labels={"component": "worker"},
            message="High memory usage detected"
        )
        
        result = event.to_dict()
        expected = {
            "id": "test-789",
            "signature": "memory.leak",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "warning",
            "source": "metrics",
            "labels": {"component": "worker"},
            "message": "High memory usage detected",
            "raw": {},
            "context_hash": None,
            "embedding_version": "v1.0"
        }
        
        assert result == expected


class TestKBItem:
    """Test KBItem class"""
    
    def test_basic_kb_item(self):
        """Test basic KBItem creation"""
        item = KBItem(
            doc_id="kb_001",
            excerpt="Database connection pool exhausted",
            score=0.85,
            metadata={"category": "database"}
        )
        
        assert item.doc_id == "kb_001"
        assert item.excerpt == "Database connection pool exhausted"
        assert item.score == 0.85
        assert item.metadata["category"] == "database"
    
    def test_kb_item_to_dict(self):
        """Test KBItem to_dict conversion"""
        item = KBItem(
            doc_id="kb_002",
            excerpt="Connection timeout error",
            score=0.7
        )
        
        result = item.to_dict()
        expected = {
            "doc_id": "kb_002",
            "excerpt": "Connection timeout error",
            "score": 0.7,
            "metadata": {}
        }
        
        assert result == expected


class TestBaseExtractor:
    """Test base extractor functionality"""
    
    def test_extractor_abstract_methods(self):
        """Test that ExtractorBase cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ExtractorBase(ImkbConfig())
    
    def test_extractor_interface(self):
        """Test extractor interface requirements"""
        # Create a minimal concrete extractor
        class MinimalExtractor(ExtractorBase):
            name = "minimal"
            prompt_template = "test:v1"
            
            async def match(self, event: Event) -> bool:
                return True
            
            async def recall(self, event: Event, k: int = 5) -> List[KBItem]:
                return []
        
        extractor = MinimalExtractor(ImkbConfig())
        assert extractor.name == "minimal"
        assert extractor.prompt_template == "test:v1"
    
    def test_get_max_results_default(self):
        """Test default max results"""
        class TestExtractorImpl(ExtractorBase):
            name = "test"
            prompt_template = "test:v1"
            
            async def match(self, event: Event) -> bool:
                return True
            
            async def recall(self, event: Event, k: int = 5) -> List[KBItem]:
                return []
        
        extractor = TestExtractorImpl(ImkbConfig())
        assert extractor.get_max_results() == 10  # Default value from ExtractorBase
    
    def test_get_prompt_context_default(self):
        """Test default prompt context"""
        class TestExtractorImpl(ExtractorBase):
            name = "test"
            prompt_template = "test:v1"
            
            async def match(self, event: Event) -> bool:
                return True
            
            async def recall(self, event: Event, k: int = 5) -> List[KBItem]:
                return []
        
        event = Event(
            id="test", 
            signature="test.event",
            timestamp="2024-01-01T12:00:00Z",
            severity="info",
            source="test",
            labels={},
            message="Test event"
        )
        kb_items = [KBItem(doc_id="kb_001", excerpt="Test knowledge", score=0.8)]
        
        extractor = TestExtractorImpl(ImkbConfig())
        context = extractor.get_prompt_context(event, kb_items)
        
        assert "event" in context
        assert "snippets" in context
        assert context["event"]["id"] == "test"
        assert len(context["snippets"]) == 1


class TestTestExtractor:
    """Test TestExtractor implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = TestExtractor(ImkbConfig())
    
    def test_extractor_properties(self):
        """Test TestExtractor properties"""
        assert self.extractor.name == "test"
        assert self.extractor.prompt_template == "test_rca:v1"
    
    @pytest.mark.asyncio
    async def test_match_always_true(self):
        """Test that TestExtractor matches any event"""
        event = Event(
            id="test",
            signature="any.event",
            timestamp="2024-01-01T12:00:00Z",
            severity="info",
            source="test",
            labels={},
            message="Any Event"
        )
        assert await self.extractor.match(event) is True
    
    @pytest.mark.asyncio
    async def test_recall_returns_mock_items(self):
        """Test TestExtractor recall returns mock knowledge items"""
        event = Event(
            id="test-incident", 
            signature="database.error",
            timestamp="2024-01-01T12:00:00Z",
            severity="critical",
            source="monitoring",
            labels={"service": "database"},
            message="Database Error: Connection failed"
        )
        
        items = await self.extractor.recall(event, k=3)
        
        assert len(items) == 3
        for item in items:
            assert isinstance(item, KBItem)
            assert 0.5 <= item.score <= 1.0
            # Just check that we got some relevant content
            assert len(item.excerpt) > 10
    
    @pytest.mark.asyncio
    async def test_recall_respects_k_parameter(self):
        """Test that recall respects the k parameter"""
        event = Event(
            id="test",
            signature="test.event",
            timestamp="2024-01-01T12:00:00Z",
            severity="info",
            source="test",
            labels={},
            message="Test"
        )
        
        items_k3 = await self.extractor.recall(event, k=3)
        items_k5 = await self.extractor.recall(event, k=5)
        
        assert len(items_k3) == 3
        assert len(items_k5) == 5


class TestMySQLKBExtractor:
    """Test MySQLKBExtractor implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.extractor = MySQLKBExtractor(ImkbConfig())
    
    def test_extractor_properties(self):
        """Test MySQLKBExtractor properties"""
        assert self.extractor.name == "mysqlkb"
        assert self.extractor.prompt_template == "mysql_rca:v1"
    
    @pytest.mark.asyncio
    async def test_match_mysql_keywords(self):
        """Test MySQL keyword matching"""
        # Should match MySQL-related events
        mysql_event = Event(
            id="mysql-1",
            signature="mysql.connection.error",
            timestamp="2024-01-01T12:00:00Z",
            severity="critical",
            source="monitoring",
            labels={"service": "mysql"},
            message="Database connection pool exhausted"
        )
        assert await self.extractor.match(mysql_event) is True
        
        # Should match connection-related events
        conn_event = Event(
            id="conn-1",
            signature="database.connection.timeout",
            timestamp="2024-01-01T12:00:00Z",
            severity="warning",
            source="monitoring",
            labels={},
            message="Failed to connect to database"
        )
        assert await self.extractor.match(conn_event) is True
        
        # Should not match unrelated events
        unrelated_event = Event(
            id="other-1",
            signature="network.latency.high",
            timestamp="2024-01-01T12:00:00Z",
            severity="info",
            source="monitoring",
            labels={},
            message="High latency detected in network"
        )
        assert await self.extractor.match(unrelated_event) is False
    
    @pytest.mark.asyncio
    async def test_match_case_insensitive(self):
        """Test case-insensitive keyword matching"""
        event = Event(
            id="test",
            signature="mysql.connection.issue",
            timestamp="2024-01-01T12:00:00Z",
            severity="warning",
            source="monitoring",
            labels={},
            message="DATABASE timeout"
        )
        assert await self.extractor.match(event) is True
    
    @pytest.mark.asyncio
    async def test_recall_returns_mysql_knowledge(self):
        """Test MySQL knowledge recall"""
        event = Event(
            id="mysql-incident",
            signature="mysql.connection.pool.exhausted",
            timestamp="2024-01-01T12:00:00Z",
            severity="critical",
            source="monitoring",
            labels={"service": "mysql"},
            message="Too many connections error"
        )
        
        items = await self.extractor.recall(event, k=4)
        
        assert len(items) == 4
        for item in items:
            assert isinstance(item, KBItem)
            assert 0.3 <= item.score <= 1.0
            assert any(keyword in item.excerpt.lower() for keyword in 
                      ["connection", "mysql", "database", "pool", "timeout"])
    
    @pytest.mark.asyncio
    async def test_get_max_results(self):
        """Test max results for MySQL extractor"""
        assert self.extractor.get_max_results() == 10  # Default value


class TestExtractorRegistry:
    """Test extractor registry system"""
    
    def test_registry_initialization(self):
        """Test registry starts with registered extractors"""
        # Registry should have both test and mysql extractors
        extractor_names = registry.get_available_extractors()
        
        assert "test" in extractor_names
        assert "mysqlkb" in extractor_names
    
    def test_get_extractor_by_name(self):
        """Test getting extractor by name"""
        config = ImkbConfig()
        
        test_extractor = registry.create_extractor("test", config)
        assert test_extractor is not None
        assert test_extractor.name == "test"
        
        mysql_extractor = registry.create_extractor("mysqlkb", config)
        assert mysql_extractor is not None
        assert mysql_extractor.name == "mysqlkb"
        
        # Non-existent extractor
        assert registry.create_extractor("nonexistent", config) is None
    
    def test_create_enabled_extractors(self):
        """Test creating enabled extractors from config"""
        config = ImkbConfig()
        config.extractors.enabled = ["mysqlkb"]
        
        extractors = registry.create_enabled_extractors(config)
        
        assert len(extractors) == 1
        assert extractors[0].name == "mysqlkb"
    
    def test_create_enabled_extractors_with_invalid(self):
        """Test creating extractors with invalid names in config"""
        config = ImkbConfig()
        config.extractors.enabled = ["mysqlkb", "nonexistent", "test"]
        
        extractors = registry.create_enabled_extractors(config)
        
        # Should only create valid extractors
        assert len(extractors) == 2
        extractor_names = [e.name for e in extractors]
        assert "mysqlkb" in extractor_names
        assert "test" in extractor_names
        assert "nonexistent" not in extractor_names
    
    def test_register_custom_extractor(self):
        """Test registering a custom extractor"""
        from imkb.extractors import register_extractor
        
        @register_extractor
        class CustomExtractor(ExtractorBase):
            name = "custom_test"
            prompt_template = "custom:v1"
            
            async def match(self, event: Event) -> bool:
                return "custom" in event.message.lower()
            
            async def recall(self, event: Event, k: int = 5) -> List[KBItem]:
                return [KBItem(doc_id="custom_001", excerpt="Custom knowledge", score=0.8)]
        
        # Verify registration
        config = ImkbConfig()
        custom_extractor = registry.create_extractor("custom_test", config)
        assert custom_extractor is not None
        assert custom_extractor.name == "custom_test"
        
        # Clean up
        if "custom_test" in registry._extractors:
            del registry._extractors["custom_test"]


class TestExtractorIntegration:
    """Integration tests for extractor system"""
    
    @pytest.mark.asyncio
    async def test_event_processing_workflow(self):
        """Test complete event processing workflow"""
        # Create event
        event_data = {
            "id": "incident-001",
            "signature": "mysql.connection.pool.exhausted",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "critical",
            "source": "monitoring",
            "labels": {
                "database": "user_db",
                "service": "mysql"
            },
            "message": "Database is refusing new connections"
        }
        
        event = Event.from_dict(event_data)
        
        # Test with MySQL extractor
        mysql_extractor = MySQLKBExtractor(ImkbConfig())
        
        # Should match
        assert await mysql_extractor.match(event) is True
        
        # Should recall relevant knowledge
        knowledge = await mysql_extractor.recall(event, k=5)
        assert len(knowledge) == 5
        assert all(isinstance(item, KBItem) for item in knowledge)
        
        # Test prompt context generation
        context = mysql_extractor.get_prompt_context(event, knowledge)
        
        assert "event" in context
        assert "snippets" in context
        assert context["event"]["id"] == "incident-001"
        assert len(context["snippets"]) == 5
    
    @pytest.mark.asyncio
    async def test_extractor_priority_and_matching(self):
        """Test extractor priority and matching logic"""
        event = Event(
            id="test",
            signature="mysql.connection.error",
            timestamp="2024-01-01T12:00:00Z",
            severity="warning",
            source="monitoring",
            labels={},
            message="Database connection failed"
        )
        
        config = ImkbConfig()
        config.extractors.enabled = ["test", "mysqlkb"]
        
        extractors = registry.create_enabled_extractors(config)
        
        # Both extractors should match this event
        matching_extractors = []
        for extractor in extractors:
            if await extractor.match(event):
                matching_extractors.append(extractor)
        
        assert len(matching_extractors) >= 1
        
        # MySQL extractor should be among them since it has specific keywords
        mysql_matches = any(e.name == "mysqlkb" for e in matching_extractors)
        assert mysql_matches is True
    
    @pytest.mark.asyncio
    async def test_knowledge_item_quality(self):
        """Test knowledge item quality and relevance"""
        event = Event(
            id="db-incident",
            signature="mysql.connection.pool.issues",
            timestamp="2024-01-01T12:00:00Z",
            severity="critical",
            source="monitoring",
            labels={"service": "mysql"},
            message="MySQL connection pool is running out of connections"
        )
        
        mysql_extractor = MySQLKBExtractor(ImkbConfig())
        knowledge_items = await mysql_extractor.recall(event, k=3)
        
        # Check knowledge quality
        for item in knowledge_items:
            # Should have reasonable score
            assert item.score >= 0.3
            assert item.score <= 1.0
            
            # Should contain relevant content
            assert len(item.excerpt) > 10  # Not empty
            
            # Should be relevant to the incident
            relevant_keywords = ["connection", "mysql", "database", "pool"]
            has_relevant_content = any(
                keyword in item.excerpt.lower() 
                for keyword in relevant_keywords
            )
            assert has_relevant_content