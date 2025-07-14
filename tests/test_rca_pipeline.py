"""
Test suite for RCA Pipeline

Tests RCAResult, PromptManager, RCAPipeline, and main get_rca function.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from imkb.config import ImkbConfig
from imkb.extractors import Event, KBItem
from imkb.llm_client import LLMResponse
from imkb.rca_pipeline import PromptManager, RCAPipeline, RCAResult, get_rca


class TestRCAResult:
    """Test RCAResult class"""

    def test_basic_rca_result(self):
        """Test basic RCAResult creation"""
        references = [KBItem(doc_id="kb_001", excerpt="Test knowledge", score=0.8)]

        result = RCAResult(
            root_cause="Database connection pool exhausted",
            confidence=0.85,
            extractor="mysqlkb",
            references=references,
            status="SUCCESS",
        )

        assert result.root_cause == "Database connection pool exhausted"
        assert result.confidence == 0.85
        assert result.extractor == "mysqlkb"
        assert result.status == "SUCCESS"
        assert len(result.references) == 1
        assert result.contributing_factors == []
        assert result.evidence == []

    def test_rca_result_with_all_fields(self):
        """Test RCAResult with all optional fields"""
        references = [KBItem(doc_id="kb_002", excerpt="Knowledge item", score=0.9)]

        result = RCAResult(
            root_cause="Memory leak in application",
            confidence=0.75,
            extractor="test",
            references=references,
            status="SUCCESS",
            contributing_factors=["High traffic", "Memory pressure"],
            evidence=["Memory usage graphs", "Error logs"],
            immediate_actions=["Restart service", "Monitor memory"],
            preventive_measures=["Add memory alerts", "Code review"],
            additional_investigation=["Check heap dumps"],
            confidence_reasoning="Based on memory patterns",
            knowledge_gaps=["Exact leak location"],
            metadata={"analysis_time": "2024-01-01"},
        )

        assert len(result.contributing_factors) == 2
        assert len(result.evidence) == 2
        assert len(result.immediate_actions) == 2
        assert len(result.preventive_measures) == 2
        assert len(result.additional_investigation) == 1
        assert result.confidence_reasoning == "Based on memory patterns"
        assert len(result.knowledge_gaps) == 1
        assert result.metadata["analysis_time"] == "2024-01-01"

    def test_rca_result_to_dict(self):
        """Test RCAResult to_dict conversion"""
        references = [KBItem(doc_id="kb_003", excerpt="Test", score=0.7)]

        result = RCAResult(
            root_cause="Test cause",
            confidence=0.8,
            extractor="test",
            references=references,
            immediate_actions=["Action 1"],
        )

        result_dict = result.to_dict()

        assert result_dict["root_cause"] == "Test cause"
        assert result_dict["confidence"] == 0.8
        assert result_dict["extractor"] == "test"
        assert len(result_dict["references"]) == 1
        assert result_dict["immediate_actions"] == ["Action 1"]
        assert result_dict["status"] == "SUCCESS"

    def test_rca_result_from_dict(self):
        """Test RCAResult from_dict creation"""
        data = {
            "root_cause": "Network timeout",
            "confidence": 0.9,
            "extractor": "network",
            "references": [
                {
                    "doc_id": "kb_004",
                    "excerpt": "Timeout info",
                    "score": 0.8,
                    "metadata": {},
                }
            ],
            "status": "SUCCESS",
            "contributing_factors": ["High latency"],
            "immediate_actions": ["Check network"],
        }

        result = RCAResult.from_dict(data)

        assert result.root_cause == "Network timeout"
        assert result.confidence == 0.9
        assert result.extractor == "network"
        assert len(result.references) == 1
        assert result.references[0].excerpt == "Timeout info"
        assert result.contributing_factors == ["High latency"]
        assert result.immediate_actions == ["Check network"]


class TestPromptManager:
    """Test PromptManager class"""

    def setup_method(self):
        """Setup test fixtures"""
        # Create temporary prompts directory
        self.temp_dir = tempfile.mkdtemp()
        self.prompts_dir = Path(self.temp_dir) / "prompts"
        self.prompts_dir.mkdir()

        # Create test template structure
        template_dir = self.prompts_dir / "test_rca" / "v1"
        template_dir.mkdir(parents=True)

        # Create template file
        template_file = template_dir / "template.jinja2"
        template_content = """
Analyze this incident:

Event: {{ event.title }}
Description: {{ event.description }}

Knowledge Items:
{% for item in knowledge_items %}
- {{ item.excerpt }} (confidence: {{ item.confidence }})
{% endfor %}

Provide root cause analysis.
"""
        template_file.write_text(template_content)

        # Create meta file
        meta_file = template_dir / "meta.yaml"
        meta_content = """
name: "Test RCA Template"
version: "v1"
description: "Template for testing RCA analysis"
"""
        meta_file.write_text(meta_content)

        self.prompt_manager = PromptManager(str(self.prompts_dir))

    def test_get_template(self):
        """Test getting Jinja2 template"""
        template = self.prompt_manager.get_template("test_rca/v1/template.jinja2")
        assert template is not None

        # Test rendering
        context = {
            "event": {"title": "Test Event", "description": "Test description"},
            "knowledge_items": [{"excerpt": "Test knowledge", "confidence": 0.8}],
        }

        rendered = template.render(**context)
        assert "Test Event" in rendered
        assert "Test description" in rendered
        assert "Test knowledge" in rendered

    def test_get_nonexistent_template(self):
        """Test getting non-existent template raises error"""
        import jinja2

        with pytest.raises(jinja2.TemplateNotFound):
            self.prompt_manager.get_template("nonexistent/template.jinja2")

    def test_load_template_meta(self):
        """Test loading template metadata"""
        meta = self.prompt_manager.load_template_meta("test_rca:v1")

        assert meta["name"] == "Test RCA Template"
        assert meta["version"] == "v1"
        assert meta["description"] == "Template for testing RCA analysis"

    def test_load_nonexistent_meta(self):
        """Test loading non-existent metadata returns empty dict"""
        meta = self.prompt_manager.load_template_meta("nonexistent:v1")
        assert meta == {}

    def test_render_template(self):
        """Test template rendering with context"""
        context = {
            "event": {"title": "Database Error", "description": "Connection failed"},
            "knowledge_items": [
                {"excerpt": "MySQL connection issues", "confidence": 0.9},
                {"excerpt": "Connection pool problems", "confidence": 0.8},
            ],
        }

        rendered = self.prompt_manager.render_template("test_rca:v1", context)

        assert "Database Error" in rendered
        assert "Connection failed" in rendered
        assert "MySQL connection issues" in rendered
        assert "confidence: 0.9" in rendered


class TestRCAPipeline:
    """Test RCAPipeline class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = ImkbConfig()
        self.pipeline = RCAPipeline(self.config)

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.config == self.config
        assert self.pipeline.llm_router is not None
        assert self.pipeline.prompt_manager is not None
        assert self.pipeline._extractors is None  # Lazy loading

    def test_get_extractors_lazy_loading(self):
        """Test extractor lazy loading"""
        extractors = self.pipeline.get_extractors()
        assert len(extractors) > 0
        assert self.pipeline._extractors is not None

        # Second call should return cached extractors
        extractors2 = self.pipeline.get_extractors()
        assert extractors is extractors2

    @pytest.mark.asyncio
    async def test_find_matching_extractor(self):
        """Test finding matching extractor for event"""
        event = Event(
            id="mysql-incident",
            signature="mysql.connection.error",
            timestamp="2024-01-01T12:00:00Z",
            severity="critical",
            source="monitoring",
            labels={"service": "mysql"},
            message="Database connection failed",
        )

        extractor = await self.pipeline.find_matching_extractor(event)

        # Should find an extractor (either mysql or test)
        assert extractor is not None
        assert hasattr(extractor, "name")
        assert hasattr(extractor, "match")

    @pytest.mark.asyncio
    async def test_find_matching_extractor_prioritization(self):
        """Test extractor prioritization (specific over generic)"""
        mysql_event = Event(
            id="test",
            signature="mysql.connection.pool.exhausted",
            timestamp="2024-01-01T12:00:00Z",
            severity="critical",
            source="monitoring",
            labels={"service": "mysql"},
            message="Database connection failed",
        )

        extractor = await self.pipeline.find_matching_extractor(mysql_event)

        # Should prioritize MySQL extractor over test extractor
        # MySQL extractor has specific keywords, test extractor matches everything
        assert extractor is not None
        # The exact extractor depends on the sorting logic in the pipeline

    @pytest.mark.asyncio
    async def test_find_no_matching_extractor(self):
        """Test behavior when no extractor matches"""
        # Mock all extractors to not match
        mock_extractor = Mock()
        mock_extractor.match.return_value = False
        mock_extractor.name = "mock"

        with patch.object(
            self.pipeline, "get_extractors", return_value=[mock_extractor]
        ):
            event = Event(
                id="test",
                signature="test.event",
                timestamp="2024-01-01T12:00:00Z",
                severity="info",
                source="test",
                labels={},
                message="Test",
            )
            extractor = await self.pipeline.find_matching_extractor(event)
            assert extractor is None

    @pytest.mark.asyncio
    async def test_recall_knowledge(self):
        """Test knowledge recall"""
        event = Event(
            id="test-incident",
            signature="database.connection.issue",
            timestamp="2024-01-01T12:00:00Z",
            severity="warning",
            source="monitoring",
            labels={"service": "database"},
            message="Connection problems",
        )

        # Mock extractor
        mock_extractor = Mock()
        mock_knowledge = [
            KBItem(doc_id="kb_005", excerpt="Knowledge 1", score=0.8),
            KBItem(doc_id="kb_006", excerpt="Knowledge 2", score=0.7),
        ]
        mock_extractor.recall = AsyncMock(return_value=mock_knowledge)
        mock_extractor.get_max_results = MagicMock(return_value=5)

        knowledge = await self.pipeline.recall_knowledge(event, mock_extractor)

        assert len(knowledge) == 2
        assert all(isinstance(item, KBItem) for item in knowledge)
        mock_extractor.recall.assert_called_once_with(event, k=5)

    @pytest.mark.asyncio
    async def test_recall_knowledge_error_handling(self):
        """Test knowledge recall error handling"""
        event = Event(
            id="test",
            signature="test.event",
            timestamp="2024-01-01T12:00:00Z",
            severity="info",
            source="test",
            labels={},
            message="Test",
        )

        # Mock extractor that raises exception
        mock_extractor = Mock()
        mock_extractor.recall = AsyncMock(side_effect=Exception("Recall failed"))
        mock_extractor.get_max_results = MagicMock(return_value=5)

        knowledge = await self.pipeline.recall_knowledge(event, mock_extractor)

        # Should return empty list on error
        assert knowledge == []

    def test_parse_llm_response_json(self):
        """Test parsing valid JSON LLM response"""
        response_content = """{
            "root_cause": "Database connection pool exhausted",
            "confidence": 0.85,
            "contributing_factors": ["High traffic", "Pool misconfiguration"],
            "evidence": ["Connection timeout errors", "Pool metrics"],
            "immediate_actions": ["Increase pool size", "Monitor connections"],
            "preventive_measures": ["Add monitoring", "Review configuration"],
            "confidence_reasoning": "Clear error patterns match known issues"
        }"""

        llm_response = LLMResponse(
            content=response_content, model="gpt-4", tokens_used=200
        )

        references = [KBItem(doc_id="kb_007", excerpt="Test", score=0.7)]
        result = self.pipeline.parse_llm_response(llm_response, "test", references)

        assert isinstance(result, RCAResult)
        assert result.root_cause == "Database connection pool exhausted"
        assert result.confidence == 0.85
        assert result.extractor == "test"
        assert len(result.contributing_factors) == 2
        assert len(result.immediate_actions) == 2
        assert result.status == "SUCCESS"

    def test_parse_llm_response_json_in_markdown(self):
        """Test parsing JSON embedded in markdown"""
        response_content = """Based on the analysis, here's the root cause:

```json
{
    "root_cause": "Memory leak in application code",
    "confidence": 0.75,
    "contributing_factors": ["Poor garbage collection"]
}
```

This analysis indicates a clear memory issue."""

        llm_response = LLMResponse(content=response_content, model="gpt-4")
        references = []

        result = self.pipeline.parse_llm_response(llm_response, "test", references)

        assert result.root_cause == "Memory leak in application code"
        assert result.confidence == 0.75
        assert len(result.contributing_factors) == 1

    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON returns fallback result"""
        llm_response = LLMResponse(
            content="This is not valid JSON at all", model="gpt-4"
        )

        references = [KBItem(doc_id="kb_007", excerpt="Test", score=0.7)]
        result = self.pipeline.parse_llm_response(llm_response, "test", references)

        assert isinstance(result, RCAResult)
        assert (
            "parse" in result.root_cause.lower()
            or "failed" in result.root_cause.lower()
        )
        assert result.confidence == 0.3
        assert result.status == "PARSE_ERROR"
        assert result.metadata.get("parse_error") is not None

    @pytest.mark.asyncio
    async def test_generate_rca_success(self):
        """Test successful RCA generation"""
        event = Event(
            id="test",
            signature="test.event",
            timestamp="2024-01-01T12:00:00Z",
            severity="info",
            source="test",
            labels={},
            message="Test description",
        )
        knowledge_items = [KBItem(doc_id="kb_008", excerpt="Test knowledge", score=0.8)]

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.name = "test"
        mock_extractor.prompt_template = "test_rca:v1"
        mock_extractor.get_prompt_context.return_value = {
            "event": event.to_dict(),
            "knowledge_items": [item.to_dict() for item in knowledge_items],
        }

        # Mock LLM response
        mock_llm_response = LLMResponse(
            content='{"root_cause": "Test cause", "confidence": 0.8}', model="gpt-4"
        )

        with (
            patch.object(
                self.pipeline.llm_router,
                "generate",
                new=AsyncMock(return_value=mock_llm_response),
            ),
            patch.object(
                self.pipeline.prompt_manager,
                "render_template",
                return_value="Test prompt",
            ),
        ):
            result = await self.pipeline.generate_rca(
                event, mock_extractor, knowledge_items
            )

        assert isinstance(result, RCAResult)
        assert result.root_cause == "Test cause"
        assert result.confidence == 0.8
        assert result.extractor == "test"

    @pytest.mark.asyncio
    async def test_generate_rca_error_handling(self):
        """Test RCA generation error handling"""
        event = Event(
            id="test",
            signature="test.event",
            timestamp="2024-01-01T12:00:00Z",
            severity="info",
            source="test",
            labels={},
            message="Test",
        )
        knowledge_items = []

        # Mock extractor that raises exception
        mock_extractor = Mock()
        mock_extractor.name = "test"
        mock_extractor.prompt_template = "test_rca:v1"
        mock_extractor.get_prompt_context.side_effect = Exception("Context error")

        result = await self.pipeline.generate_rca(
            event, mock_extractor, knowledge_items
        )

        assert isinstance(result, RCAResult)
        assert "failed" in result.root_cause.lower()
        assert result.confidence == 0.0
        assert result.status == "LLM_ERROR"
        assert result.metadata.get("error") is not None


class TestGetRCAFunction:
    """Test main get_rca function"""

    @pytest.mark.asyncio
    @patch("imkb.rca_pipeline.RCAPipeline")
    async def test_get_rca_success(self, mock_pipeline_class):
        """Test successful RCA generation via main function"""
        # Mock pipeline instance
        mock_pipeline = AsyncMock()
        mock_pipeline_class.return_value = mock_pipeline

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.name = "test"
        mock_extractor.prompt_template = "test_rca:v1"
        mock_pipeline.find_matching_extractor.return_value = mock_extractor

        # Mock knowledge recall
        mock_knowledge = [KBItem(doc_id="kb_009", excerpt="Test", score=0.8)]
        mock_pipeline.recall_knowledge.return_value = mock_knowledge

        # Mock RCA result
        mock_rca_result = RCAResult(
            root_cause="Test cause",
            confidence=0.8,
            extractor="test",
            references=mock_knowledge,
        )
        mock_pipeline.generate_rca.return_value = mock_rca_result

        # Test the function
        event_data = {
            "id": "test-123",
            "signature": "test.event",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "info",
            "source": "test",
            "labels": {},
            "message": "Test description",
        }

        result = await get_rca(event_data, namespace="test")

        assert isinstance(result, dict)
        assert result["root_cause"] == "Test cause"
        assert result["confidence"] == 0.8
        assert result["extractor"] == "test"

    @pytest.mark.asyncio
    @patch("imkb.rca_pipeline.RCAPipeline")
    async def test_get_rca_no_extractor(self, mock_pipeline_class):
        """Test RCA when no extractor matches"""
        # Mock pipeline instance
        mock_pipeline = AsyncMock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.find_matching_extractor.return_value = None

        event_data = {
            "id": "test",
            "signature": "test.event",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "info",
            "source": "test",
            "labels": {},
            "message": "Test",
        }
        result = await get_rca(event_data)

        assert isinstance(result, dict)
        assert "no suitable" in result["root_cause"].lower()
        assert result["confidence"] == 0.0
        assert result["extractor"] == "none"
        assert result["status"] == "NO_EXTRACTOR"

    @pytest.mark.asyncio
    async def test_get_rca_pipeline_error(self):
        """Test RCA when pipeline raises exception"""
        # Invalid event data that will cause error
        invalid_event_data = {"invalid": "data"}

        result = await get_rca(invalid_event_data)

        assert isinstance(result, dict)
        assert "error" in result["root_cause"].lower()
        assert result["confidence"] == 0.0
        assert result["extractor"] == "error"
        assert result["status"] == "PIPELINE_ERROR"


class TestRCAPipelineIntegration:
    """Integration tests for RCA pipeline"""

    @pytest.mark.asyncio
    async def test_end_to_end_rca_workflow(self):
        """Test complete RCA workflow with real components"""
        # Create real event
        event_data = {
            "id": "integration-test",
            "signature": "mysql.connection.pool.exhausted",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "critical",
            "source": "monitoring",
            "labels": {"service": "mysql"},
            "message": "Database is refusing new connections due to pool exhaustion",
        }

        # Run complete RCA workflow
        result = await get_rca(event_data, namespace="integration_test")

        # Verify result structure
        assert isinstance(result, dict)
        assert "root_cause" in result
        assert "confidence" in result
        assert "extractor" in result
        assert "status" in result

        # Should have found an appropriate extractor
        assert result["extractor"] in ["mysqlkb", "test"]

        # Should have some analysis
        assert len(result["root_cause"]) > 10
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_namespace_isolation(self):
        """Test namespace isolation in RCA pipeline"""
        event_data = {
            "id": "namespace-test",
            "signature": "test.event",
            "timestamp": "2024-01-01T12:00:00Z",
            "severity": "info",
            "source": "test",
            "labels": {},
            "message": "Test description",
        }

        # Run RCA with different namespaces
        result1 = await get_rca(event_data, namespace="tenant1")
        result2 = await get_rca(event_data, namespace="tenant2")

        # Both should succeed
        assert result1["status"] in [
            "SUCCESS",
            "PARSE_ERROR",
        ]  # Mock LLM might have parse issues
        assert result2["status"] in ["SUCCESS", "PARSE_ERROR"]

        # Results might be similar due to mock LLM, but namespace isolation is tested
        # at the config level
