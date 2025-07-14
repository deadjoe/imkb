"""
Test suite for Action Pipeline

Tests ActionResult, ActionPipeline, and main gen_playbook function.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imkb.action_pipeline import ActionPipeline, ActionResult, gen_playbook
from imkb.config import ImkbConfig
from imkb.extractors import KBItem
from imkb.llm_client import LLMResponse
from imkb.rca_pipeline import RCAResult


class TestActionResult:
    """Test ActionResult class"""

    def test_basic_action_result(self):
        """Test basic ActionResult creation"""
        result = ActionResult(
            actions=["Restart service", "Check logs"],
            playbook="1. Restart the service 2. Check error logs for issues",
            priority="high",
            risk_level="medium",
        )

        assert result.actions == ["Restart service", "Check logs"]
        assert (
            result.playbook == "1. Restart the service 2. Check error logs for issues"
        )
        assert result.priority == "high"
        assert result.risk_level == "medium"
        assert result.estimated_time is None
        assert result.prerequisites == []
        assert result.validation_steps == []
        assert result.confidence == 0.8  # default

    def test_action_result_with_all_fields(self):
        """Test ActionResult with all fields"""
        result = ActionResult(
            actions=["Action 1", "Action 2"],
            playbook="Detailed playbook steps",
            priority="medium",
            estimated_time="30 minutes",
            risk_level="low",
            prerequisites=["Admin access", "Backup completed"],
            validation_steps=["Check service status", "Verify functionality"],
            rollback_plan="Revert to previous configuration",
            automation_potential="semi-automated",
            confidence=0.9,
            metadata={"source": "test"},
        )

        assert result.estimated_time == "30 minutes"
        assert len(result.prerequisites) == 2
        assert len(result.validation_steps) == 2
        assert result.rollback_plan == "Revert to previous configuration"
        assert result.automation_potential == "semi-automated"
        assert result.confidence == 0.9
        assert result.metadata["source"] == "test"

    def test_action_result_to_dict(self):
        """Test ActionResult to_dict conversion"""
        result = ActionResult(
            actions=["Test action"],
            playbook="Test playbook",
            priority="high",
            risk_level="low",
            prerequisites=["Prerequisite 1"],
            validation_steps=["Validation 1"],
        )

        result_dict = result.to_dict()

        assert result_dict["actions"] == ["Test action"]
        assert result_dict["playbook"] == "Test playbook"
        assert result_dict["priority"] == "high"
        assert result_dict["risk_level"] == "low"
        assert result_dict["prerequisites"] == ["Prerequisite 1"]
        assert result_dict["validation_steps"] == ["Validation 1"]
        assert result_dict["confidence"] == 0.8
        assert "metadata" in result_dict

    def test_action_result_from_dict(self):
        """Test ActionResult from_dict creation"""
        data = {
            "actions": ["Action A", "Action B"],
            "playbook": "Step-by-step playbook",
            "priority": "medium",
            "estimated_time": "1 hour",
            "risk_level": "high",
            "prerequisites": ["Access required"],
            "validation_steps": ["Check status"],
            "rollback_plan": "Rollback procedure",
            "automation_potential": "manual",
            "confidence": 0.75,
            "metadata": {"test": "value"},
        }

        result = ActionResult.from_dict(data)

        assert result.actions == ["Action A", "Action B"]
        assert result.playbook == "Step-by-step playbook"
        assert result.priority == "medium"
        assert result.estimated_time == "1 hour"
        assert result.risk_level == "high"
        assert result.prerequisites == ["Access required"]
        assert result.validation_steps == ["Check status"]
        assert result.rollback_plan == "Rollback procedure"
        assert result.automation_potential == "manual"
        assert result.confidence == 0.75
        assert result.metadata["test"] == "value"


class TestActionPipeline:
    """Test ActionPipeline class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = ImkbConfig()
        self.pipeline = ActionPipeline(self.config)

        # Create sample RCA result
        self.rca_result = RCAResult(
            root_cause="Database connection pool exhausted",
            confidence=0.85,
            extractor="mysqlkb",
            references=[
                KBItem(doc_id="kb-1", excerpt="Connection pool issues", score=0.9)
            ],
            immediate_actions=["Increase pool size", "Monitor connections"],
            preventive_measures=["Add alerts", "Review configuration"],
            contributing_factors=["High traffic", "Misconfiguration"],
        )

    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.config == self.config
        assert self.pipeline.llm_router is not None
        assert self.pipeline.mem0_adapter is not None

    @pytest.mark.asyncio
    async def test_search_similar_actions(self):
        """Test searching for similar past actions"""
        with patch.object(self.pipeline.mem0_adapter, "search") as mock_search:
            mock_search.return_value = [
                KBItem(
                    doc_id="kb-2",
                    excerpt="Previous action for similar issue",
                    score=0.8,
                )
            ]

            # Add metadata to the mock object for testing
            mock_search.return_value[0].metadata["source"] = "actions"

            similar_actions = await self.pipeline.search_similar_actions(
                self.rca_result, limit=3
            )

            assert len(similar_actions) == 1
            assert similar_actions[0].metadata.get("source") == "actions"
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar_actions_error_handling(self):
        """Test error handling in similar actions search"""
        with patch.object(self.pipeline.mem0_adapter, "search") as mock_search:
            mock_search.side_effect = Exception("Search failed")

            similar_actions = await self.pipeline.search_similar_actions(
                self.rca_result
            )

            # Should return empty list on error
            assert similar_actions == []

    def test_build_action_prompt_context(self):
        """Test building prompt context for action generation"""
        similar_actions = [
            KBItem(doc_id="kb-3", excerpt="Previous remediation", score=0.9)
        ]

        context = self.pipeline._build_action_prompt_context(
            self.rca_result, similar_actions
        )

        assert "rca_result" in context
        assert "root_cause" in context
        assert "confidence" in context
        assert "extractor" in context
        assert "similar_actions" in context
        assert "has_similar_actions" in context

        assert context["root_cause"] == "Database connection pool exhausted"
        assert context["confidence"] == 0.85
        assert context["extractor"] == "mysqlkb"
        assert context["has_similar_actions"] is True
        assert len(context["similar_actions"]) == 1

    def test_render_action_prompt(self):
        """Test action prompt rendering"""
        context = {
            "root_cause": "Memory leak detected",
            "confidence": 0.8,
            "extractor": "memory_analyzer",
            "immediate_actions": ["Restart service", "Collect heap dump"],
            "preventive_measures": ["Add memory monitoring", "Code review"],
            "contributing_factors": ["High memory usage", "Memory leaks"],
            "similar_actions": [{"excerpt": "Previous memory fix"}],
            "has_similar_actions": True,
        }

        prompt = self.pipeline._render_action_prompt(context)

        assert "Memory leak detected" in prompt
        assert "0.8" in prompt
        assert "memory_analyzer" in prompt
        assert "Restart service" in prompt
        assert "Add memory monitoring" in prompt
        assert "High memory usage" in prompt
        assert "Previous memory fix" in prompt
        assert "JSON format" in prompt

    def test_parse_action_response_valid_json(self):
        """Test parsing valid JSON action response"""
        response_content = """{
            "actions": [
                "Increase MySQL max_connections parameter",
                "Monitor connection pool usage",
                "Restart MySQL service if needed"
            ],
            "playbook": "1. Check current connections 2. Increase max_connections 3. Monitor usage",
            "priority": "high",
            "estimated_time": "15 minutes",
            "risk_level": "medium",
            "prerequisites": ["MySQL admin access"],
            "validation_steps": ["Verify connections", "Check application health"],
            "rollback_plan": "Revert max_connections to original value",
            "automation_potential": "semi-automated"
        }"""

        llm_response = LLMResponse(
            content=response_content, model="gpt-4", tokens_used=300
        )

        result = self.pipeline._parse_action_response(llm_response, self.rca_result)

        assert isinstance(result, ActionResult)
        assert len(result.actions) == 3
        assert "Increase MySQL max_connections" in result.actions[0]
        assert result.priority == "high"
        assert result.estimated_time == "15 minutes"
        assert result.risk_level == "medium"
        assert len(result.prerequisites) == 1
        assert len(result.validation_steps) == 2
        assert result.rollback_plan == "Revert max_connections to original value"
        assert result.automation_potential == "semi-automated"

    def test_parse_action_response_json_in_markdown(self):
        """Test parsing JSON embedded in markdown"""
        response_content = """Here's the recommended action plan:

```json
{
    "actions": ["Fix the issue", "Monitor results"],
    "playbook": "Detailed steps to resolve the problem",
    "priority": "medium",
    "risk_level": "low"
}
```

This should resolve the issue."""

        llm_response = LLMResponse(content=response_content, model="gpt-4")
        result = self.pipeline._parse_action_response(llm_response, self.rca_result)

        assert len(result.actions) == 2
        assert result.actions[0] == "Fix the issue"
        assert result.priority == "medium"
        assert result.risk_level == "low"

    def test_parse_action_response_invalid_json(self):
        """Test parsing invalid JSON returns fallback result"""
        llm_response = LLMResponse(
            content="This is not valid JSON response at all", model="gpt-4"
        )

        result = self.pipeline._parse_action_response(llm_response, self.rca_result)

        assert isinstance(result, ActionResult)
        # Should use immediate actions from RCA as fallback
        assert result.actions == self.rca_result.immediate_actions
        assert "failed" in result.playbook.lower()
        assert result.confidence == 0.3
        assert result.metadata.get("parse_error") is not None

    @pytest.mark.asyncio
    async def test_generate_actions_success(self):
        """Test successful action generation"""
        # Mock similar actions search
        mock_similar_actions = [
            KBItem(doc_id="kb-4", excerpt="Previous DB fix", score=0.9)
        ]

        # Mock LLM response
        mock_llm_response = LLMResponse(
            content='{"actions": ["Action 1", "Action 2"], "playbook": "Test playbook", "priority": "high"}',
            model="gpt-4",
        )

        with patch.object(
            self.pipeline, "search_similar_actions", return_value=mock_similar_actions
        ), patch.object(
            self.pipeline.llm_router, "generate", return_value=mock_llm_response
        ), patch.object(self.pipeline, "_store_action_plan") as mock_store:
            result = await self.pipeline.generate_actions(self.rca_result)

        assert isinstance(result, ActionResult)
        assert len(result.actions) == 2
        assert result.playbook == "Test playbook"
        assert result.priority == "high"
        mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_actions_error_handling(self):
        """Test action generation error handling"""
        # Mock search to raise exception
        with patch.object(
            self.pipeline,
            "search_similar_actions",
            side_effect=Exception("Search error"),
        ):
            result = await self.pipeline.generate_actions(self.rca_result)

        assert isinstance(result, ActionResult)
        # Should use RCA immediate actions as fallback
        assert result.actions == self.rca_result.immediate_actions
        assert "error" in result.playbook.lower()
        assert result.confidence == 0.2
        assert result.metadata.get("error") is not None
        assert result.metadata.get("fallback") is True

    @pytest.mark.asyncio
    async def test_store_action_plan(self):
        """Test storing successful action plan"""
        action_result = ActionResult(
            actions=["Test action"],
            playbook="Test playbook",
            priority="medium",
            risk_level="low",
            confidence=0.8,
        )

        with patch.object(self.pipeline.mem0_adapter, "add_memory") as mock_add:
            await self.pipeline._store_action_plan(self.rca_result, action_result)

            mock_add.assert_called_once()
            call_args = mock_add.call_args

            # Verify memory content
            assert "Database connection pool exhausted" in call_args[1]["content"]
            assert call_args[1]["user_id"].endswith("_actions_mysqlkb")

            # Verify metadata
            metadata = call_args[1]["metadata"]
            assert metadata["type"] == "action_plan"
            assert metadata["extractor"] == "mysqlkb"
            assert metadata["priority"] == "medium"
            assert metadata["risk_level"] == "low"

    @pytest.mark.asyncio
    async def test_store_action_plan_error_handling(self):
        """Test action plan storage error handling"""
        action_result = ActionResult(actions=["Test"], playbook="Test")

        with patch.object(
            self.pipeline.mem0_adapter,
            "add_memory",
            side_effect=Exception("Storage error"),
        ):
            # Should not raise exception, just log warning
            await self.pipeline._store_action_plan(self.rca_result, action_result)


class TestGenPlaybookFunction:
    """Test main gen_playbook function"""

    @pytest.mark.asyncio
    async def test_gen_playbook_success(self):
        """Test successful playbook generation via main function"""
        rca_data = {
            "root_cause": "Network timeout issue",
            "confidence": 0.8,
            "extractor": "network",
            "references": [
                {
                    "doc_id": "kb-1",
                    "excerpt": "Network issue",
                    "score": 0.8,
                    "metadata": {},
                }
            ],
            "immediate_actions": ["Check network", "Restart service"],
            "preventive_measures": ["Add monitoring"],
        }

        with patch("imkb.action_pipeline.ActionPipeline") as mock_pipeline_class:
            mock_pipeline = AsyncMock()
            mock_pipeline_class.return_value = mock_pipeline

            mock_action_result = ActionResult(
                actions=["Fix network", "Monitor"],
                playbook="Network fix playbook",
                priority="high",
            )
            mock_pipeline.generate_actions.return_value = mock_action_result

            result = await gen_playbook(rca_data, namespace="test")

        assert isinstance(result, dict)
        assert result["actions"] == ["Fix network", "Monitor"]
        assert result["playbook"] == "Network fix playbook"
        assert result["priority"] == "high"

    @pytest.mark.asyncio
    async def test_gen_playbook_error(self):
        """Test playbook generation error handling"""
        # Invalid RCA data that will cause error
        invalid_rca_data = {"invalid": "data"}

        result = await gen_playbook(invalid_rca_data)

        assert isinstance(result, dict)
        assert "error" in result["playbook"].lower()
        assert result["confidence"] == 0.1
        assert result["metadata"].get("error") is not None

    @pytest.mark.asyncio
    async def test_gen_playbook_with_namespace(self):
        """Test playbook generation with custom namespace"""
        rca_data = {
            "root_cause": "Test cause",
            "confidence": 0.7,
            "extractor": "test",
            "references": [],
            "immediate_actions": ["Test action"],
        }

        with patch("imkb.action_pipeline.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            with patch("imkb.action_pipeline.ActionPipeline") as mock_pipeline_class:
                mock_pipeline = AsyncMock()
                mock_pipeline_class.return_value = mock_pipeline
                mock_pipeline.generate_actions.return_value = ActionResult(
                    actions=["Test"], playbook="Test"
                )

                await gen_playbook(rca_data, namespace="custom_tenant")

        # Verify namespace was set on config
        assert mock_config.namespace == "custom_tenant"


class TestActionPipelineIntegration:
    """Integration tests for action pipeline"""

    @pytest.mark.asyncio
    async def test_end_to_end_action_workflow(self):
        """Test complete action generation workflow with real components"""
        # Create RCA result
        rca_data = {
            "root_cause": "MySQL connection pool exhausted due to high traffic",
            "confidence": 0.85,
            "extractor": "mysqlkb",
            "references": [
                {
                    "excerpt": "Connection pool issues are common",
                    "source": "mysql_kb",
                    "confidence": 0.9,
                    "metadata": {},
                }
            ],
            "immediate_actions": ["Increase max_connections", "Monitor pool usage"],
            "preventive_measures": [
                "Add connection monitoring",
                "Review pool configuration",
            ],
            "contributing_factors": ["High concurrent load", "Undersized pool"],
        }

        # Run complete action workflow
        result = await gen_playbook(rca_data, namespace="integration_test")

        # Verify result structure
        assert isinstance(result, dict)
        assert "actions" in result
        assert "playbook" in result
        assert "priority" in result
        assert "risk_level" in result

        # Should have generated meaningful actions
        assert len(result["actions"]) > 0
        assert len(result["playbook"]) > 20  # Should be substantial

        # Priority should be reasonable
        assert result["priority"] in ["low", "medium", "high"]
        assert result["risk_level"] in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_action_generation_with_different_extractors(self):
        """Test action generation for different extractor types"""
        # Test with different RCA results from different extractors
        test_cases = [
            {
                "root_cause": "MySQL connection issue",
                "extractor": "mysqlkb",
                "immediate_actions": ["Check DB connections"],
            },
            {
                "root_cause": "Generic system error",
                "extractor": "test",
                "immediate_actions": ["Generic troubleshooting"],
            },
        ]

        for case in test_cases:
            rca_data = {
                "root_cause": case["root_cause"],
                "confidence": 0.8,
                "extractor": case["extractor"],
                "references": [],
                "immediate_actions": case["immediate_actions"],
            }

            result = await gen_playbook(rca_data)

            # Should generate valid actions for any extractor
            assert isinstance(result, dict)
            assert len(result["actions"]) > 0
            assert len(result["playbook"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_action_generation(self):
        """Test handling multiple concurrent action generations"""
        rca_data = {
            "root_cause": "Test concurrent issue",
            "confidence": 0.8,
            "extractor": "test",
            "references": [],
            "immediate_actions": ["Test action"],
        }

        # Generate multiple concurrent requests
        import asyncio

        tasks = []
        for i in range(3):
            task = gen_playbook(rca_data, namespace=f"concurrent_test_{i}")
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert "actions" in result
            assert "playbook" in result
