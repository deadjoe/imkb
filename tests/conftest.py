"""
Pytest configuration and shared fixtures for imkb tests

Provides common fixtures for testing configuration, mock objects, and test data.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
from unittest.mock import AsyncMock, MagicMock

from imkb.config import ImkbConfig
from imkb.extractors import Event, KBItem
from imkb.rca_pipeline import RCAResult
from imkb.action_pipeline import ActionResult


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide a test configuration with safe defaults"""
    config = ImkbConfig()
    config.namespace = "test"
    
    # Ensure we use mock LLM for testing
    config.llm.routers["default"].provider = "mock"
    config.llm.routers["default"].api_key = "test-key"
    
    return config


@pytest.fixture
def sample_event():
    """Provide a sample event for testing"""
    return Event(
        id="test-incident-001",
        title="MySQL Connection Pool Exhausted",
        description="Database connection pool has reached maximum capacity",
        severity="critical",
        source="monitoring",
        metadata={
            "database": "production_db",
            "max_connections": "150",
            "current_connections": "150"
        }
    )


@pytest.fixture
def sample_event_data():
    """Provide sample event data as dictionary"""
    return {
        "id": "test-incident-001", 
        "title": "MySQL Connection Pool Exhausted",
        "description": "Database connection pool has reached maximum capacity",
        "severity": "critical",
        "source": "monitoring",
        "metadata": {
            "database": "production_db",
            "max_connections": "150"
        }
    }


@pytest.fixture
def sample_kb_items():
    """Provide sample knowledge base items for testing"""
    return [
        KBItem(
            excerpt="MySQL connection pool exhaustion is often caused by application connection leaks",
            source="mysql_kb",
            confidence=0.9,
            metadata={"category": "database", "severity": "high"}
        ),
        KBItem(
            excerpt="Connection pool sizing should account for peak concurrent load",
            source="mysql_kb", 
            confidence=0.85,
            metadata={"category": "configuration", "severity": "medium"}
        ),
        KBItem(
            excerpt="SHOW PROCESSLIST can help identify blocking connections",
            source="mysql_kb",
            confidence=0.8,
            metadata={"category": "diagnostic", "severity": "low"}
        )
    ]


@pytest.fixture
def sample_rca_result(sample_kb_items):
    """Provide a sample RCA result for testing"""
    return RCAResult(
        root_cause="Database connection pool exhausted due to application connection leaks",
        confidence=0.85,
        extractor="mysqlkb", 
        references=sample_kb_items,
        status="SUCCESS",
        contributing_factors=[
            "High concurrent application load",
            "Connection pool sized too small",
            "Application not properly closing connections"
        ],
        evidence=[
            "Connection count reached max_connections limit",
            "Multiple connection timeout errors in logs",
            "SHOW PROCESSLIST shows many sleeping connections"
        ],
        immediate_actions=[
            "Increase max_connections parameter temporarily",
            "Identify and kill long-running idle connections",
            "Monitor connection usage patterns"
        ],
        preventive_measures=[
            "Implement connection pooling best practices",
            "Add connection monitoring and alerting",
            "Review application connection lifecycle management"
        ],
        confidence_reasoning="Strong evidence from connection metrics and error patterns",
        knowledge_gaps=[
            "Exact source of connection leaks in application code",
            "Historical connection usage trends"
        ]
    )


@pytest.fixture
def sample_rca_data(sample_rca_result):
    """Provide sample RCA data as dictionary"""
    return sample_rca_result.to_dict()


@pytest.fixture
def sample_action_result():
    """Provide a sample action result for testing"""
    return ActionResult(
        actions=[
            "Execute SHOW PROCESSLIST to identify active connections",
            "Increase max_connections parameter from 150 to 300",
            "Identify and terminate problematic connections using KILL command",
            "Review application connection pool configuration",
            "Implement connection monitoring and alerting"
        ],
        playbook="""1. Immediate Assessment: Run 'SHOW PROCESSLIST' and 'SHOW STATUS LIKE "Threads_connected"'
2. Emergency Relief: Increase max_connections with 'SET GLOBAL max_connections = 300'
3. Connection Cleanup: Identify problematic connections and terminate using 'KILL <connection_id>'
4. Application Review: Check application connection pools and increase timeout settings
5. Monitoring Setup: Implement alerts for connection usage above 80% threshold
6. Validation: Verify new connections can be established and application functionality restored""",
        priority="high",
        estimated_time="30 minutes",
        risk_level="medium",
        prerequisites=[
            "MySQL administrative access",
            "Application deployment pipeline access",
            "Monitoring system configuration access"
        ],
        validation_steps=[
            "Verify new database connections can be established",
            "Check application error logs for connection failures", 
            "Monitor connection count remains below new threshold",
            "Confirm all application services are operational"
        ],
        rollback_plan="If issues arise, revert max_connections to original value (150) and restart MySQL service if necessary",
        automation_potential="semi-automated",
        confidence=0.8
    )


@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client for testing"""
    client = AsyncMock()
    
    # Configure default responses
    client.generate.return_value = MagicMock(
        content='{"root_cause": "Mock analysis", "confidence": 0.8}',
        model="mock-gpt-4",
        tokens_used=100,
        finish_reason="stop",
        metadata={"mock": True}
    )
    client.health_check.return_value = True
    
    return client


@pytest.fixture
def mock_mem0_adapter():
    """Provide a mock Mem0 adapter for testing"""
    adapter = AsyncMock()
    
    # Configure default responses
    adapter.search.return_value = []
    adapter.add_memory.return_value = None
    adapter.initialize.return_value = None
    
    return adapter


@pytest.fixture
def mock_extractor():
    """Provide a mock extractor for testing"""
    extractor = AsyncMock()
    extractor.name = "mock_extractor"
    extractor.prompt_template = "mock_rca:v1"
    extractor.match.return_value = True
    extractor.recall.return_value = []
    extractor.get_max_results.return_value = 5
    extractor.get_prompt_context.return_value = {
        "event": {"id": "test", "title": "Test"},
        "knowledge_items": []
    }
    
    return extractor


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_config_file(temp_dir):
    """Provide a temporary config file for testing"""
    config_file = temp_dir / "test_config.yml"
    config_content = """
namespace: "test_environment"
llm:
  default: "test_router"
  routers:
    test_router:
      provider: "mock"
      model: "test-model"
      api_key: "test-key"
extractors:
  enabled: ["test", "mysqlkb"]
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def temp_event_file(temp_dir, sample_event_data):
    """Provide a temporary event file for CLI testing"""
    import json
    
    event_file = temp_dir / "test_event.json"
    with open(event_file, 'w') as f:
        json.dump(sample_event_data, f)
    
    return event_file


@pytest.fixture
def temp_rca_file(temp_dir, sample_rca_data):
    """Provide a temporary RCA file for CLI testing"""
    import json
    
    rca_file = temp_dir / "test_rca.json"
    with open(rca_file, 'w') as f:
        json.dump(sample_rca_data, f)
    
    return rca_file


# Pytest async support configuration
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the event loop policy for the test session"""
    return asyncio.DefaultEventLoopPolicy()


# Custom pytest markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests across multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than a few seconds"
    )
    config.addinivalue_line(
        "markers", "external: Tests that require external services"
    )


# Skip markers for conditional test execution
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests (async tests are often slower)
        if hasattr(item.function, "__code__") and "async" in str(item.function):
            if not any(m.name == "slow" for m in item.iter_markers()):
                # Only add slow marker if test might be actually slow
                if any(keyword in item.name for keyword in ["pipeline", "workflow", "end_to_end"]):
                    item.add_marker(pytest.mark.slow)


# Test data validation
@pytest.fixture(autouse=True)
def validate_test_environment():
    """Validate test environment setup"""
    # This fixture runs automatically for all tests
    # Can be used to ensure test environment is properly configured
    pass