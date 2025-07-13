#!/usr/bin/env python3
"""
MVP Test Script for imkb Phase 1

Tests the core functionality with mock data to validate the implementation.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from imkb import get_rca
from imkb.config import ImkbConfig

async def test_basic_rca():
    """Test basic RCA functionality"""
    print("🔍 Testing basic RCA pipeline...")
    
    # Sample event data (like examples/event.example.json)
    event_data = {
        "id": "test-alert-001",
        "signature": "mysql_connection_pool_exhausted",
        "context_hash": "sha256:test123...",
        "timestamp": "2025-01-15T14:30:00Z",
        "severity": "P1",
        "source": "test_prometheus",
        "labels": {
            "cluster": "test-cluster",
            "service": "mysql-primary",
            "environment": "development"
        },
        "message": "MySQL connection pool exhausted: 150/150 connections active",
        "raw": {
            "alertname": "MySQLConnectionPoolExhausted",
            "summary": "Test MySQL connection pool issue"
        },
        "embedding_version": "v1.0"
    }
    
    try:
        # Call the main RCA function
        result = await get_rca(event_data, namespace="test-dev")
        
        print("✅ RCA completed successfully!")
        print(f"   Root Cause: {result['root_cause'][:100]}...")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Extractor: {result['extractor']}")
        print(f"   Status: {result['status']}")
        print(f"   References: {len(result['references'])} knowledge items")
        
        if result['immediate_actions']:
            print(f"   Immediate Actions: {len(result['immediate_actions'])} suggested")
        
        return True
        
    except Exception as e:
        print(f"❌ RCA failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_config_loading():
    """Test configuration loading"""
    print("⚙️ Testing configuration loading...")
    
    try:
        config = ImkbConfig.load_from_file("imkb.yml")
        print(f"✅ Config loaded: LLM={config.llm.default}, Namespace={config.namespace}")
        
        # Test Mem0 config generation
        mem0_config = config.get_mem0_config()
        print(f"   Mem0 Vector Store: {mem0_config['vector_store']['provider']}")
        print(f"   Mem0 Graph Store: {mem0_config['graph_store']['provider']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

async def test_extractor_registry():
    """Test extractor registry"""
    print("🔌 Testing extractor registry...")
    
    try:
        from imkb.extractors import registry
        
        available = registry.get_available_extractors()
        print(f"✅ Available extractors: {available}")
        
        # Test creating test extractor
        config = ImkbConfig.load_from_file("imkb.yml")
        test_extractor = registry.create_extractor("test", config)
        
        if test_extractor:
            print(f"   Test extractor created: {test_extractor.name}")
            print(f"   Prompt template: {test_extractor.prompt_template}")
            return True
        else:
            print("❌ Failed to create test extractor")
            return False
            
    except Exception as e:
        print(f"❌ Extractor registry test failed: {e}")
        return False

async def test_llm_client():
    """Test LLM client with mock provider"""
    print("🤖 Testing LLM client...")
    
    try:
        from imkb.llm_client import LLMRouter
        from imkb.config import ImkbConfig
        
        config = ImkbConfig.load_from_file("imkb.yml")
        router = LLMRouter(config)
        
        # Test mock LLM generation
        response = await router.generate(
            prompt="Test prompt for RCA analysis",
            template_type="rca"
        )
        
        print(f"✅ LLM response generated")
        print(f"   Model: {response.model}")
        print(f"   Content length: {len(response.content)} chars")
        print(f"   Tokens used: {response.tokens_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM client test failed: {e}")
        return False

async def test_infrastructure():
    """Test infrastructure connectivity"""
    print("🏗️ Testing infrastructure connectivity...")
    
    try:
        # Test Qdrant connectivity
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:6333/")
            if response.status_code == 200:
                print("✅ Qdrant is accessible")
            else:
                print(f"⚠️ Qdrant returned status {response.status_code}")
        
        # Test Neo4j connectivity (basic)
        response = await client.get("http://localhost:7474/browser/")
        if response.status_code == 200:
            print("✅ Neo4j browser is accessible")
        else:
            print(f"⚠️ Neo4j returned status {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Infrastructure test had issues: {e}")
        return False

async def main():
    """Run all MVP tests"""
    print("🚀 Starting imkb MVP Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Infrastructure Connectivity", test_infrastructure),
        ("Extractor Registry", test_extractor_registry),
        ("LLM Client", test_llm_client),
        ("Basic RCA Pipeline", test_basic_rca),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        success = await test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! MVP is ready for development.")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)