"""
Tests for namespace isolation using contextvars

This test suite verifies that the new context-based namespace management
works correctly and provides proper isolation between different tenants.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from imkb.config import get_config
from imkb.context import NamespaceContext, get_namespace, set_namespace


class TestNamespaceContext:
    """Test namespace context management"""

    def test_default_namespace(self):
        """Test that default namespace is 'default'"""
        assert get_namespace() == "default"

    def test_set_namespace(self):
        """Test setting namespace"""
        token = set_namespace("test_namespace")
        assert get_namespace() == "test_namespace"

        # Reset using token
        from imkb.context import reset_namespace
        reset_namespace(token)
        assert get_namespace() == "default"

    def test_namespace_context_manager(self):
        """Test namespace context manager"""
        assert get_namespace() == "default"

        with NamespaceContext("tenant1"):
            assert get_namespace() == "tenant1"

            with NamespaceContext("tenant2"):
                assert get_namespace() == "tenant2"

            # Should restore to tenant1
            assert get_namespace() == "tenant1"

        # Should restore to default
        assert get_namespace() == "default"

    def test_nested_contexts(self):
        """Test nested namespace contexts"""
        with NamespaceContext("outer"):
            assert get_namespace() == "outer"

            with NamespaceContext("inner"):
                assert get_namespace() == "inner"

            assert get_namespace() == "outer"

    def test_config_get_current_namespace(self):
        """Test that config.get_current_namespace() uses context"""
        config = get_config()

        # Default should be config's namespace
        assert config.get_current_namespace() == "default"

        # Context should override config
        with NamespaceContext("context_override"):
            assert config.get_current_namespace() == "context_override"

        # Should return to default
        assert config.get_current_namespace() == "default"


class TestConcurrentNamespaceIsolation:
    """Test namespace isolation in concurrent scenarios"""

    def test_concurrent_namespace_isolation(self):
        """Test that namespace isolation works in concurrent threads"""
        results = {}

        def worker(namespace, worker_id):
            """Worker function that sets namespace and returns it"""
            with NamespaceContext(namespace):
                config = get_config()
                # Simulate some work
                import time
                time.sleep(0.01)
                results[worker_id] = config.get_current_namespace()

        # Run multiple workers with different namespaces concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(10):
                namespace = f"tenant_{i}"
                future = executor.submit(worker, namespace, i)
                futures.append(future)

            # Wait for all workers to complete
            for future in futures:
                future.result()

        # Verify each worker got its own namespace
        for i in range(10):
            assert results[i] == f"tenant_{i}"

    @pytest.mark.asyncio
    async def test_asyncio_namespace_isolation(self):
        """Test namespace isolation in asyncio tasks"""
        results = {}

        async def async_worker(namespace, worker_id):
            """Async worker function"""
            with NamespaceContext(namespace):
                config = get_config()
                # Simulate async work
                await asyncio.sleep(0.01)
                results[worker_id] = config.get_current_namespace()

        # Run multiple async workers concurrently
        tasks = []
        for i in range(10):
            namespace = f"async_tenant_{i}"
            task = asyncio.create_task(async_worker(namespace, i))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Verify each task got its own namespace
        for i in range(10):
            assert results[i] == f"async_tenant_{i}"

    def test_namespace_isolation_across_tasks(self):
        """Test that namespace doesn't leak between different tasks"""

        def task1():
            with NamespaceContext("task1_namespace"):
                config = get_config()
                return config.get_current_namespace()

        def task2():
            with NamespaceContext("task2_namespace"):
                config = get_config()
                return config.get_current_namespace()

        # Each task should get its own namespace
        assert task1() == "task1_namespace"
        assert task2() == "task2_namespace"

        # Default should be restored
        assert get_namespace() == "default"


class TestNamespaceErrorHandling:
    """Test error handling in namespace contexts"""

    def test_exception_in_context(self):
        """Test that namespace is properly restored after exception"""
        assert get_namespace() == "default"

        try:
            with NamespaceContext("error_test"):
                assert get_namespace() == "error_test"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should restore to default even after exception
        assert get_namespace() == "default"

    def test_multiple_exceptions(self):
        """Test namespace restoration with nested exceptions"""
        assert get_namespace() == "default"

        try:
            with NamespaceContext("outer_error"):
                try:
                    with NamespaceContext("inner_error"):
                        raise ValueError("Inner exception")
                except ValueError:
                    assert get_namespace() == "outer_error"
                    raise RuntimeError("Outer exception")
        except RuntimeError:
            pass

        # Should restore to default
        assert get_namespace() == "default"
