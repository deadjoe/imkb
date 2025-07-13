"""
Observability module for imkb

Provides OpenTelemetry tracing and Prometheus metrics collection for
monitoring and debugging imkb operations.
"""

from .config import TelemetryConfig
from .tracer import get_tracer, trace_operation, trace_async
from .metrics import get_metrics, MetricsCollector
from .init import initialize_observability, shutdown_observability, is_observability_initialized

__all__ = [
    "TelemetryConfig",
    "get_tracer", 
    "trace_operation",
    "trace_async",
    "get_metrics",
    "MetricsCollector",
    "initialize_observability",
    "shutdown_observability",
    "is_observability_initialized"
]