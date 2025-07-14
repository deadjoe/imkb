"""
Observability module for imkb

Provides OpenTelemetry tracing and Prometheus metrics collection for
monitoring and debugging imkb operations.
"""

from .config import TelemetryConfig
from .init import (
    initialize_observability,
    is_observability_initialized,
    shutdown_observability,
)
from .metrics import MetricsCollector, get_metrics
from .tracer import get_tracer, trace_async, trace_operation

__all__ = [
    "TelemetryConfig",
    "get_tracer",
    "trace_operation",
    "trace_async",
    "get_metrics",
    "MetricsCollector",
    "initialize_observability",
    "shutdown_observability",
    "is_observability_initialized",
]
