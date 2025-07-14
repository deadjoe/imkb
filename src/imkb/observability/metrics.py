"""
Prometheus metrics collection for imkb

Provides comprehensive metrics collection for monitoring imkb operations,
performance, and business metrics.
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)

from .config import TelemetryConfig

logger = logging.getLogger(__name__)


@dataclass
class MetricsCollector:
    """
    Central metrics collector for imkb operations

    Provides high-level metrics for RCA operations, LLM usage,
    extractor performance, and system health.
    """

    config: TelemetryConfig
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)
    _metrics_server: Optional[threading.Thread] = field(default=None, init=False)

    # Core operation metrics
    rca_requests_total: Counter = field(init=False)
    rca_duration: Histogram = field(init=False)
    rca_success_total: Counter = field(init=False)
    rca_errors_total: Counter = field(init=False)

    # Action pipeline metrics
    action_requests_total: Counter = field(init=False)
    action_duration: Histogram = field(init=False)
    action_success_total: Counter = field(init=False)

    # LLM metrics
    llm_requests_total: Counter = field(init=False)
    llm_duration: Histogram = field(init=False)
    llm_tokens_total: Counter = field(init=False)
    llm_errors_total: Counter = field(init=False)

    # Extractor metrics
    extractor_requests_total: Counter = field(init=False)
    extractor_duration: Histogram = field(init=False)
    extractor_matches_total: Counter = field(init=False)
    extractor_knowledge_items: Histogram = field(init=False)

    # Memory system metrics
    memory_operations_total: Counter = field(init=False)
    memory_duration: Histogram = field(init=False)
    memory_items_total: Counter = field(init=False)

    # System health metrics
    active_operations: Gauge = field(init=False)
    system_info: Info = field(init=False)

    def __post_init__(self):
        """Initialize all metrics after dataclass creation"""
        if not self.config.enabled or not self.config.metrics.enabled:
            logger.info("Metrics collection is disabled")
            return

        self._initialize_metrics()

        if self.config.should_start_metrics_server():
            self._start_metrics_server()

    def _initialize_metrics(self):
        """Initialize all Prometheus metrics"""
        labels = list(self.config.metrics.default_labels.keys())
        buckets = self.config.metrics.duration_buckets

        # RCA operation metrics
        self.rca_requests_total = Counter(
            "imkb_rca_requests_total",
            "Total number of RCA requests",
            labelnames=["extractor", "namespace", "status"] + labels,
            registry=self.registry,
        )

        self.rca_duration = Histogram(
            "imkb_rca_duration_seconds",
            "Duration of RCA operations",
            labelnames=["extractor", "namespace"] + labels,
            buckets=buckets,
            registry=self.registry,
        )

        self.rca_success_total = Counter(
            "imkb_rca_success_total",
            "Total number of successful RCA operations",
            labelnames=["extractor", "namespace"] + labels,
            registry=self.registry,
        )

        self.rca_errors_total = Counter(
            "imkb_rca_errors_total",
            "Total number of RCA errors",
            labelnames=["extractor", "namespace", "error_type"] + labels,
            registry=self.registry,
        )

        # Action pipeline metrics
        self.action_requests_total = Counter(
            "imkb_action_requests_total",
            "Total number of action generation requests",
            labelnames=["namespace", "status"] + labels,
            registry=self.registry,
        )

        self.action_duration = Histogram(
            "imkb_action_duration_seconds",
            "Duration of action generation operations",
            labelnames=["namespace"] + labels,
            buckets=buckets,
            registry=self.registry,
        )

        self.action_success_total = Counter(
            "imkb_action_success_total",
            "Total number of successful action generations",
            labelnames=["namespace", "priority", "risk_level"] + labels,
            registry=self.registry,
        )

        # LLM metrics
        self.llm_requests_total = Counter(
            "imkb_llm_requests_total",
            "Total number of LLM requests",
            labelnames=["provider", "model", "router", "template_type"] + labels,
            registry=self.registry,
        )

        self.llm_duration = Histogram(
            "imkb_llm_duration_seconds",
            "Duration of LLM requests",
            labelnames=["provider", "model", "router"] + labels,
            buckets=buckets,
            registry=self.registry,
        )

        self.llm_tokens_total = Counter(
            "imkb_llm_tokens_total",
            "Total number of LLM tokens used",
            labelnames=["provider", "model", "router", "token_type"] + labels,
            registry=self.registry,
        )

        self.llm_errors_total = Counter(
            "imkb_llm_errors_total",
            "Total number of LLM errors",
            labelnames=["provider", "model", "router", "error_type"] + labels,
            registry=self.registry,
        )

        # Extractor metrics
        self.extractor_requests_total = Counter(
            "imkb_extractor_requests_total",
            "Total number of extractor requests",
            labelnames=["extractor", "operation"] + labels,
            registry=self.registry,
        )

        self.extractor_duration = Histogram(
            "imkb_extractor_duration_seconds",
            "Duration of extractor operations",
            labelnames=["extractor", "operation"] + labels,
            buckets=buckets,
            registry=self.registry,
        )

        self.extractor_matches_total = Counter(
            "imkb_extractor_matches_total",
            "Total number of extractor matches",
            labelnames=["extractor"] + labels,
            registry=self.registry,
        )

        self.extractor_knowledge_items = Histogram(
            "imkb_extractor_knowledge_items",
            "Number of knowledge items returned by extractors",
            labelnames=["extractor"] + labels,
            buckets=[0, 1, 2, 5, 10, 20, 50, 100],
            registry=self.registry,
        )

        # Memory system metrics
        self.memory_operations_total = Counter(
            "imkb_memory_operations_total",
            "Total number of memory operations",
            labelnames=["operation", "adapter"] + labels,
            registry=self.registry,
        )

        self.memory_duration = Histogram(
            "imkb_memory_duration_seconds",
            "Duration of memory operations",
            labelnames=["operation", "adapter"] + labels,
            buckets=buckets,
            registry=self.registry,
        )

        self.memory_items_total = Counter(
            "imkb_memory_items_total",
            "Total number of memory items processed",
            labelnames=["operation", "adapter"] + labels,
            registry=self.registry,
        )

        # System health metrics
        self.active_operations = Gauge(
            "imkb_active_operations",
            "Number of currently active operations",
            labelnames=["operation_type"] + labels,
            registry=self.registry,
        )

        self.system_info = Info(
            "imkb_system", "System information", registry=self.registry
        )

        # Set system info
        self.system_info.info(
            {
                "version": self.config.tracing.service_version,
                "environment": self.config.environment,
            }
        )

        logger.info("Prometheus metrics initialized")

    def _start_metrics_server(self):
        """Start HTTP server for metrics endpoint"""
        try:
            start_http_server(port=self.config.metrics.port, registry=self.registry)
            logger.info(f"Metrics server started on port {self.config.metrics.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def _get_default_labels(self) -> dict[str, str]:
        """Get default labels for metrics"""
        return self.config.metrics.default_labels.copy()

    @contextmanager
    def time_operation(self, metric: Histogram, labels: dict[str, str]):
        """Context manager to time operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            combined_labels = {**self._get_default_labels(), **labels}
            metric.labels(**combined_labels).observe(duration)

    @contextmanager
    def track_active_operation(self, operation_type: str):
        """Context manager to track active operations"""
        labels = {**self._get_default_labels(), "operation_type": operation_type}
        self.active_operations.labels(**labels).inc()
        try:
            yield
        finally:
            self.active_operations.labels(**labels).dec()

    # RCA operation tracking
    def record_rca_request(self, extractor: str, namespace: str, status: str):
        """Record an RCA request"""
        labels = {
            **self._get_default_labels(),
            "extractor": extractor,
            "namespace": namespace,
            "status": status,
        }
        self.rca_requests_total.labels(**labels).inc()

    def record_rca_success(self, extractor: str, namespace: str, confidence: float):
        """Record a successful RCA operation"""
        labels = {
            **self._get_default_labels(),
            "extractor": extractor,
            "namespace": namespace,
        }
        self.rca_success_total.labels(**labels).inc()

    def record_rca_error(self, extractor: str, namespace: str, error_type: str):
        """Record an RCA error"""
        labels = {
            **self._get_default_labels(),
            "extractor": extractor,
            "namespace": namespace,
            "error_type": error_type,
        }
        self.rca_errors_total.labels(**labels).inc()

    # LLM operation tracking
    def record_llm_request(
        self, provider: str, model: str, router: str, template_type: str = ""
    ):
        """Record an LLM request"""
        labels = {
            **self._get_default_labels(),
            "provider": provider,
            "model": model,
            "router": router,
            "template_type": template_type,
        }
        self.llm_requests_total.labels(**labels).inc()

    def record_llm_tokens(
        self,
        provider: str,
        model: str,
        router: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        """Record LLM token usage"""
        base_labels = {
            **self._get_default_labels(),
            "provider": provider,
            "model": model,
            "router": router,
        }

        if prompt_tokens > 0:
            labels = {**base_labels, "token_type": "prompt"}
            self.llm_tokens_total.labels(**labels).inc(prompt_tokens)

        if completion_tokens > 0:
            labels = {**base_labels, "token_type": "completion"}
            self.llm_tokens_total.labels(**labels).inc(completion_tokens)

    def record_llm_error(self, provider: str, model: str, router: str, error_type: str):
        """Record an LLM error"""
        labels = {
            **self._get_default_labels(),
            "provider": provider,
            "model": model,
            "router": router,
            "error_type": error_type,
        }
        self.llm_errors_total.labels(**labels).inc()

    # Extractor operation tracking
    def record_extractor_match(self, extractor: str):
        """Record an extractor match"""
        labels = {**self._get_default_labels(), "extractor": extractor}
        self.extractor_matches_total.labels(**labels).inc()

    def record_knowledge_items(self, extractor: str, count: int):
        """Record number of knowledge items returned"""
        labels = {**self._get_default_labels(), "extractor": extractor}
        self.extractor_knowledge_items.labels(**labels).observe(count)

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        return generate_latest(self.registry).decode("utf-8")


# Global metrics instance
_metrics: Optional[MetricsCollector] = None


def initialize_metrics(config: TelemetryConfig) -> None:
    """Initialize global metrics collector"""
    global _metrics
    _metrics = MetricsCollector(config)


def get_metrics() -> Optional[MetricsCollector]:
    """Get the global metrics collector"""
    return _metrics


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled"""
    return _metrics is not None
