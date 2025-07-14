"""
Telemetry configuration for OpenTelemetry and Prometheus

Provides configuration options for tracing, metrics, and observability features.
"""

import os
from typing import Optional

from pydantic import BaseModel, Field


class TracingConfig(BaseModel):
    """OpenTelemetry tracing configuration"""

    enabled: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    service_name: str = Field(default="imkb", description="Service name for traces")
    service_version: str = Field(default="0.1.0", description="Service version")

    # OTLP Exporter configuration
    otlp_endpoint: Optional[str] = Field(
        default=None, description="OTLP endpoint URL (e.g., http://localhost:4317)"
    )
    otlp_headers: dict[str, str] = Field(
        default_factory=dict, description="OTLP headers for authentication"
    )
    otlp_insecure: bool = Field(
        default=True, description="Use insecure connection for OTLP"
    )

    # Sampling configuration
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0 = no traces, 1.0 = all traces)",
    )

    # Resource attributes
    resource_attributes: dict[str, str] = Field(
        default_factory=dict, description="Additional resource attributes for traces"
    )

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create tracing config from environment variables"""
        return cls(
            enabled=os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true",
            service_name=os.getenv("OTEL_SERVICE_NAME", "imkb"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "0.1.0"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otlp_headers=cls._parse_headers(
                os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
            ),
            otlp_insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower()
            == "true",
            sample_rate=float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "1.0")),
        )

    @staticmethod
    def _parse_headers(headers_str: str) -> dict[str, str]:
        """Parse OTLP headers from environment variable format"""
        headers = {}
        if headers_str:
            for header in headers_str.split(","):
                if "=" in header:
                    key, value = header.split("=", 1)
                    headers[key.strip()] = value.strip()
        return headers


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration"""

    enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    port: int = Field(
        default=9090, ge=1024, le=65535, description="Metrics server port"
    )
    path: str = Field(default="/metrics", description="Metrics endpoint path")

    # Metric collection settings
    collect_runtime_metrics: bool = Field(
        default=True, description="Collect Python runtime metrics"
    )
    collect_process_metrics: bool = Field(
        default=True, description="Collect process metrics"
    )

    # Custom metric labels
    default_labels: dict[str, str] = Field(
        default_factory=dict, description="Default labels added to all metrics"
    )

    # Histogram buckets for duration metrics
    duration_buckets: list[float] = Field(
        default_factory=lambda: [
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ],
        description="Histogram buckets for operation duration metrics (in seconds)",
    )

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        """Create metrics config from environment variables"""
        return cls(
            enabled=os.getenv("PROMETHEUS_METRICS_ENABLED", "true").lower() == "true",
            port=int(os.getenv("PROMETHEUS_METRICS_PORT", "9090")),
            path=os.getenv("PROMETHEUS_METRICS_PATH", "/metrics"),
            collect_runtime_metrics=os.getenv(
                "PROMETHEUS_RUNTIME_METRICS", "true"
            ).lower()
            == "true",
            collect_process_metrics=os.getenv(
                "PROMETHEUS_PROCESS_METRICS", "true"
            ).lower()
            == "true",
        )


class LoggingConfig(BaseModel):
    """Structured logging configuration"""

    enabled: bool = Field(default=True, description="Enable structured logging")
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json|text)")

    # Correlation with traces
    include_trace_id: bool = Field(
        default=True, description="Include trace ID in log records"
    )
    include_span_id: bool = Field(
        default=True, description="Include span ID in log records"
    )


class TelemetryConfig(BaseModel):
    """Complete telemetry configuration"""

    enabled: bool = Field(default=True, description="Enable all telemetry features")

    tracing: TracingConfig = Field(default_factory=TracingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Environment detection
    environment: str = Field(
        default="development",
        description="Deployment environment (development|staging|production)",
    )

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Create complete telemetry config from environment variables"""
        return cls(
            enabled=os.getenv("TELEMETRY_ENABLED", "true").lower() == "true",
            environment=os.getenv("ENVIRONMENT", "development"),
            tracing=TracingConfig.from_env(),
            metrics=MetricsConfig.from_env(),
        )

    def get_resource_attributes(self) -> dict[str, str]:
        """Get OpenTelemetry resource attributes"""
        attributes = {
            "service.name": self.tracing.service_name,
            "service.version": self.tracing.service_version,
            "deployment.environment": self.environment,
        }

        # Add custom resource attributes
        attributes.update(self.tracing.resource_attributes)

        return attributes

    def should_export_traces(self) -> bool:
        """Check if traces should be exported to external system"""
        return (
            self.enabled
            and self.tracing.enabled
            and self.tracing.otlp_endpoint is not None
        )

    def should_start_metrics_server(self) -> bool:
        """Check if metrics server should be started"""
        return self.enabled and self.metrics.enabled
