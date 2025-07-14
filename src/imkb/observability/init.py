"""
Observability initialization

Provides centralized initialization for all observability features including
tracing, metrics, and logging configuration.
"""

import logging
from typing import Optional

from .config import TelemetryConfig
from .metrics import initialize_metrics
from .tracer import initialize_tracing

logger = logging.getLogger(__name__)

_initialized = False
_config: Optional[TelemetryConfig] = None


def initialize_observability(config: TelemetryConfig) -> None:
    """
    Initialize all observability features

    Args:
        config: Telemetry configuration
    """
    global _initialized, _config

    if _initialized:
        logger.warning("Observability already initialized, skipping")
        return

    _config = config

    if not config.enabled:
        logger.info("Observability is disabled")
        return

    logger.info(f"Initializing observability for environment: {config.environment}")

    # Initialize tracing
    if config.tracing.enabled:
        try:
            initialize_tracing(config)
            logger.info("OpenTelemetry tracing initialized")
        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")

    # Initialize metrics
    if config.metrics.enabled:
        try:
            initialize_metrics(config)
            logger.info("Prometheus metrics initialized")
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")

    # Configure structured logging
    if config.logging.enabled:
        try:
            _configure_logging(config)
            logger.info("Structured logging configured")
        except Exception as e:
            logger.error(f"Failed to configure logging: {e}")

    _initialized = True
    logger.info("Observability initialization complete")


def _configure_logging(config: TelemetryConfig) -> None:
    """Configure structured logging with trace correlation"""
    import logging.config

    # Base logging configuration
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
            },
            "text": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": config.logging.level,
                "formatter": config.logging.format,
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": config.logging.level, "handlers": ["console"]},
        "loggers": {"imkb": {"level": config.logging.level, "propagate": True}},
    }

    # Apply logging configuration
    logging.config.dictConfig(log_config)

    # Add trace correlation filter if tracing is enabled
    if config.tracing.enabled and (
        config.logging.include_trace_id or config.logging.include_span_id
    ):
        _add_trace_correlation_filter()


def _add_trace_correlation_filter():
    """Add filter to include trace/span IDs in log records"""
    from opentelemetry import trace

    class TraceContextFilter(logging.Filter):
        def filter(self, record):
            span = trace.get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                record.trace_id = (
                    format(span_context.trace_id, "032x")
                    if span_context.trace_id
                    else ""
                )
                record.span_id = (
                    format(span_context.span_id, "016x") if span_context.span_id else ""
                )
            else:
                record.trace_id = ""
                record.span_id = ""
            return True

    # Add filter to all handlers
    for handler in logging.getLogger().handlers:
        handler.addFilter(TraceContextFilter())


def get_observability_config() -> Optional[TelemetryConfig]:
    """Get the current observability configuration"""
    return _config


def is_observability_initialized() -> bool:
    """Check if observability has been initialized"""
    return _initialized


def shutdown_observability() -> None:
    """Shutdown observability systems gracefully"""
    global _initialized

    if not _initialized:
        return

    logger.info("Shutting down observability systems")

    # Shutdown tracing
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import get_tracer_provider

        provider = get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.shutdown()
            logger.debug("Tracing provider shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down tracing: {e}")

    # Note: Prometheus metrics server shutdown is handled automatically
    # when the process exits

    _initialized = False
    logger.info("Observability shutdown complete")
