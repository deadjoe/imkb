"""
OpenTelemetry tracing implementation

Provides distributed tracing capabilities for imkb operations with
automatic instrumentation and trace context propagation.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.trace import Status, StatusCode

from .config import TelemetryConfig

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: Optional[trace.Tracer] = None
_config: Optional[TelemetryConfig] = None

P = ParamSpec("P")
T = TypeVar("T")


def initialize_tracing(config: TelemetryConfig) -> None:
    """Initialize OpenTelemetry tracing with the given configuration"""
    global _tracer, _config

    if not config.enabled or not config.tracing.enabled:
        logger.info("Tracing is disabled")
        return

    _config = config

    # Create resource with service information
    resource = Resource.create(config.get_resource_attributes())

    # Create tracer provider with sampling
    sampler = TraceIdRatioBased(config.tracing.sample_rate)
    provider = TracerProvider(resource=resource, sampler=sampler)

    # Configure OTLP exporter if endpoint is provided
    if config.should_export_traces():
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.tracing.otlp_endpoint,
                headers=config.tracing.otlp_headers,
                insecure=config.tracing.otlp_insecure,
            )

            # Add batch span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(span_processor)

            logger.info(
                f"OTLP trace exporter configured for {config.tracing.otlp_endpoint}"
            )
        except Exception as e:
            logger.error(f"Failed to configure OTLP exporter: {e}")

    # Set the tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer instance
    _tracer = trace.get_tracer(
        instrumenting_module_name="imkb",
        instrumenting_library_version=config.tracing.service_version,
    )

    logger.info(
        f"OpenTelemetry tracing initialized (sample_rate={config.tracing.sample_rate})"
    )


def get_tracer() -> trace.Tracer:
    """Get the configured tracer instance"""
    if _tracer is None:
        # Return no-op tracer if not initialized
        return trace.NoOpTracer()
    return _tracer


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Optional[dict[str, Any]] = None,
    set_status_on_exception: bool = True,
):
    """
    Context manager for tracing operations

    Args:
        operation_name: Name of the operation being traced
        attributes: Additional attributes to add to the span
        set_status_on_exception: Whether to set error status on exceptions
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(operation_name) as span:
        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            if set_status_on_exception:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise


def trace_async(
    operation_name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False,
):
    """
    Decorator for tracing async functions

    Args:
        operation_name: Custom operation name (defaults to function name)
        attributes: Static attributes to add to spans
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record return value as attribute
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            name = operation_name or f"{func.__module__}.{func.__qualname__}"

            span_attributes = {}
            if attributes:
                span_attributes.update(attributes)

            # Record function arguments if requested
            if record_args:
                try:
                    # Only record simple types to avoid large spans
                    for i, arg in enumerate(args):
                        if isinstance(arg, (str, int, float, bool)):
                            span_attributes[f"arg.{i}"] = arg

                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span_attributes[f"kwarg.{key}"] = value
                except Exception as e:
                    # Don't fail the operation if recording args fails
                    logger.debug(f"Failed to record trace args: {e}")
                    pass

            with trace_operation(name, span_attributes) as span:
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)

                    # Record execution time
                    duration = time.time() - start_time
                    span.set_attribute("operation.duration_ms", duration * 1000)

                    # Record result if requested
                    if record_result and result is not None:
                        try:
                            if isinstance(result, (str, int, float, bool)):
                                span.set_attribute("result", result)
                            elif hasattr(result, "__len__"):
                                span.set_attribute("result.length", len(result))
                        except Exception as e:
                            # Don't fail the operation if recording result fails
                            logger.debug(f"Failed to record trace result: {e}")
                            pass

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("operation.duration_ms", duration * 1000)
                    span.set_attribute("error.type", type(e).__name__)
                    raise

        return wrapper

    return decorator


def trace_sync(
    operation_name: Optional[str] = None,
    attributes: Optional[dict[str, Any]] = None,
    record_args: bool = False,
    record_result: bool = False,
):
    """
    Decorator for tracing synchronous functions

    Args:
        operation_name: Custom operation name (defaults to function name)
        attributes: Static attributes to add to spans
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record return value as attribute
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            name = operation_name or f"{func.__module__}.{func.__qualname__}"

            span_attributes = {}
            if attributes:
                span_attributes.update(attributes)

            # Record function arguments if requested
            if record_args:
                try:
                    for i, arg in enumerate(args):
                        if isinstance(arg, (str, int, float, bool)):
                            span_attributes[f"arg.{i}"] = arg

                    for key, value in kwargs.items():
                        if isinstance(value, (str, int, float, bool)):
                            span_attributes[f"kwarg.{key}"] = value
                except Exception as e:
                    logger.debug(f"Failed to record async trace args: {e}")
                    pass

            with trace_operation(name, span_attributes) as span:
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # Record execution time
                    duration = time.time() - start_time
                    span.set_attribute("operation.duration_ms", duration * 1000)

                    # Record result if requested
                    if record_result and result is not None:
                        try:
                            if isinstance(result, (str, int, float, bool)):
                                span.set_attribute("result", result)
                            elif hasattr(result, "__len__"):
                                span.set_attribute("result.length", len(result))
                        except Exception as e:
                            logger.debug(f"Failed to record async trace result: {e}")
                            pass

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    span.set_attribute("operation.duration_ms", duration * 1000)
                    span.set_attribute("error.type", type(e).__name__)
                    raise

        return wrapper

    return decorator


def add_event(name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Add an event to the current span"""
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, attributes or {})


def set_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current span"""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, value)


def get_trace_id() -> str:
    """Get the current trace ID as a string"""
    span = trace.get_current_span()
    if span.is_recording():
        return format(span.get_span_context().trace_id, "032x")
    return ""


def get_span_id() -> str:
    """Get the current span ID as a string"""
    span = trace.get_current_span()
    if span.is_recording():
        return format(span.get_span_context().span_id, "016x")
    return ""


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled"""
    return _tracer is not None and _config is not None and _config.tracing.enabled
