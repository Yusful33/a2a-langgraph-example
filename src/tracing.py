"""
Shared tracing setup with filtering to reduce noise.

Only exports spans that match our allowed prefixes, filtering out
auto-instrumentation noise from LangGraph, ADK, etc.
"""

import os
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

# Set up W3C trace context propagator IMMEDIATELY on module import
# This is required for inject() and extract() to work
set_global_textmap(CompositePropagator([
    TraceContextTextMapPropagator(),
    W3CBaggagePropagator()
]))
print("[TRACE] W3C trace context propagator configured")

# Span name prefixes we want to keep (our explicit spans + LLM/tool spans)
ALLOWED_SPAN_PREFIXES = (
    # Our explicit spans
    "orchestrator:",
    "bull_agent:",
    "bear_agent:",
    "call_remote_agent:",
    "cli:",
    # LangChain/LangGraph spans (for bear agent)
    "ChatOpenAI",
    "ChatAnthropic",  # Keep for backward compatibility
    "ChatModel",
    "LLM",
    "Tool",
    "Chain",
    "RunnableSequence",
    "AgentExecutor",
    "Runnable",
    "PromptTemplate",
    "invoke_agent",
    "invocation",
    "call_llm",
    "ChatCompletion",
    # Google ADK / LiteLLM spans (for bull agent and orchestrator)
    "LiteLLM",
    "litellm",
    "chat",
    "completion",
    "generate",
    "acompletion",
    # Anthropic spans (keep for backward compatibility)
    "anthropic",
    "Anthropic",
    # OpenAI spans
    "openai",
    "OpenAI",
    "Messages",
    "messages",
    # Generic LLM spans
    "llm",
    "gen_ai",
)

# Also allow spans containing these substrings (for more flexible matching)
ALLOWED_SPAN_SUBSTRINGS = (
    "llm",
    "chat",
    "completion",
    "anthropic",  # Keep for backward compatibility
    "openai",
    "tool",
)


class FilteringSpanProcessor(SpanProcessor):
    """Span processor that filters out unwanted spans before exporting."""

    def __init__(self, exporter: SpanExporter):
        self._batch_processor = BatchSpanProcessor(exporter)
        self._filtered_count = 0
        self._exported_count = 0

    def on_start(self, span, parent_context=None):
        # Allow all spans to start (needed for context propagation)
        pass

    def on_end(self, span):
        # Drop spans that only capture asyncio cancellation noise
        exception_type = (span.attributes or {}).get("exception.type", "")
        if exception_type in ("CancelledError", "asyncio.exceptions.CancelledError"):
            self._filtered_count += 1
            return

        # Only export spans that match our allowed prefixes or contain allowed substrings
        span_name = span.name
        span_name_lower = span_name.lower()

        # Drop noisy A2A spans except request handlers
        if span_name.startswith("a2a.") and not span_name.startswith("a2a.server.request_handlers."):
            self._filtered_count += 1
            return

        # Check prefixes first
        if span_name.startswith(ALLOWED_SPAN_PREFIXES):
            self._exported_count += 1
            print(f"[TRACE] Exporting span: {span_name}")
            self._batch_processor.on_end(span)
        # Check substrings (case-insensitive)
        elif any(sub in span_name_lower for sub in ALLOWED_SPAN_SUBSTRINGS):
            self._exported_count += 1
            print(f"[TRACE] Exporting span (substring match): {span_name}")
            self._batch_processor.on_end(span)
        else:
            self._filtered_count += 1
            # Uncomment to debug what's being filtered:
            # print(f"[TRACE] Filtering out span: {span_name}")

    def shutdown(self):
        print(f"[TRACE] Shutdown - Exported: {self._exported_count}, Filtered: {self._filtered_count}")
        self._batch_processor.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self._batch_processor.force_flush(timeout_millis)


class CancelledErrorDropSpanProcessor(SpanProcessor):
    """Drop spans that only represent asyncio cancellation."""

    def __init__(self, exporter: SpanExporter):
        self._batch_processor = BatchSpanProcessor(exporter)
        self._filtered_count = 0
        self._exported_count = 0

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        exception_type = (span.attributes or {}).get("exception.type", "")
        if exception_type in ("CancelledError", "asyncio.exceptions.CancelledError"):
            self._filtered_count += 1
            return

        # Drop noisy A2A spans except request handlers
        span_name = span.name
        if span_name.startswith("a2a.") and not span_name.startswith("a2a.server.request_handlers."):
            self._filtered_count += 1
            return

        self._exported_count += 1
        self._batch_processor.on_end(span)

    def shutdown(self):
        print(f"[TRACE] Shutdown - Exported: {self._exported_count}, Filtered: {self._filtered_count}")
        self._batch_processor.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self._batch_processor.force_flush(timeout_millis)


def setup_tracing(service_name: str, filter_spans: bool = True) -> trace_sdk.TracerProvider:
    """
    Set up OpenTelemetry tracing with Arize exporter.

    Args:
        service_name: Name of the service (e.g., 'orchestrator', 'bull-agent', 'bear-agent')
        filter_spans: If True, only export spans matching ALLOWED_SPAN_PREFIXES

    Returns:
        TracerProvider instance
    """
    project_name = os.environ.get("ARIZE_PROJECT_NAME", "trading-agent")
    # Require env for Arize creds; avoid hardcoding any defaults
    space_id = os.environ.get("ARIZE_SPACE_ID", "")
    api_key = os.environ.get("ARIZE_API_KEY", "")
    trace_attributes = {
        "model_id": project_name,
        "model_version": "v1",
        "service.name": service_name,
    }
    api_key_suffix = api_key[-4:] if api_key else "NONE"
    space_id_status = space_id if space_id else "MISSING"
    api_key_status = f"set (****{api_key_suffix})" if api_key else "MISSING"
    print(f"[TRACE] {service_name} using model_id={project_name}")
    print(f"[TRACE] {service_name} ARIZE_SPACE_ID={space_id_status}, ARIZE_API_KEY={api_key_status}")

    tracer_provider = trace_sdk.TracerProvider(
        resource=Resource(attributes=trace_attributes)
    )

    # Create OTLP exporter to Arize (prefer gRPC, fall back to HTTP if available)
    otlp_exporter = None
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        otlp_exporter = OTLPSpanExporter(
            endpoint="https://otlp.arize.com:443",
            headers=(
                ("space_id", space_id),
                ("api_key", api_key),
            ),
            insecure=False,
        )
    except Exception as e:
        print(f"[TRACE] OTLP gRPC exporter unavailable: {e}")
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpOTLPSpanExporter
            otlp_exporter = HttpOTLPSpanExporter(
                endpoint="https://otlp.arize.com/v1/traces",
                headers={
                    "space_id": space_id,
                    "api_key": api_key,
                },
            )
        except Exception as http_e:
            print(f"[TRACE] OTLP HTTP exporter unavailable: {http_e}")
            otlp_exporter = ConsoleSpanExporter()
            print("[TRACE] Falling back to ConsoleSpanExporter (no traces will be sent to Arize).")

    if filter_spans:
        # Use filtering processor to drop noisy auto-instrumented spans
        tracer_provider.add_span_processor(FilteringSpanProcessor(otlp_exporter))
        print(f"[TRACE] Setup: {service_name} -> Arize [FILTERED - only exporting: {ALLOWED_SPAN_PREFIXES}]")
    else:
        # Export all spans, but drop asyncio CancelledError noise
        tracer_provider.add_span_processor(CancelledErrorDropSpanProcessor(otlp_exporter))
        print(f"[TRACE] Setup: {service_name} -> Arize [unfiltered + drop CancelledError]")

    trace_api.set_tracer_provider(tracer_provider=tracer_provider)

    return tracer_provider
