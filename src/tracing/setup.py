"""
Simplified tracing setup for Arize Cloud demo.

Ensures single unified trace across CLI -> Orchestrator -> Bull/Bear agents
via W3C trace context propagation.
"""

import os
from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap, extract, inject
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.trace import SpanKind

load_dotenv(override=True)

_initialized = False


def setup_arize_tracing(service_name: str) -> trace.Tracer:
    """Configure tracing with Arize Cloud."""
    global _initialized

    if _initialized:
        return trace.get_tracer(service_name)

    arize_api_key = os.environ.get("ARIZE_API_KEY")
    arize_space_id = os.environ.get("ARIZE_SPACE_ID")
    project_name = os.environ.get("ARIZE_PROJECT_NAME", "financial-agents")

    # MUST set propagator FIRST, before any tracer provider
    set_global_textmap(CompositePropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator()
    ]))

    if not arize_api_key or not arize_space_id:
        print(f"Warning: Arize credentials not set for {service_name}")
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        trace.set_tracer_provider(provider)
    else:
        try:
            from arize.otel import register
            register(
                space_id=arize_space_id,
                api_key=arize_api_key,
                project_name=project_name,
                set_global_tracer_provider=True,
                batch=True,
                verbose=False,
            )
        except ImportError:
            # Fallback if arize-otel is not available - use basic tracer provider
            print(f"Warning: arize-otel not available for {service_name}, using basic tracer provider")
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                
                provider = TracerProvider(
                    resource=Resource.create({
                        "service.name": service_name,
                        "arize.project.name": project_name
                    })
                )
                
                # Configure OTLP exporter to send directly to Arize
                otlp_endpoint = "https://otlp.arize.com/v1/traces"
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    headers={
                        "authorization": f"Bearer {arize_api_key}",
                        "api_key": arize_api_key,
                        "arize-space-id": arize_space_id,
                        "space_id": arize_space_id,
                        "arize-interface": "python-sdk"
                    }
                )
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                trace.set_tracer_provider(provider)
            except ImportError:
                # Final fallback - just use basic provider without exporter
                provider = TracerProvider(
                    resource=Resource.create({
                        "service.name": service_name,
                        "arize.project.name": project_name
                    })
                )
                trace.set_tracer_provider(provider)
        # Re-set propagator after register (register might override it)
        set_global_textmap(CompositePropagator([
            TraceContextTextMapPropagator(),
            W3CBaggagePropagator()
        ]))

    # Instrument HTTP clients for automatic trace propagation
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except Exception:
        pass

    # Also instrument aiohttp if available (ADK might use it)
    try:
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
        AioHttpClientInstrumentor().instrument()
    except Exception:
        pass

    _initialized = True
    print(f"Tracing: {service_name}")

    return trace.get_tracer(service_name)


def instrument_app(app, service_name: str):
    """Add trace context extraction middleware."""
    return TracingMiddleware(app, service_name)


class TracingMiddleware:
    """Extracts trace context from incoming requests and creates a span."""

    def __init__(self, app, service_name: str):
        self.app = app
        self.service_name = service_name
        self.tracer = trace.get_tracer(service_name)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        # Skip tracing for discovery endpoints
        if ".well-known" in path:
            await self.app(scope, receive, send)
            return

        # Extract traceparent from headers
        headers = {}
        for key, value in scope.get("headers", []):
            k = key.decode() if isinstance(key, bytes) else key
            v = value.decode() if isinstance(value, bytes) else value
            headers[k] = v

        # Extract parent context from W3C traceparent header
        parent_ctx = extract(headers)

        # Create span as child of parent context (or new root if no parent)
        with self.tracer.start_as_current_span(
            self.service_name,
            context=parent_ctx,
            kind=SpanKind.SERVER,
        ) as span:
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("http.path", path)

            # Log trace info for debugging
            ctx = span.get_span_context()
            if ctx.is_valid:
                trace_id = format(ctx.trace_id, '032x')
                span_id = format(ctx.span_id, '016x')
                parent_span_id = headers.get("traceparent", "none")
                print(f"  [{self.service_name}] trace={trace_id[:8]}... span={span_id[:8]}... parent={parent_span_id[:20] if parent_span_id != 'none' else 'root'}")

            await self.app(scope, receive, send)


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance."""
    return trace.get_tracer(name)
