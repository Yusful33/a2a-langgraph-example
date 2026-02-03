import asyncio
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import httpx
from httpx import Timeout

from a2a.client import A2AClient
from a2a.types import (
    AgentSkill,
    TransportProtocol,
    SendMessageRequest,
    MessageSendParams,
)
from a2a.utils import new_agent_text_message
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card

# Set up tracing for the client
from src.tracing import setup_tracing
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import inject

tracer_provider = setup_tracing("cli", filter_spans=True)

skills = [
    AgentSkill(
        id='bull',
        name='bull analysis',
        description='Identify bullish patterns',
        tags=['momentum', 'screening'],
        examples=['Find opportunities for NVDA'],
    ),
    AgentSkill(
        id='bear',
        name='bear analysis',
        description='do risk analysis',
        tags=['risk'],
        examples=['Find risks for nvda'],
    )
]
session_id = "sess1689"

# --- Agent card ---
orchestrator_agent_card = create_agent_card(
    agent_name="trading_strategy_orchestrator",
    description="Coordinates bull and bear agents to form a strategy",
    skills=skills,
)

orchestrator_agent_card.url = "http://localhost:8003"
orchestrator_agent_card.preferred_transport = TransportProtocol.jsonrpc


async def main():
    tracer = trace.get_tracer("cli")

    query = """Create a combined bull + bear view for NVDA.

Bull agent: find upside catalysts and technical setup. Keep it concise.
Bear agent: you MUST call exactly one MCP tool (prefer list_top_risks; if unavailable, recent_volatility_snapshot). Do not call more than one tool total.

Return a brief orchestrated summary that highlights both sides."""

    # Create root span for the entire request
    with tracer.start_as_current_span(
        "cli:analyze_request",
        kind=trace.SpanKind.CLIENT,
    ) as root_span:
        span_ctx = root_span.get_span_context()
        print(f"[TRACE] CLI: created root span trace_id={format(span_ctx.trace_id, '032x')}")

        # Add input attribute
        root_span.set_attribute("input.value", query)
        root_span.set_attribute("session.id", session_id)

        # Inject trace context into headers
        headers = {}
        inject(headers)
        print(f"[TRACE] CLI: injected traceparent={headers.get('traceparent')}")

        async with httpx.AsyncClient(timeout=Timeout(500), headers=headers) as httpx_client:
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=orchestrator_agent_card,
            )

            request = SendMessageRequest(
                id=str(session_id),
                params=MessageSendParams(
                    message=new_agent_text_message(
                        text=query,
                        context_id=str(session_id),
                    ),
                    metadata={
                        # Pass trace context in metadata as fallback
                        "traceparent": headers.get("traceparent", ""),
                        "tracestate": headers.get("tracestate", ""),
                    },
                ),
            )

            print("sending request")
            try:
                response = await client.send_message(request)
            except asyncio.CancelledError:
                # Avoid marking the root span as error for cancellations
                root_span.set_attribute("request.cancelled", True)
                root_span.set_status(Status(StatusCode.OK))
                print("[TRACE] CLI request cancelled")
                return
            except Exception as e:
                root_span.record_exception(e)
                root_span.set_status(Status(StatusCode.ERROR, str(e)))
                print(f"[TRACE] CLI request failed: {e}")
                return

            # Add output attribute
            root_span.set_attribute("output.value", str(response))
            root_span.set_status(Status(StatusCode.OK))

            print(response)


if __name__ == "__main__":
    asyncio.run(main())
