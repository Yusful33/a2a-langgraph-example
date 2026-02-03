#!/usr/bin/env python
# coding: utf-8

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# In[3]:


import os
import vertexai
try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None

# fmt: off
PROJECT_ID = "arize-project-483603"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
LOCATION = "us-central1" # @param {type: "string", placeholder: "[your-location]", isTemplate: true}
# fmt: on

# Create the bucket
BUCKET_NAME = f"{PROJECT_ID}-agent"
BUCKET_URI = f"gs://{BUCKET_NAME}"

# Set environment variables for ADK
os.environ['GOOGLE_API_KEY'] = ""
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

# For notebook async support
if nest_asyncio is not None:
    nest_asyncio.apply()
else:
    print("[WARN] nest_asyncio not available; skipping apply()")

# Initiate the client
client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

# General
import os
from pathlib import Path
import random
import uvicorn
import threading
import time
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import List, Dict
from textwrap import dedent
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# A2A types
from a2a.types import AgentSkill
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card
from a2a.server.agent_execution import AgentExecutor, RequestContext
from google.adk.agents import InvocationContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

# ADK agent
from google.adk.models.lite_llm import litellm
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk import Runner
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor, A2aAgentExecutorConfig
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import TransportProtocol
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.tools.agent_tool import AgentTool

# Agent deployment
import vertexai
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import A2aAgent
from google.auth import default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as AuthRequest
from a2a.client.client import ClientConfig as A2AClientConfig
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.types import TransportProtocol as A2ATransport

# Trace context propagation
from opentelemetry import trace as trace_api, context as context_module
from opentelemetry.propagate import extract
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from src.tracing import setup_tracing

# Set up tracing without filtering (export all spans)
_bull_tracer_provider = setup_tracing("bull-agent", filter_spans=False)
print(f"[TRACE] Bull agent using ARIZE_PROJECT_NAME={os.environ.get('ARIZE_PROJECT_NAME', 'NOT SET')}")

# Enable Google ADK instrumentation (OpenInference)
try:
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    GoogleADKInstrumentor().instrument(tracer_provider=_bull_tracer_provider)
    print("[TRACE] Bull agent: GoogleADKInstrumentor enabled")
except ImportError:
    print("[TRACE] Bull agent: GoogleADKInstrumentor not available")

# Enable LiteLLM instrumentation for LLM call tracing (used by Google ADK)
try:
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    LiteLLMInstrumentor().instrument(tracer_provider=_bull_tracer_provider)
    print("[TRACE] Bull agent: LiteLLMInstrumentor enabled")
except ImportError:
    print("[TRACE] Bull agent: LiteLLMInstrumentor not available, trying OpenAI instrumentor")
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=_bull_tracer_provider)
        print("[TRACE] Bull agent: OpenAIInstrumentor enabled")
    except ImportError:
        print("[TRACE] Bull agent: No LLM instrumentor available")


# In[5]:


def create_bull_agent_card():
    """Create A2A Agent Card for Bull Analyst."""

    skills = [
        AgentSkill(
            id='breakout_detection',
            name='Breakout Pattern Detection',
            description='Identify bullish breakout patterns',
            tags=['technical-analysis', 'breakouts'],
            examples=['Find breakout patterns for NVDA'],
        ),
        AgentSkill(
            id='momentum_screening',
            name='Momentum Screening',
            description='Screen for stocks with strong momentum',
            tags=['momentum', 'screening'],
            examples=['Find high momentum tech stocks'],
        ),
        AgentSkill(
            id='entry_signals',
            name='Entry Signal Detection',
            description='Detect optimal entry points',
            tags=['entry-points', 'timing'],
            examples=['When should I buy AAPL?'],
        ),
    ]

    return create_agent_card(
        agent_name='Bull Market Analyst (ADK + MCP)',
        description=(
            'An optimistic analyst powered by Google ADK, '
            'focused on growth opportunities and bullish patterns.'
        ),
        skills=skills
    )

bull_agent_card = create_bull_agent_card()


print("Bull Agent Card:")
print(f"   Name: {bull_agent_card.name}")
print(f"   Skills: {len(bull_agent_card.skills)}")



class TraceContextMiddleware(BaseHTTPMiddleware):
    """Middleware to extract trace context from incoming HTTP requests."""

    async def dispatch(self, request: Request, call_next):
        headers = dict(request.headers)
        traceparent = headers.get('traceparent')

        # Skip tracing for agent card requests
        if '/.well-known/' in str(request.url):
            return await call_next(request)

        # Debug: Log received traceparent
        print(f"[TRACE] Bull agent received traceparent: {traceparent}")

        if traceparent:
            # Parse traceparent manually: 00-{trace_id}-{parent_span_id}-{flags}
            parts = traceparent.split('-')
            if len(parts) == 4:
                received_trace_id = parts[1]
                received_span_id = parts[2]
                received_flags = parts[3]
                print(f"[TRACE] Bull: parsed trace_id={received_trace_id}, parent_span_id={received_span_id}")

                # Manually create SpanContext from traceparent (bypass extract())
                from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan, set_span_in_context

                trace_id_int = int(received_trace_id, 16)
                span_id_int = int(received_span_id, 16)
                trace_flags = TraceFlags(int(received_flags, 16))

                parent_span_context = SpanContext(
                    trace_id=trace_id_int,
                    span_id=span_id_int,
                    is_remote=True,
                    trace_flags=trace_flags,
                )

                print(f"[TRACE] Bull: manually created SpanContext valid={parent_span_context.is_valid}")

                # Create a NonRecordingSpan with the parent context and set it in context
                parent_span = NonRecordingSpan(parent_span_context)
                parent_context = set_span_in_context(parent_span)

                # Create span as child of parent context
                tracer = trace_api.get_tracer("bull-agent")
                with tracer.start_as_current_span(
                    "bull_agent:handle_request",
                    context=parent_context,
                    kind=trace_api.SpanKind.SERVER,
                ) as span:
                    span.set_attribute("openinference.span.kind", "agent")
                    # Agent Graph metadata for Arize
                    span.set_attribute("graph.node.id", "bull_agent")
                    span.set_attribute("graph.node.parent_id", "orchestrator")
                    span.set_attribute("graph.node.display_name", "Bull Agent")
                    # Debug: verify trace ID matches
                    span_ctx = span.get_span_context()
                    created_trace_id = format(span_ctx.trace_id, '032x')
                    print(f"[TRACE] Bull: created span trace_id={created_trace_id}, matches={created_trace_id == received_trace_id}")
                    current_context = trace_api.set_span_in_context(span, parent_context)
                    token = context_module.attach(current_context)
                    try:
                        response = await call_next(request)
                        return response
                    finally:
                        context_module.detach(token)
            else:
                # Invalid traceparent format
                return await call_next(request)
        else:
            # No parent context - just process without tracing
            return await call_next(request)


class BullAgentExecutor(AgentExecutor):
    """Agent executor for Bull Agent."""

    def __init__(self):
        self.agent = None
        self.runner = None
        self.register = None

    async def _init_agent(self):
        """Lazy initialization of the Bull Agent and ADK Runner.
        Creates the agent and runner when first needed.
        This happens on Agent Engine after deployment, not during pickling.
        """

        if self.register is None:
            # TracerProvider is set up at module level with filtered exporter.
            # Auto-instrumentation is DISABLED to reduce noise.
            # Only our explicit spans (bull_agent:handle_request) are exported.
            self.register = True

        if self.agent is None:
            from google.adk.models.lite_llm import litellm
            from google.adk.agents import LlmAgent
            import os

            # Create agent without MCP tools for simplified testing
            from google.adk.models.lite_llm import LiteLlm
            self.agent = LlmAgent(
                model=LiteLlm(model="gpt-4o"),
                name="bull_market_analyst",
                instruction="""You are an optimistic market analyst focused on identifying growth
                opportunities, bullish catalysts, and upside potential. Provide insightful analysis
                about market opportunities.""",
                tools=[],  # No tools for simplified test
            )


        if self.runner is None:
            from google.adk import Runner
            from google.adk.artifacts import InMemoryArtifactService
            from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
            from google.adk.sessions import InMemorySessionService

            self.runner = Runner(
                app_name=self.agent.name,
                agent=self.agent,
                session_service=InMemorySessionService(),
            )

    async def cancel(self, context: InvocationContext, event_queue: EventQueue):
      raise ServerError(error=UnsupportedOperationError())

    async def execute(self, context: InvocationContext, event_queue: EventQueue) -> None:
        """Execute Bull Agent analysis."""
        print("printing bull context: ", context)
        print("printing bull context id : ", context.context_id)

        await self._init_agent()

        session_id = context.context_id

        print("session id from orchestrator : ", session_id)

        if not context.message:
            return

        # NOTE: Context extraction is handled by TraceContextMiddleware.
        # Do NOT extract context here - it causes conflicting context scopes.
        # The middleware already attaches the parent context from HTTP headers.

        # Get current span to add attributes
        current_span = trace_api.get_current_span()

        print("context.messages :", context.message)
        user_id = (
            context.message.metadata.get('user_id')
            if context.message and context.message.metadata
            else 'a2a_user'
        )
        print("extracted user id :", user_id)

        print("extracted session id :", session_id)

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # if not hasattr(context, "current_task") or not context.current_task:
        #     await updater.submit()

        await updater.start_work()

        query = context.get_user_input()

        # Add input attribute to current span
        if current_span and current_span.is_recording():
            current_span.set_attribute("input.value", query)
            current_span.set_attribute("session.id", session_id)

        try:
            await updater.update_status(
                TaskState.working,
                message=new_agent_text_message("Analyzing opportunities...")
            )

            # Get or create session
            from google.genai import types

            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=context.context_id,
            ) or await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=context.context_id,
            )

            content = types.Content(role='user', parts=[types.Part(text=query)])

            # Run ADK agent - spans should inherit from bull_agent:handle_request middleware span
            # ADK's LiteLLM instrumentation will create spans that should be children of the current context
            final_event = None
            async for event in self.runner.run_async(
                session_id=session.id,
                user_id=user_id,
                new_message=content
            ):
                if event.is_final_response():
                    final_event = event

            # Extract response
            if final_event and final_event.content and final_event.content.parts:
                response_text = "".join(
                    part.text for part in final_event.content.parts
                    if hasattr(part, 'text') and part.text
                )
                if response_text:
                    # Add output attribute to current span
                    if current_span and current_span.is_recording():
                        current_span.set_attribute("output.value", response_text)

                    await updater.add_artifact(
                        [TextPart(text=response_text)],
                        name='opportunity_analysis',
                    )
                    await updater.complete()
                    return

            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message('Failed to generate response.'),
                final=True
            )

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(f"Analysis failed: {str(e)}"),
                final=True,
            )

# Update Bull Agent card
bull_agent_card.url = "http://localhost:8002"
bull_agent_card.preferred_transport = TransportProtocol.jsonrpc


# In[22]:


def create_bull_agent_a2a_server(agent, agent_card):
    """Create an A2A server for an ADK agent.

    This wraps an ADK agent with A2A protocol handling, making it
    accessible via HTTP endpoints that follow the A2A specification.

    Args:
        agent: The ADK agent instance (LlmAgent, Agent, etc.)
        agent_card: The A2A AgentCard describing the agent's capabilities

    Returns:
        A2AStarletteApplication instance ready to serve via uvicorn
    """

    config = A2aAgentExecutorConfig()
    # executor = A2aAgentExecutor(runner=runner, config=config)


    request_handler = DefaultRequestHandler(
        agent_executor=BullAgentExecutor(),
        task_store=InMemoryTaskStore(),  # Stores task state
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

async def run_bull_server(agent, agent_card, port):
    """Run a single agent as an A2A server on the specified port."""
    app = create_bull_agent_a2a_server(agent, agent_card)

    # Build the Starlette app and add trace context middleware
    starlette_app = app.build()
    starlette_app.add_middleware(TraceContextMiddleware)

    # Configure uvicorn server
    config = uvicorn.Config(
        starlette_app,
        host='127.0.0.1',  # localhost
        port=port,
        log_level='warning',  # Quiet output
        loop='none',  # Use the current event loop
    )

    server = uvicorn.Server(config)
    await server.serve()


# In[23]:


async def start_a2a_servers():
    """Start both Bear and Bull agents as A2A servers."""
    # Create tasks for both servers
    # Bear Agent uses Pydantic AI, so it needs custom A2A server
    # Bull Agent uses ADK, so it uses the standard ADK A2A pattern
    bull_agent = None
    tasks = [
        asyncio.create_task(
            run_bull_server(bull_agent, bull_agent_card, 8002)
        )
    ]

    # Give servers time to start
    await asyncio.sleep(2)

    print("   ✓ Bull Agent A2A server: http://127.0.0.1:8002 (ADK)")

    # Keep servers running
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("Shutting down A2A servers...")

if __name__ == "__main__":
    asyncio.run(start_a2a_servers())



