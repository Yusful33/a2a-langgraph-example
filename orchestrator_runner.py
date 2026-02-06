#!/usr/bin/env python
# coding: utf-8

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# In[1]:
import json
import asyncio
from a2a.types import (
    SendStreamingMessageRequest,
    SendMessageRequest,
    MessageSendParams,
    Task,
    Message,
    TaskArtifactUpdateEvent,
)
from a2a.utils import new_agent_text_message
from httpx import Timeout
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card
# from IPython.display import Markdown, display, clear_outputvbutjdliindercvtulnktiguukdltigtrvc
from a2a.types import AgentSkill
from a2a.client import A2AClient


from yaml import Mark

import os
import vertexai
try:
    import nest_asyncio
except ImportError:
    nest_asyncio = None

# fmt: off
PROJECT_ID = ""  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}
LOCATION = "us-central1" # @param {type: "string", placeholder: "[your-location]", isTemplate: true}
# fmt: on

# Create the bucket
BUCKET_NAME = f"{PROJECT_ID}-agent"
BUCKET_URI = f"gs://{BUCKET_NAME}"

# Set environment variables for ADK
os.environ['GOOGLE_API_KEY'] = ""
# os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID
# os.environ['GOOGLE_CLOUD_LOCATION'] = LOCATION
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

# For notebook async support (commented out for standalone script)
if nest_asyncio is not None:
    # nest_asyncio.apply()
    pass
else:
    print("[WARN] nest_asyncio not available; skipping apply()")

# Initiate the client (commented out to run locally without gcloud)
# client = vertexai.Client(project=PROJECT_ID, location=LOCATION)



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
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card
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

# Observability - use filtered tracing to reduce noise
from opentelemetry import trace as trace_api
from src.tracing import setup_tracing

# Set up tracing without filtering (export all spans)
# ARIZE_PROJECT_NAME should be set via .env file
tracer_provider = setup_tracing("orchestrator", filter_spans=False)
print(f"[TRACE] Orchestrator using ARIZE_PROJECT_NAME={os.environ.get('ARIZE_PROJECT_NAME', 'NOT SET')}")

# Enable Google ADK instrumentation (OpenInference)
try:
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
    print("[TRACE] Orchestrator: GoogleADKInstrumentor enabled")
except ImportError:
    print("[TRACE] Orchestrator: GoogleADKInstrumentor not available")

# Enable LiteLLM instrumentation for LLM call tracing (used by Google ADK)
try:
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    print("[TRACE] Orchestrator: LiteLLMInstrumentor enabled")
except ImportError:
    print("[TRACE] Orchestrator: LiteLLMInstrumentor not available")


from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.tools.agent_tool import AgentTool
from opentelemetry import trace, context as context_module
from opentelemetry.propagate import inject
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


from google.genai import types

ROUTER_SYSTEM_PROMPT = """
You are a routing agent.

Your task:
Decide which specialist agent(s) should be called.
Also transform the query based on the agent extracting only part of query relevant to that agent.
Agents can be one or more than one.

Available agents:
- bull : growth, upside, positive outlook
- bear : risks, downside, warnings

Rules:
- Return ONLY valid JSON
- Do NOT explain
- Do NOT add extra keys
- dont put ```json annotation
Output schema:
{
  "agents": [{"name": "bull", "query": "analyze growth of the stock"}, {"name": "bear", "query": "analyze risks of stock"}]
}
"""

import httpx


class TraceContextMiddleware(BaseHTTPMiddleware):
    """Middleware to extract trace context from incoming HTTP requests."""

    async def dispatch(self, request: Request, call_next):
        headers = dict(request.headers)
        traceparent = headers.get('traceparent')

        # Skip tracing for agent card requests
        if '/.well-known/' in str(request.url):
            return await call_next(request)

        # Debug: Log received traceparent
        print(f"[TRACE] Orchestrator received traceparent: {traceparent}")

        if traceparent:
            # Parse traceparent manually: 00-{trace_id}-{parent_span_id}-{flags}
            parts = traceparent.split('-')
            if len(parts) == 4:
                received_trace_id = parts[1]
                received_span_id = parts[2]
                received_flags = parts[3]
                print(f"[TRACE] Orchestrator: parsed trace_id={received_trace_id}, parent_span_id={received_span_id}")

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

                print(f"[TRACE] Orchestrator: manually created SpanContext valid={parent_span_context.is_valid}")

                # Create a NonRecordingSpan with the parent context and set it in context
                parent_span = NonRecordingSpan(parent_span_context)
                parent_context = set_span_in_context(parent_span)

                # Create span as child of parent context (like bull/bear agents do)
                tracer = trace.get_tracer("orchestrator")
                with tracer.start_as_current_span(
                    "orchestrator:handle_request",
                    context=parent_context,
                    kind=trace.SpanKind.SERVER,
                ) as span:
                    span.set_attribute("openinference.span.kind", "agent")
                    # Agent Graph metadata for Arize
                    span.set_attribute("graph.node.id", "orchestrator")
                    span.set_attribute("graph.node.display_name", "Orchestrator")
                    # Debug: verify trace ID matches
                    span_ctx = span.get_span_context()
                    created_trace_id = format(span_ctx.trace_id, '032x')
                    print(f"[TRACE] Orchestrator: created span trace_id={created_trace_id}, matches={created_trace_id == received_trace_id}")
                    current_context = trace.set_span_in_context(span, parent_context)
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


class TradingOrchestratorExecutor(AgentExecutor):
    """
    A2A Executor for the Trading Strategy Orchestrator.
    This is the canonical entrypoint that defines the session / context ID.
    """

    def __init__(self):
        self.agent = None
        self.runner = None
        self.bull_agent = None
        self.bear_agent = None

    

    async def fetch_agent_card_json(self, url: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    def hydrate_agent_card(self, card_json: dict):
        return create_agent_card(
            agent_name=card_json["name"],
            description=card_json.get("description", ""),
            skills=card_json.get("skills", [])
        )

    async def build_remote_agent(self, name: str, card_url: str, url: str) -> RemoteA2aAgent:
        card_json = await self.fetch_agent_card_json(card_url)
        agent_card = self.hydrate_agent_card(card_json)

        agent_card.url = url
        agent_card.preferred_transport = TransportProtocol.jsonrpc
    
        return agent_card


    async def _init_agent(self):
        """Lazy init to avoid pickling issues in deployment."""
        # NOTE: TracerProvider is configured at module level (lines 143-154).
        # Do NOT create another TracerProvider here - that causes multiple
        # export pipelines and prevents trace unification.

        if self.agent is None:
            # ---- Remote agents (already running as A2A servers) ----
            remote_bear = RemoteA2aAgent(
                name="bear_risk_analyst",
                description="Analyzes downside risks and warnings",
                agent_card="http://localhost:18001/.well-known/agent-card.json",
            )

            remote_bull = RemoteA2aAgent(
                name="bull_market_analyst",
                description="Identifies upside and growth opportunities",
                agent_card="http://localhost:8002/.well-known/agent-card.json",
            )

            # ---- Orchestrator LLM Agent ----
            from google.adk.models.lite_llm import LiteLlm
            self.agent = LlmAgent(
                name="trading_strategy_orchestrator",
                model=LiteLlm(model="gpt-4o"),
                instruction="""
                You are a trading strategy orchestrator.
                Use the bull and bear agents to analyze opportunities and risks,
                then synthesize a balanced investment recommendation.
                """,
                tools=[
                    AgentTool(agent=remote_bear),
                    AgentTool(agent=remote_bull),
                ],
            )

        if self.runner is None:
            self.runner = Runner(
                app_name=self.agent.name,
                agent=self.agent,
                session_service=InMemorySessionService(),
            )

        self.bull_agent = await self.build_remote_agent(
            name="bull_market_analyst",
            card_url="http://localhost:8002/.well-known/agent-card.json",
            url="http://localhost:8002"
        )
        
        self.bear_agent = await self.build_remote_agent(
            name="bear_risk_analyst",
            card_url="http://localhost:18001/.well-known/agent-card.json",
            url="http://localhost:18001"
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())


    async def decide_agents_with_llm(
        self,
        runner: Runner,
        session_id: str,
        user_id: str,
        query: str,
    ) -> list[str]:
        tracer = trace.get_tracer("orchestrator")
        
        # Create a span for the routing decision - this ensures ADK spans are children
        with tracer.start_as_current_span(
            "orchestrator:decide_routing",
            kind=trace.SpanKind.INTERNAL,
        ) as routing_span:
            routing_span.set_attribute("openinference.span.kind", "chain")
            routing_span.set_attribute("input.value", query)
            
            content = types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"""
                            {ROUTER_SYSTEM_PROMPT}
                            
                            User query:
                            {query}
                            """
                    )
                ],
            )
        
            final_event = None

            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )

            if session is None:
                session = await self.runner.session_service.create_session(
                    app_name=self.runner.app_name,
                    user_id=user_id,
                    session_id=session_id,
                )

            # ADK runner.run_async() will create spans internally, and they should inherit
            # the current trace context from routing_span
            async for event in runner.run_async(
                session_id=session_id,
                user_id=user_id,
                new_message=content,
            ):
                if event.is_final_response():
                    final_event = event

            if not final_event:
                routing_span.set_attribute("output.value", "default: ['bull', 'bear']")
                return ["bull", "bear"]  # safe default
            
            raw = "".join(
                part.text
                for part in final_event.content.parts
                if hasattr(part, "text")
            )
            raw = raw.strip("```json")
            raw = raw.strip("```")
            try:
                data = json.loads(raw.strip())
                routing_span.set_attribute("output.value", str(data))
                # agents = data.get("agents", [])
                return data
            except Exception:
                routing_span.set_attribute("output.value", "parse_error: ['bull', 'bear']")
                return ["bull", "bear"]
    
    
    async def call_remote_agent(
        self,
        agent_card,
        session_id: str,
        text: str,
    ):
        tracer = trace.get_tracer("orchestrator")

        with tracer.start_as_current_span(
            f"call_remote_agent:{agent_card.name}",
            kind=trace.SpanKind.CLIENT,
        ) as span:
            span.set_attribute("openinference.span.kind", "tool")
            # Agent Graph edge from orchestrator to remote agent
            node_id = "bull_agent" if "Bull" in agent_card.name else "bear_agent" if "Bear" in agent_card.name else agent_card.name
            span.set_attribute("graph.node.id", node_id)
            span.set_attribute("graph.node.parent_id", "orchestrator")
            span.set_attribute("graph.node.display_name", agent_card.name)
            print(f"[TRACE] Created call_remote_agent:{agent_card.name} span")

            # Add input attribute
            span.set_attribute("input.value", text)
            span.set_attribute("agent.name", agent_card.name)

            # inject() MUST be inside the span context to capture the trace context
            headers = {}
            inject(headers)

            # Debug: Verify headers were populated
            print(f"[TRACE] Injected traceparent: {headers.get('traceparent')}")

            async with httpx.AsyncClient(timeout=httpx.Timeout(500), headers=headers) as httpx_client:
                client = A2AClient(
                    httpx_client=httpx_client,
                    agent_card=agent_card,
                )

                request = SendMessageRequest(
                    id=str(session_id),
                    params=MessageSendParams(
                        message=new_agent_text_message(
                            text=text,
                            context_id=str(session_id),
                        ),
                        metadata={
                            # Fallback: also pass trace context in metadata
                            "traceparent": headers.get("traceparent", ""),
                            "tracestate": headers.get("tracestate", ""),
                        },
                    ),
                )

                response = await client.send_message(request)

                task = response.root.result

                texts = []

                for artifact in task.artifacts or []:
                    for part in artifact.parts or []:
                        root = part.root
                        if hasattr(root, "text") and root.text:
                            texts.append(root.text)

                result = "\n".join(texts)
                # Add output attribute
                span.set_attribute("output.value", result)
                return result
    
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        await self._init_agent()

        if not context.message:
            return

        tracer = trace.get_tracer("orchestrator")

        # Check if there's a parent context (from middleware)
        current_span = trace.get_current_span()
        current_ctx = current_span.get_span_context()
        print(f"[TRACE] Orchestrator execute: current context trace_id={format(current_ctx.trace_id, '032x')}, valid={current_ctx.is_valid}")

        # Create span for entire orchestration - will be child of middleware span if context is valid
        # Use INTERNAL kind since middleware already created SERVER span
        with tracer.start_as_current_span(
            "orchestrator:execute",
            kind=trace.SpanKind.INTERNAL,
        ) as root_span:
            root_span.set_attribute("openinference.span.kind", "chain")
            # Agent Graph metadata (execution stage)
            root_span.set_attribute("graph.node.id", "orchestrator.execute")
            root_span.set_attribute("graph.node.parent_id", "orchestrator")
            root_span.set_attribute("graph.node.display_name", "Orchestrator Execute")
            span_ctx = root_span.get_span_context()
            print(f"[TRACE] Created orchestrator:execute span trace_id={format(span_ctx.trace_id, '032x')}")
            session_id = context.context_id
            query = context.get_user_input()

            # Add input attributes for observability
            root_span.set_attribute("input.value", query)
            root_span.set_attribute("session.id", session_id)

            user_id = (
                context.message.metadata.get("user_id", "a2a_user")
                if context.message.metadata
                else "a2a_user"
            )

            def _truncate(value: str, limit: int = 2000) -> str:
                if value is None:
                    return ""
                value = str(value)
                if len(value) <= limit:
                    return value
                return value[:limit] + "...<truncated>"

            # Log trace info for debugging
            span_context = root_span.get_span_context()
            print(f"[TRACE] Root span trace_id: {format(span_context.trace_id, '032x')}")

            updater = TaskUpdater(event_queue, context.task_id, session_id)

            if not hasattr(context, "current_task") or not context.current_task:
                await updater.submit()

            await updater.start_work()

            try:
                await updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("Deciding routing strategy..."),
                )

                # 🧠 STEP 1: LLM routing decision
                agents = await self.decide_agents_with_llm(
                    runner=self.runner,
                    session_id=session_id,
                    user_id=user_id,
                    query=query,
                )
                print("agent decided : ", agents)

                results = {}

                import asyncio

                tasks = {}

                # Handle both list format ['bull', 'bear'] and dict format {"agents": [...]}
                agent_list = []
                if isinstance(agents, list):
                    # Simple list format - use original query for all
                    agent_list = [{"name": a, "query": query} for a in agents]
                elif isinstance(agents, dict):
                    agent_list = agents.get("agents", [])

                agent_inputs = {}

                for agent in agent_list:
                    name = agent.get("name", "").lower() if isinstance(agent, dict) else str(agent).lower()
                    agent_query = agent.get("query", query) if isinstance(agent, dict) else query

                    if "bull" in name:
                        agent_inputs["bull"] = agent_query
                        tasks["bull"] = asyncio.create_task(
                            self.call_remote_agent(
                                agent_card=self.bull_agent,
                                session_id=session_id,
                                text=agent_query,
                            )
                        )

                    elif "bear" in name:
                        agent_inputs["bear"] = agent_query
                        tasks["bear"] = asyncio.create_task(
                            self.call_remote_agent(
                                agent_card=self.bear_agent,
                                session_id=session_id,
                                text=agent_query,
                            )
                        )

                # Run all present agents in parallel
                if tasks:
                    completed = await asyncio.gather(*tasks.values())

                    # Reattach results deterministically
                    for key, value in zip(tasks.keys(), completed):
                        results[key] = value


                final = ""

                if "bull" in results:
                    final += "📈 Bull Analysis\n" + results["bull"] + "\n\n"

                if "bear" in results:
                    final += "⚠️ Bear Analysis\n" + results["bear"] + "\n\n"

                final += (
                    "🧠 Orchestrated View\n"
                    "The recommendation balances opportunity with risk awareness."
                )

                # Add output attribute for observability
                root_span.set_attribute("output.value", final)
                root_span.set_attribute("agent.bull.input", _truncate(agent_inputs.get("bull", "")))
                root_span.set_attribute("agent.bull.output", _truncate(results.get("bull", "")))
                root_span.set_attribute("agent.bear.input", _truncate(agent_inputs.get("bear", "")))
                root_span.set_attribute("agent.bear.output", _truncate(results.get("bear", "")))
                root_span.set_attribute("agent.orchestrator.output", _truncate(final))

                await updater.add_artifact(
                    [TextPart(text=final)],
                    name="trading_strategy",
                )
                await updater.complete()

            except Exception as e:
                await updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message(f"Orchestration failed: {str(e)}"),
                    final=True,
                )


import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import TransportProtocol
from vertexai.preview.reasoning_engines.templates.a2a import create_agent_card


skills = [
        AgentSkill(
            id='bull',
            name='bull analysis',
            description='Identify bullish breakout patterns',
            tags=['momentum', 'screening'],
            examples=['Find opportunities for NVDA'],
        ),
        AgentSkill(
            id='bear',
            name='bear analysis',
            description='find bear analysis, risks',
            tags=['bear', 'risk'],
            examples=['Find risks for nvda'],
        )
    ]

# --- Agent card ---
orchestrator_agent_card = create_agent_card(
    agent_name="trading_strategy_orchestrator",
    description="Coordinates bull and bear agents to form a strategy",
    skills=skills,
)

orchestrator_agent_card.url = "http://localhost:8003"
orchestrator_agent_card.preferred_transport = TransportProtocol.jsonrpc

# --- A2A app ---
app = A2AStarletteApplication(
    agent_card=orchestrator_agent_card,
    http_handler=DefaultRequestHandler(
        agent_executor=TradingOrchestratorExecutor(),
        task_store=InMemoryTaskStore(),
    ),
)

if __name__ == "__main__":
    # Build the Starlette app and add trace context middleware
    starlette_app = app.build()
    starlette_app.add_middleware(TraceContextMiddleware)

    uvicorn.run(
        starlette_app,
        host="127.0.0.1",
        port=8003,
        log_level="info",
    )


