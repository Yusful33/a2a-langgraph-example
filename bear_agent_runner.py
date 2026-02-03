#!/usr/bin/env python
# coding: utf-8

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

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
import json
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
from openinference.instrumentation import using_attributes

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
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import extract
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from src.tracing import setup_tracing

# Set up tracing without filtering (export all spans)
_bear_tracer_provider = setup_tracing("bear-agent", filter_spans=False)
print(f"[TRACE] Bear agent using ARIZE_PROJECT_NAME={os.environ.get('ARIZE_PROJECT_NAME', 'NOT SET')}")

# Enable LangChain instrumentation for LLM call and tool tracing
from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument(tracer_provider=_bear_tracer_provider)
print("[TRACE] Bear agent: LangChainInstrumentor enabled")


# In[3]:


def create_bear_agent_card():
    """Create A2A Agent Card for Bear Risk Analyst."""

    # Define the agent's capabilities as A2A skills
    skills = [
        AgentSkill(
            id="risk_analysis",
            name="Risk Factor Scanner",
            description="Identifies potential downside catalysts and risk factors",
            tags=["Risk-Analysis", "Market-Analysis"],
            examples=[
                "What are the key risks for NVDA?",
                "Analyze downside catalysts for tech stocks",
            ],
        ),
        AgentSkill(
            id="divergence_detection",
            name="Divergence Detection",
            description="Finds bearish divergences and technical weakness signals",
            tags=["Technical-Analysis", "Divergence"],
            examples=[
                "Find bearish divergences in AAPL",
            ],
        ),
        AgentSkill(
            id="exit_signals",
            name="Exit Signal Monitoring",
            description="Tracks distribution patterns and exit signals",
            tags=["Exit-Strategy", "Risk-Management"],
            examples=[
                "Monitor exit signals for NVDA",
            ],
        ),
    ]

    # Create A2A agent card for capability advertisement
    return create_agent_card(
        agent_name="Bear Risk Analyst (Langgraph + MCP)",
        description=(
            "A cautious risk analyst powered by Langgraph, "
            "focused on identifying downside catalysts and warning signals."
        ),
        skills=skills,
    )

# Generate the agent card
bear_agent_card = create_bear_agent_card()


# In[4]:


print("Bear Agent Card:")
print(f"   Name: {bear_agent_card.name}")
print(f"   Skills: {len(bear_agent_card.skills)}")


# In[5]:


from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, tool
from langgraph.prebuilt import create_react_agent

from langchain_core.prompts import ChatPromptTemplate

from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core.messages import HumanMessage, BaseMessage
from openinference.instrumentation import using_session

# ============================================================================
# MCP TOOL INTEGRATION
# ============================================================================
# MCP Server URL - change this to your MCP server address
# Default matches the local finance MCP server (mcp_server/finance_server.py)
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8080/sse")

# Global MCP client (reused across requests)
_mcp_client = None
_mcp_tools = None


async def load_mcp_tools():
    """
    Load tools from the Finance MCP Server.
    
    The tools returned are LangChain-compatible and will be automatically
    instrumented by LangChainInstrumentor - you'll see Tool spans in Arize.
    """
    global _mcp_client, _mcp_tools
    
    if _mcp_tools is not None:
        return _mcp_tools
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        print(f"[MCP] Connecting to MCP server at {MCP_SERVER_URL}...")
        
        # Connect to the Finance MCP server via SSE
        _mcp_client = MultiServerMCPClient(
            {
                "finance": {
                    "url": MCP_SERVER_URL,
                    "transport": "sse",
                }
            }
        )
        
        # Start the client connection
        await _mcp_client.__aenter__()
        
        # Get tools from MCP server - these are LangChain-compatible
        _mcp_tools = _mcp_client.get_tools()
        
        print(f"[MCP] Successfully loaded {len(_mcp_tools)} tools from MCP server:")
        for t in _mcp_tools:
            print(f"[MCP]   - {t.name}: {t.description[:60]}...")
        
        return _mcp_tools
        
    except ImportError as e:
        print(f"[MCP] langchain-mcp-adapters not installed: {e}")
        return None
    except Exception as e:
        print(f"[MCP] Failed to connect to MCP server at {MCP_SERVER_URL}: {e}")
        print("[MCP] Make sure the MCP server is running:")
        print("[MCP]   docker run -p 8080:8080 finance-mcp-server")
        print("[MCP] Or: python mcp_server/finance_server.py")
        return None


def _wrap_tools_with_tracing(tools):
    """Wrap tools to ensure we emit tool spans even if upstream adapters don't."""
    tracer = trace_api.get_tracer("bear-agent")
    wrapped = []

    for t in tools:
        name = getattr(t, "name", "tool")
        description = getattr(t, "description", "")

        async def _acall(args, _tool=t, _name=name):
            with tracer.start_as_current_span(
                f"tool:{_name}",
                kind=trace_api.SpanKind.CLIENT,
            ) as span:
                span.set_attribute("openinference.span.kind", "tool")
                span.set_attribute("tool.name", _name)
                try:
                    span.set_attribute("input.value", json.dumps(args))
                except Exception:
                    span.set_attribute("input.value", str(args))
                try:
                    result = await _tool.ainvoke(args)
                    span.set_attribute("output.value", str(result))
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        # LangChain Tool wrapper preferring async; sync path delegates to async
        def _call(args, _ac=_acall):
            return asyncio.get_event_loop().run_until_complete(_ac(args))

        wrapped.append(
            Tool.from_function(
                func=_call,
                coroutine=_acall,
                name=name,
                description=description,
            )
        )

    return wrapped


# --- Fallback: Local demo tools (used if MCP server unavailable) ---
@tool
def list_top_risks(ticker: str) -> str:
    """Return a list of typical downside risks for a given stock ticker."""
    ticker = ticker.upper()
    return f"{ticker} risks (LOCAL): macro slowdown; margin compression; regulatory risk; execution risk."


@tool
def recent_volatility_snapshot(ticker: str) -> str:
    """Provide a volatility/technical snapshot for the ticker."""
    ticker = ticker.upper()
    return f"{ticker} vol (LOCAL): 20d ~32%, 60d ~28%, beta ~1.3."


@tool
def sector_peers(ticker: str) -> str:
    """Return peer tickers for relative comparison."""
    ticker = ticker.upper()
    peers = {"NVDA": "AMD, AVGO, ASML, INTC", "AAPL": "MSFT, GOOGL"}
    return f"Peers for {ticker} (LOCAL): {peers.get(ticker, 'SPY, QQQ')}"


class TraceContextMiddleware(BaseHTTPMiddleware):
    """Middleware to extract trace context from incoming HTTP requests."""

    async def dispatch(self, request: Request, call_next):
        headers = dict(request.headers)
        traceparent = headers.get('traceparent')

        # Skip tracing for agent card requests
        if '/.well-known/' in str(request.url):
            return await call_next(request)

        # Debug: Log received traceparent
        print(f"[TRACE] Bear agent received traceparent: {traceparent}")

        if traceparent:
            # Parse traceparent manually: 00-{trace_id}-{parent_span_id}-{flags}
            parts = traceparent.split('-')
            if len(parts) == 4:
                received_trace_id = parts[1]
                received_span_id = parts[2]
                received_flags = parts[3]
                print(f"[TRACE] Bear: parsed trace_id={received_trace_id}, parent_span_id={received_span_id}")

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

                print(f"[TRACE] Bear: manually created SpanContext valid={parent_span_context.is_valid}")

                # Create a NonRecordingSpan with the parent context and set it in context
                parent_span = NonRecordingSpan(parent_span_context)
                parent_context = set_span_in_context(parent_span)

                # Create span as child of parent context
                tracer = trace_api.get_tracer("bear-agent")
                with tracer.start_as_current_span(
                    "bear_agent:handle_request",
                    context=parent_context,
                    kind=trace_api.SpanKind.SERVER,
                ) as span:
                    span.set_attribute("openinference.span.kind", "agent")
                    # Agent Graph metadata for Arize
                    span.set_attribute("graph.node.id", "bear_agent")
                    span.set_attribute("graph.node.parent_id", "orchestrator")
                    span.set_attribute("graph.node.display_name", "Bear Agent")
                    # Debug: verify trace ID matches
                    span_ctx = span.get_span_context()
                    created_trace_id = format(span_ctx.trace_id, '032x')
                    print(f"[TRACE] Bear: created span trace_id={created_trace_id}, matches={created_trace_id == received_trace_id}")
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


class BearAgentExecutor(AgentExecutor):
    """Agent executor for A2A integration with Bear Agent (LangGraph version)."""

    def __init__(self):
        self.graph = None
        self.register = None

    async def _init_agent(self):
        """Initialize LangGraph Bear Agent with MCP tools + Arize tracing."""

        if self.register is None:
            # TracerProvider is set up at module level with filtered exporter.
            # Auto-instrumentation is DISABLED to reduce noise.
            # Only our explicit spans (bear_agent:handle_request) are exported.
            self.register = True

        if self.graph is None:
            # -------------------------------
            # OpenAI Model
            # -------------------------------
            import os
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )

            # Try to load tools from MCP server first
            mcp_tools = await load_mcp_tools()
            
            if mcp_tools:
                # Use MCP tools - automatically instrumented by LangChainInstrumentor
                tools = _wrap_tools_with_tracing(mcp_tools)
                print(f"[TRACE] Bear agent using {len(tools)} MCP tools")
            else:
                # Fall back to local demo tools
                tools = [list_top_risks, recent_volatility_snapshot, sector_peers]
                print(f"[TRACE] Bear agent using {len(tools)} local fallback tools")

            agent = create_react_agent(
                model=llm,
                tools=tools,
            )

            # -------------------------------
            # LangGraph State
            # -------------------------------


            class BearState(TypedDict):
                messages: List[BaseMessage]


            async def agent_node(state: BearState) -> BearState:
                result = await agent.ainvoke(state)
                return {
                        "messages": result["messages"]
                    }

            graph = StateGraph(BearState)
            graph.add_node("bear_agent", agent_node)
            graph.set_entry_point("bear_agent")
            graph.add_edge("bear_agent", END)

            self.graph = graph.compile()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute Bear Agent analysis."""

        if self.graph is None:
            await self._init_agent()

        session_id = context.context_id

        # NOTE: Context extraction is handled by TraceContextMiddleware.
        # Do NOT extract context here - it causes conflicting context scopes.
        # The middleware already attaches the parent context from HTTP headers.

        # Get current span to add attributes
        current_span = trace_api.get_current_span()

        with using_attributes(session_id=session_id):

            query = context.get_user_input()

            # Add input attribute to current span
            if current_span and current_span.is_recording():
                current_span.set_attribute("input.value", query)
                current_span.set_attribute("session.id", session_id)

            initial_state = {
                "messages": [HumanMessage(content=query)]
            }
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)

            if not hasattr(context, "current_task") or not context.current_task:
                await updater.submit()

            await updater.start_work()

            try:
                await updater.update_status(
                    TaskState.working,
                    message=new_agent_text_message("Analyzing risks...")
                )

                # Invoke LangGraph - spans should inherit from bear_agent:handle_request middleware span
                # LangChain instrumentation will create spans that should be children of the current context
                # with using_session(session_id=session_id):
                result = await self.graph.ainvoke(initial_state)
                final_answer = result["messages"][-1].content
                
                response = f"""
                    BEAR RISK ANALYSIS
                    {'=' * 50}

                    {final_answer}

                    Analysis completed
                    """

                # Add output attribute to current span
                if current_span and current_span.is_recording():
                    current_span.set_attribute("output.value", response)

                await updater.add_artifact(
                    [TextPart(text=response)],
                    name="risk_analysis"
                )
                await updater.complete()

            except Exception as e:
                await updater.update_status(
                    TaskState.failed,
                    message=new_agent_text_message(f"Analysis failed: {str(e)}"),
                )

bear_agent_card.url = "http://localhost:8001"
bear_agent_card.preferred_transport = TransportProtocol.jsonrpc


# In[7]:


def create_bear_a2a_server(agent_card):
    """Create A2A server for Pydantic AI Bear Agent.

    Since Bear Agent uses Pydantic AI (not ADK), we create the A2A server
    directly using the BearAgentExecutor we defined earlier.
    """
    request_handler = DefaultRequestHandler(
        agent_executor=BearAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )

async def run_bear_server(agent_card, port):
    """Run Bear Agent A2A server (Pydantic AI)."""
    app = create_bear_a2a_server(agent_card)

    # Build the Starlette app and add trace context middleware
    starlette_app = app.build()
    starlette_app.add_middleware(TraceContextMiddleware)

    config = uvicorn.Config(
        starlette_app,
        host='127.0.0.1',
        port=port,
        log_level='warning',
        loop='none',
    )

    server = uvicorn.Server(config)
    await server.serve()


# In[8]:


async def start_a2a_servers():
    """Start both Bear and Bull agents as A2A servers."""
    # Create tasks for both servers
    # Bear Agent uses Pydantic AI, so it needs custom A2A server
    # Bull Agent uses ADK, so it uses the standard ADK A2A pattern
    tasks = [
        asyncio.create_task(
            run_bear_server(bear_agent_card, 8001)
        )
    ]

    # Give servers time to start
    await asyncio.sleep(2)
    print("   ✓ Bear Agent A2A server: http://127.0.0.1:8001 (langgraph)")

    # Keep servers running
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("Shutting down A2A servers...")

if __name__ == "__main__":
    asyncio.run(start_a2a_servers())
