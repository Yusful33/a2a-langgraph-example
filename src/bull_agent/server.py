"""
A2A Server for the Bull Agent (LangGraph).

This server exposes a LangGraph agent via the A2A protocol, demonstrating
interoperability between different agent frameworks (LangGraph + ADK).

Run with: uvicorn src.bull_agent.server:app --port 8001
"""

import os
import sys
import json
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize tracing BEFORE creating agent
from src.tracing.setup import setup_arize_tracing, instrument_app
setup_arize_tracing("bull-agent")

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from langchain_core.messages import HumanMessage

from src.bull_agent.agent import create_bull_agent

# Create the LangGraph agent
agent = create_bull_agent()

# Agent card for A2A discovery
AGENT_CARD = {
    "name": "Bull Opportunity Analyst",
    "description": "LangGraph-based agent that identifies bullish opportunities, momentum signals, and growth catalysts.",
    "url": f"http://localhost:{os.environ.get('BULL_AGENT_PORT', 8001)}/",
    "version": "1.0.0",
    "protocolVersion": "0.3.0",
    "capabilities": {},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [
        {
            "id": "bullish_analysis",
            "name": "Bullish Analysis",
            "description": "Analyzes stocks for bullish momentum, growth catalysts, and breakout patterns",
            "tags": ["llm", "financial_analysis"],
        }
    ],
}


async def agent_card(request):
    """Serve the A2A agent card for discovery."""
    return JSONResponse(AGENT_CARD)


async def handle_message(request):
    """Handle A2A message/send requests."""
    try:
        body = await request.json()

        # Extract message from A2A JSON-RPC format
        params = body.get("params", {})
        message = params.get("message", {})
        parts = message.get("parts", [])

        # Get the text content
        query = ""
        for part in parts:
            if "text" in part:
                query = part["text"]
                break

        if not query:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "No text content in message"}
            })

        # Invoke the LangGraph agent
        result = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Extract the final response
        response_text = ""
        if "messages" in result:
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content") and msg.content:
                    response_text = msg.content
                    break

        # Return A2A response format (matching ADK's to_a2a format exactly)
        context_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        artifact_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "artifacts": [
                    {
                        "artifactId": artifact_id,
                        "parts": [
                            {
                                "kind": "text",
                                "text": response_text
                            }
                        ]
                    }
                ],
                "contextId": context_id,
                "history": [
                    {
                        "contextId": context_id,
                        "kind": "message",
                        "messageId": message_id,
                        "parts": [
                            {
                                "kind": "text",
                                "text": response_text
                            }
                        ],
                        "role": "agent",
                        "taskId": task_id
                    }
                ]
            }
        })

    except Exception as e:
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": body.get("id") if "body" in dir() else None,
            "error": {"code": -32603, "message": str(e)}
        })


# Create Starlette app with routes
routes = [
    Route("/.well-known/agent-card.json", agent_card),
    Route("/", handle_message, methods=["POST"]),
]

_app = Starlette(routes=routes)
a2a_app = instrument_app(_app, "bull-agent")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("BULL_AGENT_PORT", 8001))
    print(f"Starting Bull Agent (LangGraph) on port {port}...")
    print(f"Agent Card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(a2a_app, host="0.0.0.0", port=port)
