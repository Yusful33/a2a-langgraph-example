"""
A2A Server for the Orchestrator Agent.

This module exposes the Financial Orchestrator via the A2A protocol.
The orchestrator coordinates between Bull and Bear agents to provide
balanced financial analysis.

Run with: uvicorn src.orchestrator.server:a2a_app --port 8000
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize tracing BEFORE creating agent
from src.tracing.setup import setup_arize_tracing, instrument_app
setup_arize_tracing("orchestrator")

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from src.orchestrator.agent import create_orchestrator

agent = create_orchestrator()
_app = to_a2a(agent, port=8000)
a2a_app = instrument_app(_app, "orchestrator")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("ORCHESTRATOR_PORT", 8000))
    print(f"Starting Financial Orchestrator on port {port}...")
    print(f"Agent Card: http://localhost:{port}/.well-known/agent-card")
    print("\nMake sure Bull Agent (port 8001) and Bear Agent (port 8002) are running!")
    uvicorn.run(a2a_app, host="0.0.0.0", port=port)
