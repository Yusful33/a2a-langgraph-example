"""
A2A Server for the Bear Agent.

This module exposes the Bear Risk Analyst via the A2A protocol,
allowing it to be called by the orchestrator or other agents.

Run with: uvicorn src.bear_agent.server:a2a_app --port 8002
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize tracing BEFORE creating agent
from src.tracing.setup import setup_arize_tracing, instrument_app
setup_arize_tracing("bear-agent")

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from src.bear_agent.agent import create_bear_agent

agent = create_bear_agent()
_app = to_a2a(agent, port=8002)
a2a_app = instrument_app(_app, "bear-agent")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("BEAR_AGENT_PORT", 8002))
    print(f"Starting Bear Agent on port {port}...")
    print(f"Agent Card: http://localhost:{port}/.well-known/agent-card")
    uvicorn.run(a2a_app, host="0.0.0.0", port=port)
