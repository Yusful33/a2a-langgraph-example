"""
CLI Interface for the Financial Multi-Agent System.

This CLI allows users to query the orchestrator for balanced financial analysis.
Traces are automatically sent to Arize Cloud for observability.

Usage:
    python -m src.cli.main --symbol NVDA
    python -m src.cli.main --query "What are the risks for AAPL?"
    python -m src.cli.main  # Interactive mode
"""

import os
import sys
import asyncio
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(override=True)

import typer
import httpx
from opentelemetry import trace
from opentelemetry.propagate import inject
from opentelemetry.trace import SpanKind

from src.tracing.setup import setup_arize_tracing

app = typer.Typer(
    name="financial-agents",
    help="Multi-Agent Financial Analysis CLI with A2A Protocol",
    add_completion=False,
)


async def query_orchestrator(query: str, orchestrator_url: str) -> dict:
    """
    Send a query to the orchestrator with trace context propagation.

    Args:
        query: The user's financial query
        orchestrator_url: URL of the orchestrator's A2A endpoint

    Returns:
        The orchestrator's response
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(
        "cli",
        kind=SpanKind.CLIENT,
    ) as span:
        span.set_attribute("query", query)

        # Log trace ID for verification
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, '032x')
        print(f"  [cli] trace={trace_id[:8]}...")

        # Prepare headers with trace context injection
        headers = {"Content-Type": "application/json"}
        inject(headers)  # Inject W3C traceparent header

        # A2A message format
        import uuid
        message_id = str(uuid.uuid4())

        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": "1",
            "params": {
                "message": {
                    "messageId": message_id,
                    "parts": [{"text": query}],
                    "role": "user",
                }
            }
        }

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    orchestrator_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()

                # Extract response text from A2A result
                if "result" in result:
                    task = result["result"]
                    if "artifacts" in task and task["artifacts"]:
                        for artifact in task["artifacts"]:
                            if "parts" in artifact:
                                for part in artifact["parts"]:
                                    if "text" in part:
                                        span.set_attribute("response.length", len(part["text"]))
                                        return {"success": True, "content": part["text"]}

                    # Check for message in status
                    if "status" in task and "message" in task["status"]:
                        msg = task["status"]["message"]
                        if "parts" in msg:
                            for part in msg["parts"]:
                                if "text" in part:
                                    return {"success": True, "content": part["text"]}

                return {"success": True, "content": json.dumps(result, indent=2)}

        except httpx.TimeoutException:
            span.set_attribute("error", "timeout")
            return {"success": False, "error": "Request timed out. The analysis is taking longer than expected."}
        except httpx.HTTPStatusError as e:
            span.set_attribute("error", str(e))
            return {"success": False, "error": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            span.record_exception(e)
            return {"success": False, "error": str(e)}


def print_header():
    """Print the CLI header."""
    typer.echo("\n" + "=" * 60)
    typer.echo("  Financial Multi-Agent Analysis System")
    typer.echo("  Bull Agent (LangGraph) + Bear Agent (ADK) + Orchestrator")
    typer.echo("=" * 60 + "\n")


def print_footer(trace_id: str = None):
    """Print the CLI footer with trace info."""
    typer.echo("\n" + "-" * 60)
    typer.echo("Trace sent to Arize Cloud")
    if trace_id:
        typer.echo(f"Trace ID: {trace_id}")
    typer.echo("View traces at: https://app.arize.com")
    typer.echo("-" * 60 + "\n")


@app.command()
def analyze(
    symbol: str = typer.Option(
        None,
        "--symbol", "-s",
        help="Stock ticker symbol to analyze (e.g., NVDA, AAPL, TSLA)"
    ),
    query: str = typer.Option(
        None,
        "--query", "-q",
        help="Custom financial query"
    ),
    orchestrator_port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Orchestrator port"
    ),
):
    """
    Run financial analysis through the multi-agent system.

    The orchestrator will consult both the Bull Agent (for opportunities)
    and Bear Agent (for risks), then synthesize a balanced analysis.
    """
    # Initialize tracing for the CLI
    tracer = setup_arize_tracing("financial-cli")

    print_header()

    # Determine the query
    if symbol:
        full_query = f"Provide a comprehensive analysis of {symbol.upper()} stock, including both bullish opportunities and bearish risks."
    elif query:
        full_query = query
    else:
        # Interactive mode
        full_query = typer.prompt("Enter your financial query")

    orchestrator_url = f"http://localhost:{orchestrator_port}/"

    typer.echo(f"Query: {full_query}")
    typer.echo(f"Orchestrator: {orchestrator_url}")
    typer.echo("\nAnalyzing... (this may take a moment as we consult multiple agents)\n")
    typer.echo("-" * 60)

    # Run the async query
    result = asyncio.run(query_orchestrator(full_query, orchestrator_url))

    if result["success"]:
        typer.echo("\n" + result["content"])
    else:
        typer.echo(f"\nError: {result['error']}", err=True)
        raise typer.Exit(1)

    # Get current trace ID for reference
    current_span = trace.get_current_span()
    trace_id = None
    if current_span and current_span.get_span_context().trace_id:
        trace_id = format(current_span.get_span_context().trace_id, '032x')

    print_footer(trace_id)


@app.command()
def health():
    """Check if all agents are running and healthy."""
    import httpx

    print_header()
    typer.echo("Checking agent health...\n")

    agents = [
        ("Orchestrator", "http://localhost:8000/.well-known/agent-card"),
        ("Bull Agent", "http://localhost:8001/.well-known/agent-card"),
        ("Bear Agent", "http://localhost:8002/.well-known/agent-card"),
    ]

    all_healthy = True
    for name, url in agents:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                typer.echo(f"  [OK] {name} is running at {url}")
            else:
                typer.echo(f"  [WARN] {name} returned status {response.status_code}")
                all_healthy = False
        except Exception as e:
            typer.echo(f"  [FAIL] {name} is not reachable: {e}")
            all_healthy = False

    typer.echo("")
    if all_healthy:
        typer.echo("All agents are healthy!")
    else:
        typer.echo("Some agents are not running. Start them with: python run_all.py")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
