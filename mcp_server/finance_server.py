#!/usr/bin/env python3
"""
Finance MCP Server - Exposes finance tools via MCP protocol.

Run locally:
    python finance_server.py

Run with Docker:
    docker build -t finance-mcp-server .
    docker run -p 8080:8080 finance-mcp-server
"""

import asyncio
import os
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn


# Create the MCP server
server = Server("finance-tools")


# ============================================================================
# FINANCE TOOLS
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return list of available tools."""
    return [
        Tool(
            name="list_top_risks",
            description="Return a list of typical downside risks for a given stock ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., NVDA, AAPL)"
                    }
                },
                "required": ["ticker"]
            }
        ),
        Tool(
            name="recent_volatility_snapshot",
            description="Provide a volatility and technical snapshot for a stock ticker.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        ),
        Tool(
            name="sector_peers",
            description="Return peer tickers for relative comparison and risk analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        ),
        Tool(
            name="get_earnings_calendar",
            description="Get upcoming earnings dates for a stock.",
            inputSchema={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool and return results."""
    
    ticker = arguments.get("ticker", "UNKNOWN").upper()
    
    if name == "list_top_risks":
        risks = [
            "macro slowdown reduces demand",
            "margin compression from pricing pressure",
            "regulatory/export constraints (esp. China)",
            "execution risk on product roadmap",
            "customer concentration risk",
        ]
        result = f"{ticker} TOP RISKS:\n" + "\n".join(f"  - {r}" for r in risks)
        
    elif name == "recent_volatility_snapshot":
        result = f"""{ticker} VOLATILITY SNAPSHOT:
  - 20-day realized vol: ~32%
  - 60-day realized vol: ~28%
  - Beta vs SPY: 1.35
  - Momentum: Mixed (RSI ~55)
  - Key levels: Support -10%, Resistance +8%
  - Watch: Gap-down risk on earnings miss"""
        
    elif name == "sector_peers":
        peers_map = {
            "NVDA": ["AMD", "AVGO", "ASML", "INTC", "QCOM"],
            "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
            "TSLA": ["RIVN", "LCID", "F", "GM"],
        }
        peers = peers_map.get(ticker, ["SPY", "QQQ", "XLK"])
        result = f"{ticker} SECTOR PEERS: {', '.join(peers)}\nUse these for relative valuation and correlation analysis."
        
    elif name == "get_earnings_calendar":
        result = f"""{ticker} EARNINGS CALENDAR:
  - Next earnings: ~30 days out (estimated)
  - Consensus: Beat expected
  - Options IV: Elevated (~45% vs 30% normal)
  - Historical move: +/- 8% on earnings"""
        
    else:
        result = f"Unknown tool: {name}"
    
    return [TextContent(type="text", text=result)]


# ============================================================================
# SSE TRANSPORT (for HTTP-based MCP connections)
# ============================================================================

def create_sse_app():
    """Create Starlette app with SSE transport for MCP."""
    
    sse_transport = SseServerTransport("/messages/")
    
    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )
        return JSONResponse({"status": "disconnected"})
    
    async def handle_health(request):
        return JSONResponse({"status": "healthy", "server": "finance-mcp"})
    
    async def handle_tools(request):
        """List available tools (for debugging)."""
        tools = await list_tools()
        return JSONResponse({
            "tools": [
                {"name": t.name, "description": t.description}
                for t in tools
            ]
        })
    
    return Starlette(
        routes=[
            Route("/health", handle_health),
            Route("/tools", handle_tools),
            Route("/sse", handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ]
    )


if __name__ == "__main__":
    # Allow overriding the listen port (default 8080) via PORT env var.
    port = int(os.environ.get("PORT", "8080"))

    print(f"Starting Finance MCP Server on http://0.0.0.0:{port}")
    print(f"  - SSE endpoint: http://localhost:{port}/sse")
    print(f"  - Health check: http://localhost:{port}/health")
    print(f"  - List tools:   http://localhost:{port}/tools")
    
    app = create_sse_app()
    uvicorn.run(app, host="0.0.0.0", port=port)
