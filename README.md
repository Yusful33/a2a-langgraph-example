# zscaler agents + MCP demo

Minimal repo for running:
- Finance MCP server (simple risk/volatility/peers tools)
- Bear agent (LangGraph + MCP tools, explicit tool spans)
- Bull agent (ADK)
- Orchestrator (ADK router)
- CLI client that fans out to both agents and aggregates

## Layout
- `mcp_server/` – finance MCP server (SSE on `PORT`, default 8080)
- `bear_agent_runner.py` – LangGraph bear agent, consumes MCP tools
- `bull_agent_runner.py` – ADK bull agent (no tools)
- `orchestrator_runner.py` – ADK router calling bull and bear
- `client.py` – sends a combined bull/bear request
- `run.sh` – activates `.venv313` and runs a Python target
- `src/tracing.py` – OpenTelemetry + Arize exporter helper

## Prereqs
- Python 3.11+ (matches base image)
- `.venv313` with dependencies installed (install your stack as needed)
- Env vars for tracing/LLMs:
  - `OPENAI_API_KEY` (bear agent LLM)
  - `ARIZE_SPACE_ID`, `ARIZE_API_KEY`, `ARIZE_PROJECT_NAME` (optional tracing export; no defaults baked in)
  - `MCP_SERVER_URL` (bear defaults to `http://localhost:10080/sse`; override to your host/port)
  - Any Google/Vertex creds needed by your ADK flows if you add them

## Quick start (local)
```bash
# from repo root
python -m venv .venv313
source .venv313/bin/activate
pip install -r mcp_server/requirements.txt  # MCP server deps; add your own agent deps as needed

# start MCP (choose a free port, e.g., 10080)
PORT=10080 ./run.sh -m uvicorn mcp_server.finance_server:create_sse_app --host 0.0.0.0 --port 10080

# new shell: start bear (point to MCP URL; required)
MCP_SERVER_URL=http://localhost:10080/sse ./run.sh bear_agent_runner.py

# new shell: start bull
./run.sh bull_agent_runner.py

# new shell: start orchestrator
./run.sh orchestrator_runner.py

# new shell: run client
./run.sh client.py
```

## Docker for MCP server
```bash
docker build -t finance-mcp-server mcp_server
# remap host port to avoid conflicts; container listens on $PORT or 8080 default
docker run --rm -e PORT=10080 -p 10080:10080 finance-mcp-server
export MCP_SERVER_URL=http://localhost:10080/sse
```

## Observability notes
- Bear agent wraps MCP tools to emit `tool:<name>` spans with `openinference.span.kind=tool`, input/output attributes.
- Trace context is propagated end-to-end (client → orchestrator → bull/bear). One client run should equal one trace.
- Arize export requires `ARIZE_SPACE_ID` and `ARIZE_API_KEY` env vars; no secrets are hardcoded.

## Listing available tools
```bash
curl http://localhost:10080/tools    # adjust host/port as needed
```

## Safety / secrets
- No default Arize creds are included; set them via env.
- `.gitignore` was removed when syncing; add one if you need to exclude local artifacts (e.g., `.venv*`, `.env`).
