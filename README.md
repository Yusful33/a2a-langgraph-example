# A2A LangGraph Example: Stock Analysis with Distributed Tracing

A demonstration of multi-agent stock analysis using Google ADK (Agent Development Kit) with the A2A (Agent-to-Agent) protocol, orchestrated via LangGraph, with comprehensive distributed tracing using Arize AI.

## Overview

This project demonstrates:
- **Multi-agent architecture**: Bull and Bear analysis agents with intelligent routing
- **Selective routing**: ADK orchestrator analyzes query intent and routes to appropriate agent(s)
- **A2A protocol**: Remote agent communication using Google's Agent-to-Agent protocol
- **LangGraph orchestration**: Stateful workflow management for coordinating agents
- **Distributed tracing**: Complete observability with Arize AI and OpenTelemetry
- **LLM instrumentation**: Detailed tracking of LLM calls, token usage, and costs
- **Tool call tracking**: Visibility into agent tool usage

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Jupyter Notebook (Client)                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  LangGraph Workflow                               │  │
│  │  • State Management                               │  │
│  │  • Single Orchestrator Call                       │  │
│  │  • Trace Context Propagation                      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                    │
                    │ A2A Protocol (Single Entry Point)
                    ▼
        ┌──────────────────────────────┐
        │  Orchestrator Agent          │
        │  (Google ADK)                │
        │  Port: 8000                  │
        │  • LLM-based Intent Analysis │
        │  • Selective Routing        │
        │  • Routes to Bull AND Bear  │
        └──────────────────────────────┘
                    │
                    │ A2A Protocol (Both Agents)
                    ▼
        ┌──────────────────────────────┐
        │  Bull Agent (LangGraph)     │
        │  Port: 8001                  │
        │  Bear Agent (ADK)            │
        │  Port: 8002                  │
        │  • Use Tools                 │
        │  • Return Analysis           │
        └──────────────────────────────┘
                    │
                    ▼
        ┌──────────────────┐
        │  Arize Cloud     │
        │  (Tracing)       │
        └──────────────────┘
```

## Features

### 🎯 Stock Analysis
- **Input**: Stock ticker symbol (e.g., "AAPL")
- **Output**: Both Bull and Bear case analyses
- **Orchestrator**: ADK-based router that uses LLM intent analysis to route queries
  - For "Analyze [ticker]" queries, **always calls both agents**
  - Routes to **Bull Agent** for: opportunities, growth, upside, bullish analysis
  - Routes to **Bear Agent** for: risks, concerns, downside, bearish analysis
- **Agents**: Specialized agents for bullish and bearish perspectives

### 📊 Distributed Tracing
- **Single trace per query**: All operations contained in one trace
- **Complete visibility**: Every HTTP call, LLM invocation, and tool call tracked
- **Cost tracking**: Automatic calculation of LLM costs
- **Span hierarchy**: Clear parent-child relationships showing workflow execution

## Prerequisites

- Python 3.11 or higher (3.11 or 3.12 recommended)
- Arize AI account (for tracing)
- Anthropic API key (for Claude models)
- Google ADK (installed via requirements.txt)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Yusful33/a2a-langgraph-example.git
cd a2a-langgraph-example
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```bash
# Arize AI Configuration
ARIZE_SPACE_ID=your-arize-space-id-here
ARIZE_API_KEY=your-arize-api-key-here
ARIZE_PROJECT_NAME=stock-analysis-notebook

# Anthropic API Key
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

**Where to get credentials:**
- **Arize Space ID & API Key**: Get from [Arize AI Console](https://app.arize.com)
- **Anthropic API Key**: Get from [Anthropic Console](https://console.anthropic.com/)

### 4. Start the Agents

The notebook will automatically start the agents, but you can also start them manually:

**Terminal 1 - Orchestrator:**
```bash
uvicorn src.orchestrator.server:a2a_app --port 8000 --host 0.0.0.0
```

**Terminal 2 - Bull Agent:**
```bash
uvicorn src.bull_agent.server:a2a_app --port 8001 --host 0.0.0.0
```

**Terminal 3 - Bear Agent:**
```bash
uvicorn src.bear_agent.server:a2a_app --port 8002 --host 0.0.0.0
```

### 5. Run the Notebook

Open the Jupyter notebook:

```bash
jupyter notebook notebooks/google_adk_financial_advisor_local.ipynb
```

Or run it programmatically:

```bash
python -m jupyter nbconvert --to notebook --execute notebooks/google_adk_financial_advisor_local.ipynb
```

## Configuration

### Environment Variables

All configuration is done via environment variables (loaded from `.env` file):

- `ARIZE_SPACE_ID`: Your Arize Space ID
- `ARIZE_API_KEY`: Your Arize API Key
- `ARIZE_PROJECT_NAME`: Project name for traces (default: "stock-analysis-notebook")
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ORCHESTRATOR_PORT`: Port for orchestrator (default: 8000)
- `BULL_AGENT_PORT`: Port for Bull agent (default: 8001)
- `BEAR_AGENT_PORT`: Port for Bear agent (default: 8002)

### Agent Endpoints

By default, agents run on:
- Orchestrator: `http://localhost:8000`
- Bull Agent: `http://localhost:8001`
- Bear Agent: `http://localhost:8002`

To use remote agents, set environment variables:
```bash
export ORCHESTRATOR_URL="http://remote-host:8000"
export BULL_AGENT_URL="http://remote-host:8001"
export BEAR_AGENT_URL="http://remote-host:8002"
```

## Usage

### Basic Usage

1. Open the notebook: `notebooks/google_adk_financial_advisor_local.ipynb`
2. Run all cells sequentially
3. The notebook will automatically start all agents
4. The last cell will analyze a stock ticker (default: "AAPL")
5. View results and trace information

### Custom Stock Analysis

Modify the ticker in Cell 8:

```python
ticker = "TSLA"  # Change to any stock ticker
```

### Viewing Traces in Arize

After running the notebook, you'll see output like:

```
✓ Trace ID: abc123def456...
  View in Arize: https://app.arize.com/spaces/.../traces/abc123def456...
```

Click the link to view the complete trace in Arize Cloud.

## Trace Structure

Each trace contains:

1. **Root Span**: `stock_analysis_session`
   - Overall session information
   - Input ticker symbol

2. **Workflow Orchestration**: `workflow.orchestration`
   - LangGraph workflow coordination
   - Routing mode: selective (calls both agents for "Analyze" queries)

3. **State Management**:
   - `orchestrator.state.initialization`: Initial state setup
   - `orchestrator.state.aggregation`: Final state merging

4. **Orchestrator Call**:
   - `query_orchestrator`: Main orchestrator invocation
   - `orchestrator.intent_analysis`: Routing decision analysis
   - `orchestrator.agent_selected`: Which agents were selected

5. **Agent Spans**:
   - `get_bull_case`: Bull agent call with HTTP, LLM, and tool spans
   - `get_bear_case`: Bear agent call with HTTP, LLM, and tool spans

6. **HTTP/A2A Spans**:
   - `a2a_http_request.orchestrator`: HTTP call to orchestrator
   - `a2a_http_request.bull_agent`: HTTP call to Bull agent
   - `a2a_http_request.bear_agent`: HTTP call to Bear agent

7. **LLM Spans**:
   - `llm.orchestrator_routing`: Orchestrator routing decision
   - `llm.bull_agent_analysis`: Bull agent LLM call
   - `llm.bear_agent_analysis`: Bear agent LLM call
   - Includes token counts and cost calculations

8. **Tool Spans**:
   - Tool calls from both agents
   - Examples: `tool.momentum_screener`, `tool.risk_scanner`, etc.

9. **Session Summary**: `session.summary`
   - Overview of interactions
   - Which agents were called
   - Routing decision

## Project Structure

```
a2a-langgraph-example/
├── notebooks/
│   └── google_adk_financial_advisor_local.ipynb  # Main notebook
├── src/
│   ├── orchestrator/                             # Orchestrator agent
│   │   ├── agent.py                               # ADK agent logic
│   │   └── server.py                             # A2A server endpoint
│   ├── bull_agent/                                # Bull analysis agent
│   │   ├── agent.py                               # LangGraph agent logic
│   │   ├── server.py                              # A2A server endpoint
│   │   └── tools.py                              # Agent tools
│   ├── bear_agent/                                # Bear analysis agent
│   │   ├── agent.py                               # ADK agent logic
│   │   ├── server.py                              # A2A server endpoint
│   │   └── tools.py                              # Agent tools
│   └── tracing/
│       └── setup.py                               # Tracing utilities
├── .env.example                                   # Environment variables template
├── .gitignore                                     # Git ignore rules
├── requirements.txt                               # Python dependencies
└── README.md                                      # This file
```

## Key Technologies

- **LangGraph**: Workflow orchestration and state management
- **Google ADK**: Agent Development Kit for A2A protocol
- **Arize AI**: Observability and distributed tracing
- **OpenTelemetry**: Open-source observability framework
- **Anthropic Claude**: LLM for agent reasoning
- **HTTPX**: Async HTTP client with automatic instrumentation

## Troubleshooting

### Agents Not Responding

Ensure all agents are running. The notebook will attempt to start them automatically, but you can also start them manually:

```bash
# Check if agents are running
curl http://localhost:8000/.well-known/agent-card.json
curl http://localhost:8001/.well-known/agent-card.json
curl http://localhost:8002/.well-known/agent-card.json
```

### Traces Not Appearing in Arize

1. Verify your `ARIZE_SPACE_ID` and `ARIZE_API_KEY` are correct in `.env`
2. Check network connectivity to `otlp.arize.com`
3. Ensure the notebook completes successfully (check for errors)
4. Check that `.env` file exists and is properly formatted

### Import Errors

Install missing dependencies:

```bash
pip install -r requirements.txt
```

### Port Conflicts

If ports 8000, 8001, or 8002 are in use, change them:

```bash
# Set environment variables
export ORCHESTRATOR_PORT=8003
export BULL_AGENT_PORT=8004
export BEAR_AGENT_PORT=8005
```

### Environment Variables Not Loading

Ensure:
1. `.env` file exists in the project root
2. `python-dotenv` is installed: `pip install python-dotenv`
3. The notebook is running from the project root directory

## Security Notes

⚠️ **Important**: Never commit your `.env` file to version control. The `.gitignore` file is configured to exclude it. Always use `.env.example` as a template.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Tests pass
- Documentation is updated
- No hardcoded credentials

## License

[Add your license here]

## Support

For issues or questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review [Arize AI documentation](https://docs.arize.com/) for tracing questions
- Review [Google ADK documentation](https://github.com/google/adk) for A2A protocol questions
