"""
Bull Agent - Opportunity-focused financial analyst using LangGraph with Claude.

This agent uses LangGraph (different from the ADK-based agents) to demonstrate
A2A protocol interoperability between different agent frameworks.
"""

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from src.bull_agent.tools import (
    momentum_screener,
    growth_catalyst_finder,
    breakout_pattern_finder,
)

BULL_SYSTEM_PROMPT = """You are an optimistic, growth-focused financial analyst known as the "Bull Opportunity Analyst".

Your role is to identify opportunities, positive catalysts, and bullish signals. You take an enthusiastic but data-driven approach.

ALWAYS use your tools to gather data before providing analysis:
- momentum_screener: Analyze technical momentum signals
- growth_catalyst_finder: Identify positive catalysts
- breakout_pattern_finder: Find technical breakout setups

Present opportunities with clear reasoning and price targets."""


def create_bull_agent():
    """
    Create the Bull Agent using LangGraph's create_react_agent.

    This demonstrates LangGraph working alongside ADK agents via A2A protocol.
    """
    model = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
    )

    tools = [momentum_screener, growth_catalyst_finder, breakout_pattern_finder]

    agent = create_react_agent(
        model,
        tools=tools,
        prompt=BULL_SYSTEM_PROMPT,
    )

    return agent
