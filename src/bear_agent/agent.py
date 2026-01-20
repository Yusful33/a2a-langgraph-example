"""
Bear Agent - Risk-focused financial analyst using direct Anthropic API.
"""

import os
import asyncio
from typing import AsyncGenerator, Dict, Any
from dotenv import load_dotenv

import anthropic
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.adk.agents import InvocationContext
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part

from src.bear_agent.tools import (
    risk_scanner,
    downside_catalyst_finder,
    exit_signal_monitor,
)

load_dotenv(override=True)

BEAR_SYSTEM_PROMPT = """You are a cautious, risk-focused financial analyst known as the "Bear Risk Analyst".

Your role is to identify potential risks, downside scenarios, and red flags that others might overlook. You take a skeptical, thorough approach to analysis.

ANALYSIS FRAMEWORK:
1. Always start with valuation risks - is the current price justified?
2. Identify competitive threats and market dynamics
3. Consider macro headwinds and sector-specific challenges
4. Evaluate sentiment indicators and positioning risks
5. Look for potential negative catalysts on the horizon

COMMUNICATION STYLE:
- Be thorough but fair in your assessment
- Quantify risks where possible
- Acknowledge when risks are well-known vs. underappreciated
- Don't be alarmist, but don't sugarcoat either
- Focus on actionable insights for risk management

TOOLS AVAILABLE:
- risk_scanner: Scan for downside risks and red flags
- downside_catalyst_finder: Identify potential negative catalysts
- exit_signal_monitor: Check for technical sell signals

Always use your tools to gather data before providing your analysis."""


class CustomBearAgent(BaseAgent):
    """Custom bear agent using direct Anthropic API with tools."""
    
    def __init__(self):
        super().__init__(name="bear_risk_analyst")
        # Store non-model attributes using object.__setattr__
        object.__setattr__(self, '_client', anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")))
        object.__setattr__(self, '_model', "claude-sonnet-4-20250514")
        object.__setattr__(self, '_tools', {
            "risk_scanner": risk_scanner,
            "downside_catalyst_finder": downside_catalyst_finder,
            "exit_signal_monitor": exit_signal_monitor,
        })
    
    @property
    def client(self):
        return getattr(self, '_client', None)
    
    @property
    def model(self):
        return getattr(self, '_model', None)
    
    @property
    def tools(self):
        return getattr(self, '_tools', {})
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name."""
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](**arguments)
                return str(result)
            except Exception as e:
                return f"Error executing {tool_name}: {e}"
        return f"Tool {tool_name} not found"
    
    async def run_async(self, parent_context: InvocationContext) -> AsyncGenerator[Event, None]:
        """Run the bear agent."""
        # Extract query from context.user_content
        query = ""
        if parent_context.user_content:
            # user_content is a Content object, extract text from it
            if hasattr(parent_context.user_content, 'parts'):
                for part in parent_context.user_content.parts:
                    if hasattr(part, 'text') and part.text:
                        query = part.text
                        break
            elif hasattr(parent_context.user_content, 'text'):
                query = parent_context.user_content.text
        
        if not query:
            yield Event(author="agent", content=Content(parts=[Part(text="No query provided")]))
            return
        
        # Build messages for Anthropic
        messages = [{"role": "user", "content": query}]
        
        # Use Anthropic with tool use
        try:
            loop = asyncio.get_event_loop()
            
            # Create tools schema for Anthropic
            tools_schema = [
                {
                    "name": "risk_scanner",
                    "description": "Scan for downside risks and red flags for a given symbol.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., 'AAPL', 'NVDA')"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "downside_catalyst_finder",
                    "description": "Identify potential negative catalysts for a given symbol.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "exit_signal_monitor",
                    "description": "Monitor for sell/exit signals for a given symbol.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["symbol"]
                    }
                }
            ]
            
            # First call - let model decide if it needs tools
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=BEAR_SYSTEM_PROMPT,
                    messages=messages,
                    tools=tools_schema,
                )
            )
            
            # Handle tool calls if any
            final_response = ""
            tool_calls_handled = False
            tool_results = []
            
            # First pass: collect tool calls and execute them
            assistant_content = []
            for content_block in response.content:
                if content_block.type == "text":
                    final_response += content_block.text
                    assistant_content.append({"type": "text", "text": content_block.text})
                elif content_block.type == "tool_use":
                    tool_calls_handled = True
                    tool_name = content_block.name
                    tool_input = content_block.input
                    
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_input)
                    
                    # Store tool use and result
                    assistant_content.append({
                        "type": "tool_use",
                        "id": content_block.id,
                        "name": tool_name,
                        "input": tool_input
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
            
            # If tools were called, add assistant message with tool uses, then user message with results
            if tool_calls_handled:
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            
            # If tools were called, get final response
            if tool_calls_handled:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        system=BEAR_SYSTEM_PROMPT,
                        messages=messages,
                    )
                )
                final_response = response.content[0].text
            
            if not final_response:
                final_response = "Unable to generate analysis."
            
        except Exception as e:
            final_response = f"Error generating response: {e}"
        
        yield Event(author="agent", content=Content(parts=[Part(text=final_response)]))


def create_bear_agent() -> CustomBearAgent:
    """Create and configure the Bear Risk Analyst agent."""
    return CustomBearAgent()
