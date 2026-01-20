"""
Orchestrator Agent - Balanced financial analyst coordinator using direct Anthropic API.

This agent coordinates between the Bull and Bear agents to provide
balanced, comprehensive financial analysis.
"""

import os
import uuid
import httpx
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv

import anthropic
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.adk.agents import InvocationContext
from google.genai.types import Content, Part

load_dotenv(override=True)

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Financial Query Router that analyzes user queries and routes them to the appropriate specialist analyst.

Your role is to intelligently route financial queries to the most relevant specialist analyst based on the query's intent and focus.

You have access to:
- bull_analyst: An optimistic analyst who identifies opportunities, growth catalysts, upside potential, bullish perspectives, entry points, and momentum signals. Use for queries about: opportunities, growth, upside, bullish analysis, entry points, catalysts, momentum, positive outlook.
- bear_analyst: A cautious analyst who identifies risks, downside scenarios, red flags, valuation concerns, and potential negative catalysts. Use for queries about: risks, concerns, downside, bearish analysis, exit signals, red flags, threats, negative outlook.

ROUTING PROCESS:
1. Analyze the user's query to determine its primary intent and focus
2. Route to bull_analyst if the query focuses on opportunities, growth, upside, bullish analysis
3. Route to bear_analyst if the query focuses on risks, concerns, downside, bearish analysis
4. If the query is neutral/balanced, prefer routing to the single most relevant specialist

OUTPUT:
- Return the selected agent's analysis directly and clearly
- Present the analysis in a clear, actionable format"""


class CustomOrchestratorAgent(BaseAgent):
    """Custom orchestrator agent using direct Anthropic API."""
    
    def __init__(self):
        super().__init__(name="financial_orchestrator")
        # Store non-model attributes using object.__setattr__
        object.__setattr__(self, '_client', anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY")))
        object.__setattr__(self, '_model', "claude-sonnet-4-20250514")
        object.__setattr__(self, '_bull_agent_url', f"http://localhost:{os.environ.get('BULL_AGENT_PORT', '8001')}/")
        object.__setattr__(self, '_bear_agent_url', f"http://localhost:{os.environ.get('BEAR_AGENT_PORT', '8002')}/")
    
    @property
    def client(self):
        return getattr(self, '_client', None)
    
    @property
    def model(self):
        return getattr(self, '_model', None)
    
    @property
    def bull_agent_url(self):
        return getattr(self, '_bull_agent_url', None)
    
    @property
    def bear_agent_url(self):
        return getattr(self, '_bear_agent_url', None)
    
    async def _call_agent_via_a2a(self, agent_url: str, query: str) -> str:
        """Call a remote agent via A2A protocol."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "id": str(uuid.uuid4()),
                "params": {
                    "message": {
                        "messageId": str(uuid.uuid4()),
                        "parts": [{"text": query}],
                        "role": "user",
                    }
                }
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(agent_url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                result = response.json()
                
                # Extract text from agent response
                if "result" in result:
                    if "artifacts" in result["result"] and result["result"]["artifacts"]:
                        artifact = result["result"]["artifacts"][0]
                        if "parts" in artifact and artifact["parts"]:
                            part = artifact["parts"][0]
                            if part.get("kind") == "text" and "text" in part:
                                return part["text"]
            return ""
        except Exception as e:
            print(f"Error calling agent at {agent_url}: {e}")
            return ""
    
    async def _determine_agents(self, query: str) -> list:
        """Use Anthropic to determine which agent(s) to route to.
        
        Returns a list of agent names: ["bull_analyst"], ["bear_analyst"], or ["bull_analyst", "bear_analyst"]
        """
        # For "Analyze [ticker]" queries, ALWAYS call both agents to get both perspectives
        query_lower = query.lower()
        if any(phrase in query_lower for phrase in ["analyze", "analysis"]):
            # For analysis queries, always call both agents
            return ["bull_analyst", "bear_analyst"]
        
        # Check if query explicitly asks for both perspectives
        if any(phrase in query_lower for phrase in ["both", "bull and bear", "pros and cons", "opportunities and risks"]):
            # For explicit requests for both, call both agents
            return ["bull_analyst", "bear_analyst"]
        
        routing_prompt = f"""Based on this financial query, determine which specialist analyst should handle it:
- "bull_analyst" for opportunities, growth, upside, bullish analysis
- "bear_analyst" for risks, concerns, downside, bearish analysis
- "both" if the query requires both perspectives

Query: {query}

Respond with ONLY the agent name(s): "bull_analyst", "bear_analyst", or "both"."""
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=50,
                    messages=[{"role": "user", "content": routing_prompt}],
                )
            )
            agent_response = response.content[0].text.strip().lower()
            
            if "both" in agent_response:
                return ["bull_analyst", "bear_analyst"]
            elif "bear" in agent_response:
                return ["bear_analyst"]
            elif "bull" in agent_response:
                return ["bull_analyst"]
            else:
                # Default to both for ambiguous queries
                return ["bull_analyst", "bear_analyst"]
        except Exception as e:
            print(f"Error determining agent: {e}")
            # Default to both for error cases
            return ["bull_analyst", "bear_analyst"]
    
    async def run_async(self, parent_context: InvocationContext) -> AsyncGenerator[Event, None]:
        """Run the orchestrator agent."""
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
            # Try to get from session state as fallback
            if hasattr(parent_context, 'session') and hasattr(parent_context.session, 'state'):
                query = parent_context.session.state.get('query', '')
        
        if not query:
            yield Event(author="agent", content=Content(parts=[Part(text="No query provided")]))
            return
        
        # Determine which agent(s) to route to
        agents_to_call = await self._determine_agents(query)
        
        # Call the selected agent(s) and collect responses
        bull_response = ""
        bear_response = ""
        
        if "bull_analyst" in agents_to_call:
            bull_response = await self._call_agent_via_a2a(self.bull_agent_url, query)
        
        if "bear_analyst" in agents_to_call:
            bear_response = await self._call_agent_via_a2a(self.bear_agent_url, query)
        
        # Format the combined response
        if bull_response and bear_response:
            # Both perspectives requested
            agent_response = f"""## Bull Case

{bull_response}

---

## Bear Case

{bear_response}"""
        elif bull_response:
            agent_response = f"""## Bull Case

{bull_response}"""
        elif bear_response:
            agent_response = f"""## Bear Case

{bear_response}"""
        else:
            # If no response from agents, use Anthropic to generate a response
            try:
                full_prompt = f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\nUser Query: {query}\n\nProvide a financial analysis based on the query."
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        messages=[{"role": "user", "content": full_prompt}],
                    )
                )
                agent_response = response.content[0].text
            except Exception as e:
                agent_response = f"Error generating response: {e}"
        
        # Yield response as events
        yield Event(author="agent", content=Content(parts=[Part(text=agent_response)]))


def create_orchestrator() -> CustomOrchestratorAgent:
    """Create and configure the Financial Orchestrator agent."""
    return CustomOrchestratorAgent()
