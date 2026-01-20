"""
Mock risk analysis tools for the Bear Agent.

These tools simulate bearish/risk analysis for financial instruments.
In a production system, these would connect to real data sources.
"""

import random


def risk_scanner(symbol: str) -> str:
    """
    Scan for downside risks and red flags for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "NVDA", "AAPL")

    Returns:
        A mock risk analysis report
    """
    # Mock data - varies slightly based on symbol hash for consistency
    seed = sum(ord(c) for c in symbol.upper())
    random.seed(seed)

    pe_ratio = random.randint(25, 50)
    rsi = random.randint(60, 85)
    insider_selling = random.randint(10, 60)

    return f"""Risk Analysis for {symbol.upper()}:

VALUATION RISKS:
- Forward P/E: {pe_ratio}x ({"elevated" if pe_ratio > 35 else "moderate"} vs sector average of 22x)
- Price/Sales: {pe_ratio / 5:.1f}x (premium valuation)
- EV/EBITDA: {pe_ratio - 5}x (above historical average)

TECHNICAL WARNINGS:
- RSI: {rsi} ({"overbought territory" if rsi > 70 else "approaching overbought"})
- Price vs 200-day MA: +{random.randint(15, 35)}% (extended)
- Bollinger Band Position: Upper band breach

MACRO HEADWINDS:
- Rising interest rates pressure growth stock valuations
- Strong dollar impacts international revenue
- Sector rotation risk as investors seek value

SENTIMENT INDICATORS:
- Insider selling increased {insider_selling}% vs last quarter
- Short interest: {random.randint(3, 8)}% of float
- Put/Call ratio trending higher"""


def downside_catalyst_finder(symbol: str) -> str:
    """
    Identify potential negative catalysts for a given symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        A list of potential downside catalysts
    """
    seed = sum(ord(c) for c in symbol.upper()) + 100
    random.seed(seed)

    earnings_risk = random.randint(5, 20)
    market_share_loss = random.randint(2, 8)

    return f"""Potential Downside Catalysts for {symbol.upper()}:

EARNINGS RISK:
- Consensus estimates may be {earnings_risk}% too aggressive
- Margin compression risk from input cost inflation
- Revenue guidance could disappoint on macro weakness

COMPETITIVE THREATS:
- New market entrants gaining {market_share_loss}% share YoY
- Pricing pressure from aggressive competitors
- Technology disruption risk in core segments

REGULATORY CONCERNS:
- Antitrust scrutiny in EU and domestic markets
- Data privacy regulations impacting operations
- Potential tax policy changes affecting margins

OPERATIONAL RISKS:
- Supply chain concentration in Asia
- Key executive departure risk
- Cybersecurity vulnerabilities

MARKET STRUCTURE:
- High institutional ownership creates crowded trade risk
- Options gamma exposure could amplify downside moves
- Passive flow concentration in major indices"""


def exit_signal_monitor(symbol: str) -> str:
    """
    Monitor for sell/exit signals for a given symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Current exit signal status
    """
    seed = sum(ord(c) for c in symbol.upper()) + 200
    random.seed(seed)

    signals_triggered = random.randint(2, 5)

    return f"""Exit Signal Monitor for {symbol.upper()}:

TRIGGERED SIGNALS ({signals_triggered}/7):
{"[X]" if random.random() > 0.5 else "[ ]"} RSI Overbought (>70)
{"[X]" if random.random() > 0.5 else "[ ]"} MACD Bearish Divergence
{"[X]" if random.random() > 0.5 else "[ ]"} Volume Declining on Highs
{"[X]" if random.random() > 0.6 else "[ ]"} Breaking Below 20-day MA
{"[X]" if random.random() > 0.7 else "[ ]"} Negative Earnings Revision
{"[X]" if random.random() > 0.6 else "[ ]"} Sector Underperformance
{"[X]" if random.random() > 0.8 else "[ ]"} Insider Selling Spike

RISK ASSESSMENT: {"ELEVATED" if signals_triggered >= 4 else "MODERATE"}
RECOMMENDED ACTION: {"Consider reducing position" if signals_triggered >= 4 else "Monitor closely"}"""
