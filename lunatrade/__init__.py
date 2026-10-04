"""
LunaTrade - a multi-agent crypto/stock trading system built on top of the Kronos forecasting model.

Hierarchy (no layer may be skipped):

    DATA -> ANALYSIS (300 workers) -> SIGNAL -> CONVICTION (Lead Brain + Devil's Advocate)
         -> RISK (Risk Engine) -> PORTFOLIO (Portfolio Brain) -> EXECUTION (Execution Gateway)

An LLM can never place an order. LLM output is advisory, schema-checked and can only adjust
conviction inside fixed bounds; the kill switch and risk engine sit outside the AI layer.
"""

__version__ = "1.0.0"
