"""Finance/crypto sentiment lexicon with negation and intensifier handling (no model download needed)."""
from __future__ import annotations

import re

POSITIVE = {
    "surge": 2.0, "surges": 2.0, "soar": 2.0, "soars": 2.0, "rally": 1.8, "rallies": 1.8, "jump": 1.4, "jumps": 1.4,
    "gain": 1.0, "gains": 1.0, "rise": 1.0, "rises": 1.0, "climb": 1.0, "climbs": 1.0, "record": 1.2, "high": 0.6,
    "bullish": 2.0, "breakout": 1.5, "approve": 2.0, "approved": 2.0, "approval": 1.8, "inflow": 1.5, "inflows": 1.5,
    "adoption": 1.4, "partnership": 1.2, "partners": 1.0, "launch": 0.8, "launches": 0.8, "upgrade": 1.0,
    "integrates": 1.0, "accumulate": 1.4, "accumulation": 1.4, "buy": 0.8, "buys": 1.0, "bought": 1.0,
    "listing": 1.2, "lists": 1.0, "listed": 1.0, "win": 1.2, "wins": 1.2, "victory": 1.5, "dismissed": 1.3,
    "recover": 1.2, "recovers": 1.2, "rebound": 1.3, "outperform": 1.3, "optimism": 1.2, "support": 0.5,
    "cut": 0.6, "cuts": 0.6, "easing": 1.0, "stimulus": 1.0, "etf": 0.4, "institutional": 0.6, "reserve": 0.6,
    "strong": 0.8, "beat": 1.0, "beats": 1.0, "boost": 1.2, "boosts": 1.2, "milestone": 1.0,
}
NEGATIVE = {
    "crash": -2.5, "crashes": -2.5, "plunge": -2.2, "plunges": -2.2, "dump": -2.0, "dumps": -2.0, "slump": -1.8,
    "fall": -1.0, "falls": -1.0, "drop": -1.1, "drops": -1.1, "decline": -1.0, "declines": -1.0, "sell-off": -1.8,
    "selloff": -1.8, "bearish": -2.0, "hack": -2.5, "hacked": -2.5, "exploit": -2.4, "exploited": -2.4,
    "breach": -2.0, "stolen": -2.2, "theft": -2.2, "drain": -2.0, "drained": -2.0, "scam": -2.2, "fraud": -2.4,
    "lawsuit": -1.6, "sues": -1.6, "sued": -1.6, "charges": -1.6, "charged": -1.6, "ban": -2.0, "bans": -2.0,
    "banned": -2.0, "crackdown": -2.0, "reject": -1.8, "rejected": -1.8, "rejects": -1.8, "delay": -0.9,
    "delays": -0.9, "delist": -2.2, "delisting": -2.2, "delisted": -2.2, "outflow": -1.5, "outflows": -1.5,
    "liquidation": -1.4, "liquidations": -1.4, "insolvent": -2.5, "bankruptcy": -2.6, "bankrupt": -2.6,
    "collapse": -2.5, "collapses": -2.5, "fear": -1.2, "panic": -2.0, "warning": -1.0, "warns": -1.0,
    "investigation": -1.4, "probe": -1.3, "subpoena": -1.3, "fine": -1.0, "fined": -1.2, "penalty": -1.2,
    "depeg": -2.4, "depegged": -2.4, "halt": -1.6, "halts": -1.6, "suspend": -1.6, "suspends": -1.6,
    "outage": -1.4, "unlock": -0.8, "unlocks": -0.8, "hike": -0.8, "hikes": -0.8, "hawkish": -1.2,
    "inflation": -0.5, "recession": -1.6, "weak": -0.8, "miss": -1.0, "misses": -1.0, "loss": -1.0, "losses": -1.0,
}
NEGATIONS = {"not", "no", "never", "without", "denies", "deny", "denied", "fails", "failed", "unlikely", "won't",
             "isn't", "wasn't", "aren't", "doesn't", "didn't"}
INTENSIFIERS = {"massive": 1.5, "huge": 1.4, "record": 1.3, "major": 1.3, "sharp": 1.3, "sharply": 1.3,
                "biggest": 1.5, "historic": 1.4, "slight": 0.6, "slightly": 0.6, "minor": 0.6}
_TOKEN = re.compile(r"[a-z][a-z'\-]*")


def score(text: str) -> float:
    """Sentiment in [-1, 1]."""
    tokens = _TOKEN.findall(text.lower())
    total, hits = 0.0, 0
    for i, tok in enumerate(tokens):
        w = POSITIVE.get(tok) or NEGATIVE.get(tok)
        if w is None:
            continue
        window = tokens[max(0, i - 3):i]
        if any(t in NEGATIONS for t in window):
            w = -w * 0.8
        for t in window:
            w *= INTENSIFIERS.get(t, 1.0)
        total += w
        hits += 1
    if not hits:
        return 0.0
    return max(-1.0, min(1.0, total / (2.0 + abs(total)) * 1.6))


def label(s: float, dead: float = 0.1) -> str:
    return "BULLISH" if s > dead else "BEARISH" if s < -dead else "NEUTRAL"
