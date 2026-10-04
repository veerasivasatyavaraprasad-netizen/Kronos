"""Voice / text command parser -> intents. Pure function, no side effects (the control API acts on intents)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from lunatrade.intel.assets import extract_assets

SENSITIVE = {"emergency_close", "resume", "close_symbol", "mode_change", "reset_kill_switch"}


@dataclass
class Intent:
    name: str
    symbol: str | None = None
    args: dict = field(default_factory=dict)
    text: str = ""

    @property
    def sensitive(self) -> bool:
        return self.name in SENSITIVE


PATTERNS = [
    ("emergency_close", r"\b(emergency close|close (everything|all( positions)?)|liquidate (everything|all)|panic sell)\b"),
    ("pause", r"\b(stop (all )?(new )?(trades|trading|buying)|pause( trading)?|halt( trading)?|no new trades)\b"),
    ("resume", r"\b(resume|restart|continue|unpause|start) (trading|buying)\b|\bresume\b"),
    ("reset_kill_switch", r"\breset (the )?kill ?switch\b"),
    ("mode_change", r"\b(switch|change|set|go) (to )?(the )?(mode )?(to )?(research|paper|shadow|approval|live)( mode)?\b"),
    ("approve", r"\b(approve|yes|confirm)\b\s+([a-f0-9]{4,16})\b"),
    ("reject", r"\b(reject|no|deny|decline)\b\s+([a-f0-9]{4,16})\b"),
    ("explain", r"\b(why did (you|we) (buy|sell|open|close)|explain)\b"),
    ("close_symbol", r"\b(close|sell|exit) (my |the |our )?(position (in|on) )?([a-z$]{2,10})\b"),
    ("pnl", r"\b(pnl|p&l|profit|loss|how much (did|have) (we|i) (make|made|lose|lost)|performance)\b"),
    ("positions", r"\b(positions?|holdings|what (are|do) (we|i) (hold|own)|exposure)\b"),
    ("mode", r"\b(what|which) mode\b|\bmode\?"),
    ("regime", r"\b(regime|market condition)\b"),
    ("symbol_query", r"\b(what'?s happening with|how is|how'?s|status of|update on|outlook (for|on))\b"),
    ("status", r"\b(status|what'?s happening|how are (we|things)|summary|overview|report)\b"),
    ("help", r"\b(help|what can you do|commands)\b"),
]


def parse(text: str) -> Intent:
    raw = (text or "").strip()
    low = raw.lower()
    assets = extract_assets(raw) or [a.upper() for a in re.findall(r"\$([a-z]{2,10})", low)]
    sym = f"{assets[0]}USDT" if assets else None
    for name, pat in PATTERNS:
        m = re.search(pat, low)
        if not m:
            continue
        if name in ("approve", "reject"):
            return Intent(name, None, {"id": m.group(2)}, raw)
        if name == "mode_change":
            target = next(x for x in ("research", "paper", "shadow", "approval", "live") if x in low)
            return Intent(name, None, {"mode": target.upper()}, raw)
        if name == "close_symbol":
            if not sym:
                continue
            return Intent(name, sym, {}, raw)
        if name == "symbol_query" and not sym:
            return Intent("status", None, {}, raw)
        return Intent(name, sym if name in ("symbol_query", "explain") else None, {}, raw)
    if sym:
        return Intent("symbol_query", sym, {}, raw)
    return Intent("unknown", None, {}, raw)
