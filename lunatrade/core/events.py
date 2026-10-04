"""Event envelope and topic names for the agent communication bus."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

from lunatrade.core.types import _jsonable, new_id, utcnow


class Topic:
    MARKET_EVENT = "market.event"
    NEWS_EVENT = "news.event"
    SENTIMENT_EVENT = "sentiment.event"
    WHALE_EVENT = "whale.event"
    SIGNAL_EVENT = "signal.event"
    REGIME_EVENT = "regime.event"
    RISK_EVENT = "risk.event"
    TRADE_PROPOSAL = "trade.proposal"
    DEVILS_ADVOCATE = "trade.devils_advocate"
    RISK_DECISION = "trade.risk_decision"
    APPROVAL_REQUESTED = "trade.approval_requested"
    TRADE_APPROVAL = "trade.approval"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_UPDATED = "order.updated"
    ORDER_FILLED = "order.filled"
    POSITION_CHANGED = "position.changed"
    TRADE_CLOSED = "trade.closed"
    SYSTEM_ALERT = "system.alert"
    KILL_SWITCH = "system.kill_switch"
    MODE_CHANGED = "system.mode_changed"
    HEALTH = "system.health"

    ALL = (MARKET_EVENT, NEWS_EVENT, SENTIMENT_EVENT, WHALE_EVENT, SIGNAL_EVENT, REGIME_EVENT, RISK_EVENT,
           TRADE_PROPOSAL, DEVILS_ADVOCATE, RISK_DECISION, APPROVAL_REQUESTED, TRADE_APPROVAL,
           ORDER_SUBMITTED, ORDER_UPDATED, ORDER_FILLED, POSITION_CHANGED, TRADE_CLOSED, SYSTEM_ALERT,
           KILL_SWITCH, MODE_CHANGED, HEALTH)


@dataclass(slots=True)
class Event:
    topic: str
    payload: dict
    source: str = "system"
    correlation_id: str = ""
    id: str = field(default_factory=lambda: new_id("evt_"))
    ts: dt.datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return {"id": self.id, "topic": self.topic, "source": self.source, "correlation_id": self.correlation_id,
                "ts": self.ts.isoformat(), "payload": _jsonable(self.payload)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(topic=d["topic"], payload=d.get("payload", {}), source=d.get("source", "system"),
                   correlation_id=d.get("correlation_id", ""), id=d.get("id") or new_id("evt_"),
                   ts=dt.datetime.fromisoformat(d["ts"]) if d.get("ts") else utcnow())
