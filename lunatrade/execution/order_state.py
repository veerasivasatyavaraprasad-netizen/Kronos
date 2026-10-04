"""
Order state machine. An API response is NOT proof of execution:

    CREATED -> VALIDATED -> SUBMITTED -> ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED -> POSITION_UPDATED
                                    \\-> UNKNOWN -> (query exchange) -> confirmed state

Illegal transitions raise, so a bug can never silently mark an order filled.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from enum import Enum

from lunatrade.core.types import new_id, utcnow


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"
    POSITION_UPDATED = "POSITION_UPDATED"


S = OrderStatus
EXCHANGE_STATES = {S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.EXPIRED, S.REJECTED}
TRANSITIONS = {
    S.CREATED: {S.VALIDATED, S.REJECTED},
    S.VALIDATED: {S.SUBMITTED, S.REJECTED},
    S.SUBMITTED: EXCHANGE_STATES | {S.UNKNOWN},
    S.UNKNOWN: EXCHANGE_STATES | {S.UNKNOWN},
    S.ACKNOWLEDGED: {S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.EXPIRED, S.UNKNOWN},
    S.PARTIALLY_FILLED: {S.PARTIALLY_FILLED, S.FILLED, S.CANCELED, S.EXPIRED, S.UNKNOWN},
    S.FILLED: {S.POSITION_UPDATED},
    S.CANCELED: {S.POSITION_UPDATED},      # only when something was partially filled
    S.EXPIRED: {S.POSITION_UPDATED},
    S.REJECTED: set(),
    S.POSITION_UPDATED: set(),
}
TERMINAL = {S.REJECTED, S.POSITION_UPDATED}
OPEN_STATES = {S.CREATED, S.VALIDATED, S.SUBMITTED, S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.UNKNOWN}

# Binance order status -> ours
BINANCE_STATUS = {"NEW": S.ACKNOWLEDGED, "PARTIALLY_FILLED": S.PARTIALLY_FILLED, "FILLED": S.FILLED,
                  "CANCELED": S.CANCELED, "PENDING_CANCEL": S.ACKNOWLEDGED, "REJECTED": S.REJECTED,
                  "EXPIRED": S.EXPIRED, "EXPIRED_IN_MATCH": S.EXPIRED}
ALPACA_STATUS = {"new": S.ACKNOWLEDGED, "accepted": S.ACKNOWLEDGED, "pending_new": S.ACKNOWLEDGED,
                 "partially_filled": S.PARTIALLY_FILLED, "filled": S.FILLED, "canceled": S.CANCELED,
                 "expired": S.EXPIRED, "rejected": S.REJECTED, "done_for_day": S.EXPIRED, "replaced": S.CANCELED}


class IllegalTransition(Exception):
    pass


@dataclass
class Order:
    proposal_id: str
    symbol: str
    side: str
    broker: str
    mode: str
    order_type: str = "MARKET"
    requested_qty: float | None = None
    requested_notional: float | None = None
    limit_price: float | None = None
    reference_price: float = 0.0
    client_order_id: str = field(default_factory=lambda: new_id("lt"))
    exchange_order_id: str = ""
    status: OrderStatus = S.CREATED
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    fee_asset: str = ""
    reason: str = ""
    error: str = ""
    history: list = field(default_factory=list)
    created_at: dt.datetime = field(default_factory=utcnow)
    updated_at: dt.datetime = field(default_factory=utcnow)

    def transition(self, new: OrderStatus, detail: str = "") -> None:
        if new not in TRANSITIONS[self.status]:
            raise IllegalTransition(f"{self.client_order_id}: {self.status.value} -> {new.value} not allowed")
        self.history.append({"at": utcnow().isoformat(), "from": self.status.value, "to": new.value, "detail": detail})
        self.status = new
        self.updated_at = utcnow()

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATES

    @property
    def filled_notional(self) -> float:
        return self.filled_qty * self.avg_price

    @property
    def slippage_bps(self) -> float:
        if not self.reference_price or not self.avg_price:
            return 0.0
        sign = 1 if self.side == "BUY" else -1
        return sign * (self.avg_price / self.reference_price - 1) * 1e4

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["created_at"] = self.created_at.isoformat()
        d["updated_at"] = self.updated_at.isoformat()
        d["slippage_bps"] = round(self.slippage_bps, 2)
        return d
