"""Broker interface used by the Execution Gateway (the only caller)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from lunatrade.execution.order_state import Order, OrderStatus


class BrokerError(Exception):
    """Definite failure: the order was not placed (safe to treat as REJECTED)."""


class BrokerUnknown(Exception):
    """Timeout / connection drop after sending: the order MAY exist. Must be reconciled, never blindly retried."""


@dataclass
class BrokerResult:
    status: OrderStatus
    exchange_order_id: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fee: float = 0.0
    fee_asset: str = ""
    raw: dict = field(default_factory=dict)


class Broker:
    name = "base"
    supports_test_orders = False

    def __init__(self, mode: str):
        self.mode = mode

    def filters(self, symbol: str) -> dict:
        return {"min_notional": 0.0, "step_size": 0.0, "min_qty": 0.0, "tick_size": 0.0, "status": "TRADING",
                "base": symbol[:-4] if symbol.endswith("USDT") else symbol, "quote": "USDT"}

    def submit(self, order: Order) -> BrokerResult:
        raise NotImplementedError

    def query(self, order: Order) -> BrokerResult:
        raise NotImplementedError

    def cancel(self, order: Order) -> BrokerResult:
        raise NotImplementedError

    def balances(self) -> dict[str, float]:
        return {}

    def market_open(self, symbol: str) -> bool:
        return True

    def healthy(self) -> bool:
        return True


def floor_step(value: float, step: float) -> float:
    if not step or step <= 0:
        return value
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(math.floor(value / step + 1e-9) * step, decimals)


def round_tick(price: float, tick: float) -> float:
    return floor_step(price, tick)


def fmt(x: float) -> str:
    return f"{x:.8f}".rstrip("0").rstrip(".")
