"""Alpaca equities broker (paper by default; live needs LUNATRADE_ALLOW_LIVE=yes). Paper keys start with PK."""
from __future__ import annotations

import os

import requests

from lunatrade.config import live_trading_allowed, secret
from lunatrade.execution.brokers.base import Broker, BrokerError, BrokerResult, BrokerUnknown
from lunatrade.execution.order_state import ALPACA_STATUS, Order, OrderStatus

PAPER, LIVE = "https://paper-api.alpaca.markets", "https://api.alpaca.markets"


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, environment: str = "paper", base_url: str | None = None, timeout: float = 15):
        if environment == "live" and not live_trading_allowed():
            environment = "paper"
        super().__init__(environment)
        self.base = (base_url or (PAPER if environment == "paper" else os.getenv("ALPACA_API_BASE_URL", LIVE))).rstrip("/")
        key, sec = secret("ALPACA_API_KEY"), secret("ALPACA_SECRET_KEY")
        if not key or not sec:
            raise RuntimeError("Alpaca: ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
        self.timeout = timeout
        self.errors = 0

    def _req(self, method, path, unknown_ok=False, **kw):
        try:
            r = self.s.request(method, f"{self.base}{path}", timeout=self.timeout, **kw)
        except (requests.Timeout, requests.ConnectionError) as e:
            self.errors += 1
            if unknown_ok:
                raise BrokerUnknown(str(e)) from e
            raise BrokerError(str(e)) from e
        if r.status_code == 404:
            return None
        if r.status_code >= 500 and unknown_ok:
            raise BrokerUnknown(f"{path} {r.status_code}")
        if r.status_code >= 400:
            raise BrokerError(f"Alpaca {path} {r.status_code}: {r.text[:200]}")
        self.errors = 0
        return r.json() if r.content else {}

    def healthy(self):
        return self.errors < 3

    def market_open(self, symbol):
        return bool((self._req("GET", "/v2/clock") or {}).get("is_open"))

    def filters(self, symbol):
        return {"min_notional": 1.0, "step_size": 1e-9, "min_qty": 0.0, "tick_size": 0.01, "status": "TRADING",
                "base": symbol, "quote": "USD", "market_step_size": 1e-9}

    def balances(self):
        acct = self._req("GET", "/v2/account") or {}
        out = {"USD": float(acct.get("cash", 0))}
        for p in self._req("GET", "/v2/positions") or []:
            out[p["symbol"]] = float(p["qty"])
        return out

    def submit(self, order: Order) -> BrokerResult:
        body = {"symbol": order.symbol, "side": order.side.lower(), "type": order.order_type.lower(),
                "time_in_force": "day", "client_order_id": order.client_order_id}
        if order.side == "BUY" and order.requested_notional and not order.requested_qty:
            body["notional"] = f"{order.requested_notional:.2f}"
        else:
            body["qty"] = f"{order.requested_qty:.6f}"
        if order.order_type == "LIMIT":
            body["limit_price"] = f"{order.limit_price:.2f}"
        return self._result(self._req("POST", "/v2/orders", unknown_ok=True, json=body))

    def _result(self, d: dict | None) -> BrokerResult:
        if not d:
            return BrokerResult(OrderStatus.REJECTED, raw={"error": "not found"})
        qty = float(d.get("filled_qty") or 0)
        return BrokerResult(ALPACA_STATUS.get(d.get("status", ""), OrderStatus.UNKNOWN), d.get("id", ""), qty,
                            float(d.get("filled_avg_price") or 0), 0.0, "USD", d)

    def query(self, order: Order) -> BrokerResult:
        return self._result(self._req("GET", "/v2/orders:by_client_order_id",
                                      params={"client_order_id": order.client_order_id}))

    def cancel(self, order: Order) -> BrokerResult:
        if order.exchange_order_id:
            self._req("DELETE", f"/v2/orders/{order.exchange_order_id}")
        return self.query(order)
