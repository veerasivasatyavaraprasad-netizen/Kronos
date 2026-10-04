"""
Binance spot broker (signed REST).

environment: testnet (https://testnet.binance.vision, BINANCE_TESTNET_API_KEY/SECRET)
             live    (https://api.binance.com, BINANCE_API_KEY/BINANCE_SECRET_KEY) - needs LUNATRADE_ALLOW_LIVE=yes
validate_only=True (SHADOW mode) sends orders to /api/v3/order/test: Binance checks them, nothing executes.

Least privilege: the trading key needs only "Enable Spot & Margin Trading" (TRADE) + reading; keep withdrawals
disabled and restrict the key to your server's IP. Every order carries newClientOrderId so a timeout can be
reconciled with GET /api/v3/order instead of re-sending (which could double the position).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from urllib.parse import urlencode

import requests

from lunatrade.config import live_trading_allowed, secret
from lunatrade.data.binance_public import symbol_filters
from lunatrade.execution.brokers.base import Broker, BrokerError, BrokerResult, BrokerUnknown, floor_step, fmt
from lunatrade.execution.order_state import BINANCE_STATUS, Order, OrderStatus

log = logging.getLogger("lunatrade.binance")
HOSTS = {"testnet": "https://testnet.binance.vision", "live": "https://api.binance.com"}
UNKNOWN_CODES = {-1006, -1007}        # "unexpected response" / "timeout waiting for backend" -> status unknown
NOT_FOUND = -2013


class BinanceSpotBroker(Broker):
    name = "binance"
    supports_test_orders = True

    def __init__(self, environment: str = "testnet", validate_only: bool = False, base_url: str | None = None,
                 api_key: str | None = None, api_secret: str | None = None, timeout: float = 10):
        if environment not in HOSTS:
            raise ValueError(f"unknown Binance environment {environment}")
        if environment == "live" and not validate_only and not live_trading_allowed():
            raise RuntimeError("Binance live trading requested but LUNATRADE_ALLOW_LIVE=yes is not set - refusing.")
        super().__init__(environment)
        self.environment, self.validate_only = environment, validate_only
        if environment == "testnet":
            self.base = (base_url or os.getenv("BINANCE_TESTNET_BASE_URL") or HOSTS["testnet"]).rstrip("/")
            self.key = api_key or secret("BINANCE_TESTNET_API_KEY")
            self.secret_ = api_secret or secret("BINANCE_TESTNET_SECRET_KEY")
        else:
            self.base = (base_url or os.getenv("BINANCE_API_BASE_URL") or HOSTS["live"]).rstrip("/")
            self.key = api_key or secret("BINANCE_API_KEY")
            self.secret_ = api_secret or secret("BINANCE_SECRET_KEY")
        if not self.key or not self.secret_:
            raise RuntimeError(f"Binance {environment}: API key/secret not set in the environment")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["X-MBX-APIKEY"] = self.key
        self.time_offset = 0
        self._filters: dict = {}
        self.last_ok = 0.0
        self.consecutive_errors = 0
        self.sync_time()

    # ------------------------------------------------------------------ plumbing
    def sync_time(self) -> None:
        r = self.s.get(f"{self.base}/api/v3/time", timeout=self.timeout)
        r.raise_for_status()
        self.time_offset = int(r.json()["serverTime"]) - int(time.time() * 1000)

    def _sign(self, params: dict) -> str:
        q = urlencode(params)
        return f"{q}&signature={hmac.new(self.secret_.encode(), q.encode(), hashlib.sha256).hexdigest()}"

    def _signed(self, method: str, path: str, params: dict | None = None, may_be_unknown: bool = False):
        params = dict(params or {})
        params.setdefault("recvWindow", 5000)
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        url = f"{self.base}{path}?{self._sign(params)}"
        try:
            r = self.s.request(method, url, timeout=self.timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            self.consecutive_errors += 1
            if may_be_unknown:
                raise BrokerUnknown(f"{path}: {e}") from e
            raise BrokerError(f"{path}: {e}") from e
        if r.status_code >= 500:
            self.consecutive_errors += 1
            if may_be_unknown:
                raise BrokerUnknown(f"{path} HTTP {r.status_code}: {r.text[:200]}")
            raise BrokerError(f"{path} HTTP {r.status_code}")
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code >= 400:
            code = data.get("code") if isinstance(data, dict) else None
            if code == -1021:      # timestamp outside recvWindow -> resync for next call
                self.sync_time()
            if may_be_unknown and code in UNKNOWN_CODES:
                raise BrokerUnknown(f"{path} code {code}: {data.get('msg')}")
            err = BrokerError(f"{path} {r.status_code} code {code}: {data.get('msg') if isinstance(data, dict) else r.text[:200]}")
            err.code = code
            raise err
        self.consecutive_errors = 0
        self.last_ok = time.time()
        return data

    def healthy(self) -> bool:
        return self.consecutive_errors < 3

    # ------------------------------------------------------------------ market info
    def filters(self, symbol: str) -> dict:
        if symbol not in self._filters:
            r = self.s.get(f"{self.base}/api/v3/exchangeInfo", params={"symbol": symbol}, timeout=self.timeout)
            r.raise_for_status()
            self._filters[symbol] = symbol_filters(r.json()["symbols"][0])
        return self._filters[symbol]

    def balances(self) -> dict[str, float]:
        acct = self._signed("GET", "/api/v3/account", {"omitZeroBalances": "true"})
        return {b["asset"]: float(b["free"]) + float(b["locked"]) for b in acct.get("balances", [])}

    def free_balances(self) -> dict[str, float]:
        acct = self._signed("GET", "/api/v3/account", {"omitZeroBalances": "true"})
        return {b["asset"]: float(b["free"]) for b in acct.get("balances", [])}

    # ------------------------------------------------------------------ orders
    def _params(self, order: Order) -> dict:
        f = self.filters(order.symbol)
        p = {"symbol": order.symbol, "side": order.side, "type": order.order_type,
             "newClientOrderId": order.client_order_id, "newOrderRespType": "FULL"}
        if order.order_type == "MARKET":
            if order.side == "BUY" and order.requested_notional and not order.requested_qty:
                p["quoteOrderQty"] = f"{order.requested_notional:.2f}"
            else:
                p["quantity"] = fmt(floor_step(order.requested_qty, f["market_step_size"] or f["step_size"]))
        else:
            p["quantity"] = fmt(floor_step(order.requested_qty, f["step_size"]))
            p["price"] = fmt(floor_step(order.limit_price, f["tick_size"]))
            p["timeInForce"] = "GTC"
        return p

    def submit(self, order: Order) -> BrokerResult:
        params = self._params(order)
        if self.validate_only:
            self._signed("POST", "/api/v3/order/test", params)
            px = order.reference_price
            qty = order.requested_qty or (order.requested_notional or 0) / px
            return BrokerResult(OrderStatus.FILLED, f"test-{order.client_order_id}", qty, px, 0.0, "",
                                {"validated": True})
        data = self._signed("POST", "/api/v3/order", params, may_be_unknown=True)
        return self._result(data)

    def _result(self, data: dict) -> BrokerResult:
        qty = float(data.get("executedQty", 0) or 0)
        quote = float(data.get("cummulativeQuoteQty", 0) or 0)
        fee, fee_asset = 0.0, ""
        for fill in data.get("fills", []) or []:
            fee += float(fill.get("commission", 0))
            fee_asset = fill.get("commissionAsset", fee_asset)
        status = BINANCE_STATUS.get(data.get("status", ""), OrderStatus.UNKNOWN)
        return BrokerResult(status, str(data.get("orderId", "")), qty, quote / qty if qty else 0.0, fee, fee_asset, data)

    def query(self, order: Order) -> BrokerResult:
        if self.validate_only:
            return BrokerResult(OrderStatus.FILLED, order.exchange_order_id, order.filled_qty, order.avg_price)
        try:
            data = self._signed("GET", "/api/v3/order", {"symbol": order.symbol,
                                                        "origClientOrderId": order.client_order_id})
        except BrokerError as e:
            if getattr(e, "code", None) == NOT_FOUND:
                # the exchange never received it -> definitely not placed
                return BrokerResult(OrderStatus.REJECTED, raw={"error": "order does not exist"})
            raise
        return self._result(data)

    def cancel(self, order: Order) -> BrokerResult:
        data = self._signed("DELETE", "/api/v3/order", {"symbol": order.symbol,
                                                       "origClientOrderId": order.client_order_id})
        return self._result(data)

    def open_orders(self, symbol: str | None = None) -> list:
        return self._signed("GET", "/api/v3/openOrders", {"symbol": symbol} if symbol else {})

    # user data stream (listen key) for order/fill push updates
    def listen_key(self) -> str:
        r = self.s.post(f"{self.base}/api/v3/userDataStream", timeout=self.timeout)
        r.raise_for_status()
        return r.json()["listenKey"]
