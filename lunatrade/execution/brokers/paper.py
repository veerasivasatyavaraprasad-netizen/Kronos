"""Paper broker: real prices, simulated fills with fees, slippage and optional partial fills."""
from __future__ import annotations

import random

from lunatrade.execution.brokers.base import Broker, BrokerError, BrokerResult
from lunatrade.execution.order_state import Order, OrderStatus


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, starting_cash: float = 10_000, fee_bps: float = 10, slippage_bps: float = 5,
                 partial_fill_probability: float = 0.0, filters_provider=None, quote: str = "USDT", seed: int = 1):
        super().__init__("paper")
        self.cash = float(starting_cash)
        self.holdings: dict[str, float] = {}
        self.fee_bps, self.slippage_bps = fee_bps, slippage_bps
        self.partial_p = partial_fill_probability
        self.filters_provider = filters_provider
        self.quote = quote
        self.orders: dict[str, BrokerResult] = {}
        self.rng = random.Random(seed)
        self.extra_slippage_bps = 0.0      # stress tests widen this

    def filters(self, symbol):
        if self.filters_provider:
            f = self.filters_provider(symbol)
            if f:
                return f
        return {**super().filters(symbol), "min_notional": 5.0, "step_size": 1e-8, "min_qty": 0.0}

    def balances(self):
        return {self.quote: self.cash, **{s[: -len(self.quote)] if s.endswith(self.quote) else s: q
                                          for s, q in self.holdings.items()}}

    def submit(self, order: Order) -> BrokerResult:
        px = order.reference_price
        if not px or px <= 0:
            raise BrokerError("no reference price for paper fill")
        slip = (self.slippage_bps + self.extra_slippage_bps) / 1e4
        fill_px = px * (1 + slip) if order.side == "BUY" else px * (1 - slip)
        qty = order.requested_qty if order.requested_qty else (order.requested_notional or 0) / fill_px
        if qty <= 0:
            raise BrokerError("zero quantity")
        status = OrderStatus.FILLED
        if self.partial_p and self.rng.random() < self.partial_p:
            qty *= self.rng.uniform(0.3, 0.9)
            status = OrderStatus.PARTIALLY_FILLED
        notional = qty * fill_px
        fee = notional * self.fee_bps / 1e4
        if order.side == "BUY":
            if notional + fee > self.cash + 1e-9:
                raise BrokerError(f"insufficient paper cash {self.cash:.2f} for {notional + fee:.2f}")
            self.cash -= notional + fee
            self.holdings[order.symbol] = self.holdings.get(order.symbol, 0.0) + qty
        else:
            have = self.holdings.get(order.symbol, 0.0)
            if qty > have + 1e-12:
                qty = have
                notional = qty * fill_px
                fee = notional * self.fee_bps / 1e4
            if qty <= 0:
                raise BrokerError("nothing to sell")
            self.holdings[order.symbol] = have - qty
            self.cash += notional - fee
        res = BrokerResult(status, f"paper-{order.client_order_id}", qty, fill_px, fee, self.quote)
        self.orders[order.client_order_id] = res
        return res

    def query(self, order: Order) -> BrokerResult:
        res = self.orders.get(order.client_order_id)
        if res is None:
            return BrokerResult(OrderStatus.REJECTED, raw={"error": "unknown order"})
        if res.status == OrderStatus.PARTIALLY_FILLED:   # the rest is cancelled (IOC semantics)
            res.status = OrderStatus.CANCELED
        return res

    def cancel(self, order: Order) -> BrokerResult:
        return self.query(order)
