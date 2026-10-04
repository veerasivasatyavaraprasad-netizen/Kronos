"""
Execution Gateway - the ONLY component allowed to place orders. No analyst, brain or LLM holds exchange
credentials or calls a broker.

Validation before sending: mode allows trading, risk approval present, symbol trading, side, quantity,
precision (step/tick), min qty / min notional, balance, duplicate order, existing open order, never selling
more than LunaTrade itself bought. After sending: order state machine + reconciliation of UNKNOWN states
by querying the exchange (never a blind retry).
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time

from lunatrade.core.events import Topic
from lunatrade.core.types import Mode, OrderRequest
from lunatrade.execution.brokers.base import Broker, BrokerError, BrokerUnknown, floor_step
from lunatrade.execution.order_state import Order, OrderStatus

log = logging.getLogger("lunatrade.execution")
S = OrderStatus


class ValidationError(Exception):
    pass


class ExecutionGateway:
    def __init__(self, brokers: dict[str, Broker], book, mode: Mode, bus=None, kill_switch=None,
                 reconcile_attempts: int = 5, reconcile_delay: float = 1.0, clock=None):
        self.brokers = brokers
        self.book = book
        self.mode = mode
        self.bus = bus
        self.kill_switch = kill_switch
        self.reconcile_attempts = reconcile_attempts
        self.reconcile_delay = reconcile_delay
        self.clock = clock
        self.orders: dict[str, Order] = {}
        self.approved: dict[str, float] = {}   # proposal_id -> approved notional (from risk engine)
        self.executed_proposals: set[str] = set()
        self.consecutive_failures = 0
        self.reconciliation_failures = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ approvals
    def authorize(self, proposal_id: str, notional: float) -> None:
        """Called after Risk Engine + Portfolio Brain (+ human in APPROVAL mode) approved a proposal."""
        self.approved[proposal_id] = notional

    def _now(self) -> dt.datetime:
        return self.clock.now() if self.clock else dt.datetime.utcnow()

    def _emit(self, topic: str, order: Order) -> None:
        if self.bus:
            self.bus.emit(topic, order.to_dict(), source="execution", correlation_id=order.proposal_id)

    # ------------------------------------------------------------------ validation
    def validate(self, req: OrderRequest, broker: Broker) -> Order:
        if self.mode == Mode.RESEARCH:
            raise ValidationError("RESEARCH mode: no orders")
        if req.side not in ("BUY", "SELL"):
            raise ValidationError(f"bad side {req.side}")
        if req.order_type not in ("MARKET", "LIMIT"):
            raise ValidationError(f"unsupported order type {req.order_type}")
        if req.side == "BUY":
            if req.proposal_id not in self.approved:
                raise ValidationError("BUY without risk approval")
            if req.proposal_id in self.executed_proposals:
                raise ValidationError("duplicate: proposal already executed")
            if self.kill_switch is not None and self.kill_switch.active:
                raise ValidationError("kill switch active: " + "; ".join(self.kill_switch.reasons())[:200])
        with self._lock:
            for o in self.orders.values():
                if o.is_open and o.symbol == req.symbol:
                    raise ValidationError(f"open order {o.client_order_id} already working on {req.symbol}")
                if req.client_order_id and o.client_order_id == req.client_order_id:
                    raise ValidationError("duplicate client order id")
        if not broker.market_open(req.symbol):
            raise ValidationError(f"market closed for {req.symbol}")
        f = broker.filters(req.symbol)
        if f.get("status", "TRADING") != "TRADING":
            raise ValidationError(f"{req.symbol} status {f.get('status')}")
        px = req.reference_price
        if not px or px <= 0:
            raise ValidationError("no reference price")
        order = Order(req.proposal_id, req.symbol, req.side, broker.name, self.mode.value, req.order_type,
                      limit_price=req.limit_price, reference_price=px, reason=req.reason)
        if req.client_order_id:
            order.client_order_id = req.client_order_id
        step = f.get("market_step_size") or f.get("step_size") or 0
        if req.side == "BUY":
            notional = min(req.notional or 0.0, self.approved[req.proposal_id])
            if notional <= 0:
                raise ValidationError("zero notional")
            if notional < f.get("min_notional", 0):
                raise ValidationError(f"notional {notional:.2f} below exchange minimum {f['min_notional']}")
            order.requested_notional = round(notional, 2)
            if req.order_type == "LIMIT":
                order.requested_qty = floor_step(notional / req.limit_price, f.get("step_size", 0))
        else:
            own = self.book.qty(req.symbol)
            qty = min(req.quantity or own, own)          # never sell more than LunaTrade bought
            if self.mode in (Mode.APPROVAL, Mode.LIVE):
                try:
                    bal = broker.balances().get(f.get("base") or req.symbol.replace(f.get("quote", "USDT"), ""), None)
                    if bal is not None:
                        qty = min(qty, bal)
                except Exception as e:
                    log.warning("balance check failed: %s", e)
            qty = floor_step(qty, step)
            if qty <= 0 or qty < f.get("min_qty", 0) or qty * px < f.get("min_notional", 0):
                raise ValidationError(f"sell size {qty} below exchange minimums (dust)")
            order.requested_qty = qty
        order.transition(S.VALIDATED)
        return order

    # ------------------------------------------------------------------ execution
    def execute(self, req: OrderRequest, stop: float = 0.0, target: float = 0.0, meta: dict | None = None) -> Order:
        broker = self.brokers.get(req.broker)
        if broker is None:
            raise ValidationError(f"no broker '{req.broker}' configured for mode {self.mode.value}")
        try:
            order = self.validate(req, broker)
        except ValidationError as e:
            order = Order(req.proposal_id, req.symbol, req.side, req.broker, self.mode.value, req.order_type,
                          reference_price=req.reference_price, reason=req.reason, error=str(e))
            order.transition(S.REJECTED, str(e))
            with self._lock:
                self.orders[order.client_order_id] = order
            self._emit(Topic.ORDER_UPDATED, order)
            log.info("order rejected before sending: %s", e)
            return order
        with self._lock:
            self.orders[order.client_order_id] = order
            if req.side == "BUY":
                self.executed_proposals.add(req.proposal_id)
        order.transition(S.SUBMITTED)
        self._emit(Topic.ORDER_SUBMITTED, order)
        try:
            res = broker.submit(order)
            self._apply_result(order, res)
            self.consecutive_failures = 0
        except BrokerUnknown as e:
            order.transition(S.UNKNOWN, str(e))
            self._emit(Topic.ORDER_UPDATED, order)
            self.reconcile(order, broker)
        except BrokerError as e:
            order.error = str(e)
            order.transition(S.REJECTED, str(e))
            self.consecutive_failures += 1
            if self.kill_switch is not None:
                self.kill_switch.check_order_failures(self.consecutive_failures)
            self._emit(Topic.ORDER_UPDATED, order)
            return order
        self._settle(order, stop, target, meta or {})
        return order

    def _apply_result(self, order: Order, res) -> None:
        order.exchange_order_id = res.exchange_order_id or order.exchange_order_id
        order.filled_qty, order.avg_price = res.filled_qty, res.avg_price
        order.fee, order.fee_asset = res.fee, res.fee_asset
        if res.status != order.status or res.status == S.PARTIALLY_FILLED:
            order.transition(res.status, "exchange report")
        self._emit(Topic.ORDER_UPDATED, order)

    def reconcile(self, order: Order, broker: Broker | None = None) -> Order:
        """Query the exchange until the order's real state is known."""
        broker = broker or self.brokers[order.broker]
        for attempt in range(self.reconcile_attempts):
            try:
                res = broker.query(order)
                if res.status != S.UNKNOWN:
                    self._apply_result(order, res)
                    self.reconciliation_failures = 0
                    return order
            except (BrokerError, BrokerUnknown) as e:
                log.warning("reconcile %s attempt %d: %s", order.client_order_id, attempt + 1, e)
            if self.reconcile_delay:
                time.sleep(self.reconcile_delay * (attempt + 1))
        self.reconciliation_failures += 1
        if self.kill_switch is not None:
            self.kill_switch.check_reconciliation(self.reconciliation_failures,
                                                  f"order {order.client_order_id} state unknown")
            self.kill_switch.halt("UNRECONCILED_ORDER", order.client_order_id)
        return order

    def _settle(self, order: Order, stop: float, target: float, meta: dict) -> None:
        """Move fills into the position book (once) and finish the state machine."""
        if order.status == S.ACKNOWLEDGED or order.status == S.PARTIALLY_FILLED:
            broker = self.brokers[order.broker]
            if order.order_type == "MARKET":
                self.reconcile(order, broker)   # market orders finish quickly; confirm the final state
        if order.status not in (S.FILLED, S.CANCELED, S.EXPIRED) or order.filled_qty <= 0:
            return
        fee_quote = order.fee if order.fee_asset in ("", "USDT", "USD", "USDC", "FDUSD") else order.fee * 0
        if order.fee_asset and order.fee_asset not in ("USDT", "USD", "USDC", "FDUSD", "BNB"):
            # fee charged in the base asset reduces the quantity we hold
            if order.side == "BUY":
                order.filled_qty = max(0.0, order.filled_qty - order.fee)
            fee_quote = order.fee * order.avg_price
        cash_managed = order.broker == "paper" or self.mode in (Mode.PAPER, Mode.SHADOW)
        now = self._now()
        if order.side == "BUY":
            self.book.apply_buy(order.symbol, order.filled_qty, order.avg_price, fee_quote, now, stop, target,
                                order.proposal_id, meta.get("supporting"), meta.get("conviction", 0.0),
                                meta.get("regime", ""), cash_managed=cash_managed)
        else:
            f = self.brokers[order.broker].filters(order.symbol)
            outcome = self.book.apply_sell(order.symbol, order.filled_qty, order.avg_price, fee_quote, now,
                                           order.reason or "SELL", cash_managed=cash_managed,
                                           dust_notional=float(f.get("min_notional", 0) or 0))
            if outcome and self.bus:
                self.bus.emit(Topic.TRADE_CLOSED, outcome, source="execution", correlation_id=order.proposal_id)
        if self.kill_switch is not None and order.reference_price:
            self.kill_switch.check_slippage(order.slippage_bps)
        order.transition(S.POSITION_UPDATED, f"{order.side} {order.filled_qty} @ {order.avg_price}")
        self._emit(Topic.ORDER_FILLED, order)
        if self.bus:
            self.bus.emit(Topic.POSITION_CHANGED, {"symbol": order.symbol, "qty": self.book.qty(order.symbol)},
                          source="execution", correlation_id=order.proposal_id)

    def reconcile_open_orders(self) -> int:
        """Periodic sweep: resolve UNKNOWN/working orders."""
        n = 0
        for o in list(self.orders.values()):
            if o.status in (S.UNKNOWN, S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.SUBMITTED):
                self.reconcile(o)
                self._settle(o, 0.0, 0.0, {})
                n += 1
        if self.kill_switch is not None and not any(o.status == S.UNKNOWN for o in self.orders.values()):
            self.kill_switch.clear_halt("UNRECONCILED_ORDER")
        return n

    def reconcile_positions(self, broker_name: str, quote: str = "USDT") -> dict:
        """Compare LunaTrade's book with exchange balances: the bot can't own more than the account holds."""
        broker = self.brokers.get(broker_name)
        if not broker or broker.name == "paper":
            return {"ok": True}
        bal = broker.balances()
        mismatches = {}
        for sym, p in self.book.positions.items():
            base = sym[: -len(quote)] if sym.endswith(quote) else sym
            have = bal.get(base, 0.0)
            if have + 1e-9 < p.qty * 0.98:
                mismatches[sym] = {"book": p.qty, "exchange": have}
        if mismatches:
            self.reconciliation_failures += 1
            if self.kill_switch is not None:
                self.kill_switch.check_reconciliation(self.reconciliation_failures, f"position mismatch {mismatches}")
        else:
            self.reconciliation_failures = 0
        return {"ok": not mismatches, "mismatches": mismatches}

    def recent(self, n: int = 50) -> list[dict]:
        return [o.to_dict() for o in sorted(self.orders.values(), key=lambda o: o.created_at)[-n:]]
