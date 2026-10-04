"""
TradingEngine - one decision cycle, identical in live trading and backtests:

    exits (stops/targets/trailing/time)  ->  account marks + kill-switch checks
    -> 300 workers (phase 1)  ->  regime engine  ->  phase-2 workers (strategy eval, portfolio, adversarial)
    -> strategy memory (score past signals, meta-learn weights)
    -> Lead Brain (conviction per symbol) -> [LLM council review] -> Devil's Advocate (veto)
    -> Risk Engine -> Portfolio Brain -> (human approval in APPROVAL mode) -> Execution Gateway

Every step publishes an event on the bus, so the whole chain is auditable and explainable.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from collections import Counter, deque

import numpy as np

from lunatrade.brains.devils_advocate import DevilsAdvocate
from lunatrade.brains.lead_brain import LeadBrain
from lunatrade.brains.regime_engine import RegimeEngine
from lunatrade.control.approvals import ApprovalQueue
from lunatrade.core.bus import EventBus
from lunatrade.core.clock import Clock
from lunatrade.core.events import Topic
from lunatrade.core.types import MARKET_WIDE, Direction, Mode, OrderRequest, Regime, new_id
from lunatrade.data.market_store import to_ms
from lunatrade.execution.gateway import ExecutionGateway
from lunatrade.execution.order_state import OrderStatus
from lunatrade.indicators import ta
from lunatrade.memory.knowledge import EventMemory, prices_by_base
from lunatrade.memory.performance import StrategyMemory
from lunatrade.risk.engine import AccountState, MarketConditions, RiskEngine
from lunatrade.risk.kill_switch import KillSwitch
from lunatrade.risk.portfolio import PortfolioBrain
from lunatrade.workers.base import WorkerContext
from lunatrade.workers.registry import build_roster

log = logging.getLogger("lunatrade.engine")


class TradingEngine:
    def __init__(self, cfg, store, feeds, brokers: dict, book, mode: Mode, symbols: list[str], route=None,
                 bus: EventBus | None = None, repo=None, council=None, clock: Clock | None = None, notifier=None,
                 state_dir=None, use_llm: bool = True, max_new_positions_per_cycle: int = 2):
        self.cfg = cfg
        self.store, self.feeds = store, feeds
        self.symbols = list(symbols)
        self.mode = mode
        self.bus = bus or EventBus()
        self.repo = repo
        self.council = council if use_llm else None
        self.clock = clock or Clock()
        self.notifier = notifier
        self.book = book
        self.route = route or (lambda sym: next(iter(brokers)))
        self.market_symbol = next((s for s in ("BTCUSDT", "BTCUSD") if s in self.symbols), self.symbols[0])
        self.max_new = max_new_positions_per_cycle

        self.workers = build_roster(cfg)
        self.memory = StrategyMemory(store.interval_ms, cfg.section("meta_learning"))
        self.regime_engine = RegimeEngine()
        self.lead = LeadBrain(cfg.section("lead_brain"), cfg.section("risk"), self.council, self.memory.calibration)
        self.devil = DevilsAdvocate(cfg.section("devils_advocate") | {"llm": use_llm and cfg.get("devils_advocate.llm", True)},
                                    self.council)
        self.kill_switch = KillSwitch(state_dir or cfg.output_dir, cfg.section("kill_switch"), cfg.section("risk"),
                                      on_change=self._on_kill_switch)
        self.risk = RiskEngine(cfg.section("risk"), self.kill_switch)
        self.portfolio = PortfolioBrain(cfg.section("risk"))
        self.gateway = ExecutionGateway(brokers, book, mode, self.bus, self.kill_switch, clock=self.clock,
                                        reconcile_delay=0.0 if self.clock.simulated else 1.0)
        self.events = EventMemory(repo)
        self.approvals = ApprovalQueue(float(cfg.get("approval.timeout_minutes", 15)), notifier, repo)

        self.regimes: dict = {}
        self.assessments: dict = {}
        self.last_signals: list = []
        self.proposals: deque = deque(maxlen=300)
        self.cycle_count = 0
        self.last_cycle_at: dt.datetime | None = None
        self.last_cycle_ms = 0.0
        self.proposals_seen = 0
        self.weight_changes: deque = deque(maxlen=100)
        self._news_tracked: set = set()
        self._wire_bus()
        if repo is not None:
            try:
                repo.save_agents(self.workers)
                self.memory.load(repo.strategies())
            except Exception as e:
                log.warning("could not restore agents/strategy memory: %s", e)

    # ------------------------------------------------------------------ bus wiring / persistence
    def _wire_bus(self) -> None:
        self.bus.subscribe(Topic.TRADE_CLOSED, self._on_trade_closed)
        if self.repo is not None:
            self.bus.subscribe("*", self._persist_event)
            self.bus.subscribe("order.*", lambda e: self.repo.order(e.payload))

    def _persist_event(self, e) -> None:
        if e.topic in (Topic.SIGNAL_EVENT,):
            return   # signals are stored in their own table
        self.repo.event(e)

    def _on_trade_closed(self, e) -> None:
        o = e.payload
        self.memory.record_outcome(o)
        if self.repo is not None:
            review = None
            if self.council is not None and not self.clock.simulated:
                review = self.council.ask_json("post_trade", "post_trade", {"outcome": o})
            o = {**o, "review": review}
            self.repo.trade_outcome(o, self.mode.value)
        if self.notifier and not self.clock.simulated:
            self.notifier.send(f"Closed {o['symbol']} ({o['reason']}): PnL {o['pnl']:+.2f} ({o['return_pct']:+.2%}, "
                               f"{o['r_multiple']:+.2f}R)", "trade")

    def _on_kill_switch(self, rec: dict) -> None:
        self.bus.emit(Topic.KILL_SWITCH, rec, source="kill_switch")
        if self.repo is not None:
            self.repo.alert("CRITICAL" if rec["kind"] == "TRIP" else "WARN", "kill_switch",
                            f"{rec['kind']} {rec['key']}: {rec['detail']}", rec)
        if self.notifier and rec["kind"] in ("TRIP", "HALT"):
            self.notifier.send(f"⛔ Kill switch {rec['kind']} {rec['key']}: {rec['detail']}", "critical")

    # ------------------------------------------------------------------ helpers
    def _atr_pct(self, view, sym) -> float:
        c = view.candles(sym, 120)
        if len(c) < 20:
            return 0.02
        a = ta.atr(c.high, c.low, c.close, 14)[-1]
        return float(a / c.close[-1]) if c.close[-1] and not np.isnan(a) else 0.02

    def _ctx_value(self, signals, sym, key, default=None):
        for s in signals:
            if s.symbol == sym and key in s.context:
                return s.context[key]
        return default

    def account(self, prices: dict) -> AccountState:
        snap = self.book.snapshot(prices)
        return AccountState(snap["equity"], self.book.cash, self.book.peak_equity,
                            self.book.day_anchor.get("equity", snap["equity"]),
                            self.book.week_anchor.get("equity", snap["equity"]), snap["positions"])

    # ------------------------------------------------------------------ the cycle
    def cycle(self, now: dt.datetime | None = None) -> dict:
        t0 = time.perf_counter()
        now = now or self.clock.now()
        sim = self.clock.simulated
        view, fview = self.store.view(now), self.feeds.view(now)
        prices = {s: view.price(s) for s in self.symbols if len(view.candles(s))}
        result = {"ts": now.isoformat(), "exits": [], "proposals": [], "orders": []}

        # 1) protective exits first (they are always allowed, even with the kill switch on)
        for sym in list(self.book.positions):
            c = view.candles(sym, 2)
            if not len(c) or sym not in prices:
                continue
            ex = self.book.exit_signal(sym, prices[sym], c.low[-1], c.high[-1], self._atr_pct(view, sym) * prices[sym],
                                       self.cfg.section("risk"))
            if ex:
                reason, px = ex
                order = self.close_position(sym, reason, ref_price=px if sim else prices[sym])
                result["exits"].append({"symbol": sym, "reason": reason, "status": order.status.value if order else None})

        # 2) account marks + hard limits
        marks = self.book.mark(prices, now)
        self.kill_switch.check_account(marks["equity"], marks["peak_equity"], marks["day_start_equity"],
                                       marks["week_start_equity"])

        # 3) the worker network
        snap = self.book.snapshot(prices)
        portfolio_ctx = {**snap, "smart_money_wallets": self.cfg.get("feeds.whale.smart_money_wallets", [])}
        groups = self.memory.summary()
        ctx = WorkerContext(view, fview, self.symbols, now, self.market_symbol, self.store.interval,
                            regime=self.regimes, portfolio=portfolio_ctx, performance={"groups": groups},
                            bar_index=self.cycle_count)
        phase1 = [s for w in self.workers if w.phase == 1 for s in w.run(ctx)]
        self.regimes = self.regime_engine.compute(phase1, self.symbols, self.market_symbol)
        ctx.signals, ctx.regime = phase1, self.regimes
        phase2 = [s for w in self.workers if w.phase == 2 for s in w.run(ctx)]
        signals = phase1 + phase2
        self.last_signals = signals
        bar_ms = to_ms(now)

        # 4) strategy memory / meta-learning
        self.memory.evaluate_due(prices, bar_ms)
        self.memory.record_signals(signals, prices, bar_ms)
        if self.cfg.get("meta_learning.enabled", True) and self.cycle_count and self.cycle_count % 24 == 0:
            changes = self.memory.update_weights()
            if changes:
                self.weight_changes.append({"ts": now.isoformat(), "changes": changes})
                if self.repo is not None:
                    for name, st in self.memory.summary().items():
                        self.repo.strategy(name, st)
            priors = self.events.impact_priors()
            if priors:
                self.feeds.news.priors.update(priors)
        self._track_news(now, prices)

        # 5) conviction per symbol
        gw = self.memory.weights()
        min_conv = self.memory.min_conviction(float(self.cfg.get("lead_brain.min_conviction", 62)))
        opens, closes = [], []
        for sym in self.symbols:
            if sym not in prices:
                continue
            a = self.lead.assess(sym, signals, self.regimes.get(sym), gw)
            self.assessments[sym] = a
            p = self.lead.propose(a, prices[sym], self._atr_pct(view, sym), snap["equity"], self.book.qty(sym),
                                  self.regimes.get(sym), min_conv)
            if p is None:
                continue
            p.ts = now
            (closes if p.action == "CLOSE" else opens).append((p, a))

        for p, a in closes:
            self.proposals_seen += 1
            self._record(p, "CLOSE_SIGNAL")
            order = self.close_position(p.symbol, "SIGNAL_EXIT", ref_price=prices[p.symbol], proposal_id=p.id)
            result["proposals"].append({"id": p.id, "symbol": p.symbol, "action": "CLOSE",
                                        "status": order.status.value if order else None})

        opens.sort(key=lambda x: -x[0].conviction)
        for p, a in opens[: self.max_new]:
            if self.book.qty(p.symbol) > 0 and self.book.positions[p.symbol].qty * prices[p.symbol] >= \
                    0.9 * snap["equity"] * float(self.cfg.get("risk.max_position_pct", 0.1)):
                continue
            self.proposals_seen += 1
            status = self._handle_open(p, a, signals, view, prices, min_conv)
            result["proposals"].append({"id": p.id, "symbol": p.symbol, "action": "OPEN", "status": status,
                                        "conviction": p.conviction})

        # 6) approvals that came back YES
        if self.mode == Mode.APPROVAL:
            self.approvals.expire(now)
            result["orders"] += self.execute_approved(prices, now)

        # 7) persistence
        self.cycle_count += 1
        self.last_cycle_at = now
        self.last_cycle_ms = (time.perf_counter() - t0) * 1000
        if self.repo is not None:
            self._persist_cycle(now, signals, prices, snap)
        result["equity"] = self.book.equity(prices)
        result["signals"] = len(signals)
        result["cycle_ms"] = round(self.last_cycle_ms, 1)
        return result

    def _track_news(self, now, prices) -> None:
        pb = prices_by_base(prices)
        for item in self.feeds.news.items[-200:]:
            if item.id in self._news_tracked or item.ts > now:
                continue
            self._news_tracked.add(item.id)
            self.events.track(item, pb)
            if self.repo is not None:
                try:
                    self.repo.news(item)
                except Exception:
                    pass
        self.events.update(now, pb)

    def _record(self, p, status: str) -> None:
        self.proposals.append({"id": p.id, "ts": p.ts.isoformat(), "symbol": p.symbol, "action": p.action,
                               "conviction": p.conviction, "status": status, "rationale": p.rationale[:300],
                               "devils_advocate": p.devils_advocate, "llm_review": p.llm_review})
        if self.repo is not None:
            self.repo.proposal(p, status)

    def _handle_open(self, p, a, signals, view, prices, min_conv) -> str:
        sym = p.symbol
        self.bus.emit(Topic.TRADE_PROPOSAL, p.to_dict(), source="lead_brain", correlation_id=p.id)
        if self.council is not None:
            p = self.lead.llm_review(p, a)
            if p.conviction < min_conv:
                self._record(p, "REJECTED_LLM_REVIEW")
                return "REJECTED_LLM_REVIEW"
        snap = self.book.snapshot(prices)
        regime = self.regimes.get(sym)
        da = self.devil.review(p, signals, regime, snap)
        p.devils_advocate = da.to_dict()
        self.bus.emit(Topic.DEVILS_ADVOCATE, da.to_dict(), source="devils_advocate", correlation_id=p.id)

        broker = self.gateway.brokers.get(self.route(sym))
        try:
            min_notional = float(broker.filters(sym).get("min_notional", 0)) if broker else 0.0
        except Exception:
            min_notional = 0.0
        c = view.candles(sym, max(1, int(86_400_000 / self.store.interval_ms)))
        high_vol = bool(self._ctx_value(signals, sym, "high_vol", False)) or \
            bool(regime and regime.scores.get(Regime.HIGH_VOL.value, 0) >= 0.5)
        clusters = next((s.context["clusters"] for s in signals if s.setup == "CORRELATION_CLUSTERS"), [])
        held = set(snap["positions"])
        corr_warn = any(sym in cl and len(held & set(cl) - {sym}) >= 1 for cl in clusters)
        mkt = MarketConditions(
            price=prices[sym], atr_pct=self._atr_pct(view, sym), spread_bps=view.spread_bps(sym),
            quote_volume_24h=float(c.quote_volume.sum()) if len(c) else None,
            expected_slippage_bps=self._ctx_value(signals, sym, "expected_slippage_bps"),
            high_vol=high_vol, correlation_warning=corr_warn,
            regime_exposure_multiplier=regime.exposure_multiplier if regime else 0.5,
            exchange_ok=broker.healthy() if broker else False, min_notional=min_notional)
        d = self.risk.evaluate(p, da, self.account(prices), mkt)
        betas = {s.symbol: s.context.get("beta", 1.0) for s in signals if s.setup == "BETA_TO_MARKET"}
        d = self.portfolio.adjust(p, d, snap["equity"], snap["positions"], clusters, betas)
        self.bus.emit(Topic.RISK_DECISION, d.to_dict(), source="risk_engine", correlation_id=p.id)
        if self.repo is not None:
            self.repo.risk_decision(d, da)
        if not d.approved:
            self._record(p, "REJECTED_RISK")
            return "REJECTED_RISK"
        if self.mode == Mode.RESEARCH:
            self._record(p, "RESEARCH_ONLY")
            return "RESEARCH_ONLY"
        if self.mode == Mode.APPROVAL:
            self.approvals.request(p, d, da, self.clock.now())
            self._record(p, "PENDING_APPROVAL")
            self.bus.emit(Topic.APPROVAL_REQUESTED, {"proposal_id": p.id, "notional": d.approved_notional},
                          source="control", correlation_id=p.id)
            return "PENDING_APPROVAL"
        order = self._execute_open(p, d, prices[sym])
        status = "EXECUTED" if order.status == OrderStatus.POSITION_UPDATED else f"ORDER_{order.status.value}"
        self._record(p, status)
        return status

    def _execute_open(self, p, d, price: float):
        self.gateway.authorize(p.id, d.approved_notional)
        req = OrderRequest(p.id, p.symbol, "BUY", notional=d.approved_notional, reference_price=price,
                           broker=self.route(p.symbol), reason="OPEN")
        order = self.gateway.execute(req, stop=p.stop_price, target=p.target_price,
                                     meta={"supporting": p.supporting_signals, "conviction": p.conviction,
                                           "regime": p.regime})
        if order.status == OrderStatus.POSITION_UPDATED and self.notifier and not self.clock.simulated:
            self.notifier.send(f"Opened {p.symbol}: ${order.filled_notional:,.2f} @ {order.avg_price:.6g} "
                               f"(conviction {p.conviction:.0f}, stop {p.stop_price:.6g})", "trade")
        return order

    def execute_approved(self, prices: dict, now: dt.datetime) -> list[dict]:
        out = []
        for req in self.approvals.approved_ready():
            p = req.proposal
            px = prices.get(p.symbol)
            if not px or abs(px / p.entry_price - 1) > 0.01:
                req.status = "EXPIRED"
                self._record(p, "APPROVAL_STALE_PRICE")
                continue
            p.entry_price = px
            d = self.risk.evaluate(p, req.devils, self.account(prices), MarketConditions(px, self._atr_pct(
                self.store.view(now), p.symbol), exchange_ok=True))
            if not d.approved:
                req.status = "REJECTED"
                self._record(p, "REJECTED_RISK_AFTER_APPROVAL")
                continue
            d.approved_notional = min(d.approved_notional, req.decision.approved_notional)
            order = self._execute_open(p, d, px)
            self.approvals.mark_executed(req)
            self._record(p, "EXECUTED" if order.status == OrderStatus.POSITION_UPDATED else f"ORDER_{order.status.value}")
            out.append(order.to_dict())
        return out

    # ------------------------------------------------------------------ manual controls
    def close_position(self, symbol: str, reason: str, ref_price: float | None = None, proposal_id: str = ""):
        pos = self.book.positions.get(symbol)
        if not pos:
            return None
        if ref_price is None:
            ref_price = self.store.view(self.clock.now()).price(symbol)
        req = OrderRequest(proposal_id or pos.proposal_id or new_id("close_"), symbol, "SELL", quantity=pos.qty,
                           reference_price=ref_price, broker=self.route(symbol), reason=reason)
        return self.gateway.execute(req)

    def pause(self, by: str) -> None:
        """Stop opening new trades (exits and stops keep working)."""
        self.kill_switch.halt("PAUSED", f"paused by {by}")

    def resume(self, by: str) -> None:
        self.kill_switch.clear_halt("PAUSED")

    @property
    def paused(self) -> bool:
        return "PAUSED" in self.kill_switch.halts

    def emergency_close_all(self, by: str) -> list[dict]:
        self.kill_switch.trip("EMERGENCY_CLOSE", f"by {by}")
        out = []
        for sym in list(self.book.positions):
            o = self.close_position(sym, "EMERGENCY")
            out.append(o.to_dict() if o else {"symbol": sym, "status": "no position"})
        return out

    # ------------------------------------------------------------------ persistence + status
    def _persist_cycle(self, now, signals, prices, snap) -> None:
        try:
            keep = [s for s in signals if s.direction != Direction.NEUTRAL or s.family.value in ("regime", "adversarial")]
            self.repo.signals(f"cyc_{self.cycle_count}", keep, prices)
            for sym, a in self.assessments.items():
                r = self.regimes.get(sym)
                self.repo.snapshot(now, sym, prices.get(sym), self.store.view(now).spread_bps(sym),
                                   r.label() if r else None, a.to_dict())
            self.repo.portfolio_snapshot(now, snap, self.mode.value)
            self.repo.positions(snap)
            if self.cycle_count % 10 == 1:
                self.repo.health(self.workers)
        except Exception as e:
            log.error("persistence failed: %s", e)

    def worker_counts(self) -> dict:
        c = Counter(w.health.status for w in self.workers)
        return {"total": len(self.workers), "online": c.get("ONLINE", 0), "degraded": c.get("DEGRADED", 0),
                "offline": c.get("OFFLINE", 0), "idle": c.get("IDLE", 0)}

    def status(self) -> dict:
        view = self.store.view(self.clock.now())
        prices = {s: view.price(s) for s in self.symbols if len(view.candles(s))}
        snap = self.book.snapshot(prices)
        top = sorted(self.assessments.values(), key=lambda a: -a.conviction)
        mr = self.regimes.get(MARKET_WIDE)
        return {
            "mode": self.mode.value, "paused": self.paused, "market_regime": mr.to_dict() if mr else None,
            "prices": prices, "portfolio": snap, "kill_switch": self.kill_switch.status(),
            "workers": self.worker_counts(), "cycle": self.cycle_count,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "last_cycle_ms": round(self.last_cycle_ms, 1),
            "signals": [a.to_dict() for a in top[:30]],
            "pending_approvals": [r.to_dict() for r in self.approvals.pending()],
            "recent_proposals": list(self.proposals)[-20:],
            "strategy_weights": self.memory.weights(), "min_conviction": self.memory.min_conviction(
                float(self.cfg.get("lead_brain.min_conviction", 62))),
        }
