"""
Event-driven backtester that runs the *real* TradingEngine (all 300 workers, brains, risk, gateway) on
historical candles with a simulated clock.

No look-ahead: at each simulated time t the MarketView only exposes candles that closed at or before t,
feeds only expose items received before t, and orders fill at the decision-time price plus slippage
(what a live bot acting right after the candle close gets). Stops fill at the stop level (or worse on gaps).
"""
from __future__ import annotations

import copy
import datetime as dt
import logging
import tempfile

import numpy as np
import pandas as pd

from lunatrade.backtest.metrics import full_report
from lunatrade.config import Config
from lunatrade.core.bus import EventBus
from lunatrade.core.clock import SimClock
from lunatrade.core.events import Topic
from lunatrade.core.types import Mode
from lunatrade.data.feeds import FeedHub
from lunatrade.data.market_store import INTERVAL_MS, MarketStore, from_ms
from lunatrade.engine import TradingEngine
from lunatrade.execution.brokers.paper import PaperBroker
from lunatrade.execution.positions import PositionBook

log = logging.getLogger("lunatrade.backtest")


class BacktestResult:
    def __init__(self, metrics, equity_curve, trades, orders, proposals, weights, timestamps):
        self.metrics = metrics
        self.equity_curve = equity_curve
        self.trades = trades
        self.orders = orders
        self.proposals = proposals
        self.weights = weights
        self.timestamps = timestamps

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "trades": self.trades[-200:], "weights": self.weights,
                "equity_curve": [{"ts": t, "equity": e} for t, e in zip(self.timestamps, self.equity_curve)][-2000:]}


class Backtester:
    def __init__(self, cfg: Config, data: dict[str, pd.DataFrame], interval: str | None = None, warmup: int = 200,
                 feeds: FeedHub | None = None, starting_equity: float | None = None, decision_every: int = 1,
                 broker_overrides: dict | None = None, initial_weights: dict | None = None, learn: bool = True,
                 check_staleness: bool = True, start_index: int | None = None, end_index: int | None = None):
        self.cfg = Config(copy.deepcopy(cfg.data))
        self.cfg.set("mode", "PAPER")
        if not learn:
            self.cfg.set("meta_learning.enabled", False)
        self.interval = interval or self.cfg.get("interval", "15m")
        self.data = data
        self.warmup = warmup
        self.feeds = feeds or FeedHub()
        self.starting = float(starting_equity or self.cfg.get("paper.starting_equity", 10_000))
        self.decision_every = max(1, decision_every)
        self.broker_overrides = broker_overrides or {}
        self.initial_weights = initial_weights or {}
        self.check_staleness = check_staleness
        self.start_index, self.end_index = start_index, end_index

    def run(self) -> BacktestResult:
        n_max = max(len(df) for df in self.data.values())
        store = MarketStore(self.interval, max_bars=n_max + 10, view_cap=600)
        for sym, df in self.data.items():
            store.load_frame(sym, df)
        symbols = list(self.data)
        market = next((s for s in ("BTCUSDT",) if s in symbols), symbols[0])
        ot = store.state(market).open_time
        # regular grid of candle-close times: gaps in the data (exchange outages) become stale-data periods
        times = np.arange(int(ot[0]) + store.interval_ms, int(ot[-1]) + store.interval_ms + 1, store.interval_ms)
        lo = max(self.warmup, self.start_index or 0)
        hi = min(len(times), self.end_index or len(times))
        if hi - lo < 2:
            raise ValueError(f"not enough bars to backtest (have {len(times)}, warmup {self.warmup})")
        clock = SimClock(from_ms(int(times[lo]) - 1))
        p = self.cfg.section("paper")
        broker = PaperBroker(self.starting, float(p.get("fee_bps", 10)), float(p.get("slippage_bps", 5)),
                             float(self.broker_overrides.get("partial_fill_probability", p.get("partial_fill_probability", 0))))
        broker.extra_slippage_bps = float(self.broker_overrides.get("extra_slippage_bps", 0))
        book = PositionBook(None, self.starting)
        bus = EventBus(history=500)
        engine = TradingEngine(self.cfg, store, self.feeds, {"paper": broker}, book, Mode.PAPER, symbols,
                               route=lambda s: "paper", bus=bus, repo=None, council=None, clock=clock,
                               state_dir=tempfile.mkdtemp(prefix="lt_bt_"), use_llm=False)
        for name, w in self.initial_weights.items():
            engine.memory.groups[name].weight = w
        trades, equity, stamps, exposure = [], [], [], []
        bus.subscribe(Topic.TRADE_CLOSED, lambda e: trades.append(e.payload))
        for i in range(lo, hi):
            t_ms = int(times[i])
            now = from_ms(t_ms)
            clock.set(now)
            if self.check_staleness:
                v = store.view(now)
                stale = {}
                for s in symbols:
                    c = v.candles(s, 1)
                    if len(c):
                        stale[s] = (t_ms - int(c.open_time[-1]) - store.interval_ms) / 1000
                engine.kill_switch.check_data(stale)
            if (i - lo) % self.decision_every == 0:
                engine.cycle(now)
            view = store.view(now)
            prices = {s: view.price(s) for s in symbols if len(view.candles(s))}
            equity.append(book.equity(prices))
            exposure.append(sum(pos.qty * prices.get(s, pos.entry) for s, pos in book.positions.items()))
            stamps.append(now.isoformat())
        # liquidate at the end so every trade is counted
        for sym in list(book.positions):
            engine.close_position(sym, "END_OF_TEST", ref_price=prices.get(sym))
        if equity:
            equity[-1] = book.equity(prices)
        bars_per_year = 365 * 86_400_000 / INTERVAL_MS[self.interval]
        metrics = full_report(equity, trades, bars_per_year, engine.gateway.recent(100_000), exposure)
        metrics.update({"bars": hi - lo, "symbols": symbols, "start": stamps[0], "end": stamps[-1],
                        "starting_equity": self.starting, "final_equity": equity[-1],
                        "kill_switch_events": len(engine.kill_switch.history),
                        "proposals": engine.proposals_seen})
        return BacktestResult(metrics, equity, trades, engine.gateway.recent(100_000), list(engine.proposals),
                              engine.memory.weights(), stamps)
