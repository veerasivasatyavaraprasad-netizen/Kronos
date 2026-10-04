"""
Worker framework.

A worker is a small, deterministic analyst (statistics/rules, no LLM) that turns the current
MarketView + FeedView into structured `Signal`s. 300 of them run every decision cycle; they are cheap
because indicators are memoised per candle and most workers touch a few hundred numbers.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from lunatrade.core.types import Direction, Family, Signal
from lunatrade.data.feeds import FeedView
from lunatrade.data.market_store import MarketView

log = logging.getLogger("lunatrade.workers")

ONLINE, DEGRADED, OFFLINE, IDLE = "ONLINE", "DEGRADED", "OFFLINE", "IDLE"


@dataclass
class WorkerContext:
    view: MarketView
    feeds: FeedView
    symbols: list
    now: dt.datetime
    market_symbol: str = "BTCUSDT"
    interval: str = "15m"
    regime: dict = field(default_factory=dict)           # symbol -> RegimeState (previous cycle)
    portfolio: dict = field(default_factory=dict)        # snapshot from the portfolio brain
    performance: dict = field(default_factory=dict)      # strategy memory stats
    signals: list = field(default_factory=list)          # phase-1 signals, visible to phase-2 workers
    bar_index: int = 0

    def signal(self, worker, symbol, direction, confidence, **kw) -> Signal:
        return Signal(worker_id=worker.id, family=worker.family, symbol=symbol,
                      direction=direction if isinstance(direction, Direction) else Direction(direction),
                      confidence=confidence, setup=kw.pop("setup", worker.name), ts=self.now, **kw)


@dataclass
class WorkerHealth:
    status: str = IDLE
    runs: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_run: str | None = None
    last_error: str = ""
    avg_latency_ms: float = 0.0
    signals_emitted: int = 0
    reason: str = ""


class Worker:
    family: Family = Family.TECHNICAL
    per_symbol: bool = True
    phase: int = 1
    min_bars: int = 60
    every_bars: int = 1          # run every N decision bars (heavy workers)
    requires: tuple = ()         # feed names / capabilities, for health reporting

    def __init__(self, wid: str, name: str, params: dict | None = None):
        self.id = wid
        self.name = name
        self.params = params or {}
        self.health = WorkerHealth()

    # override one of these
    def evaluate_symbol(self, ctx: WorkerContext, symbol: str):
        return None

    def evaluate(self, ctx: WorkerContext) -> list[Signal]:
        out = []
        for sym in ctx.symbols:
            if len(ctx.view.candles(sym)) < self.min_bars and self.family not in (
                    Family.NEWS, Family.SOCIAL, Family.ONCHAIN, Family.WHALE, Family.MACRO):
                continue
            res = self.evaluate_symbol(ctx, sym)
            if res is None:
                continue
            out.extend(res if isinstance(res, list) else [res])
        return out

    def available(self, ctx: WorkerContext) -> tuple[bool, str]:
        return True, ""

    def run(self, ctx: WorkerContext) -> list[Signal]:
        h = self.health
        if self.every_bars > 1 and ctx.bar_index % self.every_bars:
            return []
        ok, why = self.available(ctx)
        if not ok:
            h.status, h.reason = OFFLINE, why
            return []
        t0 = time.perf_counter()
        try:
            sigs = [s for s in self.evaluate(ctx) if s is not None]
            h.consecutive_errors = 0
            h.status = ONLINE
            h.reason = ""
        except Exception as e:
            h.errors += 1
            h.consecutive_errors += 1
            h.last_error = f"{type(e).__name__}: {e}"[:300]
            h.status = OFFLINE if h.consecutive_errors >= 3 else DEGRADED
            log.debug("worker %s failed: %s", self.id, e, exc_info=True)
            sigs = []
        ms = (time.perf_counter() - t0) * 1000
        h.runs += 1
        h.avg_latency_ms = ms if h.runs == 1 else 0.9 * h.avg_latency_ms + 0.1 * ms
        h.last_run = ctx.now.isoformat()
        h.signals_emitted += len(sigs)
        return sigs

    def describe(self) -> dict:
        return {"id": self.id, "name": self.name, "family": self.family.value, "params": self.params,
                "phase": self.phase, "llm": False, **self.health.__dict__}


class FnWorker(Worker):
    """Worker defined by a function fn(worker, ctx, symbol) -> Signal | list | None."""

    def __init__(self, wid: str, name: str, family: Family, fn: Callable, params: dict | None = None,
                 per_symbol: bool = True, min_bars: int = 60, phase: int = 1, every_bars: int = 1,
                 requires: tuple = (), availability: Callable | None = None):
        super().__init__(wid, name, params)
        self.family = family
        self.fn = fn
        self.per_symbol = per_symbol
        self.min_bars = min_bars
        self.phase = phase
        self.every_bars = every_bars
        self.requires = requires
        self._availability = availability

    def available(self, ctx):
        return self._availability(self, ctx) if self._availability else (True, "")

    def evaluate_symbol(self, ctx, symbol):
        return self.fn(self, ctx, symbol)

    def evaluate(self, ctx):
        if self.per_symbol:
            return super().evaluate(ctx)
        res = self.fn(self, ctx, None)
        if res is None:
            return []
        return res if isinstance(res, list) else [res]


def spec(family: Family, name: str, fn: Callable, **kw) -> dict:
    return {"family": family, "name": name, "fn": fn, **kw}


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))
