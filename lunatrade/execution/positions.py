"""
Position book: what LunaTrade itself holds (it never sells coins it did not buy), cash in paper mode,
realised PnL, equity peaks and day/week anchors for the loss limits, plus exit management
(stop-loss, take-profit, break-even, trailing stop, time stop).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    stop: float
    target: float
    opened_at: str
    proposal_id: str = ""
    high_water: float = 0.0
    bars_held: int = 0
    initial_stop: float = 0.0
    fees: float = 0.0
    supporting: list = field(default_factory=list)
    conviction: float = 0.0
    regime: str = ""

    @property
    def risk_per_unit(self) -> float:
        return max(self.entry - (self.initial_stop or self.stop), 1e-12)


class PositionBook:
    def __init__(self, path: Path | None, starting_cash: float = 10_000.0):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.positions: dict[str, Position] = {}
        self.cash = float(starting_cash)
        self.realized = 0.0
        self.peak_equity = float(starting_cash)
        self.day_anchor: dict = {}
        self.week_anchor: dict = {}
        self.closed: list = []
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        d = json.loads(self.path.read_text())
        self.cash = d.get("cash", self.cash)
        self.realized = d.get("realized", 0.0)
        self.peak_equity = d.get("peak_equity", self.cash)
        self.day_anchor = d.get("day_anchor", {})
        self.week_anchor = d.get("week_anchor", {})
        self.positions = {s: Position(**p) for s, p in d.get("positions", {}).items()}
        self.closed = d.get("closed", [])[-500:]

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cash": self.cash, "realized": self.realized, "peak_equity": self.peak_equity,
                                       "day_anchor": self.day_anchor, "week_anchor": self.week_anchor,
                                       "positions": {s: asdict(p) for s, p in self.positions.items()},
                                       "closed": self.closed[-500:]}, indent=2, default=str))
            os.replace(tmp, self.path)

    # ------------------------------------------------------------------ accounting
    def qty(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.qty if p else 0.0

    def equity(self, prices: dict) -> float:
        return self.cash + sum(p.qty * (prices.get(s) or p.entry) for s, p in self.positions.items())

    def mark(self, prices: dict, now: dt.datetime) -> dict:
        """Update peak and day/week anchors; returns the numbers the kill switch needs."""
        with self._lock:
            eq = self.equity(prices)
            self.peak_equity = max(self.peak_equity, eq)
            day, week = now.strftime("%Y-%m-%d"), now.strftime("%G-W%V")
            if self.day_anchor.get("key") != day:
                self.day_anchor = {"key": day, "equity": eq}
            if self.week_anchor.get("key") != week:
                self.week_anchor = {"key": week, "equity": eq}
            for s, p in self.positions.items():
                px = prices.get(s)
                if px:
                    p.high_water = max(p.high_water, px)
            return {"equity": eq, "peak_equity": self.peak_equity, "day_start_equity": self.day_anchor["equity"],
                    "week_start_equity": self.week_anchor["equity"], "cash": self.cash}

    def apply_buy(self, symbol: str, qty: float, price: float, fee: float, now: dt.datetime, stop: float,
                  target: float, proposal_id: str = "", supporting=None, conviction: float = 0.0, regime: str = "",
                  cash_managed: bool = True) -> Position:
        with self._lock:
            if cash_managed:
                self.cash -= qty * price + fee
            p = self.positions.get(symbol)
            if p:
                new_qty = p.qty + qty
                p.entry = (p.qty * p.entry + qty * price) / new_qty
                p.qty = new_qty
                p.fees += fee
                p.stop = max(p.stop, stop) if p.stop else stop
            else:
                p = Position(symbol, qty, price, stop, target, now.isoformat(), proposal_id, price, 0, stop, fee,
                             list(supporting or [])[:25], conviction, regime)
                self.positions[symbol] = p
            self.save()
            return p

    def apply_sell(self, symbol: str, qty: float, price: float, fee: float, now: dt.datetime, reason: str,
                   cash_managed: bool = True, dust_notional: float = 0.0) -> dict | None:
        """dust_notional: a remainder worth less than this (unsellable below exchange minimums) closes the position."""
        with self._lock:
            p = self.positions.get(symbol)
            if not p:
                return None
            qty = min(qty, p.qty)
            if cash_managed:
                self.cash += qty * price - fee
            entry_fee_share = p.fees * (qty / p.qty) if p.qty else 0
            pnl = (price - p.entry) * qty - fee - entry_fee_share
            self.realized += pnl
            r_mult = (price - p.entry) / p.risk_per_unit
            outcome = {"symbol": symbol, "qty": qty, "entry": p.entry, "exit": price, "pnl": pnl,
                       "return_pct": price / p.entry - 1, "r_multiple": r_mult, "reason": reason,
                       "opened_at": p.opened_at, "closed_at": now.isoformat(), "bars_held": p.bars_held,
                       "proposal_id": p.proposal_id, "supporting": p.supporting, "conviction": p.conviction,
                       "regime": p.regime, "fees": fee + entry_fee_share}
            p.qty -= qty
            p.fees -= entry_fee_share
            if p.qty <= max(1e-12, (p.qty + qty) * 1e-6) or p.qty * price < dust_notional:
                outcome["dust_qty"] = max(p.qty, 0.0)
                self.positions.pop(symbol)
            self.closed.append(outcome)
            self.save()
            return outcome

    # ------------------------------------------------------------------ exits
    def exit_signal(self, symbol: str, price: float, low: float, high: float, atr: float, risk_cfg: dict) -> tuple | None:
        """Return (reason, exit_price) when a protective exit triggers, updating trailing stops otherwise."""
        p = self.positions.get(symbol)
        if not p:
            return None
        p.bars_held += 1
        if low <= p.stop:
            return ("STOP_LOSS" if p.stop < p.entry else "TRAILING_STOP", min(price, p.stop) if price < p.stop else p.stop)
        if high >= p.target:
            return ("TAKE_PROFIT", max(p.target, min(price, high)) if price >= p.target else p.target)
        if p.bars_held >= int(risk_cfg.get("time_stop_bars", 96)):
            return ("TIME_STOP", price)
        # (stops were checked against the old levels first - the order of high/low inside a bar is unknown)
        p.high_water = max(p.high_water, high, price)
        # break-even after +1R, then trail
        if price - p.entry >= p.risk_per_unit and p.stop < p.entry:
            p.stop = p.entry
        trail = float(risk_cfg.get("trailing_stop_atr_mult", 2.5)) * atr
        if atr and p.high_water - trail > p.stop and p.high_water - p.entry >= 1.5 * p.risk_per_unit:
            p.stop = p.high_water - trail
        return None

    def snapshot(self, prices: dict) -> dict:
        pos = {}
        for s, p in self.positions.items():
            px = prices.get(s) or p.entry
            pos[s] = {"qty": p.qty, "entry": p.entry, "price": px, "notional": p.qty * px, "stop": p.stop,
                      "target": p.target, "unrealized": (px - p.entry) * p.qty, "opened_at": p.opened_at,
                      "bars_held": p.bars_held, "proposal_id": p.proposal_id}
        eq = self.equity(prices)
        return {"equity": eq, "cash": self.cash, "realized": self.realized, "peak_equity": self.peak_equity,
                "drawdown": (self.peak_equity - eq) / self.peak_equity if self.peak_equity else 0.0,
                "positions": pos, "exposure": sum(x["notional"] for x in pos.values()),
                "exposure_pct": sum(x["notional"] for x in pos.values()) / eq if eq else 0.0}
