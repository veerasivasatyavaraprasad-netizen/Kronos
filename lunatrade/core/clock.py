"""Clocks. Every component asks the clock for 'now' so a backtest can never see the future."""
from __future__ import annotations

import datetime as dt

from lunatrade.core.types import utcnow


class Clock:
    def now(self) -> dt.datetime:
        return utcnow()

    @property
    def simulated(self) -> bool:
        return False


class SimClock(Clock):
    def __init__(self, start: dt.datetime):
        self._now = start

    def now(self) -> dt.datetime:
        return self._now

    def set(self, t: dt.datetime) -> None:
        if t < self._now:
            raise ValueError("SimClock cannot move backwards")
        self._now = t

    @property
    def simulated(self) -> bool:
        return True
