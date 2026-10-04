"""Shared value types: signals, proposals, decisions. Plain dataclasses so 300 workers stay cheap."""
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    def __str__(self) -> str:  # json/logging friendly
        return self.value


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

    @property
    def sign(self) -> int:
        return {"LONG": 1, "SHORT": -1, "NEUTRAL": 0}[self.value]

    @classmethod
    def from_score(cls, score: float, dead_zone: float = 0.0) -> "Direction":
        if score > dead_zone:
            return cls.LONG
        if score < -dead_zone:
            return cls.SHORT
        return cls.NEUTRAL


class Mode(StrEnum):
    RESEARCH = "RESEARCH"   # analysis only, no orders of any kind
    PAPER = "PAPER"         # real data, simulated account
    SHADOW = "SHADOW"       # real data, orders validated by the exchange (/order/test) but not executed
    APPROVAL = "APPROVAL"   # every order needs a human YES
    LIVE = "LIVE"           # real orders inside risk limits, human emergency override


class Regime(StrEnum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    HIGH_VOL = "HIGH_VOL"
    LOW_VOL = "LOW_VOL"
    BULL = "BULL"
    BEAR = "BEAR"
    PANIC = "PANIC"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    ILLIQUID = "ILLIQUID"
    UNKNOWN = "UNKNOWN"


class Family(StrEnum):
    """Worker families. The Lead Brain scores each family as one input category."""
    SCANNER = "scanner"
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    MICROSTRUCTURE = "microstructure"
    NEWS = "news"
    SOCIAL = "social"
    ONCHAIN = "onchain"
    WHALE = "whale"
    MACRO = "macro"
    ARBITRAGE = "arbitrage"
    REGIME = "regime"
    FORECAST = "forecast"
    RISK = "risk"
    STRATEGY_EVAL = "strategy_eval"
    PORTFOLIO = "portfolio"
    ADVERSARIAL = "adversarial"


# Families whose output is directional evidence for the Lead Brain. The others produce context
# (regime votes, risk metrics, weights, vetoes) that other components consume.
DIRECTIONAL_FAMILIES = (
    Family.SCANNER, Family.TECHNICAL, Family.MOMENTUM, Family.VOLATILITY, Family.MICROSTRUCTURE,
    Family.NEWS, Family.SOCIAL, Family.ONCHAIN, Family.WHALE, Family.MACRO, Family.ARBITRAGE,
    Family.FORECAST,
)

MARKET_WIDE = "*"  # symbol used by signals that apply to the whole market (macro, fear & greed, ...)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        return obj.item()
    return obj


@dataclass(slots=True)
class Signal:
    """Structured output of a worker. A signal is evidence, never an order."""
    worker_id: str
    family: Family
    symbol: str
    direction: Direction
    confidence: float                 # 0..1, how sure the worker is about *its own* reading
    setup: str = ""
    expected_move: float = 0.0        # fractional, e.g. 0.034 = +3.4% (sign follows direction)
    time_horizon: str = "4h"
    evidence: list = field(default_factory=list)
    invalidations: list = field(default_factory=list)
    context: dict = field(default_factory=dict)   # non-directional numbers (risk metrics, regime votes, ...)
    ts: dt.datetime = field(default_factory=utcnow)

    def __post_init__(self):
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    @property
    def score(self) -> float:
        """Signed strength in [-1, 1]."""
        return self.direction.sign * self.confidence

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class CategoryScore:
    family: str
    score: float          # -100..100
    n_signals: int
    agreement: float      # 0..1 share of signals agreeing with the category's net direction
    weight: float


@dataclass(slots=True)
class TradeProposal:
    symbol: str
    direction: Direction
    conviction: float                       # 0..100 - NOT a probability
    win_probability: float | None           # calibrated from history; None until enough trades
    expected_move: float
    time_horizon: str
    entry_price: float
    stop_price: float
    target_price: float
    requested_notional: float
    category_scores: list = field(default_factory=list)
    supporting_signals: list = field(default_factory=list)   # worker ids that agreed
    opposing_signals: list = field(default_factory=list)
    regime: str = Regime.UNKNOWN.value
    rationale: str = ""
    action: str = "OPEN"                    # OPEN | CLOSE | REDUCE
    llm_review: dict = field(default_factory=dict)
    devils_advocate: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("prop_"))
    ts: dt.datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class DevilsAdvocateReport:
    proposal_id: str
    bull_case: float          # 0..10
    bear_case: float          # 0..10
    hidden_risks: list
    veto: bool
    risk_adjustment: float    # 0 .. -1, multiplies size by (1 + adjustment)
    notes: str = ""

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class RiskDecision:
    proposal_id: str
    approved: bool
    approved_notional: float
    requested_notional: float
    checks: dict              # name -> "OK" | "WARNING" | "FAIL: reason"
    reasons: list = field(default_factory=list)
    ts: dt.datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class OrderRequest:
    proposal_id: str
    symbol: str
    side: str                 # BUY | SELL
    notional: float | None = None
    quantity: float | None = None
    order_type: str = "MARKET"
    limit_price: float | None = None
    reference_price: float = 0.0
    broker: str = "paper"
    client_order_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))
