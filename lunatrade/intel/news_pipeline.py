"""
News intelligence pipeline.

SOURCE -> ARTICLE -> ENTITY EXTRACTION -> EVENT CLASSIFICATION -> SENTIMENT -> NOVELTY
       -> SOURCE CREDIBILITY -> MARKET IMPACT -> TIME DECAY -> SIGNAL

The LLM is optional and only refines the classification of high-impact, ambiguous items;
every stage works deterministically without it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import re
from dataclasses import asdict, dataclass, field

from lunatrade.intel import sentiment
from lunatrade.intel.assets import extract_assets, mentions_market

# event type -> (keywords, base impact 0..1, half-life hours, default direction bias)
EVENT_TYPES: dict[str, tuple[list[str], float, float, int]] = {
    "ETF_FLOW": (["etf", "inflow", "outflow", "spot bitcoin etf", "ishares", "fund flows", "blackrock"], 0.75, 36, 0),
    "REGULATION": (["sec ", "regulat", "cftc", "lawmakers", "bill", "senate", "congress", "mica", "license",
                    "compliance", "framework", "stablecoin act"], 0.7, 48, 0),
    "LAWSUIT": (["lawsuit", "sues", "sued", "court", "judge", "charges", "indict", "settlement", "plea"], 0.6, 48, -1),
    "HACK_EXPLOIT": (["hack", "exploit", "breach", "stolen", "drained", "attacker", "vulnerability", "phishing"],
                     0.85, 24, -1),
    "LISTING": (["will list", "lists ", "listing", "launchpool", "new trading pair", "adds support"], 0.6, 12, 1),
    "DELISTING": (["delist", "delisting", "will remove", "cease trading", "monitoring tag"], 0.75, 24, -1),
    "PARTNERSHIP": (["partnership", "partners with", "collaborat", "integrat", "teams up"], 0.35, 24, 1),
    "TOKEN_UNLOCK": (["unlock", "vesting", "token release", "cliff"], 0.5, 72, -1),
    "CENTRAL_BANK": (["federal reserve", "fomc", "rate cut", "rate hike", "powell", "ecb", "boj", "interest rate",
                      "basis points", "monetary policy"], 0.8, 48, 0),
    "MACRO_DATA": (["cpi", "inflation", "jobs report", "nonfarm", "payrolls", "unemployment", "gdp", "pce", "ppi"],
                   0.7, 24, 0),
    "GEOPOLITICAL": (["war", "sanction", "missile", "invasion", "conflict", "tariff", "election", "geopolit"],
                     0.6, 48, -1),
    "GOVERNANCE": (["governance", "proposal", "dao vote", "snapshot vote", "improvement proposal"], 0.3, 48, 0),
    "EXCHANGE_NEWS": (["binance", "coinbase", "kraken", "okx", "bybit", "exchange"], 0.4, 24, 0),
    "INSTITUTIONAL": (["treasury", "corporate", "microstrategy", "strategy inc", "buys bitcoin", "adds bitcoin",
                       "reserve", "custody", "bank"], 0.55, 72, 1),
    "STABLECOIN": (["stablecoin", "tether", "usdt", "usdc", "depeg", "mint", "minted", "redemption"], 0.5, 24, 0),
    "TECH_UPGRADE": (["upgrade", "hard fork", "mainnet", "testnet", "layer 2", "rollup", "launches", "release"],
                     0.4, 48, 1),
    "MINING": (["miner", "mining", "hashrate", "halving", "difficulty"], 0.35, 72, 0),
    "ADOPTION": (["adoption", "accepts", "payments", "legal tender", "integration"], 0.4, 72, 1),
    "MARKET_MOVE": (["price", "rally", "surge", "plunge", "crash", "liquidation", "all-time high", "slump"], 0.3, 6, 0),
}
GENERIC = "GENERAL"


@dataclass
class NewsItem:
    id: str
    ts: dt.datetime
    source: str
    title: str
    summary: str = ""
    url: str = ""
    assets: list = field(default_factory=list)
    market_wide: bool = False
    event_type: str = GENERIC
    event_scores: dict = field(default_factory=dict)
    sentiment: float = 0.0
    sentiment_label: str = "NEUTRAL"
    novelty: float = 1.0
    credibility: float = 0.5
    corroboration: int = 1
    impact: float = 0.0
    half_life_hours: float = 12.0
    confidence: float = 0.0
    llm: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d

    def decay(self, now: dt.datetime) -> float:
        age_h = max(0.0, (now - self.ts).total_seconds() / 3600)
        return 0.5 ** (age_h / max(self.half_life_hours, 0.1))

    def signed_strength(self, now: dt.datetime) -> float:
        """sentiment x impact x credibility x novelty x decay, in [-1, 1]."""
        return max(-1.0, min(1.0, self.sentiment * self.impact * self.credibility * (0.5 + 0.5 * self.novelty)
                             * self.decay(now) * min(1.5, 1 + 0.15 * (self.corroboration - 1)) * 1.8))


def item_id(source: str, title: str, url: str = "") -> str:
    return hashlib.sha1(f"{source}|{title.strip().lower()}|{url}".encode()).hexdigest()[:16]


_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "for", "and", "this", "that", "with", "from", "week", "today", "says", "after", "amid", "into", "over",
         "will", "its", "are", "has", "have", "new"}


def _shingles(text: str) -> set:
    """Content-word set: rewordings of the same story still overlap strongly."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def classify(text: str) -> tuple[str, dict]:
    low = f" {text.lower()} "
    scores = {}
    for etype, (kws, _, _, _) in EVENT_TYPES.items():
        hits = sum(1 for k in kws if k in low)
        if hits:
            scores[etype] = hits
    if not scores:
        return GENERIC, {}
    best = max(scores, key=lambda k: (scores[k], EVENT_TYPES[k][1]))
    return best, scores


class NewsPipeline:
    def __init__(self, source_credibility: dict | None = None, impact_priors: dict | None = None,
                 novelty_window_hours: float = 24, llm_council=None):
        self.cred = source_credibility or {}
        self.priors = impact_priors or {}       # event_type -> learned impact multiplier (from event memory)
        self.window = dt.timedelta(hours=novelty_window_hours)
        self.items: list[NewsItem] = []
        self._seen: set[str] = set()
        self.llm = llm_council

    def process(self, source: str, title: str, summary: str = "", url: str = "",
                ts: dt.datetime | None = None) -> NewsItem | None:
        ts = ts or dt.datetime.utcnow()
        iid = item_id(source, title, url)
        if iid in self._seen:
            return None
        self._seen.add(iid)
        text = f"{title}. {summary}"
        item = NewsItem(id=iid, ts=ts, source=source, title=title.strip(), summary=summary[:1000], url=url)
        # entity extraction
        item.assets = extract_assets(text)
        item.market_wide = not item.assets and mentions_market(text) or \
            (bool(item.assets) and mentions_market(title) and len(item.assets) > 2)
        # event classification
        item.event_type, item.event_scores = classify(text)
        _, base_impact, half_life, bias = EVENT_TYPES.get(item.event_type, ([], 0.2, 8, 0))
        item.half_life_hours = half_life
        # sentiment (title weighs double - it is what moves markets)
        s = (2 * sentiment.score(title) + sentiment.score(summary)) / 3 if summary else sentiment.score(title)
        if abs(s) < 0.05 and bias:
            s = 0.25 * bias
        item.sentiment, item.sentiment_label = s, sentiment.label(s)
        # novelty vs. recent items + corroboration by other sources
        sh = _shingles(title)
        recent = [x for x in self.items if ts - x.ts <= self.window]
        best_sim, corroborators = 0.0, set()
        for other in recent:
            osh = _shingles(other.title)
            sim = len(sh & osh) / max(1, len(sh | osh))
            if sim > best_sim:
                best_sim = sim
            if sim > 0.3 and other.source != source:
                corroborators.add(other.source)
                other.corroboration += 1
        item.novelty = round(1 - best_sim, 3)
        item.corroboration = 1 + len(corroborators)
        # source credibility
        item.credibility = float(self.cred.get(source, 0.5))
        # market impact
        item.impact = min(1.0, base_impact * float(self.priors.get(item.event_type, 1.0)))
        item.confidence = round(min(1.0, item.credibility * (0.4 + 0.6 * abs(item.sentiment)) *
                                    (0.6 + 0.4 * item.novelty) * min(1.3, 1 + 0.1 * (item.corroboration - 1))), 3)
        if self.llm is not None and item.impact >= 0.6 and abs(item.sentiment) < 0.3:
            self._llm_refine(item)
        self.items.append(item)
        cutoff = ts - dt.timedelta(days=7)
        self.items = [x for x in self.items if x.ts >= cutoff]
        return item

    def _llm_refine(self, item: NewsItem) -> None:
        """Ask the news-analyst LLM role for a structured second opinion; bounded blend with the rule result."""
        try:
            res = self.llm.ask_json("news_analyst", "classify_news", {
                "title": item.title, "summary": item.summary[:600], "rule_event_type": item.event_type,
                "rule_sentiment": round(item.sentiment, 3), "assets": item.assets,
                "allowed_event_types": list(EVENT_TYPES) + [GENERIC]})
        except Exception:
            return
        if not res:
            return
        llm_s = res.get("sentiment")
        if isinstance(llm_s, (int, float)) and -1 <= llm_s <= 1:
            item.sentiment = 0.5 * item.sentiment + 0.5 * float(llm_s)
            item.sentiment_label = sentiment.label(item.sentiment)
        et = res.get("event_type")
        if et in EVENT_TYPES:
            item.event_type = et
        item.llm = {k: res.get(k) for k in ("event_type", "sentiment", "reason", "provider")}

    def active(self, now: dt.datetime, min_decay: float = 0.05) -> list[NewsItem]:
        return [x for x in self.items if x.ts <= now and x.decay(now) >= min_decay]

    def asset_signal(self, asset: str, now: dt.datetime, event_types: set | None = None) -> tuple[float, list]:
        """Aggregate signed strength for one asset (market-wide news counts at 40%)."""
        total, used = 0.0, []
        for x in self.active(now):
            if event_types and x.event_type not in event_types:
                continue
            if asset in x.assets:
                w = 1.0
            elif x.market_wide:
                w = 0.4
            else:
                continue
            s = x.signed_strength(now) * w
            if s:
                total += s
                used.append(x)
        return math.tanh(total), used
