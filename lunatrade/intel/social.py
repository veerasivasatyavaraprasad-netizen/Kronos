"""
Social sentiment analysis. Mentions are not truth: every reading is classified as

    organic discussion | coordinated hype | bot activity | panic | capitulation | euphoria

and only becomes a (low-weight) signal, never a trade trigger on its own.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from lunatrade.intel import sentiment
from lunatrade.intel.assets import extract_assets

PANIC_WORDS = {"crash", "dump", "rekt", "liquidated", "scam", "rug", "dead", "sell", "selling", "panic", "capitulation",
               "bottom", "worthless", "lost", "zero"}
EUPHORIA_WORDS = {"moon", "mooning", "100x", "lambo", "ath", "pump", "send", "parabolic", "rich", "1000x", "gem",
                  "ape", "all in", "to the moon", "wagmi"}
SHILL = re.compile(r"(t\.me/|discord\.gg|airdrop|giveaway|presale|dm me|join now|guaranteed)", re.I)


@dataclass
class Post:
    ts: dt.datetime
    community: str
    title: str
    text: str = ""
    author: str = ""
    score: float = 0
    comments: float = 0
    assets: list = field(default_factory=list)

    def __post_init__(self):
        if not self.assets:
            self.assets = extract_assets(f"{self.title} {self.text}")


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


@dataclass
class SocialReading:
    asset: str
    posts: int
    mention_velocity: float      # mentions in last window / previous window
    sentiment: float             # engagement-weighted, -1..1
    author_diversity: float      # unique authors / posts
    duplicate_ratio: float       # near-identical texts / posts
    shill_ratio: float
    panic: float                 # 0..1
    euphoria: float              # 0..1
    classification: str
    confidence: float


def analyze(posts: list[Post], asset: str | None, now: dt.datetime, window_h: float = 6) -> SocialReading:
    w = dt.timedelta(hours=window_h)
    cur = [p for p in posts if now - w <= p.ts <= now and (asset is None or asset in p.assets)]
    prev = [p for p in posts if now - 2 * w <= p.ts < now - w and (asset is None or asset in p.assets)]
    n = len(cur)
    if n == 0:
        return SocialReading(asset or "*", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "QUIET", 0.0)
    velocity = (n + 1) / (len(prev) + 1)
    weights = [1 + math.log1p(max(0, p.score) + max(0, p.comments)) for p in cur]
    sents = [sentiment.score(f"{p.title} {p.text[:300]}") for p in cur]
    sent = sum(s * wt for s, wt in zip(sents, weights)) / sum(weights)
    authors = {p.author for p in cur if p.author}
    diversity = len(authors) / n if authors else 1.0
    texts = Counter(_norm(p.title)[:80] for p in cur)
    dup = sum(c - 1 for c in texts.values() if c > 1) / n
    shill = sum(1 for p in cur if SHILL.search(f"{p.title} {p.text}")) / n
    low = [f" {_norm(p.title + ' ' + p.text[:300])} " for p in cur]
    panic = sum(1 for t in low if any(f" {x} " in t for x in PANIC_WORDS)) / n
    euph = sum(1 for t in low if any(f" {x} " in t for x in EUPHORIA_WORDS)) / n

    if dup > 0.3 or (diversity < 0.35 and n >= 8):
        cls = "BOT_ACTIVITY"
    elif shill > 0.2 or (velocity > 3 and diversity < 0.6 and euph > 0.3):
        cls = "COORDINATED_HYPE"
    elif panic > 0.35 and sent < -0.3 and velocity > 2:
        cls = "CAPITULATION"
    elif panic > 0.25 and sent < -0.2:
        cls = "PANIC"
    elif euph > 0.3 and sent > 0.3:
        cls = "EUPHORIA"
    else:
        cls = "ORGANIC"
    sample = min(1.0, n / 30)
    trust = {"ORGANIC": 1.0, "PANIC": 0.8, "CAPITULATION": 0.8, "EUPHORIA": 0.7, "COORDINATED_HYPE": 0.3,
             "BOT_ACTIVITY": 0.1}[cls]
    return SocialReading(asset or "*", n, round(velocity, 3), round(sent, 3), round(diversity, 3), round(dup, 3),
                         round(shill, 3), round(panic, 3), round(euph, 3), cls, round(sample * trust, 3))
