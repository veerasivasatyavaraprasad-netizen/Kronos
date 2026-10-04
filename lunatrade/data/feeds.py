"""
External intelligence feeds (news, social, on-chain, whale, macro, cross-exchange prices).

`FeedHub` holds the processed state with timestamps; workers read it through `FeedView(as_of)`, which
hides anything received after the decision time. Collectors are best-effort: a dead source degrades
its workers (reported by the supervisor) and never blocks trading on its own.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from bisect import bisect_right
from email.utils import parsedate_to_datetime

import numpy as np
import requests

from lunatrade.intel.news_pipeline import NewsPipeline
from lunatrade.intel.social import Post
from lunatrade.intel.whale import Transfer

log = logging.getLogger("lunatrade.feeds")
UA = {"User-Agent": "LunaTrade/1.0 (research bot)"}


class FeedHub:
    def __init__(self, news_pipeline: NewsPipeline | None = None):
        self.news = news_pipeline or NewsPipeline()
        self.posts: list[Post] = []
        self.transfers: list[Transfer] = []
        self.series: dict[str, list[tuple[dt.datetime, float]]] = {}
        self.calendar: list[dict] = []                  # [{name, time: datetime, impact}]
        self.arb: list[tuple[dt.datetime, dict]] = []   # snapshots {exchange: {symbol: price}}
        self.status: dict[str, dict] = {}               # collector -> {ok, last_ok, error}
        self._lock = threading.RLock()

    # -- writes
    def add_point(self, name: str, ts: dt.datetime, value: float) -> None:
        with self._lock:
            s = self.series.setdefault(name, [])
            if s and ts <= s[-1][0]:
                if ts == s[-1][0]:
                    s[-1] = (ts, float(value))
                return
            s.append((ts, float(value)))
            if len(s) > 5000:
                del s[: len(s) - 5000]

    def add_post(self, p: Post) -> None:
        with self._lock:
            self.posts.append(p)
            cutoff = p.ts - dt.timedelta(days=3)
            if len(self.posts) > 20000 or (self.posts and self.posts[0].ts < cutoff):
                self.posts = [x for x in self.posts if x.ts >= cutoff][-20000:]

    def add_transfer(self, t: Transfer) -> None:
        with self._lock:
            if t.tx_hash and any(x.tx_hash == t.tx_hash for x in self.transfers[-500:]):
                return
            self.transfers.append(t)
            self.transfers = self.transfers[-10000:]

    def add_arb(self, ts: dt.datetime, prices: dict) -> None:
        with self._lock:
            self.arb.append((ts, prices))
            self.arb = self.arb[-2000:]

    def mark(self, collector: str, ok: bool, error: str = "") -> None:
        st = self.status.setdefault(collector, {"ok": False, "last_ok": None, "error": ""})
        st["ok"] = ok
        st["error"] = error[:300]
        if ok:
            st["last_ok"] = dt.datetime.utcnow().isoformat()

    def view(self, as_of: dt.datetime) -> "FeedView":
        return FeedView(self, as_of)


class FeedView:
    def __init__(self, hub: FeedHub, as_of: dt.datetime):
        self.hub, self.as_of = hub, as_of

    def series(self, name: str, n: int | None = None) -> np.ndarray:
        s = self.hub.series.get(name, [])
        idx = bisect_right([t for t, _ in s], self.as_of)
        vals = np.array([v for _, v in s[:idx]], dtype=float)
        return vals[-n:] if n else vals

    def series_with_time(self, name: str) -> list[tuple[dt.datetime, float]]:
        return [(t, v) for t, v in self.hub.series.get(name, []) if t <= self.as_of]

    def has(self, name: str) -> bool:
        return len(self.series(name)) > 0

    def posts(self, hours: float = 48) -> list[Post]:
        lo = self.as_of - dt.timedelta(hours=hours)
        return [p for p in self.hub.posts if lo <= p.ts <= self.as_of]

    def transfers(self, hours: float = 24) -> list[Transfer]:
        lo = self.as_of - dt.timedelta(hours=hours)
        return [t for t in self.hub.transfers if lo <= t.ts <= self.as_of]

    def news(self):
        return self.hub.news.active(self.as_of)

    def news_signal(self, asset: str, event_types: set | None = None):
        return self.hub.news.asset_signal(asset, self.as_of, event_types)

    def upcoming_events(self, hours: float) -> list[dict]:
        hi = self.as_of + dt.timedelta(hours=hours)
        return [e for e in self.hub.calendar if self.as_of - dt.timedelta(hours=1) <= e["time"] <= hi]

    def arb_latest(self) -> dict | None:
        for ts, snap in reversed(self.hub.arb):
            if ts <= self.as_of and (self.as_of - ts).total_seconds() < 900:
                return snap
        return None


# =============================================================================== collectors

def _get(url, params=None, timeout=15, headers=None):
    r = requests.get(url, params=params, timeout=timeout, headers={**UA, **(headers or {})})
    r.raise_for_status()
    return r


def _parse_date(text: str | None) -> dt.datetime:
    if not text:
        return dt.datetime.utcnow()
    try:
        d = parsedate_to_datetime(text)
    except Exception:
        try:
            d = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return dt.datetime.utcnow()
    if d.tzinfo:
        d = d.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return d


def parse_rss(xml_text: str) -> list[dict]:
    """RSS 2.0 and Atom -> [{title, summary, url, ts}]."""
    out = []
    root = ET.fromstring(xml_text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for it in root.iter("item"):
        out.append({"title": (it.findtext("title") or "").strip(), "url": it.findtext("link") or "",
                    "summary": _strip_html(it.findtext("description") or ""), "ts": _parse_date(it.findtext("pubDate"))})
    for it in root.findall(".//a:entry", ns):
        link = it.find("a:link", ns)
        out.append({"title": (it.findtext("a:title", default="", namespaces=ns)).strip(),
                    "url": link.get("href") if link is not None else "",
                    "summary": _strip_html(it.findtext("a:summary", default="", namespaces=ns)),
                    "ts": _parse_date(it.findtext("a:updated", default=None, namespaces=ns))})
    return [x for x in out if x["title"]]


def _strip_html(s: str) -> str:
    import re

    return re.sub(r"<[^>]+>", " ", s or "").replace("&nbsp;", " ").strip()


class Collectors:
    """Network collectors; each returns quickly and records its health in hub.status."""

    def __init__(self, hub: FeedHub, cfg: dict, symbols_provider=lambda: []):
        self.hub, self.cfg, self.symbols = hub, cfg, symbols_provider

    # ---- news
    def news(self) -> int:
        n = 0
        ncfg = self.cfg.get("news", {})
        for src in ncfg.get("sources", []):
            self.hub.news.cred.setdefault(src["name"], float(src.get("credibility", 0.6)))
            try:
                for it in parse_rss(_get(src["url"]).text)[:40]:
                    if self.hub.news.process(src["name"], it["title"], it["summary"], it["url"], it["ts"]):
                        n += 1
                self.hub.mark(f"news:{src['name']}", True)
            except Exception as e:
                self.hub.mark(f"news:{src['name']}", False, str(e))
        if ncfg.get("binance_announcements", True):
            self.hub.news.cred.setdefault("binance", 0.95)
            try:
                r = _get("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query",
                         params={"type": 1, "pageNo": 1, "pageSize": 20})
                for cat in r.json().get("data", {}).get("catalogs", []):
                    for a in cat.get("articles", []):
                        ts = dt.datetime.utcfromtimestamp(a.get("releaseDate", time.time() * 1000) / 1000)
                        if self.hub.news.process("binance", a.get("title", ""), cat.get("catalogName", ""), "", ts):
                            n += 1
                self.hub.mark("news:binance", True)
            except Exception as e:
                self.hub.mark("news:binance", False, str(e))
        key = os.getenv("CRYPTOPANIC_API_KEY")
        if key:
            self.hub.news.cred.setdefault("cryptopanic", 0.6)
            try:
                r = _get("https://cryptopanic.com/api/v1/posts/", params={"auth_token": key, "public": "true"})
                for p in r.json().get("results", []):
                    if self.hub.news.process("cryptopanic", p.get("title", ""), "", p.get("url", ""),
                                             _parse_date(p.get("published_at"))):
                        n += 1
                self.hub.mark("news:cryptopanic", True)
            except Exception as e:
                self.hub.mark("news:cryptopanic", False, str(e))
        return n

    # ---- social
    def social(self) -> int:
        n = 0
        scfg = self.cfg.get("social", {})
        for sub in scfg.get("subreddits", []):
            try:
                r = _get(f"https://www.reddit.com/r/{sub}/new.json", params={"limit": 100})
                for c in r.json().get("data", {}).get("children", []):
                    d = c.get("data", {})
                    self.hub.add_post(Post(ts=dt.datetime.utcfromtimestamp(d.get("created_utc", time.time())),
                                           community=f"r/{sub}", title=d.get("title", ""),
                                           text=d.get("selftext", "")[:1000], author=d.get("author", ""),
                                           score=d.get("score", 0), comments=d.get("num_comments", 0)))
                    n += 1
                self.hub.mark(f"social:{sub}", True)
            except Exception as e:
                self.hub.mark(f"social:{sub}", False, str(e))
        bearer = os.getenv("X_BEARER_TOKEN")
        if bearer:
            try:
                r = _get("https://api.twitter.com/2/tweets/search/recent",
                         params={"query": "(bitcoin OR ethereum OR crypto) -is:retweet lang:en", "max_results": 100,
                                 "tweet.fields": "created_at,author_id,public_metrics"},
                         headers={"Authorization": f"Bearer {bearer}"})
                for t in r.json().get("data", []):
                    m = t.get("public_metrics", {})
                    self.hub.add_post(Post(ts=_parse_date(t.get("created_at")), community="x", title=t.get("text", ""),
                                           author=t.get("author_id", ""), score=m.get("like_count", 0),
                                           comments=m.get("reply_count", 0)))
                    n += 1
                self.hub.mark("social:x", True)
            except Exception as e:
                self.hub.mark("social:x", False, str(e))
        if scfg.get("fear_greed", True):
            try:
                for d in _get("https://api.alternative.me/fng/", params={"limit": 60}).json().get("data", [])[::-1]:
                    self.hub.add_point("fear_greed", dt.datetime.utcfromtimestamp(int(d["timestamp"])), float(d["value"]))
                self.hub.mark("social:fear_greed", True)
            except Exception as e:
                self.hub.mark("social:fear_greed", False, str(e))
        return n

    # ---- on-chain
    def onchain(self) -> int:
        n, now = 0, dt.datetime.utcnow()
        try:
            data = _get("https://stablecoins.llama.fi/stablecoincharts/all").json()
            for d in data[-120:]:
                ts = dt.datetime.utcfromtimestamp(int(d["date"]))
                self.hub.add_point("stablecoin_mcap", ts, float(d.get("totalCirculatingUSD", {}).get("peggedUSD", 0)))
                n += 1
            self.hub.mark("onchain:stablecoins", True)
        except Exception as e:
            self.hub.mark("onchain:stablecoins", False, str(e))
        for chain in ("", "Ethereum", "Solana", "BSC", "Arbitrum", "Base", "Tron"):
            name = f"tvl_{chain.lower() or 'total'}"
            try:
                url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}" if chain else \
                    "https://api.llama.fi/v2/historicalChainTvl"
                for d in _get(url).json()[-120:]:
                    self.hub.add_point(name, dt.datetime.utcfromtimestamp(int(d["date"])), float(d["tvl"]))
                    n += 1
                self.hub.mark(f"onchain:{name}", True)
            except Exception as e:
                self.hub.mark(f"onchain:{name}", False, str(e))
        try:
            fees = _get("https://mempool.space/api/v1/fees/recommended").json()
            self.hub.add_point("btc_fee_fastest", now, float(fees.get("fastestFee", 0)))
            mp = _get("https://mempool.space/api/mempool").json()
            self.hub.add_point("btc_mempool_count", now, float(mp.get("count", 0)))
            self.hub.add_point("btc_mempool_vsize", now, float(mp.get("vsize", 0)))
            hr = _get("https://mempool.space/api/v1/mining/hashrate/3m").json()
            for d in hr.get("hashrates", [])[-90:]:
                self.hub.add_point("btc_hashrate", dt.datetime.utcfromtimestamp(int(d["timestamp"])),
                                   float(d["avgHashrate"]))
            n += 4
            self.hub.mark("onchain:mempool", True)
        except Exception as e:
            self.hub.mark("onchain:mempool", False, str(e))
        return n

    # ---- whale transfers
    def whale(self) -> int:
        key, n = os.getenv("WHALE_ALERT_API_KEY"), 0
        if not key:
            self.hub.mark("whale:whale_alert", False, "WHALE_ALERT_API_KEY not set")
            return 0
        try:
            r = _get("https://api.whale-alert.io/v1/transactions",
                     params={"api_key": key, "min_value": int(self.cfg.get("whale", {}).get("min_usd", 1_000_000)),
                             "start": int(time.time()) - 3600})
            for t in r.json().get("transactions", []):
                self.hub.add_transfer(Transfer(
                    ts=dt.datetime.utcfromtimestamp(t["timestamp"]), asset=t.get("symbol", "").upper(),
                    amount_usd=float(t.get("amount_usd", 0)),
                    from_type=_owner_type(t.get("from", {})), to_type=_owner_type(t.get("to", {})),
                    from_owner=t.get("from", {}).get("owner", ""), to_owner=t.get("to", {}).get("owner", ""),
                    tx_hash=t.get("hash", "")))
                n += 1
            self.hub.mark("whale:whale_alert", True)
        except Exception as e:
            self.hub.mark("whale:whale_alert", False, str(e))
        return n

    # ---- macro
    def macro(self) -> int:
        n = 0
        mcfg = self.cfg.get("macro", {})
        try:
            import yfinance as yf
        except Exception as e:
            self.hub.mark("macro:yfinance", False, f"yfinance missing: {e}")
            return 0
        for name, ticker in mcfg.get("tickers", {}).items():
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
                if df is None or df.empty:
                    raise RuntimeError("no data")
                close = df["Close"]
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
                for ts, v in close.dropna().items():
                    # daily bar is known after the session closes -> stamp at 21:00 UTC
                    self.hub.add_point(f"macro_{name}", ts.to_pydatetime().replace(tzinfo=None, hour=21), float(v))
                    n += 1
                self.hub.mark(f"macro:{name}", True)
            except Exception as e:
                self.hub.mark(f"macro:{name}", False, str(e))
        self.hub.calendar = [{"name": e["name"], "time": dt.datetime.fromisoformat(str(e["time"])),
                              "impact": e.get("impact", "high")} for e in mcfg.get("calendar", [])]
        return n

    # ---- cross-exchange prices
    def arbitrage(self) -> int:
        now = dt.datetime.utcnow()
        snap: dict[str, dict] = {}
        pairs = [s for s in self.symbols() if s.endswith("USDT")][:10] or ["BTCUSDT", "ETHUSDT"]
        try:
            bt = {d["symbol"]: d for d in _get("https://api.binance.com/api/v3/ticker/bookTicker").json()}
            snap["binance"] = {s: (float(bt[s]["bidPrice"]) + float(bt[s]["askPrice"])) / 2 for s in pairs if s in bt}
            for tri in ("ETHBTC", "BNBBTC", "SOLBTC", "USDCUSDT", "FDUSDUSDT"):
                if tri in bt:
                    snap["binance"][tri] = (float(bt[tri]["bidPrice"]) + float(bt[tri]["askPrice"])) / 2
            self.hub.mark("arb:binance", True)
        except Exception as e:
            self.hub.mark("arb:binance", False, str(e))
        for s in pairs[:5]:
            base = s[:-4]
            try:
                d = _get(f"https://api.exchange.coinbase.com/products/{base}-USD/ticker").json()
                snap.setdefault("coinbase", {})[s] = (float(d["bid"]) + float(d["ask"])) / 2
            except Exception:
                pass
            try:
                k = "XBT" if base == "BTC" else base
                d = _get("https://api.kraken.com/0/public/Ticker", params={"pair": f"{k}USD"}).json()["result"]
                v = next(iter(d.values()))
                snap.setdefault("kraken", {})[s] = (float(v["a"][0]) + float(v["b"][0])) / 2
            except Exception:
                pass
        self.hub.mark("arb:others", bool(snap.get("coinbase") or snap.get("kraken")))
        if snap:
            self.hub.add_arb(now, snap)
        return len(snap)


def _owner_type(side: dict) -> str:
    t = (side.get("owner_type") or "unknown").lower()
    owner = (side.get("owner") or "").lower()
    if t == "exchange":
        return "exchange"
    if "treasury" in owner or owner in ("tether treasury", "usdc treasury", "circle"):
        return "treasury"
    return t if t in ("defi", "bridge", "custody") else "unknown"


class FeedScheduler:
    """Runs each collector on its own cadence in a background thread."""

    def __init__(self, collectors: Collectors, cfg: dict):
        self.c, self.cfg = collectors, cfg
        self._stop = threading.Event()
        self._last: dict[str, float] = {}

    def due(self) -> list[str]:
        now = time.time()
        out = []
        for name in ("news", "social", "onchain", "whale", "macro", "arbitrage"):
            sec = self.cfg.get(name, {})
            if not sec.get("enabled", True):
                continue
            if now - self._last.get(name, 0) >= float(sec.get("interval_seconds", 300)):
                out.append(name)
        return out

    def run_once(self) -> dict:
        res = {}
        for name in self.due():
            self._last[name] = time.time()
            try:
                res[name] = getattr(self.c, name)()
            except Exception as e:
                log.warning("collector %s failed: %s", name, e)
                res[name] = f"error: {e}"
        return res

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                self.run_once()
                self._stop.wait(10)
        threading.Thread(target=loop, name="feeds", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
