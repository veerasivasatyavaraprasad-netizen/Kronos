import datetime as dt

import numpy as np

from lunatrade.core.types import Direction, Family
from lunatrade.data.feeds import FeedHub
from lunatrade.data.market_store import MarketStore, from_ms
from lunatrade.intel.social import Post
from lunatrade.intel.whale import Transfer
from lunatrade.workers.base import WorkerContext
from lunatrade.workers.registry import EXPECTED, TOTAL, build_roster, roster_summary
from lt.helpers import SYMS, last_close_time


def run_all(store, hub, t, symbols=SYMS):
    ws = build_roster()
    ctx = WorkerContext(store.view(t), hub.view(t), symbols, t)
    p1 = [s for w in ws if w.phase == 1 for s in w.run(ctx)]
    ctx.signals = p1
    p2 = [s for w in ws if w.phase == 2 for s in w.run(ctx)]
    return ws, p1 + p2


def test_roster_is_exactly_300_with_expected_families():
    ws = build_roster()
    assert len(ws) == TOTAL == 300
    assert roster_summary(ws) == EXPECTED
    assert len({w.id for w in ws}) == 300


def test_all_workers_run_without_errors(store):
    hub = FeedHub()
    t = last_close_time(store)
    # give the intel workers something to chew on
    for i, title in enumerate(["Bitcoin ETF sees record inflows as institutions buy",
                               "Major exchange hacked, $200M stolen in exploit",
                               "SEC approves new crypto framework"]):
        hub.news.process(f"src{i}", title, "", ts=t - dt.timedelta(minutes=30 + i))
    for i in range(20):
        hub.add_post(Post(t - dt.timedelta(minutes=10 * i), "r/Bitcoin", f"BTC looking bullish breakout {i}",
                          author=f"user{i}", score=10))
    hub.add_transfer(Transfer(t - dt.timedelta(hours=1), "BTC", 60e6, "unknown", "exchange", tx_hash="a"))
    hub.add_transfer(Transfer(t - dt.timedelta(hours=2), "USDT", 100e6, "treasury", "unknown", tx_hash="b"))
    for d in range(40):
        hub.add_point("fear_greed", t - dt.timedelta(days=40 - d), 20 + d)
        hub.add_point("stablecoin_mcap", t - dt.timedelta(days=40 - d), 150e9 * (1 + 0.002 * d))
    ws, sigs = run_all(store, hub, t)
    errors = [(w.id, w.health.last_error) for w in ws if w.health.errors]
    assert not errors, errors
    fams = {s.family for s in sigs}
    for f in (Family.TECHNICAL, Family.REGIME, Family.NEWS, Family.SOCIAL, Family.WHALE, Family.RISK,
              Family.STRATEGY_EVAL, Family.PORTFOLIO):
        assert f in fams, f
    assert all(0 <= s.confidence <= 1 for s in sigs)


def test_feed_workers_report_offline_without_data(store):
    ws, _ = run_all(store, FeedHub(), last_close_time(store))
    offline = {w.family for w in ws if w.health.status == "OFFLINE"}
    assert Family.WHALE in offline and Family.MACRO in offline


def test_no_lookahead_signals_identical_with_or_without_future_data(data):
    full, cut = MarketStore("15m", 1000), MarketStore("15m", 1000)
    for sym, df in data.items():
        full.load_frame(sym, df)
        cut.load_frame(sym, df.iloc[:350])
    t = last_close_time(cut)
    _, a = run_all(full, FeedHub(), t)
    _, b = run_all(cut, FeedHub(), t)
    key = lambda s: (s.worker_id, s.symbol, s.direction.value, round(s.confidence, 9), s.setup)
    assert sorted(map(key, a)) == sorted(map(key, b))


def test_market_wide_signals_and_directions_are_valid(store):
    _, sigs = run_all(store, FeedHub(), last_close_time(store))
    assert all(isinstance(s.direction, Direction) for s in sigs)
    regime_votes = [s for s in sigs if s.family == Family.REGIME]
    assert regime_votes and all("votes" in s.context for s in regime_votes)
