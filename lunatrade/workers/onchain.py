"""On-chain intelligence (25) and whale tracking (20) workers."""
from __future__ import annotations

import numpy as np

from lunatrade.core.types import MARKET_WIDE
from lunatrade.core.types import Direction as D
from lunatrade.core.types import Family
from lunatrade.intel.assets import base_of
from lunatrade.intel.whale import STABLES, interpret
from lunatrade.workers.base import clamp, spec

L, S, N = D.LONG, D.SHORT, D.NEUTRAL
CHAIN_ASSET = {"ethereum": "ETH", "solana": "SOL", "bsc": "BNB", "arbitrum": "ARB", "base": "ETH", "tron": "TRX"}


def _series_avail(name):
    def f(w, ctx):
        return (True, "") if ctx.feeds.has(name) else (False, f"no data for {name}")
    return f


def _growth(vals: np.ndarray, n: int) -> float | None:
    if len(vals) <= n or not vals[-n - 1]:
        return None
    return vals[-1] / vals[-n - 1] - 1


def stablecoin_growth(w, ctx, sym):
    g = _growth(ctx.feeds.series("stablecoin_mcap"), w.params["days"])
    if g is None:
        return None
    thr = w.params["thr"]
    if abs(g) < thr:
        return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="STABLE_SUPPLY", context={"stable_growth": g})
    return ctx.signal(w, MARKET_WIDE, D.from_score(g), clamp(abs(g) / thr * 0.2), setup="STABLE_LIQUIDITY",
                      time_horizon="1w", evidence=[f"stablecoin_supply_{w.params['days']}d={g:+.2%}"],
                      context={"stable_growth": g})


def stablecoin_ratio(w, ctx, sym):
    """Stablecoin supply vs BTC price (SSR proxy): more dry powder per unit of BTC = bullish."""
    st = ctx.feeds.series("stablecoin_mcap")
    c = ctx.view.candles(ctx.market_symbol)
    n = w.params["days"]
    bars_per_day = max(1, int(86_400_000 / ctx.view.store.interval_ms))
    if len(st) <= n or len(c) <= n * bars_per_day:
        return None
    ratio_now = st[-1] / c.close[-1]
    ratio_then = st[-n - 1] / c.close[-n * bars_per_day - 1]
    chg = ratio_now / ratio_then - 1
    if abs(chg) < 0.05:
        return None
    return ctx.signal(w, MARKET_WIDE, D.from_score(chg), clamp(abs(chg)), setup="BUYING_POWER", time_horizon="1w",
                      context={"ssr_change": chg})


def tvl_trend(w, ctx, sym):
    chain = w.params["chain"]
    g = _growth(ctx.feeds.series(f"tvl_{chain}"), w.params["days"])
    if g is None:
        return None
    asset = CHAIN_ASSET.get(chain)
    targets = [s for s in ctx.symbols if base_of(s) == asset] if asset else [MARKET_WIDE]
    if chain == "total":
        targets = [MARKET_WIDE]
    if abs(g) < 0.03 or not targets:
        return None
    return [ctx.signal(w, t, D.from_score(g), clamp(abs(g) * 4), setup="TVL_TREND", time_horizon="1w",
                       evidence=[f"{chain}_tvl_{w.params['days']}d={g:+.1%}"], context={"tvl_growth": g}) for t in targets]


def mempool(w, ctx, sym):
    s = ctx.feeds.series(w.params["series"], 200)
    if len(s) < 10:
        return None
    z = (s[-1] - s.mean()) / (s.std() or 1)
    targets = [x for x in ctx.symbols if base_of(x) == "BTC"]
    return [ctx.signal(w, t, N, 0.0, setup="MEMPOOL", context={f"{w.params['series']}_z": float(z),
                                                                 "congestion": bool(z > 2)}) for t in targets]


def fee_spike(w, ctx, sym):
    s = ctx.feeds.series("btc_fee_fastest", w.params["n"])
    if len(s) < 10:
        return None
    z = (s[-1] - s.mean()) / (s.std() or 1)
    if z < 2.5:
        return None
    # fee spikes accompany panics and mania alike -> volatility warning, direction from recent price
    targets = [x for x in ctx.symbols if base_of(x) == "BTC"]
    return [ctx.signal(w, t, N, 0.0, setup="FEE_SPIKE", context={"fee_z": float(z), "volatility_warning": True})
            for t in targets]


def hashrate(w, ctx, sym):
    g = _growth(ctx.feeds.series("btc_hashrate"), w.params["days"])
    if g is None:
        return None
    targets = [x for x in ctx.symbols if base_of(x) == "BTC"]
    if abs(g) < 0.03:
        return None
    return [ctx.signal(w, t, D.from_score(g), clamp(abs(g) * 2), setup="HASHRATE_TREND", time_horizon="1w",
                       context={"hashrate_growth": g}) for t in targets]


def miner_stress(w, ctx, sym):
    g = _growth(ctx.feeds.series("btc_hashrate"), 14)
    c = ctx.view.candles(ctx.market_symbol)
    if g is None or len(c) < 100:
        return None
    pr = c.close[-1] / c.close[-100] - 1
    if g < -0.05 and pr < -0.1:
        targets = [x for x in ctx.symbols if base_of(x) == "BTC"]
        return [ctx.signal(w, t, S, 0.35, setup="MINER_CAPITULATION", time_horizon="1w",
                           context={"hashrate_growth": g, "price_change": pr}) for t in targets]
    return None


def active_addresses(w, ctx, sym):
    g = _growth(ctx.feeds.series("btc_active_addresses"), 7)
    if g is None or abs(g) < 0.05:
        return None
    targets = [x for x in ctx.symbols if base_of(x) == "BTC"]
    return [ctx.signal(w, t, D.from_score(g), clamp(abs(g) * 2), setup="NETWORK_ACTIVITY",
                       context={"active_address_growth": g}) for t in targets]


def bridge_flows(w, ctx, sym):
    asset = base_of(sym)
    tr = [t for t in ctx.feeds.transfers(w.params["hours"]) if t.asset == asset and "bridge" in (t.from_type, t.to_type)]
    if len(tr) < 2:
        return None
    inflow = sum(t.amount_usd for t in tr if t.from_type == "bridge")
    outflow = sum(t.amount_usd for t in tr if t.to_type == "bridge")
    net = inflow - outflow
    if abs(net) < 5e6:
        return None
    return ctx.signal(w, sym, D.from_score(net), clamp(abs(net) / 1e8), setup="BRIDGE_FLOW",
                      context={"bridge_net_usd": net})


def staking_flows(w, ctx, sym):
    asset = base_of(sym)
    tr = [t for t in ctx.feeds.transfers(w.params["hours"]) if t.asset == asset and "defi" in (t.from_type, t.to_type)]
    if not tr:
        return None
    locked = sum(t.amount_usd for t in tr if t.to_type == "defi") - sum(t.amount_usd for t in tr if t.from_type == "defi")
    if abs(locked) < 5e6:
        return None
    return ctx.signal(w, sym, D.from_score(locked), clamp(abs(locked) / 1e8), setup="STAKING_FLOW",
                      context={"net_staked_usd": locked})


# --------------------------------------------------------------------------------- whales

def _has_transfers(w, ctx):
    return (True, "") if ctx.feeds.hub.transfers else (False, "no whale transfer feed (WHALE_ALERT_API_KEY)")


def _interps(ctx, hours, min_usd=1e6):
    return [(t, interpret(t, min_usd)) for t in ctx.feeds.transfers(hours)]


def exchange_flow(w, ctx, sym):
    asset = base_of(sym)
    kind = w.params["kind"]
    items = [(t, i) for t, i in _interps(ctx, w.params["hours"]) if t.asset == asset and i.kind == kind]
    if not items:
        return None
    usd = sum(t.amount_usd for t, _ in items)
    bias = sum(i.bias * i.confidence for _, i in items)
    return ctx.signal(w, sym, D.from_score(bias), clamp(abs(bias)), setup=kind, time_horizon="1d",
                      evidence=[f"{len(items)} transfers ${usd / 1e6:.0f}M"],
                      invalidations=["custody_or_collateral_move"], context={f"{kind.lower()}_usd": usd})


def net_exchange_flow(w, ctx, sym):
    asset = base_of(sym)
    items = [(t, i) for t, i in _interps(ctx, w.params["hours"]) if t.asset == asset]
    inflow = sum(t.amount_usd for t, i in items if i.kind == "EXCHANGE_INFLOW")
    outflow = sum(t.amount_usd for t, i in items if i.kind == "EXCHANGE_OUTFLOW")
    net = outflow - inflow
    if abs(net) < 10e6:
        return None
    return ctx.signal(w, sym, D.from_score(net), clamp(abs(net) / 3e8 + 0.15), setup="NET_EXCHANGE_FLOW",
                      context={"net_outflow_usd": net, "inflow_usd": inflow, "outflow_usd": outflow})


def stable_event(w, ctx, sym):
    kinds = set(w.params["kinds"])
    items = [(t, i) for t, i in _interps(ctx, w.params["hours"]) if t.asset in STABLES and i.kind in kinds]
    if not items:
        return None
    usd = sum(t.amount_usd for t, _ in items)
    bias = sum(i.bias * i.confidence for _, i in items)
    if abs(bias) < 0.05:
        return None
    return ctx.signal(w, MARKET_WIDE, D.from_score(bias), clamp(abs(bias)), setup="_".join(sorted(kinds)),
                      time_horizon="1d", context={"stable_usd": usd})


def dormant(w, ctx, sym):
    asset = base_of(sym)
    items = [(t, i) for t, i in _interps(ctx, 48) if t.asset == asset and i.kind == "DORMANT_WAKEUP"]
    if not items:
        return None
    return ctx.signal(w, sym, S, clamp(sum(i.confidence for _, i in items) * 0.5), setup="DORMANT_WALLET",
                      context={"dormant_moves": len(items), "risk_flag": True})


def accumulation(w, ctx, sym):
    asset = base_of(sym)
    big = [(t, i) for t, i in _interps(ctx, 72) if t.asset == asset and t.amount_usd >= w.params["min_usd"]]
    score = sum(i.bias * i.confidence for _, i in big)
    if abs(score) < 0.1:
        return None
    return ctx.signal(w, sym, D.from_score(score), clamp(abs(score)),
                      setup="WHALE_ACCUMULATION" if score > 0 else "WHALE_DISTRIBUTION", time_horizon="3d",
                      context={"whale_score": score, "transfers": len(big)})


def clustering(w, ctx, sym):
    asset = base_of(sym)
    tr = sorted((t for t in ctx.feeds.transfers(w.params["hours"]) if t.asset == asset), key=lambda t: t.ts)
    to_ex = [t for t in tr if t.to_type == "exchange" and t.from_type != "exchange"]
    if len(to_ex) < w.params["min_count"]:
        return None
    return ctx.signal(w, sym, S, clamp(len(to_ex) * 0.08), setup="DEPOSIT_CLUSTER",
                      context={"clustered_deposits": len(to_ex)}, invalidations=["custody_reshuffle"])


def smart_money(w, ctx, sym):
    wallets = set(ctx.portfolio.get("smart_money_wallets", []))
    if not wallets:
        return None
    asset = base_of(sym)
    tr = [t for t in ctx.feeds.transfers(72) if t.asset == asset and (t.from_owner in wallets or t.to_owner in wallets)]
    if not tr:
        return None
    net = sum(t.amount_usd if t.to_owner in wallets else -t.amount_usd for t in tr)
    return ctx.signal(w, sym, D.from_score(net), 0.4, setup="SMART_MONEY", context={"smart_money_net_usd": net})


def token_unlock(w, ctx, sym):
    asset = base_of(sym)
    items = [x for x in ctx.feeds.news() if x.event_type == "TOKEN_UNLOCK" and asset in x.assets]
    if not items:
        return None
    return ctx.signal(w, sym, S, clamp(0.25 + 0.1 * len(items)), setup="TOKEN_UNLOCK", time_horizon="1w",
                      evidence=[x.title[:90] for x in items[:2]], context={"unlock_news": len(items)})


def custody_filter(w, ctx, sym):
    items = _interps(ctx, 24)
    if not items:
        return None
    internal = sum(1 for _, i in items if i.kind in ("INTERNAL", "EXCHANGE_REBALANCE", "WALLET_TO_WALLET"))
    return ctx.signal(w, MARKET_WIDE, N, 0.0, setup="TRANSFER_NOISE",
                      context={"noise_share": internal / len(items), "transfers": len(items)})


def major_whale(w, ctx, sym):
    asset = w.params["asset"]
    if base_of(sym) != asset:
        return None
    items = [(t, i) for t, i in _interps(ctx, 24) if t.asset == asset and t.amount_usd >= 50e6]
    if not items:
        return None
    bias = sum(i.bias * i.confidence for _, i in items)
    return ctx.signal(w, sym, D.from_score(bias), clamp(abs(bias) + 0.1), setup="MEGA_WHALE",
                      context={"mega_transfers": len(items)})


def specs() -> list[dict]:
    O, W = Family.ONCHAIN, Family.WHALE
    out = []
    for days, thr in [(7, 0.01), (30, 0.03), (90, 0.06)]:
        out.append(spec(O, f"stablecoin_growth_{days}d", stablecoin_growth, params={"days": days, "thr": thr},
                        per_symbol=False, availability=_series_avail("stablecoin_mcap")))
    for days in (7, 30):
        out.append(spec(O, f"stablecoin_ratio_{days}d", stablecoin_ratio, params={"days": days}, per_symbol=False,
                        availability=_series_avail("stablecoin_mcap")))
    for chain in ("ethereum", "solana", "bsc", "arbitrum", "base", "tron"):
        out.append(spec(O, f"tvl_{chain}", tvl_trend, params={"chain": chain, "days": 7}, per_symbol=False,
                        availability=_series_avail(f"tvl_{chain}")))
    for days in (7, 30):
        out.append(spec(O, f"tvl_total_{days}d", tvl_trend, params={"chain": "total", "days": days}, per_symbol=False,
                        availability=_series_avail("tvl_total")))
    for series in ("btc_mempool_count", "btc_mempool_vsize"):
        out.append(spec(O, f"mempool_{series.split('_')[-1]}", mempool, params={"series": series}, per_symbol=False,
                        availability=_series_avail(series)))
    for n in (50, 300):
        out.append(spec(O, f"fee_spike_{n}", fee_spike, params={"n": n}, per_symbol=False,
                        availability=_series_avail("btc_fee_fastest")))
    for days in (30, 90):
        out.append(spec(O, f"hashrate_{days}d", hashrate, params={"days": days}, per_symbol=False,
                        availability=_series_avail("btc_hashrate")))
    out.append(spec(O, "miner_stress", miner_stress, per_symbol=False, availability=_series_avail("btc_hashrate")))
    out.append(spec(O, "active_addresses", active_addresses, per_symbol=False,
                    availability=_series_avail("btc_active_addresses")))
    for h in (24, 72):
        out.append(spec(O, f"bridge_flows_{h}h", bridge_flows, params={"hours": h}, availability=_has_transfers))
    for h in (24, 72):
        out.append(spec(O, f"staking_flows_{h}h", staking_flows, params={"hours": h}, availability=_has_transfers))
    # whales (20)
    for kind in ("EXCHANGE_INFLOW", "EXCHANGE_OUTFLOW"):
        for h in (1, 24):
            out.append(spec(W, f"{kind.lower()}_{h}h", exchange_flow, params={"kind": kind, "hours": h},
                            availability=_has_transfers))
    for h in (6, 24):
        out.append(spec(W, f"net_exchange_flow_{h}h", net_exchange_flow, params={"hours": h}, availability=_has_transfers))
    out.append(spec(W, "stable_mint", stable_event, params={"kinds": ["STABLE_MINT"], "hours": 24}, per_symbol=False,
                    availability=_has_transfers))
    out.append(spec(W, "stable_burn", stable_event, params={"kinds": ["STABLE_BURN"], "hours": 24}, per_symbol=False,
                    availability=_has_transfers))
    for h in (6, 24):
        out.append(spec(W, f"stable_to_exchange_{h}h", stable_event, params={"kinds": ["STABLE_TO_EXCHANGE"], "hours": h},
                        per_symbol=False, availability=_has_transfers))
    out.append(spec(W, "dormant_wallets", dormant, availability=_has_transfers))
    for usd in (10e6, 50e6):
        out.append(spec(W, f"accumulation_{int(usd / 1e6)}m", accumulation, params={"min_usd": usd},
                        availability=_has_transfers))
    for h, cnt in [(2, 3), (12, 6)]:
        out.append(spec(W, f"deposit_cluster_{h}h", clustering, params={"hours": h, "min_count": cnt},
                        availability=_has_transfers))
    out.append(spec(W, "smart_money", smart_money, availability=_has_transfers))
    out.append(spec(W, "token_unlock", token_unlock))
    out.append(spec(W, "custody_filter", custody_filter, per_symbol=False, availability=_has_transfers))
    for asset in ("BTC", "ETH"):
        out.append(spec(W, f"mega_whale_{asset.lower()}", major_whale, params={"asset": asset},
                        availability=_has_transfers))
    return out
