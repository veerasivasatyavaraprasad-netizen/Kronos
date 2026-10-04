"""
Whale / on-chain transfer interpretation.

wallet movement != buying. Every transfer is interpreted in context:

    unknown wallet -> exchange     possible sell pressure (could also be collateral/custody)
    exchange -> unknown wallet     withdrawal, typically accumulation / cold storage
    exchange -> exchange           internal / market-maker rebalancing -> neutral
    stablecoin -> exchange         buying power arriving -> bullish for the market
    stablecoin minted (treasury)   new liquidity -> mildly bullish
    dormant wallet wakes up        uncertainty -> raises risk, small bearish tilt
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "BUSD", "PYUSD", "USDE"}


@dataclass
class Transfer:
    ts: dt.datetime
    asset: str
    amount_usd: float
    from_type: str = "unknown"     # exchange | unknown | treasury | defi | bridge | custody
    to_type: str = "unknown"
    from_owner: str = ""
    to_owner: str = ""
    tx_hash: str = ""
    dormant_days: float = 0.0
    tags: list = field(default_factory=list)


@dataclass
class Interpretation:
    kind: str
    asset: str              # asset affected ('*' = whole market)
    bias: float             # -1..1
    confidence: float       # 0..1 - transfers are ambiguous, so this stays modest
    note: str


def interpret(t: Transfer, min_usd: float = 1_000_000) -> Interpretation:
    size = min(1.0, (t.amount_usd / max(min_usd, 1)) ** 0.5 / 5)   # 1M ->0.2, 25M -> 1.0
    a, ft, tt = t.asset.upper(), t.from_type, t.to_type
    stable = a in STABLES
    if ft == "exchange" and tt == "exchange":
        if t.from_owner and t.from_owner == t.to_owner:
            return Interpretation("INTERNAL", a, 0.0, 0.05, "same-exchange internal transfer")
        return Interpretation("EXCHANGE_REBALANCE", a, 0.0, 0.1, "exchange-to-exchange, likely market making")
    if ft == "treasury" and stable:
        return Interpretation("STABLE_MINT", "*", 0.35 * size, 0.35, f"{a} minted - fresh liquidity")
    if tt == "treasury" and stable:
        return Interpretation("STABLE_BURN", "*", -0.3 * size, 0.3, f"{a} burned - liquidity leaving")
    if tt == "exchange":
        if stable:
            return Interpretation("STABLE_TO_EXCHANGE", "*", 0.5 * size, 0.4, f"{a} sent to exchange - buying power")
        conf = 0.35 if ft == "unknown" else 0.2   # custody/defi moves are less telling
        return Interpretation("EXCHANGE_INFLOW", a, -0.6 * size, conf,
                              "deposit to exchange - possible sell, could be collateral/custody")
    if ft == "exchange":
        if stable:
            return Interpretation("STABLE_FROM_EXCHANGE", "*", -0.2 * size, 0.2, f"{a} leaving exchanges")
        return Interpretation("EXCHANGE_OUTFLOW", a, 0.5 * size, 0.35, "withdrawal - likely accumulation/cold storage")
    if t.dormant_days >= 365:
        return Interpretation("DORMANT_WAKEUP", a, -0.3 * size, 0.3, f"wallet dormant {t.dormant_days:.0f}d moved")
    if ft == "bridge" or tt == "bridge":
        return Interpretation("BRIDGE_FLOW", a, 0.1 * size if tt != "bridge" else 0.0, 0.15, "cross-chain bridge flow")
    if tt == "defi" and not stable:
        return Interpretation("STAKING", a, 0.25 * size, 0.2, "moved into DeFi/staking - supply locked")
    return Interpretation("WALLET_TO_WALLET", a, 0.0, 0.05, "unknown to unknown - no clear meaning")
