"""
Risk Engine - independent of the Lead Brain. The Lead Brain says "I want BTC long"; the Risk Engine can
say "no", or "yes, but smaller".

Checks: kill switch, exchange/API health, portfolio exposure, symbol exposure, max position, open
positions, daily/weekly loss, drawdown, liquidity, spread, expected slippage, volatility, stop distance,
available balance. Size = min(requested, risk-budget size, caps, cash) x (devil's advocate, regime,
volatility, correlation multipliers).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lunatrade.core.types import DevilsAdvocateReport, RiskDecision, TradeProposal

OK, WARN = "OK", "WARNING"


def FAIL(msg: str) -> str:
    return f"FAIL: {msg}"


@dataclass
class AccountState:
    equity: float
    cash: float
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    positions: dict = field(default_factory=dict)   # symbol -> {qty, entry, notional}

    @property
    def exposure(self) -> float:
        return sum(abs(p.get("notional", 0)) for p in self.positions.values())


@dataclass
class MarketConditions:
    price: float
    atr_pct: float
    spread_bps: float | None = None
    quote_volume_24h: float | None = None
    expected_slippage_bps: float | None = None
    high_vol: bool = False
    correlation_warning: bool = False
    regime_exposure_multiplier: float = 1.0
    exchange_ok: bool = True
    min_notional: float = 0.0


class RiskEngine:
    def __init__(self, cfg: dict, kill_switch=None):
        self.cfg = cfg
        self.kill_switch = kill_switch
        self.heartbeat = 0.0

    def evaluate(self, p: TradeProposal, da: DevilsAdvocateReport | None, acct: AccountState,
                 mkt: MarketConditions) -> RiskDecision:
        import time

        self.heartbeat = time.time()
        r = self.cfg
        checks: dict[str, str] = {}
        reasons: list[str] = []

        if p.action in ("CLOSE", "REDUCE"):
            # risk-reducing orders are always allowed while the exchange is reachable
            checks["exchange_status"] = OK if mkt.exchange_ok else FAIL("exchange unhealthy")
            ok = mkt.exchange_ok
            return RiskDecision(p.id, ok, acct.positions.get(p.symbol, {}).get("notional", 0.0), p.requested_notional,
                                checks, [] if ok else ["exchange unhealthy"])

        eq = max(acct.equity, 1e-9)
        # ---- hard gates
        if self.kill_switch is not None and self.kill_switch.active:
            checks["kill_switch"] = FAIL("; ".join(self.kill_switch.reasons())[:200])
        else:
            checks["kill_switch"] = OK
        checks["exchange_status"] = OK if mkt.exchange_ok else FAIL("exchange/API unhealthy")
        checks["devils_advocate"] = FAIL("vetoed: " + ", ".join(da.hidden_risks[:3])) if da and da.veto else OK
        daily = (acct.day_start_equity - acct.equity) / acct.day_start_equity if acct.day_start_equity else 0
        weekly = (acct.week_start_equity - acct.equity) / acct.week_start_equity if acct.week_start_equity else 0
        dd = (acct.peak_equity - acct.equity) / acct.peak_equity if acct.peak_equity else 0
        checks["daily_loss"] = FAIL(f"{daily:.1%}") if daily >= r["max_daily_loss_pct"] else \
            WARN if daily >= 0.7 * r["max_daily_loss_pct"] else OK
        checks["weekly_loss"] = FAIL(f"{weekly:.1%}") if weekly >= r["max_weekly_loss_pct"] else OK
        checks["drawdown"] = FAIL(f"{dd:.1%}") if dd >= r["max_drawdown_pct"] else \
            WARN if dd >= 0.7 * r["max_drawdown_pct"] else OK
        n_open = sum(1 for s, x in acct.positions.items() if x.get("qty", 0) > 0 and s != p.symbol)
        checks["open_positions"] = FAIL(f"{n_open} open") if n_open >= r["max_open_positions"] else OK
        if mkt.spread_bps is not None:
            checks["spread"] = FAIL(f"{mkt.spread_bps:.1f}bps") if mkt.spread_bps > r["max_spread_bps"] else OK
        if mkt.quote_volume_24h is not None:
            checks["liquidity"] = FAIL(f"24h volume {mkt.quote_volume_24h:,.0f}") \
                if mkt.quote_volume_24h < r["min_liquidity_quote_volume"] else OK
        if mkt.expected_slippage_bps is not None:
            checks["slippage"] = FAIL(f"{mkt.expected_slippage_bps:.0f}bps") \
                if mkt.expected_slippage_bps > r["max_expected_slippage_bps"] else OK
        stop_pct = 1 - p.stop_price / p.entry_price if p.entry_price else 0
        checks["stop_distance"] = FAIL(f"{stop_pct:.1%}") if stop_pct <= 0 or stop_pct > r["max_stop_pct"] + 1e-9 else OK

        # ---- sizing
        risk_budget = eq * r["risk_per_trade_pct"]
        size_risk = risk_budget / stop_pct if stop_pct > 0 else 0.0
        sym_now = abs(acct.positions.get(p.symbol, {}).get("notional", 0.0))
        caps = {
            "requested": p.requested_notional,
            "risk_budget": size_risk,
            "max_position": eq * r["max_position_pct"] - sym_now,
            "symbol_exposure": eq * r["max_symbol_exposure_pct"] - sym_now,
            "portfolio_exposure": eq * r["max_portfolio_exposure_pct"] - acct.exposure,
            "available_balance": acct.cash * 0.98,
        }
        size = max(0.0, min(caps.values()))
        binding = min(caps, key=caps.get)
        if binding != "requested":
            reasons.append(f"size capped by {binding}")
        checks["max_position"] = OK if caps["max_position"] > 0 else FAIL("position limit reached")
        checks["portfolio_exposure"] = OK if caps["portfolio_exposure"] > 0 else FAIL("portfolio exposure limit")
        checks["available_balance"] = OK if caps["available_balance"] > 0 else FAIL("no free balance")

        mult = 1.0
        if da and da.risk_adjustment:
            mult *= 1 + da.risk_adjustment
            reasons.append(f"devil's advocate {da.risk_adjustment:+.0%}")
        if mkt.regime_exposure_multiplier < 1:
            mult *= mkt.regime_exposure_multiplier
            reasons.append(f"regime exposure x{mkt.regime_exposure_multiplier:.2f}")
        checks["volatility"] = WARN if mkt.high_vol else OK
        if mkt.high_vol:
            mult *= r["high_vol_size_mult"]
            reasons.append(f"high volatility x{r['high_vol_size_mult']}")
        checks["correlation"] = WARN if mkt.correlation_warning else OK
        if mkt.correlation_warning:
            mult *= r["correlation_warning_mult"]
            reasons.append(f"correlation x{r['correlation_warning_mult']}")
        if daily >= 0.7 * r["max_daily_loss_pct"] or dd >= 0.7 * r["max_drawdown_pct"]:
            mult *= 0.5
            reasons.append("near loss limit x0.5")
        size *= max(0.0, mult)

        min_order = max(float(r["min_order_notional"]), mkt.min_notional * 1.1)
        checks["min_order"] = OK if size >= min_order else FAIL(f"size {size:.2f} < minimum {min_order:.2f}")
        failed = [k for k, v in checks.items() if v.startswith("FAIL")]
        approved = not failed
        if failed:
            reasons = [f"{k}: {checks[k][6:]}" for k in failed] + reasons
        return RiskDecision(p.id, approved, round(size if approved else 0.0, 2), p.requested_notional, checks, reasons)
