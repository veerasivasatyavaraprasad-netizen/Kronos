"""
Control facade - the single, authenticated entry point for humans (dashboard, REST API, voice, Telegram).
Nothing here can bypass the risk engine: manual actions are pause/resume, closing positions (risk
reducing), approving proposals the risk engine already approved, mode changes behind promotion gates, and
kill-switch resets.
"""
from __future__ import annotations

import datetime as dt

from lunatrade.control.modes import check_promotion
from lunatrade.core.events import Topic
from lunatrade.core.types import MARKET_WIDE, Mode


class Control:
    def __init__(self, runtime):
        self.rt = runtime

    @property
    def engine(self):
        return self.rt.engine

    # ------------------------------------------------------------------ actions
    def pause(self, by: str) -> None:
        self.engine.pause(by)
        self.engine.bus.emit(Topic.SYSTEM_ALERT, {"action": "pause", "by": by}, source="control")

    def resume(self, by: str) -> None:
        self.engine.resume(by)
        self.engine.bus.emit(Topic.SYSTEM_ALERT, {"action": "resume", "by": by}, source="control")

    def emergency_close(self, by: str) -> list:
        return self.engine.emergency_close_all(by)

    def close_symbol(self, symbol: str, by: str) -> str:
        o = self.engine.close_position(symbol, f"MANUAL:{by}")
        if o is None:
            return f"We have no {symbol} position."
        return f"Close order for {symbol}: {o.status.value}."

    def reset_kill_switch(self, by: str, reason: str | None = None) -> str:
        cleared = self.engine.kill_switch.reset(by, reason)
        return f"Kill switch reset ({', '.join(cleared) or 'nothing to clear'})."

    def approve(self, ident: str, yes: bool, by: str) -> str:
        req = self.engine.approvals.decide(ident, yes, by)
        if req is None:
            return f"No pending proposal {ident}."
        self.engine.bus.emit(Topic.TRADE_APPROVAL, {"proposal_id": req.proposal.id, "status": req.status, "by": by},
                             source="control", correlation_id=req.proposal.id)
        if req.status == "APPROVED":
            self.rt.execute_approved_now()
        return f"{req.proposal.symbol} proposal {req.short_id}: {req.status}."

    def promotion_check(self, target: str) -> dict:
        outcomes = self.rt.repo.outcomes() if self.rt.repo else self.engine.memory.outcomes
        first = None
        if outcomes:
            try:
                first = dt.datetime.fromisoformat(str(outcomes[0].get("opened_at")))
            except Exception:
                first = None
        dd = self.engine.status()["portfolio"]["drawdown"]
        if self.rt.repo:
            dd = max([dd] + [r.get("drawdown") or 0.0 for r in self.rt.repo.recent("portfolio_snapshots", 5000)])
        return check_promotion(self.engine.mode, Mode(target), outcomes, first, self.rt.cfg.section("promotion"), dd)

    def set_mode(self, target: str, by: str, force_gates: bool = False) -> str:
        target = target.upper()
        chk = self.promotion_check(target)
        if not chk["allowed"] and not force_gates:
            return f"Cannot switch to {target}: " + "; ".join(chk["reasons"])
        self.rt.switch_mode(Mode(target), by)
        return f"Mode switched to {target}."

    # ------------------------------------------------------------------ answers
    def mode_name(self) -> str:
        return self.engine.mode.value.lower()

    def brief_state(self, symbol: str | None = None) -> dict:
        st = self.engine.status()
        out = {"mode": st["mode"], "regime": (st["market_regime"] or {}).get("label"),
               "equity": round(st["portfolio"]["equity"], 2), "exposure_pct": round(st["portfolio"]["exposure_pct"], 3),
               "positions": list(st["portfolio"]["positions"]), "kill_switch": st["kill_switch"]["reasons"],
               "pending_approvals": len(st["pending_approvals"])}
        if symbol and symbol in self.engine.assessments:
            out["symbol"] = self.engine.assessments[symbol].to_dict()
        return out

    def status_sentence(self) -> str:
        st = self.engine.status()
        reg = (st["market_regime"] or {}).get("label", "unknown")
        longs = sum(1 for a in self.engine.assessments.values() if a.direction.value == "LONG" and a.conviction >= 55)
        shorts = sum(1 for a in self.engine.assessments.values() if a.direction.value == "SHORT" and a.conviction >= 55)
        neutral = len(self.engine.assessments) - longs - shorts
        pf = st["portfolio"]
        ks = " New trades are blocked by the kill switch." if st["kill_switch"]["active"] else ""
        pend = f" {len(st['pending_approvals'])} trade awaits your approval." if st["pending_approvals"] else \
            " No trade is currently pending."
        return (f"Market regime is {reg.lower().replace('_', ' ')}. {longs} symbols lean long, {shorts} lean bearish "
                f"and {neutral} are neutral. Equity is {pf['equity']:,.0f} with exposure at {pf['exposure_pct']:.0%}."
                f"{pend}{ks}")

    def symbol_report(self, symbol: str | None) -> str:
        if not symbol:
            return self.status_sentence()
        a = self.engine.assessments.get(symbol)
        if not a:
            return f"I'm not tracking {symbol}."
        r = self.engine.regimes.get(symbol)
        cats = sorted(a.category_scores, key=lambda c: -abs(c.score))[:3]
        drivers = ", ".join(f"{c.family} {c.score:+.0f}" for c in cats)
        held = self.engine.book.qty(symbol)
        pos = f" We hold {held:.6g}." if held else ""
        return (f"{symbol[:-4] if symbol.endswith('USDT') else symbol} is in a {r.label().lower().replace('_', ' ') if r else 'unknown'} "
                f"regime. Net view {a.direction.value.lower()} with conviction {a.conviction:.0f} of 100; "
                f"{a.families_agreeing} families agree. Main drivers: {drivers}.{pos}")

    def pnl_sentence(self) -> str:
        snap = self.engine.status()["portfolio"]
        unreal = sum(p["unrealized"] for p in snap["positions"].values())
        return (f"Equity {snap['equity']:,.2f}. Realised P and L {snap['realized']:+,.2f}, unrealised {unreal:+,.2f}. "
                f"Drawdown from peak {snap['drawdown']:.1%}.")

    def positions_sentence(self) -> str:
        pos = self.engine.status()["portfolio"]["positions"]
        if not pos:
            return "We have no open positions."
        parts = [f"{s}: {p['notional']:,.0f} ({p['unrealized']:+,.0f})" for s, p in pos.items()]
        return "Open positions: " + "; ".join(parts) + "."

    def regime_sentence(self) -> str:
        r = self.engine.regimes.get(MARKET_WIDE)
        if not r:
            return "No regime reading yet."
        return f"Market regime {r.label().lower().replace('_', ' ')}, exposure multiplier {r.exposure_multiplier:.1f}."

    def explain_sentence(self, symbol: str | None) -> str:
        props = [p for p in self.engine.proposals if (not symbol or p["symbol"] == symbol) and p["action"] == "OPEN"]
        if not props:
            return "There is no recent trade to explain."
        p = props[-1]
        return f"{p['symbol']} at conviction {p['conviction']:.0f}: {p['rationale']}. Status {p['status'].lower()}."

    # ------------------------------------------------------------------ Telegram
    def telegram(self, text: str) -> str:
        parts = text.split()
        cmd = parts[0].lower().split("@")[0]
        arg = parts[1] if len(parts) > 1 else ""
        if cmd in ("/yes", "/approve"):
            return self.approve(arg, True, "telegram")
        if cmd in ("/no", "/reject"):
            return self.approve(arg, False, "telegram")
        if cmd == "/status":
            return self.status_sentence()
        if cmd == "/stop":
            self.pause("telegram")
            return "New trades stopped."
        if cmd == "/resume":
            pin = self.rt.control_pin
            if not pin or arg != pin:
                return "Resume needs /resume <PIN> (LUNATRADE_CONTROL_PIN)."
            self.resume("telegram")
            return "Trading resumed."
        if cmd == "/positions":
            return self.positions_sentence()
        if cmd == "/pnl":
            return self.pnl_sentence()
        return "Commands: /status /positions /pnl /stop /resume <pin> /yes <id> /no <id>"
