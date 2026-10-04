"""
Automatic kill switch - deliberately outside the AI decision layer.

* Hard trips (loss limits, drawdown, abnormal slippage, reconciliation failure, manual emergency) block
  new orders until a human resets them through the authenticated control API / CLI.
* Soft halts (stale data, exchange/API unhealthy, risk engine heartbeat lost) block new orders and clear
  automatically once the supervisor has seen enough consecutive healthy checks.

No LLM output, worker or brain can call reset(); only the control API (human, authenticated) can.
A file named STOP in the output directory also blocks new orders (manual switch).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("lunatrade.kill_switch")


class KillSwitch:
    def __init__(self, state_dir: Path, cfg: dict, risk_cfg: dict, on_change=None):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "kill_switch.json"
        self.cfg, self.risk_cfg = cfg, risk_cfg
        self.on_change = on_change
        self._lock = threading.RLock()
        state = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.trips: dict = state.get("trips", {})     # reason -> {at, detail}
        self.halts: dict = {}                         # source -> detail (soft, not persisted)
        self.history: list = state.get("history", [])[-200:]

    # --------------------------------------------------------------- state
    @property
    def stop_file(self) -> Path:
        return self.dir / "STOP"

    @property
    def active(self) -> bool:
        return bool(self.trips or self.halts or self.stop_file.exists())

    def reasons(self) -> list[str]:
        out = [f"TRIP {k}: {v.get('detail', '')}" for k, v in self.trips.items()]
        out += [f"HALT {k}: {v}" for k, v in self.halts.items()]
        if self.stop_file.exists():
            out.append("STOP file present")
        return out

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"trips": self.trips, "history": self.history[-200:]}, indent=2, default=str))
        os.replace(tmp, self.path)

    def _event(self, kind: str, key: str, detail: str) -> None:
        rec = {"at": dt.datetime.utcnow().isoformat(), "kind": kind, "key": key, "detail": detail}
        self.history.append(rec)
        log.warning("kill switch %s %s: %s", kind, key, detail)
        if self.on_change:
            try:
                self.on_change(rec)
            except Exception:
                log.exception("kill switch callback failed")

    # --------------------------------------------------------------- hard trips
    def trip(self, reason: str, detail: str = "") -> None:
        with self._lock:
            if reason in self.trips:
                return
            self.trips[reason] = {"at": dt.datetime.utcnow().isoformat(), "detail": detail}
            self._save()
            self._event("TRIP", reason, detail)

    def reset(self, by: str, reason: str | None = None) -> list[str]:
        """Human reset (control API only). Returns the cleared reasons."""
        with self._lock:
            cleared = [reason] if reason and reason in self.trips else list(self.trips) if not reason else []
            for r in cleared:
                self.trips.pop(r, None)
            if self.stop_file.exists() and not reason:
                self.stop_file.unlink()
                cleared.append("STOP file")
            self._save()
            self._event("RESET", ",".join(cleared) or "-", f"by {by}")
            return cleared

    def manual_stop(self, by: str) -> None:
        self.trip("MANUAL_STOP", f"new trades stopped by {by}")

    # --------------------------------------------------------------- soft halts
    def halt(self, source: str, detail: str) -> None:
        with self._lock:
            if self.halts.get(source) != detail:
                new = source not in self.halts
                self.halts[source] = detail
                if new:
                    self._event("HALT", source, detail)

    def clear_halt(self, source: str) -> None:
        with self._lock:
            if self.halts.pop(source, None) is not None:
                self._event("RESUME", source, "healthy again")

    # --------------------------------------------------------------- automatic checks
    def check_account(self, equity: float, peak_equity: float, day_start_equity: float,
                      week_start_equity: float) -> None:
        r = self.risk_cfg
        if day_start_equity > 0 and (day_start_equity - equity) / day_start_equity > float(r.get("max_daily_loss_pct", 0.03)):
            self.trip("MAX_DAILY_LOSS", f"equity {equity:.2f} vs day start {day_start_equity:.2f}")
        if week_start_equity > 0 and (week_start_equity - equity) / week_start_equity > float(r.get("max_weekly_loss_pct", 0.07)):
            self.trip("MAX_WEEKLY_LOSS", f"equity {equity:.2f} vs week start {week_start_equity:.2f}")
        if peak_equity > 0 and (peak_equity - equity) / peak_equity > float(r.get("max_drawdown_pct", 0.15)):
            self.trip("MAX_DRAWDOWN", f"equity {equity:.2f} vs peak {peak_equity:.2f}")

    def check_slippage(self, slippage_bps: float) -> None:
        if abs(slippage_bps) > float(self.cfg.get("max_abnormal_slippage_bps", 150)):
            self.trip("ABNORMAL_SLIPPAGE", f"{slippage_bps:.0f} bps")

    def check_order_failures(self, consecutive: int) -> None:
        if consecutive >= int(self.cfg.get("max_consecutive_order_failures", 4)):
            self.trip("ORDER_FAILURES", f"{consecutive} consecutive order failures")

    def check_reconciliation(self, failures: int, detail: str = "") -> None:
        if failures >= int(self.cfg.get("max_reconciliation_failures", 2)):
            self.trip("RECONCILIATION_FAILED", detail or f"{failures} failed reconciliations")

    def check_data(self, stale: dict[str, float]) -> None:
        limit = float(self.cfg.get("max_data_staleness_seconds", 300))
        bad = {s: v for s, v in stale.items() if v > limit}
        if bad:
            self.halt("STALE_DATA", ", ".join(f"{s} {v:.0f}s" for s, v in list(bad.items())[:5]))
        else:
            self.clear_halt("STALE_DATA")

    def status(self) -> dict:
        return {"active": self.active, "trips": self.trips, "halts": self.halts, "stop_file": self.stop_file.exists(),
                "reasons": self.reasons(), "history": self.history[-20:]}
