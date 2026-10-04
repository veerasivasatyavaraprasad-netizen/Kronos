"""Human approval queue: the bot proposes, the owner answers YES / NO, then execution happens (or not)."""
from __future__ import annotations

import datetime as dt
import threading
from dataclasses import dataclass, field

PENDING, APPROVED, REJECTED, EXPIRED, EXECUTED = "PENDING", "APPROVED", "REJECTED", "EXPIRED", "EXECUTED"


@dataclass
class ApprovalRequest:
    proposal: object
    decision: object
    devils: object
    requested_at: dt.datetime
    expires_at: dt.datetime
    status: str = PENDING
    decided_by: str = ""
    decided_at: dt.datetime | None = None
    extra: dict = field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return self.proposal.id.replace("prop_", "")[:6]

    def summary(self) -> str:
        p, d = self.proposal, self.decision
        lines = [f"Trade needs your YES/NO  [{self.short_id}]",
                 f"{p.action} {p.symbol} {p.direction.value}  conviction {p.conviction:.0f}/100",
                 f"size ${d.approved_notional:,.2f}  entry ~{p.entry_price:.6g}  stop {p.stop_price:.6g}  target {p.target_price:.6g}",
                 f"regime {p.regime}"]
        if self.devils is not None:
            lines.append(f"devil's advocate: bull {self.devils.bull_case} / bear {self.devils.bear_case}")
            if self.devils.hidden_risks:
                lines.append("risks: " + "; ".join(self.devils.hidden_risks[:3]))
        lines.append(f"Reply /yes {self.short_id} or /no {self.short_id} (expires {self.expires_at:%H:%M} UTC)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"proposal_id": self.proposal.id, "short_id": self.short_id, "status": self.status,
                "symbol": self.proposal.symbol, "action": self.proposal.action,
                "conviction": self.proposal.conviction, "notional": self.decision.approved_notional,
                "requested_at": self.requested_at.isoformat(), "expires_at": self.expires_at.isoformat(),
                "decided_by": self.decided_by, "summary": self.summary()}


class ApprovalQueue:
    def __init__(self, timeout_minutes: float = 15, notifier=None, repo=None):
        self.timeout = dt.timedelta(minutes=timeout_minutes)
        self.notifier = notifier
        self.repo = repo
        self.items: dict[str, ApprovalRequest] = {}
        self._lock = threading.RLock()

    def request(self, proposal, decision, devils, now: dt.datetime | None = None) -> ApprovalRequest:
        now = now or dt.datetime.utcnow()
        req = ApprovalRequest(proposal, decision, devils, now, now + self.timeout)
        with self._lock:
            self.items[proposal.id] = req
        self._persist(req)
        if self.notifier:
            self.notifier.send(req.summary(), "approval")
        return req

    def resolve(self, ident: str) -> ApprovalRequest | None:
        with self._lock:
            if ident in self.items:
                return self.items[ident]
            matches = [r for r in self.items.values() if r.short_id == ident or r.proposal.id.endswith(ident)]
            return matches[0] if len(matches) == 1 else None

    def decide(self, ident: str, approve: bool, by: str, now: dt.datetime | None = None) -> ApprovalRequest | None:
        now = now or dt.datetime.utcnow()
        req = self.resolve(ident)
        if not req or req.status != PENDING:
            return req
        if now > req.expires_at:
            req.status = EXPIRED
        else:
            req.status = APPROVED if approve else REJECTED
        req.decided_by, req.decided_at = by, now
        self._persist(req)
        return req

    def expire(self, now: dt.datetime | None = None) -> list[ApprovalRequest]:
        now = now or dt.datetime.utcnow()
        out = []
        with self._lock:
            for r in self.items.values():
                if r.status == PENDING and now > r.expires_at:
                    r.status = EXPIRED
                    out.append(r)
                    self._persist(r)
        return out

    def pending(self) -> list[ApprovalRequest]:
        return [r for r in self.items.values() if r.status == PENDING]

    def approved_ready(self) -> list[ApprovalRequest]:
        return [r for r in self.items.values() if r.status == APPROVED]

    def mark_executed(self, req: ApprovalRequest) -> None:
        req.status = EXECUTED
        self._persist(req)

    def _persist(self, r: ApprovalRequest) -> None:
        if self.repo:
            try:
                self.repo.approval({"proposal_id": r.proposal.id, "requested_at": r.requested_at,
                                    "expires_at": r.expires_at, "status": r.status, "decided_by": r.decided_by,
                                    "decided_at": r.decided_at, "summary": r.summary()})
            except Exception:
                pass
