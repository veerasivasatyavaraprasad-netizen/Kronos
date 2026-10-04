"""
Control API + dashboard (FastAPI).

Auth: every /api route needs `Authorization: Bearer <token>` (LUNATRADE_API_TOKEN, or the generated token in
<output_dir>/api_token). Sensitive actions (emergency close, resume, mode change, kill-switch reset, closing a
position) additionally need the control PIN (LUNATRADE_CONTROL_PIN) in the `X-Control-Pin` header.
"""
from __future__ import annotations

import hmac
import math
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

STATIC = Path(__file__).with_name("static")


def _clean(o):
    """JSON-safe (NaN/inf -> None)."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_clean(v) for v in (sorted(o, key=str) if isinstance(o, (set, frozenset)) else o)]
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


class TextCommand(BaseModel):
    text: str


class ModeChange(BaseModel):
    mode: str
    force_gates: bool = False


class ResetBody(BaseModel):
    reason: str | None = None


def create_app(rt) -> FastAPI:
    app = FastAPI(title="LunaTrade", version="1.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def auth(authorization: str = Header(default="")) -> str:
        token = authorization.removeprefix("Bearer ").strip()
        if not token or not hmac.compare_digest(token, rt.api_token):
            raise HTTPException(401, "invalid or missing API token")
        return "api"

    def pin_required(x_control_pin: str = Header(default="")) -> None:
        if not rt.control_pin:
            raise HTTPException(403, "set LUNATRADE_CONTROL_PIN to enable sensitive actions")
        if not hmac.compare_digest(x_control_pin, rt.control_pin):
            raise HTTPException(403, "wrong control PIN")

    ok = lambda data: JSONResponse(_clean(data))

    # ------------------------------------------------------------------ public
    @app.get("/")
    def dashboard():
        return FileResponse(STATIC / "dashboard.html")

    @app.get("/health")
    def health():
        st = rt.supervisor.last or {}
        healthy = st.get("healthy", True)
        return JSONResponse({"status": "ok" if healthy else "degraded", "mode": rt.mode.value,
                             "kill_switch": rt.engine.kill_switch.active}, status_code=200 if healthy else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        e = rt.engine
        st = e.status()
        wc = e.worker_counts()
        lines = [f"lunatrade_equity {st['portfolio']['equity']}", f"lunatrade_exposure_pct {st['portfolio']['exposure_pct']}",
                 f"lunatrade_drawdown {st['portfolio']['drawdown']}", f"lunatrade_cycle_ms {e.last_cycle_ms}",
                 f"lunatrade_cycles_total {e.cycle_count}", f"lunatrade_kill_switch {int(e.kill_switch.active)}",
                 f"lunatrade_open_positions {len(st['portfolio']['positions'])}"]
        lines += [f'lunatrade_workers{{status="{k}"}} {v}' for k, v in wc.items() if k != "total"]
        lines += [f'lunatrade_events_total{{topic="{t}"}} {n}' for t, n in e.bus.counts.items()]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ read
    @app.get("/api/status")
    def status(_=Depends(auth)):
        return ok({**rt.engine.status(), "health": rt.supervisor.last, "symbols": rt.symbols,
                   "started_at": rt.started_at.isoformat()})

    @app.get("/api/signals")
    def signals(symbol: str | None = None, limit: int = 200, _=Depends(auth)):
        sigs = [s.to_dict() for s in rt.engine.last_signals if not symbol or s.symbol in (symbol, "*")]
        return ok(sorted(sigs, key=lambda s: -s["confidence"])[:limit])

    @app.get("/api/assessments")
    def assessments(_=Depends(auth)):
        return ok({s: a.to_dict() for s, a in rt.engine.assessments.items()})

    @app.get("/api/regime")
    def regime(_=Depends(auth)):
        return ok({s: r.to_dict() for s, r in rt.engine.regimes.items()})

    @app.get("/api/portfolio")
    def portfolio(_=Depends(auth)):
        return ok(rt.engine.status()["portfolio"])

    @app.get("/api/orders")
    def orders(limit: int = 100, _=Depends(auth)):
        return ok(rt.engine.gateway.recent(limit))

    @app.get("/api/trades")
    def trades(limit: int = 100, _=Depends(auth)):
        return ok(rt.repo.recent("trade_outcomes", limit, "id"))

    @app.get("/api/proposals")
    def proposals(_=Depends(auth)):
        return ok(list(rt.engine.proposals))

    @app.get("/api/agents")
    def agents(family: str | None = None, _=Depends(auth)):
        ws = [w.describe() for w in rt.engine.workers if not family or w.family.value == family]
        return ok({"summary": rt.engine.worker_counts(), "workers": ws, "llm": rt.council.status()})

    @app.get("/api/strategies")
    def strategies(_=Depends(auth)):
        return ok({"memory": rt.engine.memory.summary(), "weight_changes": list(rt.engine.weight_changes),
                   "min_conviction": rt.engine.memory.min_conviction(float(rt.cfg.get("lead_brain.min_conviction", 62)))})

    @app.get("/api/news")
    def news(limit: int = 50, _=Depends(auth)):
        items = sorted(rt.feeds.news.items, key=lambda x: x.ts, reverse=True)[:limit]
        return ok([x.to_dict() for x in items])

    @app.get("/api/events")
    def events(topic: str = "*", limit: int = 100, _=Depends(auth)):
        return ok([e.to_dict() for e in rt.bus.recent(topic, limit)])

    @app.get("/api/alerts")
    def alerts(limit: int = 50, _=Depends(auth)):
        return ok(rt.repo.recent("system_alerts", limit))

    @app.get("/api/explain/{proposal_id}")
    def explain(proposal_id: str, _=Depends(auth)):
        data = rt.repo.explain(proposal_id)
        if not data:
            raise HTTPException(404, "unknown proposal")
        return ok(data)

    @app.get("/api/approvals")
    def approvals(_=Depends(auth)):
        return ok([r.to_dict() for r in rt.engine.approvals.items.values()])

    @app.get("/api/promotion/{target}")
    def promotion(target: str, _=Depends(auth)):
        return ok(rt.control.promotion_check(target.upper()))

    @app.get("/api/equity")
    def equity(limit: int = 500, _=Depends(auth)):
        rows = rt.repo.recent("portfolio_snapshots", limit)
        return ok([{"ts": r["ts"], "equity": r["equity"], "drawdown": r["drawdown"]} for r in rows][::-1])

    @app.get("/api/db")
    def db(_=Depends(auth)):
        return ok(rt.repo.counts())

    # ------------------------------------------------------------------ actions
    @app.post("/api/approvals/{ident}/approve")
    def approve(ident: str, _=Depends(auth)):
        return ok({"result": rt.control.approve(ident, True, "dashboard")})

    @app.post("/api/approvals/{ident}/reject")
    def reject(ident: str, _=Depends(auth)):
        return ok({"result": rt.control.approve(ident, False, "dashboard")})

    @app.post("/api/control/pause")
    def pause(_=Depends(auth)):
        rt.control.pause("dashboard")
        return ok({"paused": True})

    @app.post("/api/control/resume")
    def resume(_=Depends(auth), __=Depends(pin_required)):
        rt.control.resume("dashboard")
        return ok({"paused": False})

    @app.post("/api/control/emergency-close")
    def emergency(_=Depends(auth), __=Depends(pin_required)):
        return ok({"orders": rt.control.emergency_close("dashboard")})

    @app.post("/api/control/close/{symbol}")
    def close(symbol: str, _=Depends(auth), __=Depends(pin_required)):
        return ok({"result": rt.control.close_symbol(symbol.upper(), "dashboard")})

    @app.post("/api/control/kill-switch/reset")
    def reset(body: ResetBody, _=Depends(auth), __=Depends(pin_required)):
        return ok({"result": rt.control.reset_kill_switch("dashboard", body.reason)})

    @app.post("/api/control/mode")
    def mode(body: ModeChange, _=Depends(auth), __=Depends(pin_required)):
        return ok({"result": rt.control.set_mode(body.mode, "dashboard", body.force_gates)})

    @app.post("/api/voice/command")
    def voice_command(body: TextCommand, _=Depends(auth), x_control_pin: str = Header(default="")):
        return ok(rt.voice.handle_text(body.text, x_control_pin or None, "dashboard"))

    @app.post("/api/voice/audio")
    async def voice_audio(file: UploadFile = File(...), _=Depends(auth), x_control_pin: str = Header(default="")):
        audio = await file.read()
        try:
            out = rt.voice.handle_audio(audio, file.content_type or "audio/wav", x_control_pin or None, "dashboard")
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return ok(out)

    @app.post("/api/voice/speak")
    def speak(body: TextCommand, _=Depends(auth)):
        audio = rt.voice.speech.speak(body.text)
        if audio is None:
            raise HTTPException(400, "text-to-speech not configured (VOICE_TTS_PROVIDER + key)")
        return Response(audio, media_type="audio/mpeg")

    @app.exception_handler(Exception)
    async def errors(request: Request, exc: Exception):
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app
