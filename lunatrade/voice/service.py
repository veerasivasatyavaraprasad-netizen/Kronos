"""
Voice agent. It has NO trading access:

    VOICE -> speech-to-text -> COMMAND PARSER -> AUTHORIZATION (PIN for sensitive intents) -> CONTROL API

The provider key lives only in the VOICE_API_KEY environment variable. Supported providers:
speech-to-text: assemblyai | deepgram | openai ; text-to-speech: elevenlabs | openai.
"""
from __future__ import annotations

import hmac
import logging
import os
import time

import requests

from lunatrade.config import secret
from lunatrade.voice.parser import Intent, parse

log = logging.getLogger("lunatrade.voice")


class SpeechProviders:
    def __init__(self, stt: str = "", tts: str = ""):
        self.stt, self.tts = (stt or "").lower(), (tts or "").lower()
        self.key = secret("VOICE_API_KEY")

    @property
    def stt_ready(self) -> bool:
        return bool(self.key and self.stt in ("assemblyai", "deepgram", "openai"))

    @property
    def tts_ready(self) -> bool:
        return bool((self.key or secret("ELEVENLABS_API_KEY")) and self.tts in ("elevenlabs", "openai"))

    def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        if not self.stt_ready:
            raise RuntimeError("speech-to-text not configured (set VOICE_PROVIDER and VOICE_API_KEY)")
        if self.stt == "assemblyai":
            h = {"authorization": self.key}
            up = requests.post("https://api.assemblyai.com/v2/upload", headers=h, data=audio, timeout=60)
            up.raise_for_status()
            job = requests.post("https://api.assemblyai.com/v2/transcript", headers=h, timeout=30,
                                json={"audio_url": up.json()["upload_url"]})
            job.raise_for_status()
            tid = job.json()["id"]
            for _ in range(60):
                r = requests.get(f"https://api.assemblyai.com/v2/transcript/{tid}", headers=h, timeout=30).json()
                if r.get("status") == "completed":
                    return r.get("text") or ""
                if r.get("status") == "error":
                    raise RuntimeError(r.get("error"))
                time.sleep(1)
            raise TimeoutError("transcription timed out")
        if self.stt == "deepgram":
            r = requests.post("https://api.deepgram.com/v1/listen?smart_format=true", data=audio, timeout=60,
                              headers={"Authorization": f"Token {self.key}", "Content-Type": mime})
            r.raise_for_status()
            return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
        r = requests.post("https://api.openai.com/v1/audio/transcriptions", timeout=60,
                          headers={"Authorization": f"Bearer {self.key}"},
                          files={"file": ("speech.wav", audio, mime)}, data={"model": "whisper-1"})
        r.raise_for_status()
        return r.json()["text"]

    def speak(self, text: str) -> bytes | None:
        if not self.tts_ready:
            return None
        if self.tts == "elevenlabs":
            key = secret("ELEVENLABS_API_KEY") or self.key
            voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
            r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}", timeout=60,
                              headers={"xi-api-key": key, "accept": "audio/mpeg"},
                              json={"text": text, "model_id": os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")})
            r.raise_for_status()
            return r.content
        r = requests.post("https://api.openai.com/v1/audio/speech", timeout=60,
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": "tts-1", "voice": "alloy", "input": text})
        r.raise_for_status()
        return r.content


class VoiceAgent:
    def __init__(self, control, council=None, cfg: dict | None = None):
        """control: object exposing status(), symbol_report(sym), pause(by), resume(by), emergency_close(by),
        close_symbol(sym, by), set_mode(mode, by), approve(id, yes, by), explain(sym), reset_kill_switch(by)."""
        self.control = control
        self.council = council
        self.cfg = cfg or {}
        self.speech = SpeechProviders(self.cfg.get("provider"), self.cfg.get("tts_provider"))
        self.pin = secret("LUNATRADE_CONTROL_PIN")
        self.history: list = []

    def authorized(self, intent: Intent, pin: str | None) -> bool:
        if not intent.sensitive:
            return True
        if not self.pin:
            return False          # sensitive voice commands are disabled until a PIN is configured
        return bool(pin) and hmac.compare_digest(str(pin), self.pin)

    def handle_text(self, text: str, pin: str | None = None, user: str = "voice") -> dict:
        intent = parse(text)
        if not self.authorized(intent, pin):
            answer = ("That command needs your control PIN." if self.pin else
                      "That command is disabled until LUNATRADE_CONTROL_PIN is configured.")
            return self._out(intent, answer, executed=False)
        c, by = self.control, f"{user}:voice"
        n = intent.name
        try:
            if n == "status":
                answer = c.status_sentence()
            elif n == "symbol_query":
                answer = c.symbol_report(intent.symbol)
            elif n == "pause":
                c.pause(by)
                answer = "Done. New trades are stopped. Open positions keep their stop-losses."
            elif n == "resume":
                c.resume(by)
                answer = "Trading resumed."
            elif n == "emergency_close":
                res = c.emergency_close(by)
                answer = f"Emergency close sent for {len(res)} positions and new trades are blocked."
            elif n == "close_symbol":
                answer = c.close_symbol(intent.symbol, by)
            elif n == "mode_change":
                answer = c.set_mode(intent.args["mode"], by)
            elif n == "reset_kill_switch":
                answer = c.reset_kill_switch(by)
            elif n in ("approve", "reject"):
                answer = c.approve(intent.args["id"], n == "approve", by)
            elif n == "explain":
                answer = c.explain_sentence(intent.symbol)
            elif n == "pnl":
                answer = c.pnl_sentence()
            elif n == "positions":
                answer = c.positions_sentence()
            elif n == "mode":
                answer = f"We are in {c.mode_name()} mode."
            elif n == "regime":
                answer = c.regime_sentence()
            elif n == "help":
                answer = ("You can ask for status, a coin's outlook, P and L, positions, or the regime. You can say "
                          "stop all new trades, resume trading, close BTC, or emergency close with your PIN.")
            else:
                answer = "Sorry, I didn't understand. Say help to hear what I can do."
        except Exception as e:
            log.exception("voice command failed")
            return self._out(intent, f"That failed: {e}", executed=False)
        if self.council is not None and n in ("status", "symbol_query", "regime"):
            res = self.council.ask_json("voice_assistant", "voice_answer",
                                        {"question": text, "state": c.brief_state(intent.symbol),
                                         "fallback_answer": answer})
            if res and isinstance(res.get("answer"), str) and res["answer"].strip():
                answer = res["answer"].strip()[:600]
        return self._out(intent, answer, executed=True)

    def handle_audio(self, audio: bytes, mime: str = "audio/wav", pin: str | None = None, user: str = "voice") -> dict:
        text = self.speech.transcribe(audio, mime)
        out = self.handle_text(text, pin, user)
        out["transcript"] = text
        return out

    def _out(self, intent: Intent, answer: str, executed: bool) -> dict:
        rec = {"intent": intent.name, "symbol": intent.symbol, "args": intent.args, "text": intent.text,
               "answer": answer, "executed": executed, "ts": time.time()}
        self.history.append(rec)
        self.history = self.history[-100:]
        return rec
