"""Owner notifications - reuses the Kronos alert channels (Telegram, email, Discord, Slack)."""
from __future__ import annotations

import logging
import os

import requests

from automation import notify

log = logging.getLogger("lunatrade.notify")


class Notifier:
    def __init__(self, enabled: bool | None = None, channels: list | None = None):
        self.cfg = {"enabled": True if enabled is None else enabled,
                    "channels": channels or ["telegram", "email", "discord", "slack"]}
        self.sent: list = []

    @property
    def configured(self) -> list:
        return notify.configured_channels(self.cfg)

    def send(self, text: str, level: str = "info") -> dict:
        self.sent.append({"level": level, "text": text})
        self.sent = self.sent[-200:]
        if not self.cfg["enabled"] or not self.configured:
            log.info("[%s] %s", level, text)
            return {}
        return notify.send_text(f"🌙 LunaTrade\n{text}", self.cfg)


class TelegramCommands:
    """Polls Telegram for owner commands: /yes <id>, /no <id>, /status, /stop, /resume <pin>."""

    def __init__(self, handler):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat = os.getenv("TELEGRAM_CHAT_ID")
        self.handler = handler
        self.offset = 0

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat)

    def poll(self) -> int:
        if not self.enabled:
            return 0
        try:
            r = requests.get(f"https://api.telegram.org/bot{self.token}/getUpdates",
                             params={"offset": self.offset, "timeout": 0}, timeout=15)
            r.raise_for_status()
            updates = r.json().get("result", [])
        except Exception as e:
            log.debug("telegram poll failed: %s", e)
            return 0
        n = 0
        for u in updates:
            self.offset = max(self.offset, u["update_id"] + 1)
            msg = u.get("message") or {}
            if str(msg.get("chat", {}).get("id")) != str(self.chat):
                continue           # only the owner's chat may command the bot
            text = (msg.get("text") or "").strip()
            if text.startswith("/"):
                reply = self.handler(text)
                n += 1
                if reply:
                    try:
                        requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                      data={"chat_id": self.chat, "text": reply[:4000]}, timeout=15)
                    except Exception:
                        pass
        return n
