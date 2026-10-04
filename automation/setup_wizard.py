"""
Interactive setup: writes API keys into .env (hidden input) and checks every connection.

    python -m automation.kronos_auto configure     # ask for keys, then run the checks
    python -m automation.kronos_auto check         # only run the checks
"""
import getpass
import os
import secrets
import time
from pathlib import Path

import requests

from automation.env import ROOT, load_env

ENV_PATH = ROOT / ".env"

QUESTIONS = [
    ("Binance (real-time crypto trading)", [
        ("BINANCE_API_KEY", "Binance API key", True),
        ("BINANCE_SECRET_KEY", "Binance secret key", True),
    ]),
    ("Binance testnet (optional practice exchange, keys from testnet.binance.vision)", [
        ("BINANCE_TESTNET_API_KEY", "Testnet API key", True),
        ("BINANCE_TESTNET_SECRET_KEY", "Testnet secret key", True),
    ]),
    ("Alpaca (US stocks, paper account)", [
        ("ALPACA_API_KEY", "Alpaca API key", True),
        ("ALPACA_SECRET_KEY", "Alpaca secret key", True),
    ]),
    ("Telegram alerts (optional)", [
        ("TELEGRAM_BOT_TOKEN", "Bot token from @BotFather", True),
        ("TELEGRAM_CHAT_ID", "Your chat id", False),
    ]),
    ("Web UI login (optional, needed only if you host it online)", [
        ("KRONOS_USERS", "username:password", True),
    ]),
]


def read_env_file(path=None) -> dict:
    path = Path(path or ENV_PATH)
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                values[k.strip()] = v.split(" #", 1)[0].strip()
    return values


def write_env_file(updates: dict, path=None):
    """Update keys in place (keeps comments/order of .env.example), append unknown keys."""
    path = Path(path or ENV_PATH)
    template = path if path.exists() else ROOT / ".env.example"
    lines = template.read_text(encoding="utf-8").splitlines() if template.exists() else []
    done = set()
    for i, line in enumerate(lines):
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in updates:
                lines[i] = f"{key}={updates[key]}"
                done.add(key)
    lines += [f"{k}={v}" for k, v in updates.items() if k not in done]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def mask(v):
    return "(not set)" if not v else (v[:4] + "…" + v[-2:] if len(v) > 8 else "set")


KEY_FORMATS = {  # (regex, human description)
    "BINANCE_API_KEY": (r"[A-Za-z0-9]{64}", "64 letters/digits"),
    "BINANCE_SECRET_KEY": (r"[A-Za-z0-9]{64}", "64 letters/digits"),
    "BINANCE_TESTNET_API_KEY": (r"[A-Za-z0-9]{64}", "64 letters/digits"),
    "BINANCE_TESTNET_SECRET_KEY": (r"[A-Za-z0-9]{64}", "64 letters/digits"),
    "ALPACA_API_KEY": (r"[A-Z0-9]{16,40}", "16-40 capital letters/digits"),
    "ALPACA_SECRET_KEY": (r"[A-Za-z0-9]{30,60}", "30-60 letters/digits"),
    "TELEGRAM_BOT_TOKEN": (r"\d+:[A-Za-z0-9_-]{30,}", "like 123456:ABC..."),
    "TELEGRAM_CHAT_ID": (r"-?\d+", "a number"),
}


def clean_value(key: str, raw: str):
    """Strip what commonly sneaks in when pasting: Ctrl+V control chars, spaces, quotes, 'KEY=' prefixes.
    Returns (value, had_control_chars)."""
    had_ctrl = any(ord(c) < 32 or ord(c) == 127 for c in raw)
    v = "".join(c for c in raw if ord(c) >= 32 and ord(c) != 127).strip().strip('"').strip("'").strip()
    if "=" in v and v.split("=", 1)[0].strip().upper() == key:
        v = v.split("=", 1)[1].strip().strip('"').strip("'")
    if key in KEY_FORMATS and key != "TELEGRAM_BOT_TOKEN":
        v = v.replace(" ", "")
    return v, had_ctrl


def valid_format(key: str, value: str) -> bool:
    import re
    fmt = KEY_FORMATS.get(key)
    return fmt is None or re.fullmatch(fmt[0], value) is not None


def ask(key, label, hidden, current):
    for _ in range(3):
        prompt = f"  {label} [{mask(current)}]: "
        raw = getpass.getpass(prompt) if hidden else input(prompt)
        value, had_ctrl = clean_value(key, raw)
        if not value:
            return None
        if valid_format(key, value):
            print(f"    received {len(value)} characters - format OK")
            return value
        print(f"    That does not look right: got {len(value)} characters, expected {KEY_FORMATS[key][1]}.")
        if had_ctrl or not value:
            print("    Ctrl+V does not paste into this hidden prompt - paste with a RIGHT-CLICK instead.")
        else:
            print("    Copy the key again from the website (no spaces, no quotes) and paste with a right-click.")
        print("    Try again, or press Enter to skip.")
    print("    Skipped.")
    return None


def configure():
    current = read_env_file()
    print("Kronos setup. Press Enter to keep the current value. Keys are typed hidden and saved only in")
    print("PASTE WITH A RIGHT-CLICK (Ctrl+V does not work in hidden prompts).")
    print(f"{ENV_PATH} on this computer.\n")
    print("Binance key settings: enable 'Spot & Margin Trading', keep 'Enable Withdrawals' OFF,")
    print("and choose 'Restrict access to trusted IPs only' with this PC's IP.\n")
    updates = {}
    for section, items in QUESTIONS:
        print(f"== {section}")
        for key, label, hidden in items:
            value = ask(key, label, hidden, current.get(key))
            if value:
                updates[key] = value
    if not (current.get("KRONOS_SECRET_KEY") or updates.get("KRONOS_SECRET_KEY")):
        updates["KRONOS_SECRET_KEY"] = secrets.token_hex(32)
    if updates:
        write_env_file(updates)
        print(f"\nSaved {len(updates)} value(s) to {ENV_PATH}.")
    for k, v in updates.items():
        os.environ[k] = v
    print()
    return check()


# ----------------------------------------------------------------------------- checks

def _binance_signed(base, key, secret, path, extra=None):
    import hashlib
    import hmac
    from urllib.parse import urlencode
    server = requests.get(f"{base}/api/v3/time", timeout=10).json()["serverTime"]
    params = {**(extra or {}), "timestamp": server, "recvWindow": 10000}
    q = urlencode(params)
    sig = hmac.new(secret.encode(), q.encode(), hashlib.sha256).hexdigest()
    return requests.get(f"{base}{path}?{q}&signature={sig}", headers={"X-MBX-APIKEY": key}, timeout=15)


def _env_key(name):
    return clean_value(name, os.getenv(name) or "")[0]


def check_binance():
    key, secret = _env_key("BINANCE_API_KEY"), _env_key("BINANCE_SECRET_KEY")
    if not key or not secret:
        return "skip", "no BINANCE_API_KEY / BINANCE_SECRET_KEY"
    bad = [n for n, v in (("API key", key), ("secret key", secret)) if not valid_format("BINANCE_API_KEY", v)]
    if bad:
        return "fail", (f"{' and '.join(bad)} not in Binance format (64 letters/digits; yours: {len(key)} and "
                        f"{len(secret)} characters). Re-enter with scripts\\configure.bat, pasting with a right-click")
    base = os.getenv("BINANCE_API_BASE_URL", "https://api.binance.com").rstrip("/")
    try:
        drift = abs(requests.get(f"{base}/api/v3/time", timeout=10).json()["serverTime"] - time.time() * 1000)
        r = _binance_signed(base, key, secret, "/api/v3/account")
        if r.status_code != 200:
            return "fail", f"{r.status_code} {r.text[:200]}"
        acct = r.json()
        usdt = next((b["free"] for b in acct["balances"] if b["asset"] == "USDT"), "0")
        msgs = [f"connected, canTrade={acct.get('canTrade')}, free USDT={float(usdt):.2f}"]
        status = "ok" if acct.get("canTrade") else "warn"
        if not acct.get("canTrade"):
            msgs.append("enable 'Spot & Margin Trading' on this key")
        rr = _binance_signed(base, key, secret, "/sapi/v1/account/apiRestrictions")
        if rr.status_code == 200:
            rest = rr.json()
            if rest.get("enableWithdrawals"):
                status = "warn"
                msgs.append("WITHDRAWALS ARE ENABLED on this key - turn them off in Binance API Management")
            if not rest.get("ipRestrict"):
                status = "warn"
                msgs.append("key is not IP-restricted - restrict it to this PC's IP")
        if drift > 3000:
            msgs.append(f"PC clock is {drift / 1000:.1f}s off - turn on automatic time sync in Windows")
        return status, "; ".join(msgs)
    except Exception as e:
        return "fail", f"{type(e).__name__}: {e}"


def check_binance_testnet():
    key, secret = _env_key("BINANCE_TESTNET_API_KEY"), _env_key("BINANCE_TESTNET_SECRET_KEY")
    if not key or not secret:
        return "skip", "no testnet keys (only needed for mode: testnet)"
    # Optional account: problems here are warnings, not failures.
    try:
        r = _binance_signed("https://testnet.binance.vision", key, secret, "/api/v3/account")
        if r.status_code == 200:
            return "ok", "testnet connected"
        return "warn", (f"{r.status_code} {r.text[:150]} - testnet keys come from testnet.binance.vision "
                        "(your normal Binance keys do not work there); only needed for mode: testnet")
    except Exception as e:
        return "warn", f"{type(e).__name__}: {e}"


def check_alpaca():
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return "skip", "no ALPACA_API_KEY / ALPACA_SECRET_KEY"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    for name, base in (("paper", "https://paper-api.alpaca.markets"), ("live", "https://api.alpaca.markets")):
        try:
            r = requests.get(f"{base}/v2/account", headers=headers, timeout=15)
        except Exception as e:
            return "fail", f"{type(e).__name__}: {e}"
        if r.status_code == 200:
            a = r.json()
            note = "" if name == "paper" else " - this is a LIVE key; the bot still uses paper unless live is double-opted-in"
            return ("ok" if name == "paper" else "warn"), \
                f"{name} account {a.get('status')}, equity {a.get('equity')}, buying power {a.get('buying_power')}{note}"
    return "fail", f"key rejected by both paper and live endpoints ({r.status_code})"


def check_telegram():
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token:
        return "skip", "no TELEGRAM_BOT_TOKEN"
    try:
        me = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15).json()
        if not me.get("ok"):
            return "fail", "bot token rejected"
        if not chat:
            ups = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15).json().get("result", [])
            ids = sorted({str(u["message"]["chat"]["id"]) for u in ups if "message" in u})
            hint = f"; chat ids that messaged the bot: {', '.join(ids)}" if ids else "; send any message to the bot, then re-run"
            return "warn", f"bot @{me['result']['username']} ok, TELEGRAM_CHAT_ID missing{hint}"
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": "✅ Kronos is connected to Telegram."}, timeout=15).json()
        return ("ok", "test message sent") if r.get("ok") else ("fail", r.get("description", "send failed"))
    except Exception as e:
        return "fail", f"{type(e).__name__}: {e}"


def check_model():
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache("NeoQuasar/Kronos-small", "config.json")
        return ("ok", "Kronos-small cached") if isinstance(cached, str) else \
            ("warn", "model not downloaded yet - it downloads on first start (needs internet)")
    except Exception as e:
        return "warn", str(e)


CHECKS = [("Binance", check_binance), ("Binance testnet", check_binance_testnet), ("Alpaca", check_alpaca),
          ("Telegram", check_telegram), ("Kronos model", check_model)]
ICON = {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "skip": "➖"}


def check() -> int:
    load_env()
    if os.getenv("KRONOS_ALLOW_LIVE_TRADING", "").lower() == "yes":
        print("⚠️  KRONOS_ALLOW_LIVE_TRADING=yes - real-money trading is permitted.\n")
    worst = 0
    for name, fn in CHECKS:
        status, msg = fn()
        print(f"{ICON[status]} {name:16} {msg}")
        worst = max(worst, {"ok": 0, "skip": 0, "warn": 0, "fail": 1}[status])
    return worst
