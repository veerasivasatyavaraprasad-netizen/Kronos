"""
Forecast alerts: Telegram, email (SMTP), Discord and Slack.

Secrets come from environment variables (never from config.yaml); a channel is used
when its variables are set and it is enabled in the `alerts:` section of the config:

    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO, ALERT_EMAIL_FROM (optional)
    DISCORD_WEBHOOK_URL
    SLACK_WEBHOOK_URL
"""
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import requests

CHANNELS = ("telegram", "email", "discord", "slack")
REQUIRED_ENV = {
    "telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"),
    "email": ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_EMAIL_TO"),
    "discord": ("DISCORD_WEBHOOK_URL",),
    "slack": ("SLACK_WEBHOOK_URL",),
}


def configured_channels(alert_cfg: dict) -> list:
    """Channels that are enabled in config (default: all) and have their env vars set."""
    wanted = alert_cfg.get("channels", list(CHANNELS))
    return [c for c in wanted if c in REQUIRED_ENV and all(os.getenv(v) for v in REQUIRED_ENV[c])]


def select_results(results: list, alert_cfg: dict) -> list:
    """Apply alert filters: only_signals (e.g. [UP, DOWN]) and min_abs_change_pct."""
    only = set(alert_cfg.get("only_signals") or [])
    min_move = float(alert_cfg.get("min_abs_change_pct", 0))
    picked = []
    for r in results:
        if "error" in r:
            if alert_cfg.get("include_errors", True):
                picked.append(r)
            continue
        if only and r.get("signal") not in only:
            continue
        if abs(r.get("forecast_change_pct", 0)) < min_move:
            continue
        picked.append(r)
    return picked


def format_text(results: list, meta: dict, trades: list = None) -> str:
    icon = {"UP": "🟢", "DOWN": "🔴", "FLAT": "⚪"}
    lines = [f"Kronos forecast — {meta.get('generated', '')}", f"Model: {meta.get('model', '')}", ""]
    for r in results:
        if "error" in r:
            lines.append(f"⚠️ {r['name']}: {r['error'][:120]}")
            continue
        bt = r.get("backtest") or {}
        bt_txt = f" | backtest MAPE {bt['mape_close_pct']:.2f}%" if "mape_close_pct" in bt else ""
        lines.append(
            f"{icon.get(r['signal'], '')} {r['name']}: {r['last_close']:.4g} → {r['forecast_close']:.4g} "
            f"({r['forecast_change_pct']:+.2f}%, {r['signal']}, {r['paths_up_pct']:.0f}% paths up) "
            f"by {r['horizon_end']}{bt_txt}")
    if trades:
        lines += ["", "Orders:"]
        for t in trades:
            lines.append(f"  {t['mode'].upper()} {t['side'].upper()} {t['qty']} {t['symbol']} @ ~{t['price']:.4g} "
                         f"({t['status']})")
    lines += ["", "Research output only, not financial advice."]
    return "\n".join(lines)


def _charts(results, run_dir, limit=5):
    out = []
    for r in results:
        p = Path(run_dir) / f"{r['name']}_forecast.png"
        if "error" not in r and p.exists():
            out.append(p)
    return out[:limit]


def send_telegram(text, charts):
    token, chat = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    base = f"https://api.telegram.org/bot{token}"
    r = requests.post(f"{base}/sendMessage", data={"chat_id": chat, "text": text[:4000]}, timeout=30)
    r.raise_for_status()
    for p in charts:
        with open(p, "rb") as f:
            requests.post(f"{base}/sendPhoto", data={"chat_id": chat, "caption": p.stem},
                          files={"photo": f}, timeout=60).raise_for_status()


def send_email(text, charts):
    msg = EmailMessage()
    msg["Subject"] = text.splitlines()[0]
    msg["From"] = os.getenv("ALERT_EMAIL_FROM") or os.environ["SMTP_USER"]
    msg["To"] = os.environ["ALERT_EMAIL_TO"]
    msg.set_content(text)
    for p in charts:
        msg.add_attachment(p.read_bytes(), maintype="image", subtype="png", filename=p.name)
    port = int(os.getenv("SMTP_PORT") or "587")
    if port == 465:
        server = smtplib.SMTP_SSL(os.environ["SMTP_HOST"], port, timeout=30)
    else:
        server = smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=30)
        server.starttls()
    with server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(msg)


def send_discord(text, charts):
    url = os.environ["DISCORD_WEBHOOK_URL"]
    files = {f"file{i}": (p.name, p.read_bytes(), "image/png") for i, p in enumerate(charts[:10])}
    r = requests.post(url, data={"content": text[:1900]}, files=files or None, timeout=60)
    r.raise_for_status()


def send_slack(text, charts):
    # Incoming webhooks accept text only; charts stay in the run folder / artifact.
    r = requests.post(os.environ["SLACK_WEBHOOK_URL"], json={"text": f"```{text[:3500]}```"}, timeout=30)
    r.raise_for_status()


SENDERS = {"telegram": send_telegram, "email": send_email, "discord": send_discord, "slack": send_slack}


def send_alerts(results, meta, run_dir, alert_cfg, trades=None) -> dict:
    """Send the run summary to every configured channel. Returns {channel: 'ok' | error}."""
    if not alert_cfg or not alert_cfg.get("enabled", False):
        return {}
    picked = select_results(results, alert_cfg)
    if not picked and not trades:
        print("Alerts: nothing matched the alert filters.")
        return {}
    text = format_text(picked, meta, trades)
    charts = _charts(picked, run_dir) if alert_cfg.get("attach_charts", True) else []
    status = {}
    channels = configured_channels(alert_cfg)
    if not channels:
        print("Alerts enabled but no channel has its environment variables set (see automation/notify.py).")
    for ch in channels:
        try:
            SENDERS[ch](text, charts)
            status[ch] = "ok"
        except Exception as e:  # one failing channel must not stop the others
            status[ch] = f"{type(e).__name__}: {e}"
        print(f"Alert via {ch}: {status[ch]}")
    return status


def send_text(text: str, alert_cfg: dict, charts=None) -> dict:
    """Send a free-form message (live-trading fills, errors) to every configured channel."""
    if not alert_cfg or not alert_cfg.get("enabled", False):
        return {}
    status = {}
    for ch in configured_channels(alert_cfg):
        try:
            SENDERS[ch](text, list(charts or []))
            status[ch] = "ok"
        except Exception as e:
            status[ch] = f"{type(e).__name__}: {e}"
    return status
