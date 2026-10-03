# Deploying Kronos online

Every option runs the same Docker image: the web UI is served by gunicorn on `$PORT`
(default 7070), Kronos-small is preloaded, and `/healthz` is the health check.

**Always turn on the login before going public.** Set these as secrets on the host:

| Variable | Value |
|---|---|
| `KRONOS_USERS` | `you:a-strong-password` (comma-separate more users) |
| `KRONOS_SECRET_KEY` | random string: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `KRONOS_SECURE_COOKIES` | `1` (the site is served over HTTPS) |

Memory: about 2 GB is needed for PyTorch + Kronos-small. Use `KRONOS_AUTOLOAD_MODEL=kronos-mini`
on smaller machines, or `kronos-base` if you have 4 GB+.

## 1. Any VPS (AWS, DigitalOcean, Hetzner, ...) — HTTPS via Caddy

Recommended if you also want the hourly scheduler (forecasts + alerts + auto-trading) running 24/7.

1. Create an Ubuntu server with 2 GB+ RAM. Point your domain's DNS `A` record at its IP.
2. On the server:
   ```bash
   git clone <your repo> kronos && cd kronos
   sudo ./deploy/vps/setup_vps.sh      # first run creates .env with a random KRONOS_SECRET_KEY
   nano .env                           # set DOMAIN, KRONOS_USERS, alert/broker keys
   sudo ./deploy/vps/setup_vps.sh      # builds and starts web + scheduler + Caddy
   ```
3. Open `https://<your domain>`. Caddy obtains and renews the certificate automatically.

Update later with `git pull && sudo docker compose --env-file .env -f deploy/vps/docker-compose.prod.yml up -d --build`.
Forecast reports are written to `outputs/` on the server; the paper-trading account persists there too.

## 2. Hugging Face Spaces (free)

Free CPU hardware (16 GB RAM) is enough. The Space sleeps after inactivity and wakes on the next visit.

```bash
pip install huggingface_hub
huggingface-cli login                       # token with "write" permission
./deploy/huggingface/deploy.sh your-username/kronos
```

Then in the Space → **Settings → Variables and secrets** add `KRONOS_USERS`, `KRONOS_SECRET_KEY`
(secrets) and `KRONOS_SECURE_COOKIES=1` (variable). The URL is `https://your-username-kronos.hf.space`.
Uploaded files are lost when the Space restarts (no persistent disk on the free tier).

## 3. Render

1. Push this repo to GitHub.
2. Render dashboard → **New → Blueprint** → select the repo. `render.yaml` creates:
   - `kronos-web` — the web UI (Standard plan, 2 GB RAM) with an auto-generated `KRONOS_SECRET_KEY`;
   - `kronos-daily-forecast` — a cron job running `kronos_auto run` daily that delivers results by alert.
3. Enter `KRONOS_USERS` and any alert keys when prompted.

## 4. Railway

1. Railway dashboard → **New Project → Deploy from GitHub repo**. `railway.json` makes it build the
   Dockerfile and use `/healthz`.
2. In **Variables**, add `KRONOS_USERS`, `KRONOS_SECRET_KEY`, `KRONOS_SECURE_COOKIES=1`.
3. **Settings → Networking → Generate Domain** for a public URL.
4. Optional scheduled forecasts: add a second service from the same repo with start command
   `python -m automation.kronos_auto run` and a cron schedule (e.g. `17 1 * * *`).

## Scheduled forecasts without a server

The GitHub Actions workflow `.github/workflows/scheduled-forecast.yml` already runs daily. Add your
alert secrets under **Settings → Secrets and variables → Actions** (same names as in `.env.example`)
and set `alerts.enabled: true` in `automation/config.yaml`. Note the paper-trading account is not
kept between GitHub runs; use a VPS for continuous paper/auto trading.
