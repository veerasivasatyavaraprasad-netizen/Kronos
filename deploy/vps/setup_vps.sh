#!/usr/bin/env bash
# Run on a fresh Ubuntu/Debian VPS (2+ GB RAM) from the repo root:
#   sudo ./deploy/vps/setup_vps.sh
# Point your domain's DNS A record at the server first so HTTPS certificates can be issued.
set -euo pipefail
cd "$(dirname "$0")/../.."

if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
if [ ! -f .env ]; then
  cp .env.example .env
  secret=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
  sed -i "s/^KRONOS_SECRET_KEY=.*/KRONOS_SECRET_KEY=$secret/" .env
  echo "Created .env - edit DOMAIN and KRONOS_USERS (and alert/broker keys), then re-run this script."
  exit 0
fi
grep -q '^DOMAIN=.\+' .env || { echo "Set DOMAIN in .env first"; exit 1; }
grep -q '^KRONOS_USERS=admin:change-me' .env && { echo "Change the default KRONOS_USERS password in .env first"; exit 1; }
mkdir -p outputs data
docker compose --env-file .env -f deploy/vps/docker-compose.prod.yml up -d --build
echo "Deployed. Open https://$(grep '^DOMAIN=' .env | cut -d= -f2)"
