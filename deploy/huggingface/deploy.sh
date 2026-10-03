#!/usr/bin/env bash
# Deploy the web UI to a Hugging Face Docker Space (free CPU hardware is enough for Kronos-small).
#   huggingface-cli login            # once, with a write token
#   ./deploy/huggingface/deploy.sh your-username/kronos
# Then in the Space settings add secrets: KRONOS_USERS=you:strong-password, KRONOS_SECRET_KEY=<random>,
# and variable KRONOS_SECURE_COOKIES=1.
set -euo pipefail
SPACE=${1:?usage: deploy.sh <username/space-name>}
cd "$(dirname "$0")/../.."
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

cp -r Dockerfile .dockerignore requirements.txt requirements-app.txt conftest.py model automation webui data tests "$STAGE"/
mkdir -p "$STAGE/finetune_csv" && cp -r finetune_csv/*.py finetune_csv/configs finetune_csv/data "$STAGE/finetune_csv/"
rm -rf "$STAGE"/webui/prediction_results/*.json
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
cp deploy/huggingface/README.md "$STAGE/README.md"

python - "$SPACE" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
space, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=space, repo_type="space", folder_path=folder, commit_message="Deploy Kronos")
print(f"Deployed: https://huggingface.co/spaces/{space}")
PY
