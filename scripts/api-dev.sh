#!/usr/bin/env bash
# ── API dev server (outside Docker) ──────────────────────────────────────────
# Starts uvicorn with hot reload for rapid iteration without building images.
# Requires a local Python 3.12+ venv with deps installed:
#   pip install -e ".[api,dev]"
#
# Expects Postgres and Redis to be running (use `make up` to start infra only,
# or run them directly if you prefer).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env into the shell environment
if [ -f "$REPO_ROOT/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +o allexport
fi

# Point at local Postgres / Redis instead of Docker service names
export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://regintel:regintel@localhost:5433/regintel}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://localhost:6379/1}"
export PYTHONPATH="$REPO_ROOT"

cd "$REPO_ROOT/apps/api"
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir "$REPO_ROOT/apps/api/app" \
  --reload-dir "$REPO_ROOT/packages"
