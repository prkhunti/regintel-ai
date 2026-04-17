#!/usr/bin/env bash
# ── Celery worker dev (outside Docker) ───────────────────────────────────────
# Starts the ingestion worker for rapid iteration without building images.
# Requires a local Python 3.12+ venv with deps installed:
#   pip install -e ".[worker,dev]"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_ROOT/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +o allexport
fi

export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://regintel:regintel@localhost:5433/regintel}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://localhost:6379/1}"
export PYTHONPATH="$REPO_ROOT"

cd "$REPO_ROOT/apps/worker"
exec celery -A tasks.celery_app worker \
  --loglevel=info \
  --concurrency=2 \
  -Q ingestion,celery
