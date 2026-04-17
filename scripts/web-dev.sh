#!/usr/bin/env bash
# ── Next.js dev server (outside Docker) ──────────────────────────────────────
# Starts the Next.js HMR dev server for rapid UI iteration without Docker.
# Requires Node.js 20+.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# next.config.js reads API_URL at runtime for the /api/* rewrite.
# Point at the local API server (started via scripts/api-dev.sh or make up).
export API_URL="${API_URL:-http://localhost:8000}"

cd "$REPO_ROOT/apps/web"

if [ ! -d node_modules ]; then
  echo "node_modules not found — running npm install..."
  npm install
fi

exec npm run dev
