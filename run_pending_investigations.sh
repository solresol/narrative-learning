#!/bin/bash
# Run all investigations that have no inference data, skipping
# those that use ollama-backed models.
#
# Usage: POSTGRES_DSN=postgres://user@host/db ./run_pending_investigations.sh
#        or pass the DSN as the first argument. If neither is supplied the
#        libpq environment variables and defaults are used.
set -euo pipefail


DSN="${1:-${POSTGRES_DSN:-}}"
ARGS=()
if [ -n "$DSN" ]; then
    ARGS+=(--dsn "$DSN")
fi
EMPTY_IDS=$(python list_empty_investigations.py "${ARGS[@]}" --skip-ollama 2>/dev/null | awk '/^[0-9]+$/')

if [ -z "$EMPTY_IDS" ]; then
    echo "No pending investigations to run." >&2
    exit 0
fi

for id in $EMPTY_IDS; do
    echo "Running investigation $id"
    uv run investigate.py "$id" "${ARGS[@]}"
done
