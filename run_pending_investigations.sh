#!/bin/bash
# Run all investigations that have no inference data, skipping
# those that use ollama-backed models.
#
# Usage: POSTGRES_DSN=postgres://user@host/db ./run_pending_investigations.sh
#        or pass the DSN as the first argument.
set -euo pipefail

DSN="${1:-${POSTGRES_DSN:-}}"
if [ -z "$DSN" ]; then
    echo "PostgreSQL DSN must be provided as POSTGRES_DSN or argument" >&2
    exit 1
fi

EMPTY_IDS=$(python list_empty_investigations.py --dsn "$DSN" --skip-ollama 2>/dev/null | awk '/^[0-9]+$/')

if [ -z "$EMPTY_IDS" ]; then
    echo "No pending investigations to run." >&2
    exit 0
fi

for id in $EMPTY_IDS; do
    echo "Running investigation $id"
    uv run investigate.py "$id" --dsn "$DSN"
done
