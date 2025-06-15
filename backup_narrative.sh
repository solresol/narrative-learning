#!/usr/bin/env bash
# Dump the PostgreSQL 'narrative' database to a staging file and copy it to the remote host.
set -euo pipefail

DUMP_DIR="${1:-dumps}"
mkdir -p "$DUMP_DIR"

DUMP_FILE="narrative.sql.gz"

pg_dump narrative | gzip > "$DUMP_DIR/$DUMP_FILE"

scp "$DUMP_DIR/$DUMP_FILE" merah.cassia.ifost.org.au:/var/www/vhosts/datadumps.ifost.org.au/htdocs/narrative-learning/
