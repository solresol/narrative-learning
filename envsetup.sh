#!/usr/bin/env bash
# Environment setup script for narrative-learning
set -euo pipefail

# Update package lists and install PostgreSQL
sudo apt-get update
sudo apt-get install -y postgresql postgresql-client
curl -LsSf https://astral.sh/uv/install.sh | sh

uv run uvbootstrapper.py

# Start PostgreSQL
sudo service postgresql start

# Ensure roles exist for the restore and for the current user
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='narrative'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE ROLE narrative LOGIN CREATEDB"
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='root'" | grep -q 1 || \
  sudo -u postgres psql -c "CREATE ROLE root SUPERUSER LOGIN"

# Download and restore the narrative database
DUMP_URL="https://datadumps.ifost.org.au/narrative-learning/narrative.sql.gz"
DUMP_DIR="dumps"
DUMP_FILE="$DUMP_DIR/narrative.sql.gz"
mkdir -p "$DUMP_DIR"
curl -L "$DUMP_URL" -o "$DUMP_FILE"

sudo -u postgres dropdb --if-exists narrative
sudo -u postgres createdb -O narrative narrative
gunzip -c "$DUMP_FILE" | sudo -u postgres psql narrative

