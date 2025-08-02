#!/bin/bash
set -euo pipefail

export PGUSER="root"
export PGDATABASE="narrative"

# Update repository
git pull -q

# Record incomplete investigation counts
all_count=$(uv run find_incomplete_investigations.py | grep -E '^ ' | wc -l)
hosted_count=$(uv run find_incomplete_investigations.py --hosted-only | grep -E '^ ' | wc -l)
psql -c "INSERT INTO incomplete_investigation_counts(recorded_at, total, hosted_only) VALUES (now(), $all_count, $hosted_count)"

# Run ensembling
./run_ensembling.sh

# Backup database
./backup_narrative.sh

# Export website and deploy
uv run export_website.py
rsync -av website/ narrative@merah.cassia.ifost.org.au:/var/www/vhosts/narrative-learning.symmachus.org/htdocs/
