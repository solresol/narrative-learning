#!/bin/bash
set -euo pipefail

cd $(dirname $0)

# Update repository
git pull -q

# Record incomplete investigation counts
all_count=$(uv run find_incomplete_investigations.py | grep -c '^ ')
hosted_count=$(uv run find_incomplete_investigations.py --hosted-only | grep -c '^ ')
psql narrative -c "INSERT INTO incomplete_investigation_counts(recorded_at, total, hosted_only) VALUES (now(), $all_count, $hosted_count)"

# Run ensembling
./run_ensembling.sh

# Backup database
./backup_narrative.sh

# Export website and deploy
uv run export_website.py
rsync -av website/ narrative@merah.cassia.ifost.org.au:/var/www/vhosts/narrative-learning.symmachus.org/htdocs/
