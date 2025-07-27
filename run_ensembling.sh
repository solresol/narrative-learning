#!/bin/bash


uv run results_ensembling.py --progress titanic && \
uv run results_ensembling.py --progress wisconsin && \
uv run results_ensembling.py --progress southgermancredit && \
uv run results_ensembling.py --progress --no-decodex espionage && \
uv run results_ensembling.py --progress --no-decodex timetravel_insurance && \
uv run results_ensembling.py --progress --no-decodex potions && \
psql -d narrative -Atc \
  "SELECT DISTINCT models FROM ensemble_results WHERE best_yet" | \
while IFS= read -r models; do
    [ -z "$models" ] && continue
    echo "Running lexicostatistics for $models"
    uv run lexicostatistics.py "$models" || exit 1
done
