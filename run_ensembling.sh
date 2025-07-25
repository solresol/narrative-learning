#!/bin/bash

uv run results_ensembling.py --progress titanic && uv run results_ensembling.py --progress wisconsin && uv run results_ensembling.py --progress southgermancredit && uv run results_ensembling.py --progress -no-decodex espionage && uv run results_ensembling.py --progress --no-decodex timetravel_insurance && uv run results_ensembling.py --progress --no-decodex potions
