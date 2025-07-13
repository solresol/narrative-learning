#!/usr/bin/env python3
"""Populate dataset provenance information in PostgreSQL."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from modules.postgres import get_connection


SYNTHETIC_DATASETS = {
    "espionage": "$(ESPIONAGE_DATASET)",
    "timetravel_insurance": "$(TIMETRAVEL_INSURANCE_DATASET)",
    "potions": "$(POTIONS_DATASET)",
}

OBFUSCATION_FILES = {
    "wisconsin": Path("obfuscations/breast_cancer"),
    "titanic": Path("obfuscations/titanic"),
    "southgermancredit": Path("obfuscations/southgermancredit"),
}


def parse_makefile(path: Path) -> dict[str, str]:
    """Return mapping of dataset to generation command from the Makefile."""
    info: dict[str, str] = {}
    text = path.read_text()
    for dataset, token in SYNTHETIC_DATASETS.items():
        for line in text.splitlines():
            if "random_classification_data_generator.py" in line and token in line:
                info[dataset] = line.strip()
                break
    return info


def read_obfuscation(path: Path) -> str:
    """Return obfuscation plan JSON from a SQLite file."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM obfuscation_metadata LIMIT 1")
    meta_row = cur.fetchone()
    meta_cols = [d[0] for d in cur.description]
    metadata = dict(zip(meta_cols, meta_row)) if meta_row else {}

    cur.execute(
        "SELECT original_column, remove, obfuscated_column, transformation FROM column_transformations ORDER BY id"
    )
    rows = cur.fetchall()
    columns = [
        {
            "original_column": r[0],
            "remove": bool(r[1]),
            "obfuscated_column": r[2],
            "transformation": r[3],
        }
        for r in rows
    ]
    conn.close()
    return json.dumps(
        {"metadata": metadata, "column_transformations": columns}, indent=2
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Store dataset provenance")
    parser.add_argument("--dsn")
    parser.add_argument("--config")
    parser.add_argument("--makefile", default="Makefile")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_provenance (
            dataset TEXT PRIMARY KEY REFERENCES datasets(dataset),
            provenance TEXT NOT NULL
        )
        """
    )

    synthetic = parse_makefile(Path(args.makefile))
    for dataset, command in synthetic.items():
        cur.execute(
            "INSERT INTO dataset_provenance(dataset, provenance) VALUES (%s, %s)"
            " ON CONFLICT (dataset) DO UPDATE SET provenance=EXCLUDED.provenance",
            (dataset, command),
        )

    for dataset, path in OBFUSCATION_FILES.items():
        if path.exists():
            info = read_obfuscation(path)
            cur.execute(
                "INSERT INTO dataset_provenance(dataset, provenance) VALUES (%s, %s)"
                " ON CONFLICT (dataset) DO UPDATE SET provenance=EXCLUDED.provenance",
                (dataset, info),
            )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
