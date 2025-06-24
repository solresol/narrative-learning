#!/usr/bin/env python3
"""List investigation IDs with no inference data.

Optional flags allow skipping ollama-powered models or outputting SQL to
create a temporary view for use with other scripts.
"""

import argparse
import os
import sqlite3
import sys
import fnmatch

from typing import Iterable

from modules.postgres import get_connection

# Training model names that rely on ollama for inference.
OLLAMA_TRAINING_MODELS: list[str] = [
    "phi4:latest",
    "llama3.3:latest",
    "falcon3:1b",
    "falcon3:10b",
    "gemma2:27b",
    "gemma2:2b",
    "phi4-mini",
    "deepseek-r1:70b",
    "qwq:32b",
    "gemma3:27b",
    "cogito:70b",
]


def has_inferences(db_path: str) -> bool:
    """Return True if the SQLite file has any inferences."""
    if not db_path or not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.execute("SELECT COUNT(*) FROM inferences")
        count = cur.fetchone()[0]
        conn.close()
        return count > 0
    except sqlite3.Error:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find investigations without inferences"
    )
    parser.add_argument("--dsn", help="PostgreSQL connection string")
    parser.add_argument("--config", help="Path to PostgreSQL config JSON")
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Exclude investigations that use ollama-backed models",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Only include investigations matching this dataset glob pattern",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Only include investigations matching this model glob pattern",
    )
    parser.add_argument(
        "--print-view",
        action="store_true",
        help="Output SQL for a temporary view instead of plain IDs",
    )
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.id, i.sqlite_database, m.training_model, i.dataset, i.model
          FROM investigations i
          JOIN models m ON i.model = m.model
         ORDER BY i.id
        """
    )
    missing: list[tuple[int, str]] = []
    for inv_id, db_path, training_model, dataset, model in cur.fetchall():
        if args.dataset and not any(fnmatch.fnmatch(dataset, pat) for pat in args.dataset):
            continue
        if args.model and not any(fnmatch.fnmatch(model, pat) for pat in args.model):
            continue
        if args.skip_ollama and training_model in OLLAMA_TRAINING_MODELS:
            continue
        if not has_inferences(db_path):
            missing.append((inv_id, training_model))

    conn.close()

    ids: Iterable[int] = [inv_id for inv_id, _ in missing]

    if args.print_view:
        if ids:
            print("CREATE TEMP VIEW empty_investigations(id) AS VALUES")
            print(",\n".join(f"    ({i})" for i in ids) + ";")
        else:
            print("CREATE TEMP VIEW empty_investigations(id) AS VALUES (NULL) WHERE FALSE;")
        return

    if ids:
        for inv_id in ids:
            print(inv_id)
    else:
        sys.stderr.write("All investigations have inference data.\n")


if __name__ == "__main__":
    main()
