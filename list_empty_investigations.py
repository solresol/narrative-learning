#!/usr/bin/env python3
"""List investigation IDs with no inference data."""
import argparse
import os
import sqlite3

from modules.postgres import get_connection


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
    parser = argparse.ArgumentParser(description="Find investigations without inferences")
    parser.add_argument("--dsn", help="PostgreSQL connection string")
    parser.add_argument("--config", help="Path to PostgreSQL config JSON")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()
    cur.execute("SELECT id, sqlite_database FROM investigations ORDER BY id")
    missing = []
    for inv_id, db_path in cur.fetchall():
        if not has_inferences(db_path):
            missing.append(inv_id)

    conn.close()

    if missing:
        print("Investigations with no inferences:")
        for inv_id in missing:
            print(inv_id)
    else:
        print("All investigations have inference data.")


if __name__ == "__main__":
    main()
