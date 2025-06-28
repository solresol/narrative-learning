#!/usr/bin/env python3
"""Populate investigations.round_uuid based on round_number."""
from __future__ import annotations

import argparse

from modules.postgres import get_connection


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync round_uuid from rounds tables")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="PostgreSQL config JSON")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        "SELECT id, dataset, round_number FROM investigations WHERE round_number IS NOT NULL"
    )
    for inv_id, dataset, round_no in cur.fetchall():
        rounds_table = f"{dataset}_rounds"
        cur.execute(
            f"SELECT round_uuid FROM {rounds_table} WHERE round_id = %s",
            (round_no,),
        )
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE investigations SET round_uuid=%s WHERE id=%s",
                (row[0], inv_id),
            )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
