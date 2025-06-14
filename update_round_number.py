#!/usr/bin/env python3
"""Update the round number for an investigation based on its tracking file."""
import argparse
import os

from modules.postgres import get_connection


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync round number from file")
    parser.add_argument("investigation_id", type=int, help="ID from investigations table")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="JSON config file with postgres_dsn")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("SELECT round_tracking_file FROM investigations WHERE id=%s", (args.investigation_id,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"investigation {args.investigation_id} not found")

    filename = row[0]
    round_no = None
    if filename and os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.isdigit():
                round_no = int(content)

    if round_no is None:
        print("No round number found; nothing updated")
    else:
        cur.execute(
            "UPDATE investigations SET round_number=%s WHERE id=%s",
            (round_no, args.investigation_id),
        )
        print(f"Updated investigation {args.investigation_id} to round {round_no}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
