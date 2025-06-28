#!/usr/bin/env python3
"""Update the round number for an investigation using database records."""
import argparse

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

    cur.execute(
        "SELECT dataset FROM investigations WHERE id=%s",
        (args.investigation_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"investigation {args.investigation_id} not found")

    dataset = row[0]
    rounds_table = f"{dataset}_rounds"
    cur.execute(
        f"SELECT MAX(round_id) FROM {rounds_table} WHERE investigation_id=%s",
        (args.investigation_id,),
    )
    round_row = cur.fetchone()
    round_no = round_row[0] if round_row else None

    if round_no is None:
        print("No round number found; nothing updated")
    else:
        cur.execute(
            f"SELECT round_uuid FROM {rounds_table} WHERE round_id=%s AND investigation_id=%s",
            (round_no, args.investigation_id),
        )
        uuid_row = cur.fetchone()
        round_uuid = uuid_row[0] if uuid_row else None
        cur.execute(
            "UPDATE investigations SET round_number=%s, round_uuid=%s WHERE id=%s",
            (round_no, round_uuid, args.investigation_id),
        )
        print(f"Updated investigation {args.investigation_id} to round {round_no}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
