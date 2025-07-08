#!/usr/bin/env python3
"""Export and delete a round from the database."""
from __future__ import annotations

import argparse
import json
from typing import Any

from modules.postgres import get_connection, get_investigation_settings
from datasetconfig import DatasetConfig


def fetch_all_dict(cur) -> list[dict[str, Any]]:
    """Return list of rows as dicts for a cursor."""
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete a round after exporting")
    parser.add_argument("round_uuid", help="Round UUID to remove")
    parser.add_argument("--output", help="File to write round data")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="PostgreSQL config JSON")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        "SELECT investigation_id FROM round_investigations WHERE round_uuid = %s",
        (args.round_uuid,),
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"round {args.round_uuid} not found")
    if len(rows) > 1:
        raise SystemExit(f"round {args.round_uuid} maps to multiple investigations")
    investigation_id = rows[0][0]

    dataset, cfg_path = get_investigation_settings(conn, investigation_id)
    cfg = DatasetConfig(conn, cfg_path, dataset, investigation_id)

    cur.execute(
        f"SELECT round_id FROM {cfg.rounds_table} WHERE round_uuid = %s",
        (args.round_uuid,),
    )
    round_row = cur.fetchone()
    if round_row is None:
        raise SystemExit(f"uuid {args.round_uuid} not found in {cfg.rounds_table}")
    round_id = round_row[0]

    cur.execute(
        f"SELECT * FROM {cfg.rounds_table} WHERE round_id = %s",
        (round_id,),
    )
    round_data = fetch_all_dict(cur)[0]

    inf_table = f"{dataset}_inferences" if dataset else "inferences"
    cur.execute(
        f"SELECT * FROM {inf_table} WHERE round_id = %s",
        (round_id,),
    )
    inferences = fetch_all_dict(cur)

    if args.output:
      with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"round": round_data, "inferences": inferences}, f, indent=2, default=str)

    cur.execute(f"DELETE FROM {inf_table} WHERE round_id = %s", (round_id,))
    cur.execute(f"DELETE FROM {cfg.rounds_table} WHERE round_id = %s", (round_id,))

    if args.output:
       print(f"Round {round_id} exported to {args.output} and deleted")
    else:
       print(f"Round {round_id} deleted")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
