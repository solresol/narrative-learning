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
    parser.add_argument(
        "round_uuid",
        nargs="*",
        help="Round UUID(s) to remove",
    )
    parser.add_argument("--investigation-id", type=int, help="Investigation ID")
    parser.add_argument(
        "--round-id",
        type=int,
        nargs="+",
        help="Round ID(s) within investigation",
    )
    parser.add_argument("--output", help="File to write round data")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="PostgreSQL config JSON")
    args = parser.parse_args()

    if args.round_uuid:
        if args.investigation_id or args.round_id:
            parser.error(
                "Use round UUID(s) or both --investigation-id and --round-id"
            )
    elif args.round_id:
        if args.investigation_id is None:
            parser.error("--round-id requires --investigation-id")
    else:
        parser.error(
            "Specify round UUID(s) or both --investigation-id and --round-id"
        )

    conn = get_connection(args.dsn, args.config)
    conn.autocommit = True
    cur = conn.cursor()

    export_data = []

    if args.round_uuid:
        for uuid in args.round_uuid:
            cur.execute(
                "SELECT investigation_id FROM round_investigations WHERE round_uuid = %s",
                (uuid,),
            )
            rows = cur.fetchall()
            if not rows:
                raise SystemExit(f"round {uuid} not found")
            if len(rows) > 1:
                raise SystemExit(f"round {uuid} maps to multiple investigations")
            investigation_id = rows[0][0]
            dataset, cfg_path = get_investigation_settings(conn, investigation_id)
            cfg = DatasetConfig(conn, cfg_path, dataset, investigation_id)
            cur.execute(
                f"SELECT round_id FROM {cfg.rounds_table} WHERE round_uuid = %s",
                (uuid,),
            )
            row = cur.fetchone()
            if row is None:
                raise SystemExit(f"uuid {uuid} not found in {cfg.rounds_table}")
            round_id = row[0]

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

            export_data.append({"round": round_data, "inferences": inferences})

            cur.execute(f"DELETE FROM {inf_table} WHERE round_id = %s", (round_id,))
            cur.execute(f"DELETE FROM {cfg.rounds_table} WHERE round_id = %s", (round_id,))

            print(f"Round {round_id} deleted")
    else:
        dataset, cfg_path = get_investigation_settings(conn, args.investigation_id)
        cfg = DatasetConfig(conn, cfg_path, dataset, args.investigation_id)
        for rid in args.round_id:
            cur.execute(
                f"SELECT round_uuid FROM {cfg.rounds_table} WHERE round_id = %s AND investigation_id = %s",
                (rid, args.investigation_id),
            )
            row = cur.fetchone()
            if row is None:
                raise SystemExit(
                    f"round {rid} not found for investigation {args.investigation_id}"
                )

            cur.execute(
                f"SELECT * FROM {cfg.rounds_table} WHERE round_id = %s",
                (rid,),
            )
            round_data = fetch_all_dict(cur)[0]

            inf_table = f"{dataset}_inferences" if dataset else "inferences"
            cur.execute(
                f"SELECT * FROM {inf_table} WHERE round_id = %s",
                (rid,),
            )
            inferences = fetch_all_dict(cur)

            export_data.append({"round": round_data, "inferences": inferences})

            cur.execute(f"DELETE FROM {inf_table} WHERE round_id = %s", (rid,))
            cur.execute(f"DELETE FROM {cfg.rounds_table} WHERE round_id = %s", (rid,))

            print(f"Round {rid} deleted")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export_data[0] if len(export_data) == 1 else export_data, f, indent=2, default=str)
        print(f"Exported {len(export_data)} round{'s' if len(export_data) != 1 else ''} to {args.output}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
