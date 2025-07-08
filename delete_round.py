#!/usr/bin/env python3
"""Export and delete one or more rounds from the database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from modules.postgres import get_connection, get_investigation_settings
from datasetconfig import DatasetConfig


def fetch_all_dict(cur) -> list[dict[str, Any]]:
    """Return list of rows as dicts for a cursor."""
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def export_and_delete(
    conn,
    investigation_id: int,
    round_id: int | None = None,
    round_uuid: str | None = None,
    output_file: Path | None = None,
) -> None:
    """Export (optional) and delete a single round."""
    if round_id is None and round_uuid is None:
        raise ValueError("round_id or round_uuid required")

    cur = conn.cursor()

    dataset, cfg_path = get_investigation_settings(conn, investigation_id)
    cfg = DatasetConfig(conn, cfg_path, dataset, investigation_id)

    if round_id is None:
        cur.execute(
            f"SELECT round_id FROM {cfg.rounds_table} WHERE round_uuid = %s",
            (round_uuid,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(
                f"uuid {round_uuid} not found in {cfg.rounds_table}"
            )
        round_id = row[0]
    else:
        cur.execute(
            f"SELECT round_uuid FROM {cfg.rounds_table} "
            "WHERE round_id = %s AND investigation_id = %s",
            (round_id, investigation_id),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(
                f"round {round_id} not found for investigation {investigation_id}"
            )
        round_uuid = row[0]

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

    if output_file:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump({"round": round_data, "inferences": inferences}, f, indent=2, default=str)

    cur.execute(f"DELETE FROM {inf_table} WHERE round_id = %s", (round_id,))
    cur.execute(f"DELETE FROM {cfg.rounds_table} WHERE round_id = %s", (round_id,))

    if output_file:
        print(f"Round {round_id} exported to {output_file} and deleted")
    else:
        print(f"Round {round_id} deleted")

    cur.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete rounds after optionally exporting them")
    parser.add_argument("round_uuid", nargs="*", help="Round UUID(s) to remove")
    parser.add_argument("--investigation-id", type=int, help="Investigation ID when using --round-id")
    parser.add_argument("--round-id", nargs="+", type=int, help="Round ID(s) within the investigation")
    parser.add_argument("--output", help="File to write round data (single round only)")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="PostgreSQL config JSON")
    args = parser.parse_args()

    if args.round_uuid:
        if args.investigation_id or args.round_id:
            parser.error("Use either round UUIDs or --investigation-id with --round-id")
        identifiers: Iterable[tuple[int, int | None, str | None]] = []
        conn = get_connection(args.dsn, args.config)
        conn.autocommit = True
        cur = conn.cursor()
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
            inv_id = rows[0][0]
            identifiers.append((inv_id, None, uuid))
        cur.close()
    else:
        if args.round_id is None or args.investigation_id is None:
            parser.error("Specify --round-id and --investigation-id")
        identifiers = [
            (args.investigation_id, rid, None) for rid in args.round_id
        ]
        conn = get_connection(args.dsn, args.config)
        conn.autocommit = True

    if args.output and len(identifiers) > 1:
        parser.error("--output can only be used with a single round")

    for inv_id, r_id, r_uuid in identifiers:
        out_file = Path(args.output) if args.output else None
        export_and_delete(conn, inv_id, r_id, r_uuid, out_file)

    conn.close()


if __name__ == "__main__":
    main()
