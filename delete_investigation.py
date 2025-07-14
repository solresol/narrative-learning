#!/usr/bin/env python3
"""Delete an investigation and all related data from PostgreSQL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modules.postgres import get_connection, get_investigation_settings
from datasetconfig import DatasetConfig


def fetch_all_dict(cur) -> list[dict[str, Any]]:
    """Return list of rows as dicts for a cursor."""
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def export_and_delete(conn, investigation_id: int, output_file: Path | None = None) -> None:
    """Export (optional) and delete an investigation and its data."""
    cur = conn.cursor()

    dataset, cfg_path = get_investigation_settings(conn, investigation_id)
    cfg = DatasetConfig(conn, cfg_path, dataset, investigation_id)

    cur.execute("SELECT * FROM investigations WHERE id = %s", (investigation_id,))
    inv_data = fetch_all_dict(cur)[0]

    cur.execute(
        f"SELECT * FROM {cfg.rounds_table} WHERE investigation_id = %s ORDER BY round_id",
        (investigation_id,),
    )
    rounds = fetch_all_dict(cur)

    inf_table = f"{dataset}_inferences" if dataset else "inferences"
    cur.execute(
        f"SELECT * FROM {inf_table} WHERE investigation_id = %s ORDER BY round_id",
        (investigation_id,),
    )
    inferences = fetch_all_dict(cur)

    if output_file:
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "investigation": inv_data,
                    "rounds": rounds,
                    "inferences": inferences,
                },
                f,
                indent=2,
                default=str,
            )

    cur.execute(f"DELETE FROM {inf_table} WHERE investigation_id = %s", (investigation_id,))
    cur.execute(f"DELETE FROM {cfg.rounds_table} WHERE investigation_id = %s", (investigation_id,))
    cur.execute("DELETE FROM investigations WHERE id = %s", (investigation_id,))

    if output_file:
        print(f"Investigation {investigation_id} exported to {output_file} and deleted")
    else:
        print(f"Investigation {investigation_id} deleted")

    cur.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete an investigation and its associated rounds and inferences"
    )
    parser.add_argument("investigation_id", type=int, help="ID from investigations table")
    parser.add_argument("--output", help="File to write investigation data before deletion")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="PostgreSQL config JSON")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    conn.autocommit = True

    out_file = Path(args.output) if args.output else None
    export_and_delete(conn, args.investigation_id, out_file)

    conn.close()


if __name__ == "__main__":
    main()
