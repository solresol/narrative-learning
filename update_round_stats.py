#!/usr/bin/env python3
"""Update accuracy metrics and completion time for a round."""
import argparse

import datasetconfig
from modules.postgres import get_connection, get_investigation_settings
from modules.round_utils import update_round_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Update round statistics")
    parser.add_argument("round_uuid", help="Round UUID to update")
    args = parser.parse_args()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT investigation_id FROM round_investigations WHERE round_uuid = %s",
        (args.round_uuid,),
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"round {args.round_uuid} not found")
    if len(rows) > 1:
        raise SystemExit(
            f"round {args.round_uuid} maps to multiple investigations"
        )
    investigation_id = rows[0][0]

    dataset, config_file = get_investigation_settings(conn, investigation_id)
    config_obj = datasetconfig.DatasetConfig(
        conn, config_file, dataset, investigation_id
    )
    cur.execute(
        f"SELECT round_id FROM {dataset}_rounds WHERE round_uuid = %s",
        (args.round_uuid,),
    )
    row = cur.fetchone()
    if row is None:
        raise SystemExit(
            f"uuid {args.round_uuid} not found in {dataset}_rounds"
        )
    round_id = row[0]
    update_round_statistics(config_obj, round_id)
    conn.close()


if __name__ == "__main__":
    main()

