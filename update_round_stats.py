#!/usr/bin/env python3
"""Update accuracy metrics and completion time for a round."""
import argparse

import datasetconfig
from modules.postgres import get_connection, get_investigation_settings
from modules.round_utils import update_round_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Update round statistics")
    parser.add_argument("round_id", type=int, help="Round ID to update")
    args = parser.parse_args()

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT investigation_id FROM round_investigations WHERE round_id = %s",
        (args.round_id,),
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"round {args.round_id} not found")
    if len(rows) > 1:
        raise SystemExit(f"round {args.round_id} maps to multiple investigations")
    investigation_id = rows[0][0]

    dataset, config_file = get_investigation_settings(conn, investigation_id)
    config_obj = datasetconfig.DatasetConfig(
        conn, config_file, dataset, investigation_id
    )
    update_round_statistics(config_obj, args.round_id)
    conn.close()


if __name__ == "__main__":
    main()

