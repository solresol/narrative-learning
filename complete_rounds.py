#!/usr/bin/env python3
"""Mark finished rounds as completed across all datasets."""
import argparse
import json
from modules.postgres import get_connection
from datasetconfig import DatasetConfig
from modules.round_utils import update_round_statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update stats for all rounds with complete inferences"
    )
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="JSON config file with postgres_dsn")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()

    cur.execute("SELECT dataset, config_file FROM datasets")
    datasets = cur.fetchall()

    for dataset, cfg_path in datasets:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        rounds_table = cfg_data.get("rounds_table", f"{dataset}_rounds")
        cur.execute(
            f"SELECT round_id, investigation_id FROM {rounds_table} "
            "WHERE round_completed IS NULL"
        )
        for round_id, inv_id in cur.fetchall():
            cfg = DatasetConfig(conn, cfg_path, dataset, inv_id)
            update_round_statistics(cfg, round_id)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
