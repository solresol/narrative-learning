#!/usr/bin/env python3
"""Update accuracy metrics and completion time for a round."""
import argparse
import os

import datasetconfig
from modules.postgres import get_connection, get_investigation_settings
from modules.round_utils import update_round_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="Update round statistics")
    parser.add_argument("round_id", type=int, help="Round ID to update")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--config", help="Dataset config file")
    parser.add_argument("--investigation-id", type=int, help="Investigation ID")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--pg-config", help="JSON file containing postgres_dsn")
    args = parser.parse_args()

    if args.investigation_id is not None:
        conn = get_connection(args.dsn, args.pg_config)
        ds, cfg = get_investigation_settings(conn, args.investigation_id)
        dataset = args.dataset or ds
        config_file = args.config or cfg
    else:
        if not args.dataset or not args.config:
            parser.error("--dataset and --config are required without --investigation-id")
        conn = get_connection(args.dsn, args.pg_config)
        dataset = args.dataset
        config_file = args.config

    config_obj = datasetconfig.DatasetConfig(conn, config_file, dataset, args.investigation_id)
    update_round_statistics(config_obj, args.round_id)
    conn.close()


if __name__ == "__main__":
    main()

