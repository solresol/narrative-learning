#!/usr/bin/env python3
"""Compare data consistency across SQLite databases for a dataset.

Usage:
    python compare_dataset_tables.py CONFIG DB1 DB2 [DB3 ...]

The CONFIG should be one of the JSON files in the ``configs/`` directory.
The script compares the rows in the main table and the splits table defined
in that configuration. All provided database files are compared against the
first database listed.
"""
import argparse
import json
import sqlite3
from typing import List, Tuple


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur]


def get_ordered_rows(conn: sqlite3.Connection, table: str) -> List[Tuple]:
    cols = get_columns(conn, table)
    col_list = ", ".join(f'"{c}"' for c in cols)
    order_by = ", ".join(f'"{c}"' for c in cols)
    cur = conn.execute(f"SELECT {col_list} FROM {table} ORDER BY {order_by}")
    return [tuple(row) for row in cur]


def compare_tables(config_path: str, db_paths: List[str]) -> None:
    with open(config_path, "r") as f:
        cfg = json.load(f)
    table = cfg["table_name"]
    splits_table = cfg["splits_table"]

    base_conn = sqlite3.connect(db_paths[0])
    base_agents = get_ordered_rows(base_conn, table)
    base_splits = get_ordered_rows(base_conn, splits_table)
    base_conn.close()

    for path in db_paths[1:]:
        conn = sqlite3.connect(path)
        agents = get_ordered_rows(conn, table)
        splits = get_ordered_rows(conn, splits_table)
        conn.close()

        mismatches = []
        if agents != base_agents:
            mismatches.append(table)
        if splits != base_splits:
            mismatches.append(splits_table)

        if mismatches:
            print(f"{path}: differences in {', '.join(mismatches)}")
        else:
            print(f"{path}: matches")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dataset tables across SQLite files")
    parser.add_argument("config", help="Path to dataset configuration JSON")
    parser.add_argument("databases", nargs='+', help="SQLite database files to compare")
    args = parser.parse_args()

    if len(args.databases) < 2:
        parser.error("Provide at least two database files to compare")

    compare_tables(args.config, args.databases)


if __name__ == "__main__":
    main()
