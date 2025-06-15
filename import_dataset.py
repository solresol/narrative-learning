#!/usr/bin/env python3
"""Import a dataset's SQLite results into PostgreSQL."""
import argparse
import os
import json
import sqlite3
from typing import List, Tuple

from modules.postgres import get_connection


def sqlite_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [r[0] for r in cur.fetchall()]

def fetch_all(conn: sqlite3.Connection, table: str) -> Tuple[List[str], List[Tuple]]:
    cur = conn.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return cols, rows




def insert_rows(cur, table: str, columns: List[str], rows: List[Tuple]) -> None:
    if not rows:
        return
    placeholders = ", ".join(["%s"] * len(columns))
    cols = ", ".join([f'"{c}"' for c in columns])
    cur.executemany(f'INSERT INTO {table} ({cols}) VALUES ({placeholders})', rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SQLite dataset into Postgres")
    parser.add_argument("investigation_id", type=int, help="Investigation ID")
    parser.add_argument("sqlite_db", nargs="*", help="Path to SQLite file(s). Defaults to the path from the investigations table")
    parser.add_argument("--dsn")
    parser.add_argument("--config")
    parser.add_argument("--schema", help="SQL file defining Postgres tables")
    args = parser.parse_args()

    conn_pg = get_connection(args.dsn, args.config)
    conn_pg.autocommit = True
    cur_pg = conn_pg.cursor()

    cur_pg.execute(
        """
        SELECT i.dataset, i.sqlite_database, d.config_file
        FROM investigations i
        JOIN datasets d ON i.dataset = d.dataset
        WHERE i.id = %s
        """,
        (args.investigation_id,),
    )
    row = cur_pg.fetchone()
    if row is None:
        raise SystemExit(f"investigation {args.investigation_id} not found")

    dataset, default_sqlite_db, config_file = row

    sqlite_paths = args.sqlite_db or [default_sqlite_db]
    schema_path = args.schema or f"postgres-schemas/{dataset}_schema.sql"

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    dataset_table = config_data["table_name"]
    dataset_splits = config_data.get("splits_table", dataset_table + "_splits")

    reference_rows = None
    reference_cols = None
    schema_loaded = False

    for path in sqlite_paths:
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found")
        conn_sq = sqlite3.connect(path)
        tables = sqlite_tables(conn_sq)
        if dataset_table not in tables:
            raise SystemExit(f"{dataset_table} not found in {path}")
        if not schema_loaded:
            with open(schema_path, "r", encoding="utf-8") as f:
                cur_pg.execute(f.read())
            schema_loaded = True
            cols, rows = fetch_all(conn_sq, dataset_table)
            reference_cols = cols
            reference_rows = rows
            insert_rows(cur_pg, dataset_table, cols, rows)
            if dataset_splits in tables:
                cols, rows = fetch_all(conn_sq, dataset_splits)
                insert_rows(cur_pg, dataset_splits, cols, rows)
        else:
            cols, rows = fetch_all(conn_sq, dataset_table)
            if reference_rows != rows or reference_cols != cols:
                raise SystemExit(f"Data mismatch in {dataset_table} for {path}")
        for table in ("rounds", "inferences"):
            cols, rows = fetch_all(conn_sq, table)
            cols.append("investigation_id")
            rows = [tuple(list(r) + [args.investigation_id]) for r in rows]
            dest = f"{dataset}_{table}"
            insert_rows(cur_pg, dest, cols, rows)
        conn_sq.close()

    cur_pg.close()
    conn_pg.close()


if __name__ == "__main__":
    main()

