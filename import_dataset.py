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
    """Insert rows into ``table`` ignoring duplicates.

    The script can be run multiple times for the same investigation, so records
    may already exist.  Using ``ON CONFLICT DO NOTHING`` allows the import to be
    idempotent.
    """

    if not rows:
        return
    placeholders = ", ".join(["%s"] * len(columns))
    cols = ", ".join(columns)

    # Convert integer representations of booleans coming from SQLite (0/1)
    # into Python booleans so that psycopg2 can map them to PostgreSQL
    # boolean values correctly. Only known boolean columns are converted.
    bool_indices = [i for i, c in enumerate(columns) if c in {"holdout", "validation"}]
    if bool_indices:
        converted_rows = []
        for row in rows:
            row_list = list(row)
            for idx in bool_indices:
                row_list[idx] = bool(row_list[idx])
            converted_rows.append(tuple(row_list))
        rows = converted_rows

    # Remove any NUL characters from string fields to avoid psycopg2 errors
    sanitized_rows = []
    for row in rows:
        sanitized = []
        for value in row:
            if isinstance(value, str):
                sanitized.append(value.replace("\x00", ""))
            else:
                sanitized.append(value)
        sanitized_rows.append(tuple(sanitized))
    rows = sanitized_rows

    cur.executemany(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import SQLite dataset into Postgres")
    parser.add_argument(
        "--investigation-id",
        dest="investigation_id",
        type=int,
        required=True,
        help="Investigation ID",
    )
    parser.add_argument(
        "sqlite_db",
        nargs="*",
        help="Path to SQLite file(s). Defaults to the path from the investigations table",
    )
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
        # Insert rounds individually so PostgreSQL can generate new IDs and
        # we can remap them for the inferences table.  Each SQLite file starts
        # numbering rounds from 1 which means many files have duplicate IDs.
        # If we inserted them directly, they would conflict with the primary
        # key and be skipped.  Instead we omit ``round_id`` from the insert so
        # the database assigns a unique identifier and remember the mapping.

        cols, rows = fetch_all(conn_sq, "rounds")
        if not rows:
            id_map = {}
        else:
            round_id_idx = cols.index("round_id")
            insert_cols = [c for c in cols if c != "round_id"] + ["investigation_id"]
            id_map = {}
            for r in rows:
                old_id = r[round_id_idx]
                values = [v for i, v in enumerate(r) if i != round_id_idx]
                values.append(args.investigation_id)
                placeholders = ", ".join(["%s"] * len(values))
                cur_pg.execute(
                    f"INSERT INTO {dataset}_rounds ({', '.join(insert_cols)}) "
                    f"VALUES ({placeholders}) RETURNING round_id"
                , values)
                new_id = cur_pg.fetchone()[0]
                id_map[old_id] = new_id

        # Insert inferences using the newly generated round IDs
        cols, rows = fetch_all(conn_sq, "inferences")
        if rows:
            round_idx = cols.index("round_id")
            cols.append("investigation_id")
            new_rows = []
            for r in rows:
                r = list(r)
                r[round_idx] = id_map.get(r[round_idx])
                r.append(args.investigation_id)
                new_rows.append(tuple(r))
            insert_rows(cur_pg, f"{dataset}_inferences", cols, new_rows)
        conn_sq.close()

    cur_pg.close()
    conn_pg.close()


if __name__ == "__main__":
    main()
