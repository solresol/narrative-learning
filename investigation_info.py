#!/usr/bin/env python3
"""Display details about an investigation.

This is useful to check investigation settings and progress while the
system is migrating between SQLite and PostgreSQL.
"""
from __future__ import annotations
import argparse
import os
import sqlite3

from modules.postgres import get_connection
from datasetconfig import DatasetConfig
from modules.exceptions import NoProcessedRoundsException, NonexistentRoundException


def main() -> None:
    parser = argparse.ArgumentParser(description="Show investigation details")
    parser.add_argument("investigation_id", type=int, help="ID from investigations table")
    parser.add_argument("--dsn", help="PostgreSQL connection string")
    parser.add_argument("--config", help="Path to PostgreSQL config JSON")
    args = parser.parse_args()

    # Connect to PostgreSQL and fetch investigation settings
    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.round_number,
               i.sqlite_database,
               i.round_tracking_file,
               d.dataset,
               d.config_file,
               m.model,
               m.training_model,
               m.inference_model,
               m.example_count,
               m.patience
          FROM investigations i
          JOIN datasets d ON i.dataset = d.dataset
          JOIN models m ON i.model = m.model
         WHERE i.id = %s
        """,
        (args.investigation_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"investigation {args.investigation_id} not found")

    (
        round_no,
        sqlite_db,
        round_file,
        dataset,
        config_file,
        model,
        training_model,
        inference_model,
        example_count,
        patience,
    ) = row

    print(f"Investigation {args.investigation_id}")
    print(f"  Dataset: {dataset}")
    print(f"  Config: {config_file}")
    print(f"  Model: {model}")
    print(f"  Training model: {training_model}")
    print(f"  Inference model: {inference_model}")
    print(f"  Example count: {example_count}")
    print(f"  Patience: {patience}")
    print(f"  Current round: {round_no}")
    print(f"  SQLite DB: {sqlite_db}")
    print(f"  Round tracking file: {round_file}")

    if sqlite_db and os.path.exists(sqlite_db):
        try:
            db_conn = sqlite3.connect(f"file:{sqlite_db}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            print(f"  Could not open SQLite database: {exc}")
        else:
            cfg = DatasetConfig(db_conn, config_file, dataset)
            try:
                split_id = cfg.get_latest_split_id()
                rounds = cfg.get_rounds_for_split(split_id)
                processed = cfg.get_processed_rounds_for_split(split_id)
                print(f"  Total rounds in DB: {len(rounds)}")
                print(f"  Rounds with data: {len(processed)}")
                try:
                    best = cfg.get_best_round_id(split_id, "accuracy")
                    print(f"  Best round: {best}")
                except NoProcessedRoundsException:
                    print("  Best round: none (no processed rounds)")
                except Exception as exc:
                    print(f"  Best round: error ({exc})")
            except SystemExit:
                print("  No rounds found in SQLite database")
            except Exception as exc:
                print(f"  Error reading SQLite data: {exc}")
            finally:
                db_conn.close()
    else:
        print("  SQLite database not found")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
