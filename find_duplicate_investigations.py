#!/usr/bin/env python3
"""List duplicate investigations and check for round data."""
import argparse
import json
from collections import defaultdict
from modules.postgres import get_connection


def main() -> None:
    parser = argparse.ArgumentParser(description="Find duplicate investigations")
    parser.add_argument("--dsn", help="PostgreSQL connection string")
    parser.add_argument("--config", help="Path to PostgreSQL config JSON")
    args = parser.parse_args()

    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, dataset, model, sqlite_database, round_tracking_file, dump_file, round_number
        FROM investigations
        ORDER BY id
        """
    )
    rows = cur.fetchall()

    groups: defaultdict[tuple, list[int]] = defaultdict(list)
    for row in rows:
        inv_id = row[0]
        key = tuple(row[1:])
        groups[key].append(inv_id)

    # Map dataset -> rounds table from config
    cur.execute("SELECT dataset, config_file FROM datasets")
    dataset_cfg_files = dict(cur.fetchall())

    dataset_round_tables: dict[str, str] = {}
    for dataset, cfg_path in dataset_cfg_files.items():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dataset_round_tables[dataset] = cfg.get("rounds_table", f"{dataset}_rounds")

    print(
        "dataset|model|sqlite_database|round_tracking_file|dump_file|round_number|ids|ids_with_data"
    )
    for key, ids in sorted(groups.items(), key=lambda kv: kv[0]):
        if len(ids) <= 1:
            continue
        dataset, model, db, round_file, dump_file, round_no = key
        round_table = dataset_round_tables.get(dataset, f"{dataset}_rounds")
        ids_with_data: list[int] = []
        for inv_id in ids:
            cur.execute(
                f"SELECT 1 FROM {round_table} WHERE investigation_id = %s LIMIT 1",
                (inv_id,),
            )
            if cur.fetchone() is not None:
                ids_with_data.append(inv_id)
        ids_str = ",".join(str(i) for i in ids)
        ids_with_data_str = ",".join(str(i) for i in ids_with_data)
        print(
            f"{dataset}|{model}|{db}|{round_file}|{dump_file}|{round_no}|{ids_str}|{ids_with_data_str}"
        )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
