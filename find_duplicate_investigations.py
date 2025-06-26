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
    parser.add_argument(
        "--delete-sql",
        action="store_true",
        help="Output SQL to delete duplicate investigations without data",
    )
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

    if not args.delete_sql:
        print(
            "dataset|model|sqlite_database|round_tracking_file|dump_file|round_number|ids|ids_with_data"
        )
    def sort_key(item: tuple[tuple, list[int]]) -> tuple:
        key = item[0]
        # Sort elements safely even if they have mixed types or None
        # by converting each part of the key to a string. None becomes
        # an empty string so that keys remain comparable.
        return tuple("" if part is None else str(part) for part in key)

    for key, ids in sorted(groups.items(), key=sort_key):
        if len(ids) <= 1:
            continue
        dataset, model, db, round_file, dump_file, round_no = key
        round_table = dataset_round_tables.get(dataset, f"{dataset}_rounds")
        ids_with_data: list[int] = []
        for inv_id in ids:
            cur.execute(
                f"SELECT count(*) FROM {round_table} WHERE investigation_id = %s and prompt != 'Choose randomly'",
                (inv_id,),
            )
            n = cur.fetchone()
            if n is not None and n[0] > 0:
                ids_with_data.append(inv_id)
        ids_str = ",".join(str(i) for i in ids)
        ids_with_data_str = ",".join(str(i) for i in ids_with_data)

        if args.delete_sql:
            duplicates_without_data = [i for i in ids if i not in ids_with_data]
            if ids_with_data:
                print(f"-- there is data in {ids_with_data}")
                to_delete = duplicates_without_data
            else:
                print(f"-- no investigation has data, so let's keep {duplicates_without_data[0]}")
                to_delete = duplicates_without_data[1:]
            if to_delete:
                for del_id in to_delete:
                    cur.execute(f"select round_id, prompt from {round_table} where investigation_id = %s", [del_id])
                    for r in cur:
                        print(f"-- round id = {r[0]} has the prompt '{r[1]}' so no loss if we kill it")
                        print(f"DELETE FROM {round_table} where round_id = {r[0]};")
                    print(f"DELETE FROM investigations WHERE id = {del_id};")
        else:
            print(
                f"{dataset}|{model}|{db}|{round_file}|{dump_file}|{round_no}|{ids_str}|{ids_with_data_str}"
            )
                

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
