#!/usr/bin/env python3
"""Check and repair investigations.round_number values.

For each dataset, the corresponding ``*_rounds`` table holds a ``round_id`` for
every round of training. The ``investigations.round_number`` column should equal
the most recent ``round_id`` for that investigation.

This helper prints how many investigations are out of sync and outputs ``UPDATE``
statements to fix them.
"""
from __future__ import annotations

import json
from modules.postgres import get_connection


def main() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT dataset, config_file FROM datasets")
    dataset_cfg_files = dict(cur.fetchall())

    # Map dataset -> rounds_table from config
    round_tables: dict[str, str] = {}
    for dataset, cfg_path in dataset_cfg_files.items():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        round_tables[dataset] = cfg.get("rounds_table", f"{dataset}_rounds")

    for dataset, table in round_tables.items():
        cur.execute(
            f"""
            SELECT i.id, i.round_number, MAX(r.round_id) AS max_round
              FROM investigations i
              LEFT JOIN {table} r ON i.id = r.investigation_id
             WHERE i.dataset = %s
             GROUP BY i.id, i.round_number
            """,
            (dataset,),
        )
        rows = cur.fetchall()

        mismatches = [r for r in rows if (r[2] or None) != r[1]]
        print(f"-- {dataset}: {len(mismatches)} mismatches")

        for inv_id, round_no, max_round in mismatches:
            correct = max_round or None
            if correct == round_no:
                continue
            cur.execute(
                f"SELECT round_uuid FROM {table} WHERE round_id=%s",
                (correct,),
            )
            uuid_row = cur.fetchone()
            round_uuid = uuid_row[0] if uuid_row else None
            print(
                f"-- investigation {inv_id}: found max round {correct} (currently {round_no})"
            )
            print(
                f"UPDATE investigations SET round_number={correct}, round_uuid='{round_uuid}' WHERE id={inv_id};"
            )

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
