#!/usr/bin/env python3
"""List investigations that appear to have missing data.

The script verifies that all processed rounds have inference results and
that early stopping conditions were met.  Early stopping is evaluated
before flagging missing inference data so investigations that legitimately
halted early are not reported as incomplete.
"""
import datasetconfig
from modules.postgres import get_connection


def gather_missing(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.id, i.dataset, d.config_file, m.patience
          FROM investigations i
          JOIN datasets d ON i.dataset = d.dataset
          JOIN models m ON i.model = m.model
         ORDER BY i.dataset, i.id
        """
    )
    rows = cur.fetchall()
    results: dict[str, list[tuple[int, str]]] = {}
    for inv_id, dataset, cfg_path, patience in rows:
        cfg = datasetconfig.DatasetConfig(conn, cfg_path, dataset, inv_id)
        try:
            split_id = cfg.get_latest_split_id()
        except SystemExit:
            results.setdefault(dataset, []).append((inv_id, "no rounds"))
            continue
        rounds = cfg.get_rounds_for_split(split_id)
        processed = cfg.get_processed_rounds_for_split(split_id)

        # Check early stopping before complaining about missing inference data.
        # If training should halt, any later rounds without inferences are
        # expected and not an error.
        if not cfg.check_early_stopping(split_id, "accuracy", patience):
            if len(processed) < len(rounds):
                results.setdefault(dataset, []).append((inv_id, "missing inferences"))
                continue
            results.setdefault(dataset, []).append((inv_id, "should continue"))
    return results


def main() -> None:
    conn = get_connection()
    missing = gather_missing(conn)
    conn.close()

    if not missing:
        print("All investigations appear complete.")
        return
    for dataset in sorted(missing):
        print(dataset)
        for inv_id, reason in missing[dataset]:
            print(f"  {inv_id}\t{reason}")


if __name__ == "__main__":
    main()
