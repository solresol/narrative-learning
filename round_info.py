#!/usr/bin/env python3
"""Display details about a specific round."""
import argparse

from modules.postgres import get_connection, get_investigation_settings
from datasetconfig import DatasetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Show round information")
    parser.add_argument("--round-uuid", help="Round UUID")
    parser.add_argument("--investigation-id", type=int, help="Investigation ID")
    parser.add_argument("--round-id", type=int, help="Round ID within investigation")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="JSON config file with postgres_dsn")
    parser.add_argument(
        "--show-inferences",
        action="store_true",
        help="Display each inference for this round",
    )
    args = parser.parse_args()

    if args.round_uuid:
        if args.investigation_id or args.round_id:
            parser.error("Use either --round-uuid or both --investigation-id and --round-id")
    else:
        if args.investigation_id is None or args.round_id is None:
            parser.error("Specify --round-uuid or both --investigation-id and --round-id")

    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()

    if args.round_uuid:
        cur.execute(
            "SELECT investigation_id FROM round_investigations WHERE round_uuid = %s",
            (args.round_uuid,),
        )
        rows = cur.fetchall()
        if not rows:
            raise SystemExit(f"round {args.round_uuid} not found")
        if len(rows) > 1:
            raise SystemExit(f"round {args.round_uuid} maps to multiple investigations")
        investigation_id = rows[0][0]
        dataset, cfg_file = get_investigation_settings(conn, investigation_id)
        cfg = DatasetConfig(conn, cfg_file, dataset, investigation_id)
        cur.execute(
            f"SELECT round_id FROM {dataset}_rounds WHERE round_uuid = %s",
            (args.round_uuid,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"uuid {args.round_uuid} not found in {dataset}_rounds")
        round_id = row[0]
        round_uuid = args.round_uuid
    else:
        investigation_id = args.investigation_id
        dataset, cfg_file = get_investigation_settings(conn, investigation_id)
        cfg = DatasetConfig(conn, cfg_file, dataset, investigation_id)
        cur.execute(
            f"SELECT round_uuid FROM {dataset}_rounds WHERE round_id = %s AND investigation_id = %s",
            (args.round_id, investigation_id),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(
                f"round {args.round_id} not found for investigation {investigation_id}"
            )
        round_id = args.round_id
        round_uuid = row[0]

    rounds_table = cfg.rounds_table
    inf_table = f"{dataset}_inferences" if dataset else "inferences"

    cfg._execute(
        cur,
        f"""
        SELECT prompt, investigation_id, train_accuracy, validation_accuracy,
               test_accuracy, round_completed
          FROM {rounds_table}
         WHERE round_id = ? AND investigation_id = ?
        """,
        (round_id, investigation_id),
    )
    round_row = cur.fetchone()
    if not round_row:
        raise SystemExit("Round information not found")

    prompt, inv_id, train_acc, val_acc, test_acc, completed = round_row

    cfg._execute(
        cur,
        f"SELECT COUNT(*) FROM {inf_table} WHERE round_id = ? AND investigation_id = ?",
        (round_id, investigation_id),
    )
    inf_count = cur.fetchone()[0]

    print(f"Round ID: {round_id}")
    print(f"Round UUID: {round_uuid}")
    print(f"Investigation ID: {inv_id}")
    print(f"Prompt:\n{prompt}")
    print(f"Inferences: {inf_count}")
    print(f"Completed: {'yes' if completed else 'no'}")
    print(f"Train accuracy: {train_acc if train_acc is not None else 'n/a'}")
    print(f"Validation accuracy: {val_acc if val_acc is not None else 'n/a'}")
    print(f"Test accuracy: {test_acc if test_acc is not None else 'n/a'}")

    if args.show_inferences:
        split_id = cfg.get_split_id(round_id)
        cfg._execute(
            cur,
            f"""
            SELECT i.{cfg.primary_key}, m.{cfg.target_field}, i.prediction,
                   i.narrative_text, s.holdout, s.validation
              FROM {inf_table} i
              JOIN {cfg.table_name} m ON i.{cfg.primary_key} = m.{cfg.primary_key}
              JOIN {cfg.splits_table} s ON (
                      s.{cfg.primary_key} = i.{cfg.primary_key} AND s.split_id = ?)
             WHERE i.round_id = ? AND i.investigation_id = ?
             ORDER BY i.{cfg.primary_key}
            """,
            (split_id, round_id, investigation_id),
        )
        rows = cur.fetchall()
        print("\nInferences:")
        for pid, outcome, pred, narrative, holdout, validation in rows:
            data_split = (
                "test" if holdout and not validation else
                "validation" if holdout and validation else "train"
            )
            print(f"- {pid} [{data_split}] truth={outcome} prediction={pred}")
            if narrative:
                print(f"  text: {narrative}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
