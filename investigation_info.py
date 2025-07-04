#!/usr/bin/env python3
"""Display details about an investigation.

This is useful to check investigation settings and progress while the
system is migrating between SQLite and PostgreSQL.
"""
from __future__ import annotations
import argparse

from modules.postgres import get_connection
from datasetconfig import DatasetConfig
from modules.exceptions import NoProcessedRoundsException


def main() -> None:
    parser = argparse.ArgumentParser(description="Show investigation details")
    parser.add_argument("investigation_id", type=int, help="ID from investigations table")
    parser.add_argument("--dsn", help="PostgreSQL connection string")
    parser.add_argument("--config", help="Path to PostgreSQL config JSON")
    parser.add_argument(
        "--round-details",
        action="store_true",
        help="Show information about all rounds for this investigation",
    )
    args = parser.parse_args()

    # Connect to PostgreSQL and fetch investigation settings
    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.round_number,
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

    cfg = DatasetConfig(conn, config_file, dataset, args.investigation_id)
    best_round = None
    try:
        split_id = cfg.get_latest_split_id()
        rounds = cfg.get_rounds_for_split(split_id)
        processed = cfg.get_processed_rounds_for_split(split_id)
        print(f"  Total rounds in DB: {len(rounds)}")
        print(f"  Rounds with data: {len(processed)}")

        if processed:
            try:
                best_round = cfg.get_best_round_id(split_id, "accuracy")
                print(f"  Best round: {best_round}")

                val_df = cfg.generate_metrics_data(split_id, "accuracy", "validation")
                val_acc = val_df[val_df.round_id == best_round].metric.iloc[0]
                print(f"  Best validation accuracy: {val_acc:.3f}")

                try:
                    test_acc = cfg.get_test_metric_for_best_validation_round(split_id, "accuracy")
                    print(f"  Best test accuracy: {test_acc:.3f}")
                except Exception as exc:
                    print(f"  Best test accuracy: error ({exc})")

                if cfg.check_early_stopping(split_id, "accuracy", patience):
                    print(f"  Final validation accuracy: {val_acc:.3f}")
                    print(f"  Final test accuracy: {test_acc:.3f}")
            except NoProcessedRoundsException:
                print("  Best round: none (no processed rounds)")
        else:
            print("  Best round: none (no processed rounds)")

        if args.round_details:
            cur = conn.cursor()
            table = cfg.rounds_table
            cfg._execute(
                cur,
                f"SELECT round_id, round_start, round_completed, validation_accuracy, test_accuracy "
                f"FROM {table} WHERE investigation_id = ? ORDER BY round_start",
                (args.investigation_id,),
            )
            rows = cur.fetchall()
            prev_start = None
            from datetime import timedelta

            for r_id, r_start, r_completed, v_acc, t_acc in rows:
                tags = []
                if best_round is not None and r_id == best_round:
                    tags.append("best")
                if prev_start and r_start and r_start - prev_start > timedelta(days=31):
                    tags.append(">1 month gap")
                tag_str = f" ({', '.join(tags)})" if tags else ""
                c_str = r_completed.strftime('%Y-%m-%d') if r_completed else 'in progress'
                print(
                    f"  Round {r_id}: {r_start:%Y-%m-%d} -> {c_str}, val={v_acc if v_acc is not None else 'n/a'}"
                    f", test={t_acc if t_acc is not None else 'n/a'}{tag_str}"
                )
                prev_start = r_start
    except SystemExit:
        print("  No rounds found in database")
    except Exception as exc:
        print(f"  Error reading data: {exc}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
