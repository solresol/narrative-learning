#!/usr/bin/env python3
"""Run the training loop for a given investigation.

The script expects an investigation ID which references a row in the
``investigations`` table defined in ``postgres-schemas/investigations_schema.sql``.
All configuration settings are read from that row. Progress is written back by
updating the ``round_number`` column.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import json

import datasetconfig
from modules.round_utils import update_round_statistics

from modules.postgres import get_connection


def run_cmd(cmd: list[str], quiet: bool = False, inv_id: int | None = None) -> int:
    """Run command, optionally suppressing output."""
    if quiet:
        label = f"investigation {inv_id}" if inv_id is not None else "investigation"
        print(f"{label}: starting {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{label}: finished {' '.join(cmd)} (exit {proc.returncode})")
        return proc.returncode
    return subprocess.call(cmd)

def capture_cmd(cmd: list[str], quiet: bool = False, inv_id: int | None = None) -> tuple[int, str]:
    if quiet:
        label = f"investigation {inv_id}" if inv_id is not None else "investigation"
        print(f"{label}: starting {' '.join(cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        print(f"{label}: finished {' '.join(cmd)} (exit {proc.returncode})")
        return proc.returncode, proc.stdout.strip()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training loop")
    parser.add_argument(
        "investigation_id", type=int, help="ID from investigations table"
    )
    parser.add_argument(
        "--dsn",
        help="PostgreSQL connection string (defaults to libpq environment variables)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print when subprocesses start and finish",
    )
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("POSTGRES_DSN")

    # Use modules.postgres helper to fall back to the local narrative database if
    # no DSN is provided.
    conn = get_connection(dsn)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        """
        SELECT i.round_number,
               i.dataset,
               d.config_file,
               m.training_model,
               m.inference_model,
               m.example_count,
               m.patience,
               i.dump_file
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
        config,
        training_model,
        inference_model,
        example_count,
        patience,
        dump_path,
    ) = row

    with open(config) as f:
        config_json = json.load(f)
    rounds_table = config_json.get("rounds_table", f"{dataset}_rounds")
    splits_table = config_json.get("splits_table", f"{dataset}_splits")
    config_obj = datasetconfig.DatasetConfig(conn, config, dataset, args.investigation_id)


    # Ensure there is at least one round for this investigation and that
    # ``round_no`` references a valid round.
    cur.execute(
        f"SELECT round_id FROM {rounds_table} WHERE investigation_id=%s ORDER BY round_id LIMIT 1",
        (args.investigation_id,),
    )
    first_round = cur.fetchone()
    if first_round is None:
        cur.execute(f"SELECT MIN(split_id) FROM {splits_table}")
        default_split = cur.fetchone()[0] or 0
        cur.execute(f"SELECT COALESCE(max(round_id), 0) + 1 FROM {rounds_table}")
        next_id = cur.fetchone()[0]
        cur.execute(
            f"SELECT setval('{rounds_table}_round_id_seq', %s, true)",
            (next_id,),
        )
        cur.execute(
            f"INSERT INTO {rounds_table} (round_id, split_id, prompt, investigation_id) VALUES (%s, %s, 'Choose randomly', %s)",
            (next_id, default_split, args.investigation_id),
        )
        round_no = next_id
        cur.execute(
            "UPDATE investigations SET round_number=%s WHERE id=%s",
            (round_no, args.investigation_id),
        )
    elif round_no is None:
        round_no = first_round[0]
        cur.execute(
            "UPDATE investigations SET round_number=%s WHERE id=%s",
            (round_no, args.investigation_id),
        )

    while True:
        ret = run_cmd(
            [
                "uv",
                "run",
                "report-script.py",
                "--investigation-id",
                str(args.investigation_id),
                "--metric",
                "accuracy",
                "--validation",
                "--patience",
                str(patience if patience is not None else 3),
            ],
            args.quiet,
            args.investigation_id,
        )
        if ret != 0:
            break

        if (
            run_cmd(
                [
                    "uv",
                    "run",
                    "process_round.py",
                    "--round-id",
                    str(round_no),
                    "--loop",
                    "--progress-bar",
                    "--investigation-id",
                    str(args.investigation_id),
                ],
                args.quiet,
                args.investigation_id,
            )
            != 0
        ):
            sys.exit(1)

        update_round_statistics(config_obj, round_no)

        if (
            run_cmd(
                [
                    "uv",
                    "run",
                    "train.py",
                    "--round-id",
                    str(round_no),
                    "--investigation-id",
                    str(args.investigation_id),
                    "--model",
                    training_model,
                    *(["--example-count", str(example_count)] if example_count else []),
                    "--verbose",
                ],
                args.quiet,
                args.investigation_id,
            )
            != 0
        ):
            sys.exit(1)
        cur.execute(
            f"SELECT max(round_id) FROM {rounds_table} WHERE investigation_id=%s",
            (args.investigation_id,),
        )
        round_no = cur.fetchone()[0]

        cur.execute(
            f"SELECT round_uuid FROM {rounds_table} WHERE round_id=%s",
            (round_no,),
        )
        uuid_row = cur.fetchone()
        round_uuid = uuid_row[0] if uuid_row else None
        cur.execute(
            "UPDATE investigations SET round_number=%s, round_uuid=%s WHERE id=%s",
            (round_no, round_uuid, args.investigation_id),
        )

    ret, best = capture_cmd([
        "uv",
        "run",
        "report-script.py",
        "--investigation-id",
        str(args.investigation_id),
        "--best",
    ],
        args.quiet,
        args.investigation_id,
    )
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
