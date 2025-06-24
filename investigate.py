#!/usr/bin/env python3
"""Run the training loop for a given investigation.

The script expects an investigation ID which references a row in the
``investigations`` table defined in ``postgres-schemas/investigations_schema.sql``.
All required
environment variables are read from that row. Progress is written back by
updating the ``round_number`` column.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import json

from modules.postgres import get_connection


def run_cmd(cmd: list[str]) -> int:
    """Run command and stream output."""
    return subprocess.call(cmd)


def capture_cmd(cmd: list[str]) -> tuple[int, str]:
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
               i.sqlite_database,
               m.training_model,
               m.inference_model,
               m.example_count,
               m.patience,
               i.round_tracking_file,
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
        sqlite_db,
        training_model,
        inference_model,
        example_count,
        patience,
        round_file,
        dump_path,
    ) = row

    with open(config) as f:
        config_json = json.load(f)
    rounds_table = config_json.get("rounds_table", f"{dataset}_rounds")

    env = {
        "NARRATIVE_LEARNING_CONFIG": config,
        "NARRATIVE_LEARNING_DATABASE": sqlite_db,
        "NARRATIVE_LEARNING_TRAINING_MODEL": training_model,
        "NARRATIVE_LEARNING_INFERENCE_MODEL": inference_model,
        "ROUND_TRACKING_FILE": round_file,
    }
    if example_count:
        env["NARRATIVE_LEARNING_EXAMPLE_COUNT"] = str(example_count)
    if patience:
        env["NARRATIVE_LEARNING_PATIENCE"] = str(patience)
    if dump_path:
        env["NARRATIVE_LEARNING_DUMP"] = dump_path

    os.environ.update(env)

    # Ensure there is at least one round for this investigation.
    cur.execute(
        f"SELECT 1 FROM {rounds_table} WHERE investigation_id=%s LIMIT 1",
        (args.investigation_id,),
    )
    if cur.fetchone() is None:
        cur.execute(f"SELECT COALESCE(max(round_id), 0) + 1 FROM {rounds_table}")
        next_id = cur.fetchone()[0]
        cur.execute(
            f"SELECT setval('{rounds_table}_round_id_seq', %s, true)",
            (next_id,),
        )
        cur.execute(
            f"INSERT INTO {rounds_table} (round_id, split_id, prompt, investigation_id) VALUES (%s, 0, 'Choose randomly', %s)",
            (next_id, args.investigation_id),
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
                os.environ.get("NARRATIVE_LEARNING_PATIENCE", "3"),
            ]
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
                ]
            )
            != 0
        ):
            sys.exit(1)

        with tempfile.NamedTemporaryFile() as tmp:
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
                        "--round-tracking-file",
                        tmp.name,
                        "--verbose",
                    ]
                )
                != 0
            ):
                sys.exit(1)
            tmp.seek(0)
            content = tmp.read().decode().strip()
            round_no = int(content) if content else round_no + 1

        cur.execute(
            "UPDATE investigations SET round_number=%s WHERE id=%s",
            (round_no, args.investigation_id),
        )

    ret, best = capture_cmd([
        "uv",
        "run",
        "report-script.py",
        "--investigation-id",
        str(args.investigation_id),
        "--best",
    ])
    if ret == 0 and best:
        outfile = sqlite_db[: -len(".sqlite")] + ".best-round.txt"
        with open(outfile, "w") as f:
            f.write(best + "\n")

    if dump_path:
        with open(dump_path, "w") as f:
            subprocess.run(["sqlite3", sqlite_db, ".dump"], stdout=f)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
