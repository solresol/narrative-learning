#!/usr/bin/env python3
"""Display progress for investigations stored in PostgreSQL."""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

SYDNEY_TZ = ZoneInfo("Australia/Sydney")
from typing import Any, Dict, Iterable, Tuple

from datasetconfig import DatasetConfig
from modules.postgres import get_connection


def check_investigation_status(
    conn: Any,
    investigation_id: int,
    dataset: str,
    model: str,
    config_path: str,
    patience: int,
    terminal_width: int = 80,
    use_color: bool = True,
) -> str:
    """Return a one-line status summary for an investigation."""

    cfg = DatasetConfig(conn, config_path, dataset, investigation_id)
    inf_table = f"{dataset}_inferences"
    cur = conn.cursor()

    # Determine if any rounds exist
    try:
        split_id = cfg.get_latest_split_id()
    except SystemExit:
        return f"{dataset}/{model}: no rounds"

    all_rounds = cfg.get_rounds_for_split(split_id)
    if not all_rounds:
        return f"{dataset}/{model}: no rounds"

    # If only one round and the prompt is "Choose randomly"
    if len(all_rounds) == 1:
        cur.execute(
            f"SELECT prompt FROM {cfg.rounds_table} WHERE round_id = %s AND investigation_id = %s",
            (all_rounds[0], investigation_id),
        )
        row = cur.fetchone()
        prompt = row[0] if row else ""
        if prompt.strip() == "Choose randomly":
            return f"{dataset}/{model}: only 'Choose randomly' round"

    processed_rounds = cfg.get_processed_rounds_for_split(split_id)

    # Early stopping check if we have any processed rounds
    if processed_rounds and cfg.check_early_stopping(split_id, "accuracy", patience):
        try:
            test_acc = cfg.get_test_metric_for_best_validation_round(split_id, "accuracy")
            return f"{dataset}/{model}: complete ({test_acc:.3f} test accuracy)"
        except Exception:
            return f"{dataset}/{model}: complete"

    # Not complete - show progress
    cur.execute(
        f"SELECT MAX(creation_time) FROM {inf_table} WHERE investigation_id = %s",
        (investigation_id,),
    )
    row = cur.fetchone()
    last_time = row[0]
    local_time = None
    seconds_ago = None
    if last_time is not None:
        local_time = (
            last_time.replace(tzinfo=SYDNEY_TZ)
            if last_time.tzinfo is None
            else last_time.astimezone(SYDNEY_TZ)
        )
        seconds_ago = (datetime.now(SYDNEY_TZ) - local_time).total_seconds()
    is_recent = seconds_ago is not None and seconds_ago < 1800
    time_str = local_time.strftime("%Y-%m-%d %H:%M:%S") if local_time else "never"

    result = (
        f"{dataset}/{model}: {len(processed_rounds)} rounds so far, "
        f"last inference {time_str}"
    )
    if use_color and is_recent:
        return f"\033[1;32m{result}\033[0m"
    return result


def matches(value: str, patterns: Iterable[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(value, p) for p in patterns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor investigations")
    parser.add_argument("--dsn", help="PostgreSQL DSN")
    parser.add_argument("--config", help="JSON file containing postgres_dsn")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset glob pattern to include (can be repeated)",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Model glob pattern to include (can be repeated)",
    )
    parser.add_argument("--width", type=int, default=None, help="Override terminal width")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    width = args.width
    if width is None:
        try:
            width, _ = shutil.get_terminal_size()
        except Exception:
            width = 80
    print(f"Terminal width detected: {width} columns")

    conn = get_connection(args.dsn, args.config)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT i.id, i.dataset, i.model, d.config_file, m.patience
          FROM investigations i
          JOIN datasets d ON i.dataset = d.dataset
          JOIN models m ON i.model = m.model
         ORDER BY i.dataset, i.model
        """
    )
    rows = cur.fetchall()

    grouped: Dict[str, Tuple[str, list[Tuple[int, str, int]]]] = {}
    for inv_id, dataset, model, cfg, patience in rows:
        if not matches(dataset, args.dataset):
            continue
        if not matches(model, args.model):
            continue
        grouped.setdefault(dataset, (cfg, []) )[1].append((inv_id, model, patience))

    if not grouped:
        print("No matching investigations found")
        return

    for dataset in sorted(grouped):
        cfg_path, investigations = grouped[dataset]
        for inv_id, model, patience in sorted(investigations, key=lambda x: x[1]):
            status = check_investigation_status(
                conn,
                inv_id,
                dataset,
                model,
                cfg_path,
                patience,
                width,
                not args.no_color,
            )
            print(status)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
