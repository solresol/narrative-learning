#!/usr/bin/env python3
"""Display progress for investigations stored in PostgreSQL."""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from typing import Any, Dict, Iterable, Tuple

from modules.postgres import get_connection


def check_investigation_status(
    conn: Any,
    investigation_id: int,
    dataset: str,
    model: str,
    rounds_table: str,
    terminal_width: int = 80,
    use_color: bool = True,
) -> str:
    """Return a one-line status summary for an investigation."""
    inf_table = f"{dataset}_inferences"
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT round_id, count(*),
               EXTRACT(EPOCH FROM NOW() - MIN(creation_time)) AS seconds_ago
          FROM {inf_table}
         WHERE investigation_id = %s
         GROUP BY round_id
         ORDER BY round_id DESC
        """,
        (investigation_id,),
    )
    rounds = cur.fetchall()
    rounds = [r for r in rounds if r[1] > 0]
    if not rounds:
        return f"{dataset}/{model}: no inferences found"

    latest_round, inference_count, seconds_ago = rounds[0]
    is_recent = seconds_ago is not None and seconds_ago < 1800

    cur.execute(
        f"SELECT prompt FROM {rounds_table} WHERE round_id = %s AND investigation_id = %s",
        (latest_round, investigation_id),
    )
    row = cur.fetchone()
    prompt = row[0] if row else "N/A"
    prompt = prompt.replace("\n", " ")

    base_output = (
        f"{dataset}/{model}: {inference_count} inferences, round #{latest_round}, prompt: \""
    )
    remaining_width = terminal_width - len(base_output) - 1
    if remaining_width < 10:
        remaining_width = 40
    prompt_preview = (
        prompt[: remaining_width - 3] + "..." if len(prompt) > remaining_width else prompt
    )
    result = f"{dataset}/{model}: {inference_count} inferences, round #{latest_round}, prompt: \"{prompt_preview}\""
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
        SELECT i.id, i.dataset, i.model, d.config_file
          FROM investigations i
          JOIN datasets d ON i.dataset = d.dataset
          JOIN models m ON i.model = m.model
         ORDER BY i.dataset, i.model
        """
    )
    rows = cur.fetchall()

    grouped: Dict[str, Tuple[str, list[Tuple[int, str]]]] = {}
    for inv_id, dataset, model, cfg in rows:
        if not matches(dataset, args.dataset):
            continue
        if not matches(model, args.model):
            continue
        grouped.setdefault(dataset, (cfg, []) )[1].append((inv_id, model))

    if not grouped:
        print("No matching investigations found")
        return

    for dataset in sorted(grouped):
        cfg_path, investigations = grouped[dataset]
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                rounds_table = json.load(f)["rounds_table"]
        except Exception as exc:
            print(f"{dataset}: error reading config ({exc})")
            continue
        for inv_id, model in sorted(investigations, key=lambda x: x[1]):
            status = check_investigation_status(
                conn,
                inv_id,
                dataset,
                model,
                rounds_table,
                width,
                not args.no_color,
            )
            print(status)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
