#!/usr/bin/env python3
"""List investigations that appear to have missing data.

The script verifies that all processed rounds have inference results and
that early stopping conditions were met.  Early stopping is evaluated
before flagging missing inference data so investigations that legitimately
halted early are not reported as incomplete.
"""
import argparse
from modules.postgres import get_connection
from modules.investigation_status import gather_incomplete_investigations


def gather_missing(conn, hosted_only: bool = False):
    """Deprecated wrapper for backward compatibility."""
    return gather_incomplete_investigations(conn, hosted_only=hosted_only)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List investigations that appear to have missing data"
    )
    parser.add_argument(
        "--hosted-only",
        action="store_true",
        help=(
            "Only include investigations using models whose language model "
            "is not marked as ollama_hosted"
        ),
    )
    args = parser.parse_args()

    conn = get_connection()
    missing = gather_missing(conn, hosted_only=args.hosted_only)
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
