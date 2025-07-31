#!/usr/bin/env python3
"""Utility for selecting interesting ensembles for plotting."""

from __future__ import annotations

import argparse
from typing import List, Tuple, Iterable

import pandas as pd


def get_interesting_ensembles(conn, dataset: str) -> pd.DataFrame:
    """Return ensembles with strictly improving validation accuracy.

    The input table ``ensemble_results`` may contain multiple rows per
    release date. This function keeps the ensemble with the highest
    validation accuracy for each date and then filters so that the
    remaining rows show a monotonically increasing validation accuracy
    over time.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT release_date, train_accuracy, validation_accuracy, validation_correct,
               validation_total, test_correct, test_total, model_names
          FROM ensemble_results
         WHERE dataset = %s
         ORDER BY release_date, validation_accuracy DESC
        """,
        (dataset,),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(
            columns=[
                "release_date",
                "validation_accuracy",
                "validation_correct",
                "validation_total",
                "test_correct",
                "test_total",
                "model_names",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "release_date",
            "train_accuracy",
            "validation_accuracy",
            "validation_correct",
            "validation_total",
            "test_correct",
            "test_total",
            "model_names",
        ],
    )

    df.sort_values(["release_date", "validation_accuracy"], ascending=[True, False], inplace=True)
    df = df.drop_duplicates(subset=["release_date"], keep="first")
    df.sort_values("release_date", inplace=True)

    best_so_far = -1.0
    keep_rows = []
    for idx, row in df.iterrows():
        train_acc = row["train_accuracy"]
        val_acc = row["validation_accuracy"]
        if val_acc is None or train_acc is None:
            continue
        combo = min(train_acc, val_acc)
        if combo > best_so_far:
            best_so_far = combo
            keep_rows.append(idx)

    return df.loc[keep_rows]


def main(argv: Iterable[str] | None = None) -> None:
    from postgres import get_connection

    parser = argparse.ArgumentParser(description="List interesting ensembles")
    parser.add_argument("dataset", help="Dataset name")
    args = parser.parse_args(argv)

    conn = get_connection()
    df = get_interesting_ensembles(conn, args.dataset)
    if df.empty:
        print("No ensemble data found")
    else:
        for row in df.itertuples(index=False):
            print(
                row.release_date,
                row.validation_accuracy,
                row.test_correct,
                row.test_total,
                row.model_names,
                sep="\t",
            )


if __name__ == "__main__":
    main()
