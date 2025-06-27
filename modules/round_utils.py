#!/usr/bin/env python3
"""Helper for updating round statistics."""
from __future__ import annotations

from datasetconfig import DatasetConfig


def update_round_statistics(config: DatasetConfig, round_id: int) -> None:
    """Update completion time and accuracies for a round."""

    conn = config.conn
    cur = conn.cursor()
    inf_table = f"{config.dataset}_inferences" if config.dataset else "inferences"

    params = [round_id]
    query = f"SELECT count(*) FROM {inf_table} WHERE round_id = ?"
    if config.investigation_id is not None:
        query += " AND investigation_id = ?"
        params.append(config.investigation_id)
    config._execute(cur, query, tuple(params))
    inf_count = cur.fetchone()[0]

    total = config.get_data_point_count()

    rounds_table = config.rounds_table

    if inf_count == total:
        params = [round_id]
        query = f"SELECT max(creation_time) FROM {inf_table} WHERE round_id = ?"
        if config.investigation_id is not None:
            query += " AND investigation_id = ?"
            params.append(config.investigation_id)
        config._execute(cur, query, tuple(params))
        round_completed = cur.fetchone()[0]

        train_matrix = config.get_confusion_matrix(round_id)
        val_matrix = config.get_confusion_matrix(round_id, on_holdout_data=True)
        test_matrix = config.get_confusion_matrix(
            round_id, on_holdout_data=True, on_test_data=True
        )

        train_accuracy = config.calculate_metric(train_matrix, "accuracy")
        validation_accuracy = config.calculate_metric(val_matrix, "accuracy")
        test_accuracy = config.calculate_metric(test_matrix, "accuracy")
    else:
        round_completed = None
        train_accuracy = None
        validation_accuracy = None
        test_accuracy = None

    update_query = (
        f"UPDATE {rounds_table} "
        "SET round_completed = ?, "
        "train_accuracy = ?, "
        "validation_accuracy = ?, "
        "test_accuracy = ? "
        "WHERE round_id = ?"
    )
    config._execute(
        cur,
        update_query,
        (
            round_completed,
            train_accuracy,
            validation_accuracy,
            test_accuracy,
            round_id,
        ),
    )
    conn.commit()

