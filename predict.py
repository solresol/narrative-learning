#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import os
import json
import llmcall
from common import create_config_from_database

def predict(conn, round_id, entity_id, model='phi4:latest', dry_run=False):
    """
    Predict the outcome for a specific entity in a specific round.

    Args:
        conn: Database connection
        round_id: Round ID
        entity_id: Entity ID
        model: Model to use for prediction
        dry_run: If True, just print the prompt and don't actually run prediction
    """
    # Create a DatasetConfig object
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list")
    db_path = cursor.fetchone()[2]  # Get the database file path

    dataset_config = create_config_from_database(db_path)

    # Get the round instructions
    instructions = dataset_config.get_round_prompt(round_id)
    if instructions is None:
        sys.exit(f"Round ID {round_id} not found.")

    # Check if the entity has already been evaluated in this round
    entity_field = dataset_config.entity_id_field

    query = f"SELECT COUNT(*) FROM inferences WHERE round_id = ? AND {entity_field}_id = ?"
    cursor.execute(query, [round_id, entity_id])
    row = cursor.fetchone()
    if row[0] == 1:
        sys.exit(f"Entity {entity_id} has already been evaluated in round {round_id}")

    # Get the entity features
    entity_features = dataset_config.get_entity_features(entity_id)

    # Create the prompt
    prompt = f"""This is an experiment in identifying whether an LLM can predict outcomes. Use the following methodology for predicting the outcome for this entity.

```
{instructions}
```

Entity Data:
{entity_features}
"""

    if dry_run:
        print(prompt)
        return

    # Valid predictions should be derived from the target field values
    # For now, hardcoding to Success/Failure which matches the Titanic dataset
    valid_predictions = ['Success', 'Failure']

    try:
        prediction_output, run_info = llmcall.dispatch_prediction_prompt(model, prompt, valid_predictions)

        # Insert the prediction into the database
        insert_query = f"""
        INSERT INTO inferences (round_id, {entity_field}_id, narrative_text, llm_stderr, prediction)
        VALUES (?, ?, ?, ?, ?)
        """

        cursor.execute(
            insert_query,
            (round_id, entity_id, prediction_output['narrative_text'], run_info, prediction_output['prediction'])
        )
        conn.commit()

    except llmcall.MissingPrediction:
        sys.stderr.write("There was a field missing in the response. Skipping.\n")
        # we'll pick it up next time through the loop
    except llmcall.InvalidPrediction:
        sys.stderr.write("There was an invalid prediction.\n")


if __name__ == '__main__':
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    default_model = os.environ.get('NARRATIVE_LEARNING_INFERENCE_MODEL', None)

    parser = argparse.ArgumentParser(
        description="Make predictions for an entity based on the round prompt"
    )
    parser.add_argument('--database', default=default_database,
                        required=(default_database is None),
                        help="Path to the SQLite database file")
    parser.add_argument('--round-id', type=int, required=True,
                        help="Round ID")
    parser.add_argument('--entity-id', required=True,
                        help="Entity ID (e.g., Patient_ID)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show the prompt, then exit")
    parser.add_argument("--model", default=default_model,
                        help="AI model to use for prediction")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)

    predict(conn, args.round_id, args.entity_id, args.model, args.dry_run)
