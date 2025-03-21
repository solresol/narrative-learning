#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import os
import json
import llmcall
import datasetconfig
import random

class AlreadyPredictedException(Exception):
    """Exception raised when a prediction has already been made for a specific primary key in a round.

    Attributes:
        primary_key_value -- The value of the primary key that already has a prediction
        round_id -- The ID of the round where the prediction exists
    """

    def __init__(self, primary_key_value, round_id):
        self.primary_key_value = primary_key_value
        self.round_id = round_id
        self.message = f"A prediction for primary key '{primary_key_value}' already exists in round '{round_id}'"
        super().__init__(self.message)

    def __str__(self):
        return self.message

def predict(config, round_id, entity_id, model='phi4:latest', dry_run=False):
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
    cursor = config.conn.cursor()

    # Get the round instructions
    instructions = config.get_round_prompt(round_id)
    if instructions is None:
        sys.exit(f"Round ID {round_id} not found.")

    query = f"SELECT COUNT(*) FROM inferences WHERE round_id = ? AND {config.primary_key} = ?"
    cursor.execute(query, [round_id, entity_id])
    row = cursor.fetchone()
    if row[0] == 1:
        raise AlreadyPredictedException(entity_id, round_id)

    # Get the entity features
    entity_features = config.get_entity_features(entity_id)

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

    try:
        if instructions == 'Choose randomly':
            # We don't need the LLM for that
            prediction_output = {'narrative_text': "Random choice", 'prediction': random.choice(config.valid_predictions) }
            run_info = "Instructions were to pick randomly. LLM was not used."
        else:
            prediction_output, run_info = llmcall.dispatch_prediction_prompt(model, prompt, config.valid_predictions)
    except llmcall.MissingPrediction:
        sys.stderr.write("There was a field missing in the response. Generating a random answer.\n")
        prediction_output = {'prediction': random.choice(config.valid_predictions),
                             'narrative_text': "Prediction missing from LLM output" }
        run_info = ""
    except llmcall.InvalidPrediction:
        sys.stderr.write("There was an invalid prediction. Generating a random answer.\n")
        prediction_output = {'prediction': random.choice(config.valid_predictions),
                             'narrative_text': "Prediction missing from LLM output" }
        run_info = ""
        
    # Insert the prediction into the database
    insert_query = f"""
        INSERT INTO inferences (round_id, {config.primary_key}, narrative_text, llm_stderr, prediction)
        VALUES (?, ?, ?, ?, ?)
        """

    cursor.execute(
            insert_query,
            (round_id, entity_id, prediction_output['narrative_text'], run_info, prediction_output['prediction'])
        )
    config.conn.commit()



if __name__ == '__main__':
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    default_model = os.environ.get('NARRATIVE_LEARNING_INFERENCE_MODEL', None)
    default_config = os.environ.get('NARRATIVE_LEARNING_CONFIG', None)

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
    parser.add_argument("--config", default=default_config, help="The JSON config file that says what columns exist and what the tables are called")
    args = parser.parse_args()

    if args.database is None:
        sys.exit("Must specify --database or set the env variable NARRATIVE_LEARNING_DATABASE")
    if args.config is None:
        sys.exit("Must specify --config or set the env variable NARRATIVE_LEARNING_CONFIG")

    conn = sqlite3.connect(args.database)
    config = datasetconfig.DatasetConfig(conn, args.config)

    predict(config, args.round_id, args.entity_id, args.model, args.dry_run)
