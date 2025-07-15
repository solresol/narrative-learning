#!/usr/bin/env python3
import argparse
import sys
import os
import json
import random
import tempfile
import llmcall
import datasetconfig
from modules.postgres import get_connection, get_investigation_settings


def _make_prompt(config, round_id: int, entity_id: str) -> tuple[str, str]:
    """Return the prompt and round instructions for ``entity_id``."""

    instructions = config.get_round_prompt(round_id)
    if instructions is None:
        sys.exit(f"Round ID {round_id} not found.")

    entity_features = config.get_entity_features(entity_id)

    prompt = f"""This is an experiment in identifying whether an LLM can predict outcomes. Use the following methodology for predicting the outcome for this entity.

```
{instructions}
```

Entity Data:
{entity_features}
"""

    return prompt, instructions


def _insert_prediction(
    config: datasetconfig.DatasetConfig,
    round_id: int,
    entity_id: str,
    narrative_text: str,
    run_info: str,
    prediction: str,
    investigation_id: int | None = None,
) -> None:
    """Insert a prediction result into the database."""
    cursor = config.conn.cursor()
    inf_table = (
        f"{config.dataset}_inferences" if getattr(config, "dataset", "") else "inferences"
    )
    fields = f"round_id, {config.primary_key}, narrative_text, llm_stderr, prediction"
    placeholders = "?, ?, ?, ?, ?"
    if investigation_id is not None:
        fields += ", investigation_id"
        placeholders += ", ?"
    insert_query = f"INSERT INTO {inf_table} ({fields}) VALUES ({placeholders})"
    params = [round_id, entity_id, narrative_text, run_info, prediction]
    if investigation_id is not None:
        params.append(investigation_id)
    config._execute(cursor, insert_query, tuple(params))
    config.conn.commit()


def _check_existing_prediction(config: datasetconfig.DatasetConfig, round_id: int, entity_id: str) -> bool:
    """Return True if there is already a prediction for this entity."""
    cursor = config.conn.cursor()
    inf_table = (
        f"{config.dataset}_inferences" if getattr(config, "dataset", "") else "inferences"
    )
    query = f"SELECT COUNT(*) FROM {inf_table} WHERE round_id = ? AND {config.primary_key} = ?"
    config._execute(cursor, query, (round_id, entity_id))
    row = cursor.fetchone()
    return row[0] == 1


def predict(config, round_id, entity_id, model='gpt-4.1-mini', dry_run=False,
            prompt_only: bool = False,            
            investigation_id: int | None = None):
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

    prompt, instructions = _make_prompt(config, round_id, entity_id)

    if _check_existing_prediction(config, round_id, entity_id) and not (
        dry_run or prompt_only
    ):
        raise llmcall.AlreadyPredictedException(entity_id, round_id)



    if prompt_only:
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
    fields = f"round_id, {config.primary_key}, narrative_text, llm_stderr, prediction"
    placeholders = "?, ?, ?, ?, ?"
    if investigation_id is not None:
        fields += ", investigation_id"
        placeholders += ", ?"
    insert_query = f"INSERT INTO {inf_table} ({fields}) VALUES ({placeholders})"

    params = [round_id, entity_id, prediction_output['narrative_text'], run_info, prediction_output['prediction']]
    if investigation_id is not None:
        params.append(investigation_id)
    if dry_run:
        print(entity_id, prediction_output['prediction'], prediction_output['narrative_text'])
    else:
        config._execute(cursor, insert_query, tuple(params))
        config.conn.commit()


def predict_many(
    config,
    round_id,
    entity_ids,
    model: str = "gpt-4.1-mini",
    dry_run: bool = False,
    prompt_only: bool = False,
    investigation_id: int | None = None,
    immediate: bool = False,
    progressbar=None,
):
    """Run :func:`predict` for multiple entity IDs.

    ``entity_ids`` can be an iterable of IDs or ``(id, unique_id)`` tuples. The
    unique identifier is ignored for now but allows callers to verify that the
    prediction corresponds to the expected entity in future implementations.
    """

    if not immediate:
        instructions = config.get_round_prompt(round_id)
        if instructions == "Choose randomly":
            immediate = True

    if immediate or not llmcall.is_openai_model(model):
        for item in entity_ids:
            entity_id = item[0] if isinstance(item, tuple) else item
            predict(
                config,
                round_id,
                entity_id,
                model=model,
                dry_run=dry_run,
                prompt_only=prompt_only,
                investigation_id=investigation_id,
            )
            if progressbar is not None:
                progressbar.update(1)
        return

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        jsonl_many(config, round_id, entity_ids, model, tmp.name)
        tmp_name = tmp.name

    try:
        for result in llmcall.openai_batch_predict(
            config.dataset,
            tmp_name,
            dry_run=dry_run,
            progress_bar=progressbar,
        ):
            if dry_run:
                continue
            _insert_prediction(
                config,
                round_id,
                result["entity_id"],
                result["narrative_text"],
                json.dumps(result["usage"]),
                result["prediction"],
                investigation_id,
            )
    finally:
        os.unlink(tmp_name)


def jsonl_many(config, round_id, entity_ids, model: str, output_file: str):
    """Write OpenAI batch requests for the given entity IDs."""

    for item in entity_ids:
        entity_id = item[0] if isinstance(item, tuple) else item

        if _check_existing_prediction(config, round_id, entity_id):
            raise llmcall.AlreadyPredictedException(entity_id, round_id)

        prompt, instructions = _make_prompt(config, round_id, entity_id)

        if instructions == "Choose randomly":
            raise ValueError("Cannot create JSONL for 'Choose randomly' instructions")

        body = llmcall.openai_request_json(model, prompt, config.valid_predictions)
        batch_text = {
            "custom_id": str(entity_id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(batch_text) + "\n")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Make predictions for an entity based on the round prompt"
    )
    parser.add_argument('--dsn', help='PostgreSQL DSN')
    parser.add_argument('--pg-config', help='JSON file containing postgres_dsn')
    parser.add_argument('--round-id', type=int, required=True,
                        help="Round ID")
    parser.add_argument('--investigation-id', type=int,
                        help='ID from investigations table')
    parser.add_argument(
        "--entity-id",
        dest="entity_id",
        nargs="+",
        required=True,
        help="Entity ID(s) (e.g., Patient_ID)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Don't store anything in the database")
    parser.add_argument("--prompt-only", action="store_true",
                        help="Just show the prompt, then exit")
    parser.add_argument("--model", help="AI model to use for prediction", default="gpt-4.1-mini")
    parser.add_argument("--config", help="The JSON config file that says what columns exist and what the tables are called")
    parser.add_argument("--jsonl", help="Write OpenAI batch requests to this file instead of running predictions")
    parser.add_argument("--immediate", action="store_true", help="Run predictions immediately instead of using OpenAI batch")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar")
    args = parser.parse_args()

    if args.investigation_id is not None:
        conn = get_connection(args.dsn, args.pg_config)
        dataset, config_file = get_investigation_settings(conn, args.investigation_id)
        config = datasetconfig.DatasetConfig(conn, config_file, dataset, args.investigation_id)
    else:
        if not (args.dsn or args.pg_config or os.environ.get('POSTGRES_DSN')):
            sys.exit("Must specify --dsn/--pg-config for PostgreSQL")
        if args.config is None:
            sys.exit("Must specify --config or set the env variable NARRATIVE_LEARNING_CONFIG")
        conn = get_connection(args.dsn, args.pg_config)
        config = datasetconfig.DatasetConfig(conn, args.config)

    if args.jsonl:
        if not llmcall.is_openai_model(args.model):
            sys.exit("--jsonl can only be used with OpenAI models")
        jsonl_many(config, args.round_id, args.entity_id, args.model, args.jsonl)
        sys.exit(0)

    if args.progress:
        import tqdm
        progressbar = tqdm.tqdm()
    else:
        progressbar = None

    predict_many(
        config,
        args.round_id,
        args.entity_id,
        model=args.model,
        dry_run=args.dry_run,
        prompt_only=args.prompt_only,
        investigation_id=args.investigation_id,
        immediate=args.immediate,
        progressbar=progressbar,
    )
