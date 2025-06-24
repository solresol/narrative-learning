#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import json
import os
import llmcall
import datasetconfig
from modules.postgres import get_connection, get_investigation_settings

def get_prompt_for_updating_model(config, round_id, example_count, history_rounds):
    answer = """

You are part of a program that is trying to learn inference rules on
this dataset. At each round, a prompt is shown to an LLM together with
one row of data at a time. It then attempts to predict the outcome based on the
rules in the prompt. This process works well if the prompt has very
explicit and clear rules: aim for unambiguous thresholds for values,
clear criteria for labels and careful wording.

We would like to improve the prompt that is being used.

Please create a new prompt that will reduce the number of false
positives and false negatives in this dataset. You can see the
prompt(s) that have been used previously, and how effective they
were. There are also some examples of where those prompt(s) did and
didn't work.

Remember: you need to create rules. Don't just waffle about what
changes need to happen. Look at the examples where the previous prediction
system got it wrong, and try to come up with at least one new rule that
would handle one of those situations correctly.

----------------------------

"""
    prompt = config.get_round_prompt(round_id)
    matrix = config.get_confusion_matrix(round_id, example_count=example_count)
    answer += config.get_printable_confusion_matrix_and_examples(round_id, matrix, show_examples=True)

    # History rounds (without examples)
    if history_rounds > 0:
        cur = config.conn.cursor()
        table = config.rounds_table
        query = f"""
            SELECT round_id, prompt
              FROM {table}
             WHERE round_id < ?
        """
        params = [round_id]
        if getattr(config, "investigation_id", None) is not None:
            query += " AND investigation_id = ?"
            params.append(config.investigation_id)
        query += " ORDER BY round_id DESC LIMIT ?"
        params.append(history_rounds)
        config._execute(cur, query, tuple(params))
        history_rounds = cur.fetchall()

        if history_rounds:
            answer += "# Historical results:"
            for r_id, r_prompt in history_rounds:
                hist_matrix = config.get_confusion_matrix(r_id, example_count=0)
                answer += config.get_printable_confusion_matrix_and_examples(r_id, hist_matrix, show_examples=False)
        else:
            #print("No previous rounds found.")
            pass
    answer += "\n\n---------------------\n\n"
    return answer

class PromptCreationFailure(Exception):
    def __init__(self):
        pass

def run_reprompt(config, prompting_creation_prompt, old_round_id, model, verbose, investigation_id: int | None = None):
    split_id = config.get_split_id(old_round_id)
    sanity_check_sample_id = config.get_random_non_holdout_id(split_id)
    sanity_check_sample = config.get_entity_features(sanity_check_sample_id)
    history = []
    remaining_attempts = 20
    while True:
        if history:
            extra = "\n\n" + "\n\n".join([f"THIS DID NOT PASS QUALITY CONTROL, IT IS NOT A GOOD ENOUGH PROMPT: {x}" for x in history])
        else:
            extra = ""
        new_prompt, process_info = llmcall.dispatch_reprompt_prompt(model, prompting_creation_prompt + extra)
        if llmcall.sanity_check_prompt(new_prompt['updated_prompt'], sanity_check_sample, config.valid_predictions):
            break
        if verbose:
            print(f"The prompt that was generated was:\n\n```\n{new_prompt['updated_prompt']}\n```\n This did not pass quality control.")
        else:
            sys.stderr.write("Generated prompt failed quality control.\n")
        remaining_attempts = remaining_attempts - 1
        if remaining_attempts == 0:
            raise PromptCreationFailure

    if verbose:
        print(f"Quality control passed the following prompt:\n\n```\n{new_prompt}\n```")
    cur = config.conn.cursor()
    table = config.rounds_table
    fields = "split_id, prompt, reasoning_for_this_prompt, stderr_from_prompt_creation"
    placeholders = "?, ?, ?, ?"
    params = [split_id, new_prompt['updated_prompt'], new_prompt['reasoning'], process_info]
    if investigation_id is not None:
        fields += ", investigation_id"
        placeholders += ", ?"
        params.append(investigation_id)
    query = f"insert into {table} ({fields}) values ({placeholders}) returning round_id"
    config._execute(cur, query, tuple(params))
    row = cur.fetchone()
    if row is None:
        sys.exit("Failed to create a new round")
    new_round_id = row[0]
    config.conn.commit()
    return (new_round_id, new_prompt)


def main():
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    default_model = os.environ.get('NARRATIVE_LEARNING_TRAINING_MODEL', None)
    default_example_count = int(os.environ.get('NARRATIVE_LEARNING_EXAMPLE_COUNT', '3'))
    default_config = os.environ.get('NARRATIVE_LEARNING_CONFIG', None)
    parser = argparse.ArgumentParser(description="Show confusion matrix for a round")
    parser.add_argument('--database', default=default_database, help="Path to the SQLite database file")
    parser.add_argument('--dsn', help='PostgreSQL DSN')
    parser.add_argument('--pg-config', help='JSON file containing postgres_dsn')
    parser.add_argument('--investigation-id', type=int, help='ID from investigations table')
    parser.add_argument('--round-id', type=int, required=True, help="Round ID")
    parser.add_argument('--example-count', type=int, default=default_example_count, help="Number of examples per cell")
    parser.add_argument('--show-history', type=int, default=2, help="Show confusion matrices for the previous N rounds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default=default_model, help="Model to do the retraining, not the model that does the inference")
    parser.add_argument("--round-tracking-file", help="A file that will have the new round ID written to it")
    parser.add_argument("--config", default=default_config, help="The JSON config file that says what columns exist and what the tables are called")
    args = parser.parse_args()

    if args.investigation_id is not None:
        conn = get_connection(args.dsn, args.pg_config)
        dataset, config_file = get_investigation_settings(conn, args.investigation_id)
        config = datasetconfig.DatasetConfig(conn, config_file, dataset, args.investigation_id)
    else:
        if not (args.database or args.dsn or args.pg_config or os.environ.get('POSTGRES_DSN')):
            sys.exit("Must specify --database or --dsn/--pg-config for PostgreSQL")
        if args.model is None:
            sys.exit("Must specify a model via --model or NARRATIVE_LEARNING_TRAINING_MODEL env")
        if args.config is None:
            sys.exit("Must specify --config or set the env variable NARRATIVE_LEARNING_CONFIG")

        if args.dsn or args.pg_config or os.environ.get('POSTGRES_DSN'):
            conn = get_connection(args.dsn, args.pg_config)
        else:
            conn = sqlite3.connect(args.database)
        config = datasetconfig.DatasetConfig(conn, args.config)

    prompting_creation_prompt = get_prompt_for_updating_model(config, args.round_id, args.example_count, args.show_history)
    if args.dry_run or args.verbose:
        print(prompting_creation_prompt)
    if args.dry_run:
        sys.exit(0)
    new_round_id, details = run_reprompt(config, prompting_creation_prompt, args.round_id, args.model, args.verbose, args.investigation_id)
    if args.verbose:
        print(f"REASONING: {details['reasoning']}")
        print()
        print(f"NEW PROMPT: {details['updated_prompt']}")
        print()
        print(f"NEW ROUND ID: {new_round_id}")
    if args.round_tracking_file:
        with open(args.round_tracking_file, 'w') as f:
            f.write(f"{new_round_id}")

if __name__ == '__main__':
    main()
