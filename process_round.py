#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import os
import predict
import datasetconfig
from modules.postgres import get_connection

def main():
    # One day I'll make --database mandatory, with no fallback to titanic_medical.sqlite
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    default_model = os.environ.get('NARRATIVE_LEARNING_INFERENCE_MODEL', None)
    default_config = os.environ.get('NARRATIVE_LEARNING_CONFIG', None)
    parser = argparse.ArgumentParser(description="Manipulate inference rounds")
    parser.add_argument("--list", action="store_true", help="Just list the patients yet to be processed")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after this many predictions")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID to check")
    parser.add_argument("--loop", action="store_true", help="Loop until there are no more patients to process")
    parser.add_argument("--config", default=default_config, help="The JSON config file that says what columns exist and what the tables are called")
    parser.add_argument("--progress-bar", action="store_true", help="Show a progress bar")
    parser.add_argument('--database', default=default_database, help="Path to the SQLite database file")
    parser.add_argument('--dsn', help='PostgreSQL DSN')
    parser.add_argument('--pg-config', help='JSON file containing postgres_dsn')
    parser.add_argument('--investigation-id', type=int, help='ID from investigations table')
    # Maybe one day I will find the latest round by default, or have an option for a split_id
    parser.add_argument("--model", default=default_model)
    args = parser.parse_args()

    if args.list and args.loop:
        sys.exit("Can't loop if we aren't going to process anything")

    if args.investigation_id is not None:
        conn = get_connection(args.dsn, args.pg_config)
        dataset, config_file = get_investigation_settings(conn, args.investigation_id)
        cur = conn.cursor()
        config = datasetconfig.DatasetConfig(conn, config_file, dataset, args.investigation_id)
    else:
        if not (args.database or args.dsn or args.pg_config or os.environ.get('POSTGRES_DSN')):
            sys.exit("Must specify --database or --dsn/--pg-config for PostgreSQL")
        if args.model is None and not args.list:
            sys.exit("Must specify --model or set the env variable NARRATIVE_LEARNING_INFERENCE_MODEL")
        if args.config is None:
            sys.exit("Must specify --config or set the env variable NARRATIVE_LEARNING_CONFIG")
        if args.dsn or args.pg_config or os.environ.get('POSTGRES_DSN'):
            conn = get_connection(args.dsn, args.pg_config)
        else:
            conn = sqlite3.connect(args.database)
        cur = conn.cursor()
        config = datasetconfig.DatasetConfig(conn, args.config)


    predictions_done = 0

    while True:
        # Get IDs that are not already inferred
        inf_table = f"{config.dataset}_inferences" if getattr(config, 'dataset', '') else "inferences"
        query = f"""
        SELECT {config.primary_key}
        FROM {config.table_name}
        WHERE {config.primary_key} NOT IN (
           SELECT {config.primary_key} FROM {inf_table} WHERE round_id = ?
        );
        """
        config._execute(cur, query, (args.round_id,))
        # I think I should try doing a fetchall so that tqdm knows the appropriate length
        iterator = cur
        if args.progress_bar:
            import tqdm
            iterator = tqdm.tqdm(cur.fetchall())
        anything_left = False
        for row in iterator:
            anything_left = True
            if args.list:
                print(row[0])
                continue
            predict.predict(config, args.round_id, row[0], model=args.model, investigation_id=args.investigation_id)
            predictions_done += 1
            if args.stop_after is not None and predictions_done >= args.stop_after:
                sys.exit(0)
        if not anything_left:
            if args.loop:
                # we succeeded
                sys.exit(0)
            # Otherwise, we weren't looping and there was nothing to do
            sys.exit(f"No missing patientIDs found for round {args.round_id}.")
        if args.loop:
            # Try again, in case there were errors
            continue
        break

if __name__ == '__main__':
    main()
