#!/usr/bin/env python3
import argparse
import sys
import os
import predict
import datasetconfig
from modules.postgres import get_connection, get_investigation_settings

def main():
    default_model = os.environ.get('NARRATIVE_LEARNING_INFERENCE_MODEL', None)
    default_config = os.environ.get('NARRATIVE_LEARNING_CONFIG', None)
    parser = argparse.ArgumentParser(description="Manipulate inference rounds")
    parser.add_argument("--list", action="store_true", help="Just list the patients yet to be processed")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after this many predictions")
    parser.add_argument('--round-id', type=int, required=True, help="Round ID to check")
    parser.add_argument("--loop", action="store_true", help="Loop until there are no more patients to process")
    parser.add_argument("--config", default=default_config, help="The JSON config file that says what columns exist and what the tables are called")
    parser.add_argument("--progress-bar", action="store_true", help="Show a progress bar")
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
        cur = conn.cursor()
        cur.execute(
            """
            SELECT i.dataset, d.config_file, m.inference_model
              FROM investigations i
              JOIN datasets d ON i.dataset = d.dataset
              JOIN models m ON i.model = m.model
             WHERE i.id = %s
            """,
            (args.investigation_id,),
        )
        row = cur.fetchone()
        if row is None:
            sys.exit(f"investigation {args.investigation_id} not found")
        dataset, config_file, inference_model = row
        if "--model" not in sys.argv:
            args.model = inference_model
        config = datasetconfig.DatasetConfig(conn, config_file, dataset, args.investigation_id)
    else:
        if args.model is None and not args.list:
            sys.exit("Must specify --model or set the env variable NARRATIVE_LEARNING_INFERENCE_MODEL")
        if args.config is None:
            sys.exit("Must specify --config or set the env variable NARRATIVE_LEARNING_CONFIG")
        conn = get_connection(args.dsn, args.pg_config)
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
        rows = cur.fetchall() if args.progress_bar else cur
        iterator = rows
        pb = None
        if args.progress_bar:
            import tqdm
            iterator = rows
            pb = tqdm.tqdm(total=len(rows))
        anything_left = False
        ids_to_predict = []
        for row in iterator:
            anything_left = True
            if args.list:
                print(row[0])
                continue
            ids_to_predict.append(row[0])
            if args.stop_after is not None and predictions_done + len(ids_to_predict) >= args.stop_after:
                break
        if ids_to_predict:
            predict.predict_many(
                config,
                args.round_id,
                ids_to_predict,
                model=args.model,
                investigation_id=args.investigation_id,
                immediate=True,
                progressbar=pb,
            )
            predictions_done += len(ids_to_predict)
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
