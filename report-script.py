#!/usr/bin/env python3
import argparse
import sqlite3
import sys
import datasetconfig
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
from modules.postgres import get_connection, get_investigation_settings

def main():
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    default_config = os.environ.get('NARRATIVE_LEARNING_CONFIG', None)
    parser = argparse.ArgumentParser(description="Report on classification metrics across rounds")
    parser.add_argument('--database', default=default_database,
                       help="Path to the SQLite database file")
    parser.add_argument('--dsn', help='PostgreSQL DSN')
    parser.add_argument('--pg-config', help='JSON file containing postgres_dsn')
    parser.add_argument('--investigation-id', type=int, help='ID from investigations table')
    parser.add_argument('--split-id', type=int, help="Split ID to analyze")
    parser.add_argument('--metric', default='accuracy',
                       choices=['accuracy', 'precision', 'recall', 'f1', 'count'],
                       help="Metric to calculate")
    parser.add_argument('--validation', action='store_true',
                       help="Report on validation data")
    parser.add_argument('--test', action='store_true',
                       help="Report on test data")
    parser.add_argument('--train', action='store_true',
                       help="Report on training data")
    parser.add_argument('--patience', type=int,
                       help="Number of rounds to look back for early stopping")
    parser.add_argument("--best", action="store_true",
                       help="Show the round where the validation result was best")
    parser.add_argument("--estimate", help="Show the value of the metric given as an argument, based on the round for which the validation result was best (for the --metric argument)")
    parser.add_argument('--csv', type=str,
                       help="Output CSV file path")
    parser.add_argument('--chart', type=str,
                       help="Output chart PNG file path")
    parser.add_argument("--show", action="store_true", help="Display the metric data to stdout")
    parser.add_argument("--config", default=default_config, help="The JSON config file that says what columns exist and what the tables are called")
    args = parser.parse_args()

    if args.investigation_id is not None:
        conn = get_connection(args.dsn, args.pg_config)
        dataset, config_file = get_investigation_settings(conn, args.investigation_id)
        config = datasetconfig.DatasetConfig(conn, config_file, dataset, args.investigation_id)
    else:
        if not (args.database or args.dsn or args.pg_config or os.environ.get('POSTGRES_DSN')):
            sys.exit("Must specify --database or --dsn/--pg-config for PostgreSQL")
        if args.config is None:
            sys.exit("Must specify --config or set the env variable NARRATIVE_LEARNING_CONFIG")

        if args.dsn or args.pg_config or os.environ.get('POSTGRES_DSN'):
            conn = get_connection(args.dsn, args.pg_config)
        else:
            conn = sqlite3.connect(f'file:{args.database}?mode=ro', uri=True)
        config = datasetconfig.DatasetConfig(conn, args.config)

    # If split_id not provided, use the most recent
    split_id = args.split_id if args.split_id is not None else config.get_latest_split_id()

    # Early stopping check
    if args.patience:
        args.validation = True
        should_stop = config.check_early_stopping(split_id, args.metric, args.patience)
        if should_stop:
            print(f"Early stopping triggered: No improvement in {args.patience} rounds")
            sys.exit(1)
        sys.exit(0)

    # Get best round ID (for validation metric)
    if args.best:
        try:
            best_round = config.get_best_round_id(split_id, args.metric)
            print(best_round)
        except ValueError as e:
            sys.exit(f"Error: {e}")
        sys.exit(0)

    # Estimate specific metric on test set for best validation round
    if args.estimate:
        try:
            test_metric = config.get_test_metric_for_best_validation_round(
                split_id, args.metric, args.estimate)
            print(test_metric)
        except ValueError as e:
            sys.exit(f"Error: {e}")
        sys.exit(0)

    # If no data type specified, default to training data
    if not any([args.validation, args.test, args.train]):
        args.train = True

    # Determine which data types to include
    data_types = []
    if args.train:
        data_types.append('train')
    if args.validation:
        data_types.append('validation')
    if args.test:
        data_types.append('test')

    # Generate metrics DataFrame
    df = config.create_metrics_dataframe(split_id, args.metric, data_types)

    did_something = False
    if args.csv:
        df.to_csv(args.csv)
        print(f"CSV output written to {args.csv}")
        did_something = True

    if args.chart:
        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot(ax=ax)
        ax.set_xlabel('Round ID')
        ax.grid(True)
        ax.set_ylim((0,1))
        fig.savefig(args.chart)
        print(f"Chart saved to {args.chart}")
        did_something = True

    if args.show:
        print(df)
        did_something = True

    if not did_something:
        sys.exit("Specify --patience, --csv, --chart, --show, --estimate or --best")

if __name__ == '__main__':
    main()
