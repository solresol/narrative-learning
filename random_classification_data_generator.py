#t!/usr/bin/env python

import argparse
import json
import numpy as np
import random
import sqlite3
import uuid


class InvalidExpressionError(Exception):
    """Raised when the provided class-deciding expression is invalid."""
    pass


def create_tables(conn, args):
    """Create the main data table and splits table in the database."""
    cursor = conn.cursor()
    
    # Create the main data table
    cursor.execute(f'''
    CREATE TABLE {args.table_name} (
        {args.primary_key_name} TEXT PRIMARY KEY,
        {args.feature1_name} REAL,
        {args.feature2_name} REAL,
        {args.target_column_name} TEXT
    )
    ''')
    
    # Create the splits table
    cursor.execute(f'''
    CREATE TABLE {args.splits_table_name} (
        split_id INTEGER,
        {args.primary_key_name} TEXT REFERENCES {args.table_name}({args.primary_key_name}),
        holdout BOOLEAN NOT NULL DEFAULT FALSE,
        validation BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (split_id, {args.primary_key_name})
    )
    ''')
    
    # Create the splits reference table
    cursor.execute('''
    CREATE TABLE splits (
        split_id INTEGER PRIMARY KEY,
        name TEXT
    )
    ''')
    
    # Insert default split
    cursor.execute("INSERT INTO splits (split_id, name) VALUES (0, 'default')")

    cursor.execute(f'''
      CREATE TABLE inferences (
          round_id integer references rounds(round_id),
          creation_time datetime default current_timestamp,
          {args.primary_key_name} text references {args.table_name}({args.primary_key_name}),
          narrative_text text,
          llm_stderr text,
          prediction text,
          primary key (round_id, {args.primary_key_name})
        );''')

    cursor.execute('''
      CREATE TABLE rounds (
         round_id integer primary key autoincrement,
         round_start datetime default current_timestamp,
         split_id integer references splits(split_id),
         prompt text,
         reasoning_for_this_prompt text,
         stderr_from_prompt_creation text
      );''')

    cursor.execute("INSERT INTO rounds (prompt, split_id) VALUES (?, 0)", [args.prompt])
    
    conn.commit()


def generate_data(args):
    """Generate random data based on the provided arguments and save to SQLite database."""
    # Set random seed for reproducibility
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
    
    # Generate features
    feature1_values = np.random.normal(
        args.feature1_mean, 
        args.feature1_stddev,
        args.number_of_data_points
    )
    
    feature2_values = np.random.normal(
        args.feature2_mean,
        args.feature2_stddev,
        args.number_of_data_points
    )
    
    # Determine classes based on the provided expression
    target_values = []
    
    # Safely evaluate the class-deciding expression for each data point
    for i in range(args.number_of_data_points):
        # Create a local scope with feature values
        local_scope = {
            args.feature1_name: feature1_values[i],
            args.feature2_name: feature2_values[i]
        }
        
        try:
            
            # Evaluate the expression with the feature values
            result = eval(args.class_deciding_expression, {"__builtins__": {}}, local_scope)
            result = bool(result)
            if not isinstance(result, bool):
                raise InvalidExpressionError(
                    f"Expression must evaluate to a boolean, but got {type(result)}"
                )
            target_values.append(result)
        except Exception as e:
            raise InvalidExpressionError(
                f"Failed to evaluate expression '{args.class_deciding_expression}' with "
                f"values {local_scope}: {str(e)}"
            )
    
    # Apply noise by flipping some target values
    if args.noise > 0:
        flip_indices = random.sample(
            range(args.number_of_data_points),
            int(args.number_of_data_points * args.noise)
        )
        for idx in flip_indices:
            target_values[idx] = not target_values[idx]
    
    # Convert boolean values to specified class names
    class_labels = [
        args.true_class_name if val else args.false_class_name
        for val in target_values
    ]
    
    # Create SQLite database
    conn = sqlite3.connect(args.output_db_file)
    create_tables(conn, args)
    cursor = conn.cursor()
    
    # Generate IDs and insert data
    ids = []
    for i in range(args.number_of_data_points):
        # Generate a unique ID
        unique_id = str(uuid.uuid4()) if args.use_uuid else f"ID_{i+1}"
        ids.append(unique_id)
        
        # Insert into main table
        cursor.execute(
            f"INSERT INTO {args.table_name} VALUES (?, ?, ?, ?)",
            (unique_id, feature1_values[i], feature2_values[i], class_labels[i])
        )
    
    # Handle splits
    # Calculate how many points to hold out
    holdout_count = int(args.number_of_data_points * args.holdout)
    validation_count = int(holdout_count * args.validation)
    test_count = holdout_count - validation_count
    
    # Randomly select points for holdout
    random.shuffle(ids)
    holdout_ids = ids[:holdout_count]
    
    # From the holdout set, determine validation and test
    validation_ids = holdout_ids[:validation_count]
    test_ids = holdout_ids[validation_count:]
    
    # Insert splits information
    for id_val in ids:
        is_holdout = id_val in holdout_ids
        is_validation = id_val in validation_ids
        
        cursor.execute(
            f"INSERT INTO {args.splits_table_name} VALUES (?, ?, ?, ?)",
            (0, id_val, is_holdout, is_validation)
        )
    
    conn.commit()
    
    # Create and save config file
    config = {
        "table_name": args.table_name,
        "primary_key": args.primary_key_name,
        "target_field": args.target_column_name,
        "splits_table": args.splits_table_name,
        "columns": [
            args.primary_key_name,
            args.feature1_name,
            args.feature2_name,
            args.target_column_name
        ]
    }
    
    with open(args.config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    conn.close()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate random data for a 2-feature classification problem in SQLite.'
    )
    
    parser.add_argument('--number-of-data-points', type=int, required=True,
                        help='Number of data points to generate')
    parser.add_argument('--feature1-name', type=str, required=True,
                        help='Name of the first feature')
    parser.add_argument('--feature1-mean', type=float, required=True,
                        help='Mean of the first feature')
    parser.add_argument('--feature1-stddev', type=float, required=True,
                        help='Standard deviation of the first feature')
    parser.add_argument('--feature2-name', type=str, required=True,
                        help='Name of the second feature')
    parser.add_argument('--feature2-mean', type=float, required=True,
                        help='Mean of the second feature')
    parser.add_argument('--feature2-stddev', type=float, required=True,
                        help='Standard deviation of the second feature')
    parser.add_argument('--target-column-name', type=str, required=True,
                        help='Name of the target column')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--class-deciding-expression', type=str, required=True,
                        help='Python expression that decides the class (must use feature names)')
    parser.add_argument('--false-class-name', type=str, required=True,
                        help='Value to use for the False class')
    parser.add_argument('--true-class-name', type=str, required=True,
                        help='Value to use for the True class')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Proportion of data points to have their targets flipped')
    parser.add_argument('--output-db-file', type=str, required=True,
                        help='Path to output SQLite database file')
    parser.add_argument('--config-file', type=str, required=True,
                        help='Path to output config JSON file')
    parser.add_argument('--table-name', type=str, required=True,
                        help='Name of the main data table')
    parser.add_argument('--primary-key-name', type=str, required=True,
                        help='Name of the primary key column')
    parser.add_argument('--splits-table-name', type=str, required=True,
                        help='Name of the table for splits information')
    parser.add_argument('--use-uuid', action='store_true',
                        help='Use UUID for primary key instead of sequential IDs')
    parser.add_argument('--holdout', type=float, default=0.2,
                        help='Proportion of the data to mark as held out (never to be used in training)')
    parser.add_argument('--validation', type=float, default=0.5,
                        help='Proportion of the holdout data to use for validation')
    parser.add_argument("--prompt", default="Choose randomly", help="Prompt to insert into the rounds table (default: 'Choose randomly').") 
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    generate_data(args)
