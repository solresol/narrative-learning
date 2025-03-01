#!/usr/bin/env python3
import argparse
import sqlite3
import pandas as pd
import uuid
import sys
import random
import os
import json

def extract_obfuscation_plan(db_path, obfuscation_id=None):
    """
    Extract the obfuscation plan from the SQLite database.
    If obfuscation_id is not provided, it uses the most recent plan.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the obfuscation ID if not provided
    if obfuscation_id is None:
        cursor.execute("""
            SELECT id FROM obfuscation_metadata 
            ORDER BY created_at DESC LIMIT 1
        """)
        result = cursor.fetchone()
        if not result:
            sys.exit(f"No obfuscation plans found in {db_path}")
        obfuscation_id = result[0]
    
    # Get the metadata
    cursor.execute("""
        SELECT primary_key, original_dataset_type, obfuscated_dataset_type, target_variable 
        FROM obfuscation_metadata 
        WHERE id = ?
    """, (obfuscation_id,))
    
    metadata_row = cursor.fetchone()
    if not metadata_row:
        sys.exit(f"Obfuscation plan with ID {obfuscation_id} not found")
    
    primary_key, original_dataset_type, obfuscated_dataset_type, target_variable = metadata_row
    
    # Get the column transformations
    cursor.execute("""
        SELECT original_column, remove, obfuscated_column, transformation 
        FROM column_transformations 
        WHERE obfuscation_id = ?
    """, (obfuscation_id,))
    
    columns = []
    for row in cursor.fetchall():
        original_column, remove, obfuscated_column, transformation = row
        columns.append({
            "original_column": original_column,
            "remove": bool(remove),
            "obfuscated_column": obfuscated_column,
            "transformation": transformation
        })
    
    conn.close()
    
    return {
        "primary_key": primary_key,
        "original_dataset_type": original_dataset_type,
        "obfuscated_dataset_type": obfuscated_dataset_type,
        "target_variable": target_variable,
        "columns": columns
    }

def apply_transformation(original_df, obfuscation_plan):
    """
    Apply transformations to the original DataFrame according to the obfuscation plan.
    Returns a new DataFrame with the transformed data.
    """
    # Create a new DataFrame to store the transformed data
    obfuscated_df = pd.DataFrame()
    
    # Keep track of the original DataFrame as it gets modified during transformation
    source_df = original_df.copy()
    
    # For each column transformation in the plan
    for column_info in obfuscation_plan["columns"]:
        original_column = column_info["original_column"]
        
        # Skip if column should be removed
        if column_info["remove"]:
            if original_column in source_df.columns:
                source_df = source_df.drop(columns=[original_column])
            continue
        
        # Get obfuscated column name
        obfuscated_column = column_info.get("obfuscated_column", original_column)
        
        # If there's a transformation, apply it
        if column_info.get("transformation"):
            # Create local variables for use in eval
            original = source_df
            # Execute the transformation
            try:
                result = eval(column_info["transformation"])
                obfuscated_df[obfuscated_column] = result
            except Exception as e:
                sys.exit(f"Error applying transformation for {original_column}: {e}")
        else:
            # No transformation, just copy the column
            obfuscated_df[obfuscated_column] = source_df[original_column]
    
    return obfuscated_df

def main():
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    
    parser = argparse.ArgumentParser(
        description="Import CSV data into database using an obfuscation plan."
    )
    parser.add_argument(
        "--database", default=default_database,
        help="Path to the SQLite database file for storing transformed data."
    )
    parser.add_argument(
        "--schema", default="schema.sql",
        help="SQL file containing schema initialization commands (default: schema.sql)."
    )
    parser.add_argument(
        "--source", required=True,
        help="CSV source file to transform."
    )
    parser.add_argument(
        "--obfuscation-plan", required=True,
        help="SQLite database containing the obfuscation plan."
    )
    parser.add_argument(
        "--obfuscation-id", type=int,
        help="ID of the obfuscation plan to use. If not provided, uses the most recent plan."
    )
    parser.add_argument(
        "--prompt", default="Choose randomly",
        help="Prompt to insert into the rounds table (default: 'Choose randomly')."
    )
    parser.add_argument(
        "--holdout", default=0.2, type=float,
        help="Proportion of the data to mark as held out (never to be used in training)"
    )
    parser.add_argument(
        "--validation", default=0.5, type=float,
        help="Proportion of the holdout data to use for validation"
    )
    parser.add_argument(
        "--table-name", default=None,
        help="Name of the table to create. If not provided, derived from obfuscated dataset type."
    )
    parser.add_argument(
        "--primary-key-field", default=None,
        help="Field name for generated primary key. If not provided, derived from obfuscated dataset type."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.database is None:
        sys.exit("Must specify a database via --database or NARRATIVE_LEARNING_DATABASE env")
    
    # Extract the obfuscation plan
    try:
        obfuscation_plan = extract_obfuscation_plan(args.obfuscation_plan, args.obfuscation_id)
        if args.verbose:
            print("Obfuscation plan extracted:")
            print(json.dumps(obfuscation_plan, indent=2))
    except Exception as e:
        sys.exit(f"Failed to extract obfuscation plan: {e}")
    
    # Load the source CSV
    try:
        source_df = pd.read_csv(args.source)
        if args.verbose:
            print(f"Loaded source data from {args.source} with {len(source_df)} rows")
    except Exception as e:
        sys.exit(f"Failed to read CSV file '{args.source}': {e}")
    
    # Apply transformations
    try:
        obfuscated_df = apply_transformation(source_df, obfuscation_plan)
        if args.verbose:
            print(f"Transformations applied, resulting in DataFrame with {len(obfuscated_df)} rows")
    except Exception as e:
        sys.exit(f"Failed to apply transformations: {e}")
    
    # Connect to the database
    try:
        conn = sqlite3.connect(args.database)
        if args.verbose:
            print(f"Connected to database '{args.database}'")
    except Exception as e:
        sys.exit(f"Failed to connect to database '{args.database}': {e}")
    
    # Initialize the schema
    try:
        with open(args.schema, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        if args.verbose:
            print(f"Schema initialized from '{args.schema}'")
    except Exception as e:
        sys.exit(f"Failed to initialize schema from '{args.schema}': {e}")
    
    # Determine table name and primary key field
    table_name = args.table_name or f"{obfuscation_plan['obfuscated_dataset_type'].lower().replace(' ', '_')}_data"
    primary_key_field = args.primary_key_field or f"{obfuscation_plan['obfuscated_dataset_type'].split()[0]}_ID"
    
    # Generate a unique ID for each row if needed
    if primary_key_field not in obfuscated_df.columns:
        obfuscated_df[primary_key_field] = [str(uuid.uuid4()) for _ in range(len(obfuscated_df))]
    
    # Set primary key as index
    obfuscated_df.set_index(primary_key_field, inplace=True)
    
    # Create a new split
    cur = conn.cursor()
    cur.execute("INSERT INTO splits DEFAULT VALUES RETURNING split_id")
    row = cur.fetchone()
    split_id = row[0]
    
    # Insert a new round
    cur.execute("INSERT INTO rounds (prompt, split_id) VALUES (?, ?)", (args.prompt, split_id))
    
    # Create column list for insert query
    columns = [primary_key_field] + [col for col in obfuscated_df.columns]
    placeholders = ", ".join(["?"] * len(columns))
    column_str = ", ".join([f'"{col}"' for col in columns])
    
    # Create insert query
    insert_sql = f'INSERT INTO {table_name} ({column_str}) VALUES ({placeholders})'
    
    # Insert each row
    for index, row in obfuscated_df.iterrows():
        values = [index] + list(row)
        cur.execute(insert_sql, values)
        
        # Handle holdout and validation
        holdout = random.random() < args.holdout
        validation = holdout and (random.random() < args.validation)
        cur.execute("INSERT INTO patient_split (split_id, patient_id, holdout, validation) VALUES (?, ?, ?, ?)",
                   (split_id, index, holdout, validation))
    
    conn.commit()
    
    if args.verbose:
        print(f"Database initialized successfully with table '{table_name}'!")
        print(f"Created split with ID {split_id}")
    else:
        print("Database initialized successfully!")
    
    conn.close()

if __name__ == '__main__':
    main()
