#!/usr/bin/env python3
import argparse
import sqlite3
import pandas as pd
import uuid
import sys
import random
import os
import json
import pathlib
import numpy as np

def extract_obfuscation_plan(db_path):
    """
    Extract the obfuscation plan from the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the most recent obfuscation plan
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

def generate_schema_sql(obfuscation_plan, table_name, primary_key_field):
    """
    Generate SQL schema based on the obfuscation plan.
    
    Args:
        obfuscation_plan: The obfuscation plan dictionary
        table_name: Name of the main data table
        primary_key_field: Name of the primary key field
    
    Returns:
        SQL string for creating the schema
    """
    # Start with the main data table
    schema_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    schema_sql += f"  {primary_key_field} TEXT primary key,\n"
    
    # Add columns from the obfuscation plan
    columns = []
    for column_info in obfuscation_plan["columns"]:
        if column_info["remove"]:
            continue
            
        col_name = column_info.get("obfuscated_column", column_info["original_column"])
        
        # Try to infer data type from original column name or transformation
        if "integer" in col_name.lower() or "_id" in col_name.lower():
            data_type = "INTEGER"
        elif any(term in col_name.lower() for term in ["date", "time"]):
            data_type = "DATETIME"
        elif any(term in col_name.lower() for term in ["price", "amount", "cost", "rate", "mass", "weight", "height", "months"]):
            data_type = "REAL"
        else:
            data_type = "TEXT"
            
        columns.append(f"  {col_name} {data_type}")
    
    schema_sql += ",\n".join(columns)
    schema_sql += "\n);\n\n"
    
    # Add the standard tables with updated references
    entity_id_field = primary_key_field.split('_')[0].lower()  # e.g., "Patient" from "Patient_ID"
    entity_split_table = f"{entity_id_field}_split"
    
    # Add inferences table with updated references
    schema_sql += f"""create table if not exists inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  {entity_id_field}_id text references {table_name}({primary_key_field}),
  narrative_text text,
  llm_stderr text,
  prediction text,
  primary key (round_id, {entity_id_field}_id)
);\n\n"""

    # Add splits table (unchanged)
    schema_sql += """create table if not exists splits (
  split_id integer primary key autoincrement
);\n\n"""

    # Add entity_split table with updated references
    schema_sql += f"""create table if not exists {entity_split_table} (
  split_id integer references splits(split_id),
  {entity_id_field}_id text references {table_name}({primary_key_field}),
  holdout bool not null default false, -- holdout either for validation or test
  validation bool not null default false,
  primary key (split_id, {entity_id_field}_id)
);\n\n"""

    # Add rounds table (unchanged)
    schema_sql += """create table if not exists rounds (
  round_id integer primary key autoincrement,
  round_start datetime default current_timestamp,
  split_id integer references splits(split_id),
  prompt text,
  reasoning_for_this_prompt text,
  stderr_from_prompt_creation text
);\n"""

    return schema_sql
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
    obfuscated = obfuscated_df
    
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
            print("Running", column_info['transformation'])
            exec(column_info["transformation"])
        else:
            # No transformation, just copy the column
            obfuscated_df[obfuscated_column] = source_df[original_column]
    
    return obfuscated_df

def create_config_file(database_path, table_name, primary_key_field, target_field, entity_id_field, entity_split_table, output_path=None):
    """
    Create a configuration file with essential metadata for other scripts.
    
    Args:
        database_path: Path to the SQLite database
        table_name: Name of the table containing transformed data
        primary_key_field: Name of the primary key field
        target_field: Name of the target variable field
        entity_id_field: Name of the entity ID field (e.g., "patient")
        entity_split_table: Name of the entity split table (e.g., "patient_split")
        output_path: Path to save config file. If None, saves alongside database
    
    Returns:
        Path to the created config file
    """
    config = {
        "database_path": database_path,
        "table_name": table_name,
        "primary_key_field": primary_key_field,
        "target_field": target_field,
        "entity_id_field": entity_id_field,
        "entity_split_table": entity_split_table
    }
    
    if output_path is None:
        # Create config file alongside database
        db_path = pathlib.Path(database_path)
        output_path = db_path.with_suffix('.config.json')
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_path

def main():
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    
    parser = argparse.ArgumentParser(
        description="Import CSV data into database using an obfuscation plan."
    )
    parser.add_argument(
        "--database", default=default_database, required=(default_database is None),
        help="Path to the SQLite database file to create with transformed data."
    )
    parser.add_argument(
        "--schema", default=None,
        help="Optional SQL file containing additional schema commands."
    )
    parser.add_argument(
        "--source", required=True,
        help="CSV source file to transform."
    )
    parser.add_argument(
        "--obfuscation", required=True,
        help="Path to SQLite database containing the obfuscation plan."
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
        "--config-file", default=None,
        help="Path to save the configuration file. If not provided, saves alongside database."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.database is None:
        sys.exit("Must specify a database via --database or NARRATIVE_LEARNING_DATABASE env")
    
    # Extract the obfuscation plan
    obfuscation_plan = extract_obfuscation_plan(args.obfuscation)
    if args.verbose:
        print("Obfuscation plan extracted:")
        print(json.dumps(obfuscation_plan, indent=2))
    
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
    
    # Determine table name and primary key field
    table_name = args.table_name or f"{obfuscation_plan['obfuscated_dataset_type'].lower().replace(' ', '_')}_data"
    primary_key_field = args.primary_key_field or f"{obfuscation_plan['obfuscated_dataset_type'].split()[0]}_ID"
    
    # Apply additional schema if provided
    if args.schema:
        try:
            with open(args.schema, "r", encoding="utf-8") as f:
                additional_schema = f.read()
            conn.executescript(additional_schema)
            if args.verbose:
                print(f"Applied additional schema from '{args.schema}'")
        except Exception as e:
            sys.exit(f"Failed to apply additional schema from '{args.schema}': {e}")
        
    # Generate dynamic schema SQL
    schema_sql = generate_schema_sql(obfuscation_plan, table_name, primary_key_field)
    if args.verbose:
        print("Generated schema SQL:")
        print(schema_sql)
        
    # Initialize the schema
    conn.executescript(schema_sql)
    if args.verbose:
        print("Schema initialized dynamically")
    
    # Generate a unique ID for each row if needed
    
    # Generate a unique ID for each row if needed
    if primary_key_field not in obfuscated_df.columns:
        obfuscated_df[primary_key_field] = [str(uuid.uuid4()) for _ in range(len(obfuscated_df))]
    
    # Set primary key as index
    obfuscated_df.set_index(primary_key_field, inplace=True)
    
    # Extract entity name for config and split table
    entity_id_field = primary_key_field.split('_')[0].lower()  # e.g., "Patient" from "Patient_ID"
    entity_split_table = f"{entity_id_field}_split"
    
    # Create a columns mapping dictionary (used during transformation but not saved to config)
    columns_mapping = {}
    for column_info in obfuscation_plan["columns"]:
        if not column_info["remove"] and column_info.get("obfuscated_column"):
            columns_mapping[column_info["original_column"]] = column_info["obfuscated_column"]
    
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
        cur.execute(f"INSERT INTO {entity_split_table} (split_id, {entity_id_field}_id, holdout, validation) VALUES (?, ?, ?, ?)",
                   (split_id, index, holdout, validation))
    
    conn.commit()
    
    # Create and save configuration file
    target_field = None
    for column_info in obfuscation_plan["columns"]:
        if column_info["original_column"] == obfuscation_plan["target_variable"]:
            target_field = column_info.get("obfuscated_column", column_info["original_column"])
            break
    
    if not target_field:
        print(f"Warning: Target field '{obfuscation_plan['target_variable']}' not found in obfuscation plan")
        target_field = obfuscation_plan["target_variable"]
    
    config_path = create_config_file(
        args.database,
        table_name,
        primary_key_field,
        target_field,
        entity_id_field,
        entity_split_table,
        args.config_file
    )
    
    # Add additional configuration elements needed by other scripts
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get a list of all columns in the transformed dataset for reference
    config["columns"] = [col for col in obfuscated_df.columns]
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.verbose:
        print(f"Database initialized successfully with table '{table_name}'!")
        print(f"Created split with ID {split_id}")
        print(f"Configuration saved to {config_path}")
    else:
        print("Database initialized successfully!")
        print(f"Configuration saved to {config_path}")
    
    conn.close()

if __name__ == '__main__':
    main()
