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
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    
    # Get the most recent obfuscation plan. There should only
    # be one, so this is kind of overly-defensive programming.
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
        SELECT primary_key, original_dataset_type, obfuscated_dataset_type, target_variable, obfuscated_table_name, obfuscated_split_table_name
        FROM obfuscation_metadata 
        WHERE id = ?
    """, (obfuscation_id,))
    
    metadata_row = cursor.fetchone()
    if not metadata_row:
        sys.exit(f"Obfuscation plan with ID {obfuscation_id} not found")
    
    primary_key, original_dataset_type, obfuscated_dataset_type, target_variable, obfuscated_table_name, obfuscated_split_table_name = metadata_row
    
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
        "obfuscated_table_name": obfuscated_table_name,
        "obfuscated_split_table_name": obfuscated_split_table_name,
        "columns": columns
    }



def generate_schema_sql(obfuscation_plan, source_df):
    """
    Generate SQL schema based on the obfuscation plan.
    
    Args:
        obfuscation_plan: The obfuscation plan dictionary
    
    Returns:
        SQL string for creating the schema
    """
    # Start with the main data table
    column_lookup = { x['original_column']: x['obfuscated_column'] for x in obfuscation_plan['columns'] }
    obfuscated_table_name = obfuscation_plan['obfuscated_table_name']
    obfuscated_split_table_name = obfuscation_plan['obfuscated_split_table_name']
    original_primary_key = obfuscation_plan['primary_key']
    obfuscated_primary_key = column_lookup[original_primary_key]
    schema_sql = f"CREATE TABLE IF NOT EXISTS {obfuscated_table_name} (\n"
    schema_sql += "  decodex "
    original_primary_type = str(source_df.index.dtype)
    if original_primary_type == 'int64':
        schema_sql += "INTEGER"
    else:
        schema_sql += "TEXT"
    schema_sql += ",\n  "
    
    # Add columns from the obfuscation plan
    columns = []
    for column_info in obfuscation_plan["columns"]:
        if column_info["remove"]:
            continue

        original_name = column_info.get("original_column")
        col_name = column_info.get("obfuscated_column")
        
        primkey = "PRIMARY KEY" if col_name == obfuscated_primary_key else ""
        columns.append(f"  {col_name} text {primkey}".strip())
    
    schema_sql += ",\n  ".join(columns)
    schema_sql += "\n);\n\n"
    
    # Add inferences table with updated references
    schema_sql += f"""create table if not exists inferences (
  round_id integer references rounds(round_id),
  creation_time datetime default current_timestamp,
  {obfuscated_primary_key} text references {obfuscated_table_name}({obfuscated_primary_key}),
  narrative_text text,
  llm_stderr text,
  prediction text,
  primary key (round_id, {obfuscated_primary_key})
);\n\n"""

    # Add splits table (unchanged)
    schema_sql += """create table if not exists splits (
  split_id integer primary key autoincrement
);\n\n"""

    # Add entity_split table with updated references
    schema_sql += f"""create table if not exists {obfuscated_split_table_name} (
  split_id integer references splits(split_id),
  {obfuscated_primary_key} text references {obfuscated_table_name}({obfuscated_primary_key}),
  holdout bool not null default false, -- holdout either for validation or test
  validation bool not null default false,
  primary key (split_id, {obfuscated_primary_key})
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

def apply_transformation(original_df, obfuscation_plan):
    """
    Apply transformations to the original DataFrame according to the obfuscation plan.
    Returns a new DataFrame with the transformed data.
    """
    # Create a new DataFrame to store the transformed data
    obfuscated_df = pd.DataFrame()
    # Keep track of the original DataFrame as it gets modified during transformation
    # (Actually, I don't think it does, but let's be overly safe in case we end up
    # creating a weird bug elsewhere).
    source_df = original_df.copy()

    # Create local variables for use in eval
    # Make an alias for it, because that's what the eval ops do.
    obfuscated = obfuscated_df
    original = source_df
    
    # For each column transformation in the plan
    for column_info in obfuscation_plan["columns"]:
        original_column = column_info["original_column"]
        
        # Skip if column should be removed
        if column_info["remove"]:
            continue
        
        # Get obfuscated column name
        obfuscated_column = column_info.get("obfuscated_column", original_column)

        # Regardless of what they said to do to the primary key column, we
        # are going to splat over it with a uuid
        if original_column == obfuscation_plan['primary_key']:
            continue

        # There better be a transformation...
        result = exec(column_info["transformation"])
    return obfuscated_df

def create_config_file(output_path, obfuscation_plan):
    # Get a list of all columns in the transformed dataset for reference
    column_lookup = { x['original_column']: x['obfuscated_column'] for x in obfuscation_plan['columns'] }

    config = {
        "table_name": obfuscation_plan['obfuscated_table_name'],
        "primary_key": column_lookup[obfuscation_plan['primary_key']],
        "target_field": column_lookup[obfuscation_plan['target_variable']],
        "splits_table": obfuscation_plan['obfuscated_split_table_name']
    }
    config["columns"] = [col['obfuscated_column'] for col in obfuscation_plan['columns'] if not col['remove']]
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_path

def main():
    default_database = os.environ.get('NARRATIVE_LEARNING_DATABASE', None)
    
    parser = argparse.ArgumentParser(description="Import CSV data into database using an obfuscation plan.")
    parser.add_argument("--database", default=default_database, required=(default_database is None),
        help="Path to the SQLite database file to create with transformed data."
    )
    parser.add_argument("--source", required=True, help="CSV source file to transform.")
    parser.add_argument( "--obfuscation", required=True, help="Path to SQLite database containing the obfuscation plan.")
    parser.add_argument("--prompt", default="Choose randomly", help="Prompt to insert into the rounds table (default: 'Choose randomly').")
    parser.add_argument("--holdout", default=0.2, type=float, help="Proportion of the data to mark as held out (never to be used in training)")
    parser.add_argument("--validation", default=0.5, type=float, help="Proportion of the holdout data to use for validation")
    parser.add_argument("--config-file", help="Path to save the configuration file.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.database is None:
        sys.exit("Must specify a database via --database or NARRATIVE_LEARNING_DATABASE env")
    
    # Extract the obfuscation plan
    obfuscation_plan = extract_obfuscation_plan(args.obfuscation)
    if args.verbose:
        print("Obfuscation plan extracted:")
        print(json.dumps(obfuscation_plan, indent=2))
    
    # Load the source CSV
    source_df = pd.read_csv(args.source,index_col=obfuscation_plan['primary_key'])
    if args.verbose:
        print(f"Loaded source data from {args.source} with {len(source_df)} rows")
    
    # Apply transformations
    obfuscated_df = apply_transformation(source_df, obfuscation_plan)
    if args.verbose:
        print(f"Transformations applied, resulting in DataFrame with {len(obfuscated_df)} rows")
    
    # Connect to the database
    conn = sqlite3.connect(args.database)
    if args.verbose:
        print(f"Connected to database '{args.database}'")
    
    column_lookup = { x['original_column']: x['obfuscated_column'] for x in obfuscation_plan['columns'] }
    # Determine table name and primary key field
    obfuscated_table_name = obfuscation_plan['obfuscated_table_name']
    obfuscated_primary_key = column_lookup[obfuscation_plan['primary_key']]
        
    # Generate dynamic schema SQL
    schema_sql = generate_schema_sql(obfuscation_plan, source_df)
    if args.verbose:
        print("Generated schema SQL:")
        print(schema_sql)
        
    # Initialize the schema
    conn.executescript(schema_sql)
    if args.verbose:
            print("Schema initialized dynamically")
    
    # Generate a unique ID for each row if needed
    obfuscated_df[obfuscated_primary_key] = [str(uuid.uuid4()) for _ in range(len(obfuscated_df))]
    
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
    columns = ["decodex"] + [col for col in obfuscated_df.columns]
    placeholders = ", ".join(["?"] * len(columns))
    column_str = ", ".join(columns)
    
    # Create insert query
    insert_sql = f'INSERT INTO {obfuscated_table_name} ({column_str}) VALUES ({placeholders})'

    obfuscated_split_table_name = obfuscation_plan['obfuscated_split_table_name']
    
    # Insert each row
    for index, row in obfuscated_df.iterrows():
        values = [index] + list(row)
        cur.execute(insert_sql, values)
        
        # Handle holdout and validation
        holdout = random.random() < args.holdout
        validation = holdout and (random.random() < args.validation)
        cur.execute(f"INSERT INTO {obfuscated_split_table_name} (split_id, {obfuscated_primary_key}, holdout, validation) VALUES (?, ?, ?, ?)",
                   (split_id, row[obfuscated_primary_key], holdout, validation))
    
    conn.commit()
    
    # Create and save configuration file
    target_field = None
    for column_info in obfuscation_plan["columns"]:
        if column_info["original_column"] == obfuscation_plan["target_variable"]:
            target_field = column_info.get("obfuscated_column", column_info["original_column"])
            break
    
    if not target_field:
        print(f"Warning: Target field '{obfuscation_plan['target_variable']}' not found in obfuscation plan")
        # Default to the original target variable name
        target_field = obfuscation_plan["target_variable"]
    
    if args.verbose:
        print(f"Target field identified as: {target_field}")

    
    config_path = create_config_file(args.config_file, obfuscation_plan)
    
    if args.verbose:
        print(f"Database initialized successfully with table '{obfuscated_table_name}'!")
        print(f"Created splits table {obfuscated_split_table_name}")
    print(f"Configuration saved to {config_path}")
    
    conn.close()

if __name__ == '__main__':
    main()
