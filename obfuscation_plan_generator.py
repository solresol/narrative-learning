#!/usr/bin/env python3

import pandas
import openai
import json
import sys
import argparse
import os
import sqlite3

save_obfuscation_plan_documentation = {
    "type": "function",
    "function": {
        "name": "save_obfuscation_plan",
        "description": "Save the obfuscation plan",
        "parameters": {
            "type": "object",
            "properties": {
                "primary_key": {
                    "type": "string",
                    "description": "The name of the primary key column in the original dataset."
                },
                "original_dataset_type": {
                    "type": "string",
                    "description": "A generic name for the nature of the original dataset (e.g., 'medical')."
                },
                "obfuscated_dataset_type": {
                    "type": "string",
                    "description": "A generic name for the obfuscated dataset (e.g., 'planetary nebular')."
                },
                "target_variable": {
                    "type": "string",
                    "description": "The name of the target variable in the original dataset."
                },
                "columns": {
                    "type": "array",
                    "description": "List of column transformations.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "original_column": {
                                "type": "string",
                                "description": "The name of the column in the original dataset."
                            },
                            "remove": {
                                "type": "boolean",
                                "description": "Whether the column is removed altogether."
                            },
                            "obfuscated_column": {
                                "type": "string",
                                "description": "The name of the column in the obfuscated dataset.",
                                "nullable": True
                            },
                            "transformation": {
                                "type": "string",
                                "description": "Python code for transforming the original column into the obfuscated form. It assumes a pandas DataFrame where 'original' is the input DataFrame and 'obfuscated' is the output dataframe.",
                                "nullable": True
                            }
                        },
                    }
                }
            }
        }
    }
}


def create_database_schema(db_path):
    """Create the SQLite database schema for storing the obfuscation plan."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table for the obfuscation plan metadata
    cursor.execute('''
    CREATE TABLE obfuscation_metadata (
        id INTEGER PRIMARY KEY,
        primary_key TEXT NOT NULL,
        original_dataset_type TEXT NOT NULL,
        obfuscated_dataset_type TEXT NOT NULL,
        target_variable TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create table for the column transformations
    cursor.execute('''
    CREATE TABLE column_transformations (
        id INTEGER PRIMARY KEY,
        obfuscation_id INTEGER NOT NULL,
        original_column TEXT NOT NULL,
        remove BOOLEAN NOT NULL,
        obfuscated_column TEXT,
        transformation TEXT,
        FOREIGN KEY (obfuscation_id) REFERENCES obfuscation_metadata(id)
    )
    ''')
    
    conn.commit()
    conn.close()

def save_obfuscation_to_db(db_path, obfuscation_json):
    """Save the obfuscation plan to the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert obfuscation metadata
    cursor.execute('''
    INSERT INTO obfuscation_metadata (
        primary_key, 
        original_dataset_type, 
        obfuscated_dataset_type, 
        target_variable
    ) VALUES (?, ?, ?, ?)
    ''', (
        obfuscation_json['primary_key'],
        obfuscation_json['original_dataset_type'],
        obfuscation_json['obfuscated_dataset_type'],
        obfuscation_json['target_variable']
    ))
    
    # Get the ID of the inserted metadata
    obfuscation_id = cursor.lastrowid
    
    # Insert column transformations
    for column in obfuscation_json['columns']:
        cursor.execute('''
        INSERT INTO column_transformations (
            obfuscation_id,
            original_column,
            remove,
            obfuscated_column,
            transformation
        ) VALUES (?, ?, ?, ?, ?)
        ''', (
            obfuscation_id,
            column['original_column'],
            column['remove'],
            column.get('obfuscated_column'),  # Using get() to handle None values
            column.get('transformation')      # Using get() to handle None values
        ))
    
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", required=True, help="The data file you want obfuscated")
    parser.add_argument("--obfuscation-plan", required=True, help="Where we store the obfuscation plan")
    parser.add_argument("--guidelines", required=True, help="The instructions to give the LLM about how to do the obfuscation")
    args = parser.parse_args()
    client = openai.OpenAI(api_key=open(os.path.expanduser("~/.openai.key")).read().strip())
    guidelines = open(args.guidelines).read()
    csv_file = pandas.read_csv(args.csv_file)
    prompt = f"{guidelines}\nFor reference, here are the columns and their types:\n{csv_file.dtypes}"
    
    messages = [
        {
            "role": "system",
            "content": "You are an assistant tasked with obfuscating a dataset. This is the first stage in a pipeline to test out a new machine learning algorithm that uses LLMs to do 'narrative learning'. Unfortunately any famous dataset is easy for an LLM to predict, so we need to turn famous datasets into other equivalent forms that are different enough that the LLM can't learn from it."
        },
        {
            "role": "user",
            "content": open(args.guidelines).read()
        }
    ]


    # Call the OpenAI API with function calling.
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        tools=[save_obfuscation_plan_documentation],
        tool_choice={'type': 'function', 'function': {"name": "save_obfuscation_plan"}},
        temperature=0
    )


    # Debug print (can remove when it is not longer needed)
    print(response)

    obfuscation_json = json.loads(response.choices[0].message.tool_calls[0].function.arguments)

    # Another debug print (should remove when it is no longer needed)
    print(json.dumps(obfuscation_json, indent=2))
    create_database_schema(args.obfuscation_plan)
    save_obfuscation_to_db(args.obfuscation_plan, obfuscation_json)
    
    print(f"Obfuscation plan successfully saved to {args.obfuscation_plan}")
