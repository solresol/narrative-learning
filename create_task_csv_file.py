#!/usr/bin/env python3
import os
import re
import argparse
import json
import csv
import sys
import sqlite3
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datasetconfig
from statsmodels.stats.proportion import proportion_confint

def parse_args():
    parser = argparse.ArgumentParser(description='Create CSV file from env files')
    parser.add_argument('--task', required=True, help='Task name (e.g., titanic, wisconsin)')
    parser.add_argument('--env-dir', required=True, help='Directory containing env files')
    parser.add_argument('--output', required=True, help='Output path for CSV file')
    parser.add_argument('--model-details', default="model_details.json", help='Path to model details file')
    return parser.parse_args()

def extract_env_settings(env_file_path: str) -> Dict:
    """Extract configuration settings from an env file."""
    settings = {}
    try:
        with open(env_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract database path
        db_match = re.search(r'NARRATIVE_LEARNING_DATABASE=([^\s]+)', content)
        if db_match:
            settings['database'] = db_match.group(1)

        # Extract config path
        config_match = re.search(r'NARRATIVE_LEARNING_CONFIG=([^\s]+)', content)
        if config_match:
            settings['config'] = config_match.group(1)

        # Extract training model
        model_match = re.search(r'NARRATIVE_LEARNING_TRAINING_MODEL=([^\s]+)', content)
        if model_match:
            settings['model'] = model_match.group(1)

        # Extract example count (sampler)
        example_match = re.search(r'NARRATIVE_LEARNING_EXAMPLE_COUNT=(\d+)', content)
        if example_match:
            settings['sampler'] = int(example_match.group(1))

        return settings
    except Exception as e:
        print(f"Error processing env file {env_file_path}: {e}")
        return {}

def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(re.findall(r'\w+', text))

def get_model_data(env_file_path: str, task: str, model_details: Dict) -> Optional[Dict]:
    """Get all required data for a model from its env file."""
    # Extract settings from env file
    settings = extract_env_settings(env_file_path)

    # Skip incomplete env files
    if not all(key in settings for key in ['database', 'config', 'model']):
        sys.exit(f"{env_file_path} is missing required settings")

    # Skip if database doesn't exist
    if not os.path.exists(settings['database']):
        print(f"Skipping {env_file_path} - database {settings['database']} not found")
        return None

    # Skip if config doesn't exist
    if not os.path.exists(settings['config']):
        print(f"Skipping {env_file_path} - config {settings['config']} not found")
        return None

    # Connect to database
    conn = sqlite3.connect(f"file:{settings['database']}?mode=ro", uri=True)
    print("Connecting to",settings['database'])
    config = datasetconfig.DatasetConfig(conn, settings['config'])

    # Get the latest split ID
    split_id = config.get_latest_split_id()

    # Get best round ID based on validation accuracy
    best_round_id = config.get_best_round_id(split_id, 'accuracy')

    # Get test accuracy for the best round
    test_accuracy = config.get_test_metric_for_best_validation_round(split_id, 'accuracy')

    # Get the prompt and reasoning from the best round and count words
    print(f"{config=} {split_id=} {best_round_id=} {test_accuracy=}")
    prompt = config.get_round_prompt(best_round_id)
    reasoning = config.get_round_reasoning(best_round_id)
    prompt_word_count = count_words(prompt)
    reasoning_word_count = count_words(reasoning)
    
    # Calculate total word count up to best round
    total_words = config.get_total_word_count(split_id, best_round_id)
    cumulative_reasoning_word_count = total_words['reasoning_words']
    
    # Calculate Herdan and Zipf's law coefficients
    herdan_result = config.calculate_herdans_law(split_id)
    zipf_result = config.calculate_zipfs_law(split_id)

    # Get model size from model details
    model_size = model_details.get(settings['model'], {}).get('parameters', '')

    # Extract model name (simplify if needed)
    model_name = settings['model']
    if model_name.startswith('gpt-4o'):
        model_name = 'openai'
    elif 'claude' in model_name:
        model_name = 'anthropic'
    elif ':' in model_name:
        model_name = model_name.split(':')[0]

    # Get the count of data points
    data_point_count = config.get_data_point_count()

    # Calculate 95% confidence lower bound for accuracy
    count_correct = round(test_accuracy * data_point_count)
    # Using the Clopper-Pearson method to find the lower bound of the 95% confidence interval
    lower_bound, _ = proportion_confint(count=count_correct, nobs=data_point_count, 
                                       alpha=0.05, method='beta')
    
    # Return all required data
    return {
        'Task': task,
        'Model': model_name,
        'Sampler': settings.get('sampler', 3),
        'Accuracy': test_accuracy,
        'Accuracy Lower Bound': lower_bound,
        'Rounds': best_round_id,
        'Prompt Word Count': prompt_word_count,
        'Reasoning Word Count': reasoning_word_count,
        'Cumulative Reasoning Words': cumulative_reasoning_word_count,
        'Herdan Coefficient': herdan_result['coefficient'],
        'Herdan R-squared': herdan_result['r_squared'],
        'Zipf Coefficient': zipf_result['coefficient'],
        'Zipf R-squared': zipf_result['r_squared'],
        'Model Size': model_size,
        'Data Points': data_point_count
    }


def write_csv(data: List[Dict], output_path: str):
    """Write data to CSV file."""
    if not data:
        print("No data to write to CSV")
        return

    # Get fieldnames from first data item
    fieldnames = list(data[0].keys())

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"CSV file written to {output_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

if __name__ == "__main__":
    args = parse_args()

    # Load model details
    with open(args.model_details, 'r', encoding='utf-8') as f:
        model_details = json.load(f)

    # Get list of env files to process
    env_files = [os.path.join(args.env_dir, f) for f in os.listdir(args.env_dir)
                    if f.endswith('.env')]

    # Process each env file
    results = []
    for env_file in env_files:
        print(f"Processing {env_file}...")
        model_data = get_model_data(env_file, args.task, model_details)
        if model_data:
            results.append(model_data)

    # Write results to CSV
    write_csv(results, args.output)
    print(f"Processed {len(results)} models")
