#!/usr/bin/env python3
import os
import argparse
import json
import csv
import sys
import math
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datasetconfig
from statsmodels.stats.proportion import proportion_confint
from env_settings import EnvSettings
from modules.postgres import get_connection

def parse_args():
    parser = argparse.ArgumentParser(description='Create CSV file from env files')
    parser.add_argument('--task', required=True, help='Task name (e.g., titanic, wisconsin)')
    parser.add_argument('--env-dir', required=True, help='Directory containing env files')
    parser.add_argument('--output', required=True, help='Output path for CSV file')
    parser.add_argument('--model-details', default="model_details.json", help='Path to model details file')
    parser.add_argument('--baseline', help='Path to baseline JSON file with additional columns')
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()

def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(re.findall(r'\w+', text))

def get_model_data(env_file_path: str, task: str, model_details: Dict) -> Optional[Dict]:
    """Get all required data for a model from its env file."""
    # Extract settings from env file
    run_name = os.path.basename(env_file_path)[:-3]
    settings = EnvSettings.from_file(env_file_path)

    # Validate settings
    is_valid, error_message = settings.validate()
    if not is_valid:
        sys.stderr.write(f"Skipping {env_file_path} - {error_message}\n")
        return None

    # Connect to PostgreSQL using libpq environment defaults
    conn = get_connection()
    dataset = os.path.basename(os.path.dirname(env_file_path))
    config = datasetconfig.DatasetConfig(conn, settings.config, dataset)

    # Get the latest split ID
    split_id = config.get_latest_split_id()

    # Get best round ID based on validation accuracy
    try:
        best_round_id = config.get_best_round_id(split_id, 'accuracy')
    except ValueError as e:
        sys.stderr.write(f"Could not get best round for {split_id}: {e}\n")
        return None

    # Get test accuracy for the best round
    test_accuracy = config.get_test_metric_for_best_validation_round(split_id, 'accuracy')

    # Get the prompt and reasoning from the best round and count words
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

    model_size = model_details.get(settings.model, {}).get('parameters', '')

    # Extract model name (simplify if needed)
    model_name = settings.model

    # Get the count of data points
    data_point_count = config.get_data_point_count()

    # Calculate 95% confidence lower bound for accuracy
    count_correct = round(test_accuracy * data_point_count)
    # Using the Clopper-Pearson method to find the lower bound of the 95% confidence interval
    lower_bound, _ = proportion_confint(count=count_correct, nobs=data_point_count, 
                                       alpha=0.05, method='beta')
    
    # Calculate negative log of the 95th percentile error rate
    neg_log_error = -math.log10(1 - lower_bound) if lower_bound < 1 else float('inf')
    
    # Return all required data
    return {
        'Task': task,
        'Model': model_name,
        'Run Name': run_name,
        'Patience': settings.patience,
        'Sampler': settings.sampler,
        'Accuracy': test_accuracy,
        'Accuracy Lower Bound': lower_bound,
        'Neg Log Error': neg_log_error,
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
        sys.stderr.write(f"No data to write to {output_path}\n")
        return

    # Get fieldnames from first data item
    fieldnames = list(data[0].keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    args = parse_args()

    # Load model details
    with open(args.model_details, 'r', encoding='utf-8') as f:
        model_details = json.load(f)
        
    # Load baseline data if provided
    baseline_data = {}
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, 'r', encoding='utf-8') as f:
            baseline_json = json.load(f)
            # Convert to use lower bounds instead of accuracy values
            baseline_data = {
                'logistic regression': baseline_json['logistic regression']['lower_bound'],
                'decision trees': baseline_json['decision trees']['lower_bound'],
                'dummy': baseline_json['dummy']['lower_bound']
            }

    # Get list of env files to process
    env_files = [os.path.join(args.env_dir, f) for f in os.listdir(args.env_dir)
                    if f.endswith('.env')]

    # Process each env file
    results = []
    iterator = env_files
    if args.progress_bar:
        import tqdm
        iterator = tqdm.tqdm(env_files)
    for env_file in iterator:
        if args.progress_bar:
            iterator.set_description(env_file)
        model_data = get_model_data(env_file, args.task, model_details)
        if model_data:
            # Add baseline data as additional columns
            for key, value in baseline_data.items():
                model_data[key] = value
            results.append(model_data)

    # Write results to CSV
    write_csv(results, args.output)
