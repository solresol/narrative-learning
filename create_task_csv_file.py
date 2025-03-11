#!/usr/bin/env python3
import os
import re
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description='Create CSV file from result files')
    parser.add_argument('--task', required=True, help='Task name (e.g., titanic)')
    parser.add_argument('--env-dir', required=True, help='Directory containing env files')
    parser.add_argument('--results-files', nargs='+', required=True, help='Path to result files')
    parser.add_argument('--model-details', default="model_details.json", help='Path to model details file')
    parser.add_argument('--output', required=True, help='Output path for CSV file')
    return parser.parse_args()


def read_file_content(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def count_words(text: str) -> int:
    """Count the number of words in a text."""
    # Split on whitespace and filter out empty strings
    words = [word for word in re.split(r'\s+', text) if word]
    return len(words)


def extract_env_data(env_file: str) -> Dict:
    """Extract relevant data from an env file."""
    content = read_file_content(env_file)
    data = {}
    
    # Extract example count
    example_count_match = re.search(r'NARRATIVE_LEARNING_EXAMPLE_COUNT=(\d+)', content)
    data['sampler'] = int(example_count_match.group(1)) if example_count_match else 3
    
    # Extract training model
    model_match = re.search(r'NARRATIVE_LEARNING_TRAINING_MODEL=([^\s]+)', content)
    data['model_name'] = model_match.group(1) if model_match else ""
    
    return data


def parse_results_files(results_files: List[str], env_dir: str, model_details: Dict) -> List[Dict]:
    """Parse results files and extract relevant data."""
    # Group files by base name (e.g., titanic_medical-anthropic-10example)
    file_groups = {}
    for filepath in results_files:
        filename = os.path.basename(filepath)
        # Extract the base name (everything before the first period)
        match = re.match(r'(.+?)\.([^.]+)\.txt$', filename)
        if not match:
            continue
            
        base_name, file_type = match.groups()
        if base_name not in file_groups:
            file_groups[base_name] = {}
        file_groups[base_name][file_type] = filepath
    
    results = []
    for base_name, files in file_groups.items():
        # Skip baseline files or any others that don't match our pattern
        if 'baseline' in base_name or 'dectree' in base_name:
            continue
            
        # Extract model name from base name
        # Format is typically: task_subtype-model-variant
        model_match = re.search(r'[^-]+-([^-]+)(?:-([^.]+))?', base_name)
        if not model_match:
            continue
            
        model_base = model_match.group(1)
        variant = model_match.group(2) if model_match.group(2) else ""
        
        # Determine which env file to use
        env_file_name = None
        if "10example" in base_name or "10examples" in base_name:
            env_file_name = f"{model_base}10.env"
        elif "o1-10example" in base_name or "o1-10examples" in base_name:
            env_file_name = "openai-o1-10.env"
        else:
            env_file_name = f"{model_base}.env"
            
        env_file_path = os.path.join(env_dir, env_file_name)
        
        # Check if env file exists
        if not os.path.exists(env_file_path):
            print(f"Warning: Env file {env_file_path} not found, trying alternatives")
            # Try to find a matching env file
            for file in os.listdir(env_dir):
                if model_base in file.lower():
                    env_file_path = os.path.join(env_dir, file)
                    print(f"Using alternative env file: {env_file_path}")
                    break
        
        # If we still don't have an env file, skip this result
        if not os.path.exists(env_file_path):
            print(f"Warning: No env file found for {base_name}, skipping")
            continue
            
        # Extract data from env file
        env_data = extract_env_data(env_file_path)
        
        # Get accuracy from estimate.txt
        accuracy = ""
        if 'estimate' in files:
            accuracy = read_file_content(files['estimate'])
            
        # Get rounds from best-round.txt
        rounds = ""
        if 'best-round' in files:
            rounds = read_file_content(files['best-round'])
            
        # Get prompt word count from decoded-best-prompt.txt
        prompt_word_count = 0
        if 'decoded-best-prompt' in files:
            prompt_text = read_file_content(files['decoded-best-prompt'])
            prompt_word_count = count_words(prompt_text)
            
        # Get model size from model details
        model_size = ""
        if env_data['model_name'] in model_details:
            model_size = model_details[env_data['model_name']]['parameters']
        
        # Simplify the model name for the CSV
        simple_model = model_base
        if "o1" in base_name:
            simple_model = "openai-o1"
            
        # Create result entry
        result = {
            'Task': args.task,
            'Model': simple_model,
            'Sampler': env_data['sampler'],
            'Accuracy': accuracy,
            'Rounds': rounds,
            'Prompt Word Count': prompt_word_count,
            'Model Size': model_size
        }
        
        results.append(result)
        
    return results


def load_model_details(model_details_path: str) -> Dict:
    """Load model details from JSON file."""
    with open(model_details_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    model_details = load_model_details(args.model_details)
    
    # Parse results files
    results = parse_results_files(args.results_files, args.env_dir, model_details)
    
    # Write to CSV
    write_csv(results, args.output)
