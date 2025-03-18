#!/usr/bin/env python3
import argparse
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from env_settings import EnvSettings
import datasetconfig
import fitter

def parse_args():
    parser = argparse.ArgumentParser(description='Generate accuracy distribution for a model')
    parser.add_argument('--env-file', required=True, help='Path to the model env file')
    parser.add_argument('--distribution-image', required=True, help='Output path for distribution histogram')
    parser.add_argument('--fitted-distribution', required=True, help='Output path for fitted distribution summary')
    return parser.parse_args()

def get_round_accuracies(database_path: str, config_path: str) -> List[float]:
    """
    Get accuracy for all rounds except round 1 from the database.
    
    Args:
        database_path: Path to SQLite database
        config_path: Path to config JSON file
        
    Returns:
        List of accuracy values for all rounds except the first one
    """
    conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    config = datasetconfig.DatasetConfig(conn, config_path)
    
    # Get the latest split ID
    split_id = config.get_latest_split_id()
    
    # Get all processed rounds for this split
    all_rounds = config.get_processed_rounds_for_split(split_id)
    
    # Skip round 1 as it's random
    if len(all_rounds) > 0:
        all_rounds = all_rounds[1:]
    
    accuracies = []
    for round_id in all_rounds:
        # Get confusion matrix for non-holdout data (training data)
        matrix = config.get_confusion_matrix(round_id, on_holdout_data=False)
        accuracy = config.calculate_metric(matrix, 'accuracy')
        accuracies.append(accuracy)
    
    conn.close()
    return accuracies

def plot_distribution(accuracies: List[float], output_path: str) -> None:
    """
    Plot histogram of accuracies and save to file.
    
    Args:
        accuracies: List of accuracy values
        output_path: Path to save the histogram image
    """
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Round Accuracies')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def fit_distribution(accuracies: List[float], output_path: str) -> None:
    """
    Fit common distributions to the accuracy values and save the summary.
    
    Args:
        accuracies: List of accuracy values
        output_path: Path to save the fitted distribution summary
    """
    if len(accuracies) < 3:
        with open(output_path, 'w') as f:
            f.write("Not enough data points to fit distributions (minimum 3 required).")
        return
    
    # Convert to numpy array
    data = np.array(accuracies)
    
    # Create a Fitter object 
    f = fitter.Fitter(data)
    
    # Fit the distributions
    f.fit()
    
    # Write the summary to file
    with open(output_path, 'w') as file:
        file.write("Fitted Distribution Summary\n")
        file.write("==========================\n\n")
        file.write(f"Data: {len(accuracies)} accuracy values\n")
        file.write(f"Range: {min(accuracies):.4f} to {max(accuracies):.4f}\n")
        file.write(f"Mean: {np.mean(accuracies):.4f}\n")
        file.write(f"Standard Deviation: {np.std(accuracies):.4f}\n\n")
        
        # Write the fitter summary 
        summary_df = f.summary()
        file.write(str(summary_df))

def main():
    args = parse_args()
    
    # Extract settings from env file
    settings = EnvSettings.from_file(args.env_file)
    
    # Validate settings
    is_valid, error_message = settings.validate()
    if not is_valid:
        print(f"Error: {error_message}")
        return
    
    # Get accuracies for all rounds except round 1
    accuracies = get_round_accuracies(settings.database, settings.config)
    
    if not accuracies:
        print("No accuracies found for rounds after round 1")
        return
    
    # Plot and save distribution histogram
    plot_distribution(accuracies, args.distribution_image)
    print(f"Distribution histogram saved to {args.distribution_image}")
    
    # Fit distributions and save summary
    fit_distribution(accuracies, args.fitted_distribution)
    print(f"Fitted distribution summary saved to {args.fitted_distribution}")

if __name__ == "__main__":
    main()