#!/usr/bin/env python3

import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import datasetconfig
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze correlation between validation and test scores')
    parser.add_argument('--database', required=True, help='Path to the SQLite database file')
    parser.add_argument('--config', required=True, help='Path to the configuration file')
    parser.add_argument('--task', required=True, help='Task name (for labels and commands)')
    parser.add_argument('--image', required=True, help='Path to save the scatter plot image')
    parser.add_argument('--latex-definitions', required=True, help='Path to save LaTeX definitions')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def generate_scatter_plot(df, task_name, output_path):
    """Generate a scatter plot of validation vs test scores."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with regression line
    sns.set_style('whitegrid')
    ax = sns.regplot(
        x='validation_score', 
        y='test_score', 
        data=df,
        scatter_kws={'alpha': 0.7, 's': 100},
        line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}
    )
    
    # Add correlation coefficient to the plot
    corr = df['validation_score'].corr(df['test_score'])
    ax.annotate(
        f'Pearson r = {corr:.3f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
    )
    
    # Label the points with round IDs
    for i, row in df.iterrows():
        ax.annotate(
            str(row['round_id']),
            (row['validation_score'], row['test_score']),
            fontsize=9,
            xytext=(5, 0),
            textcoords='offset points'
        )
    
    # Add reference line (y=x)
    min_val = min(df['validation_score'].min(), df['test_score'].min()) - 0.05
    max_val = max(df['validation_score'].max(), df['test_score'].max()) + 0.05
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
    
    # Set labels and title
    plt.xlabel('Validation Score', fontsize=14)
    plt.ylabel('Test Score', fontsize=14)
    plt.title(f'{task_name}: Validation vs Test Score Correlation', fontsize=16)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Set axis limits (with slight padding)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def calculate_top_percentile_overlap(validation_scores, test_scores, percentile=90):
    """
    Calculate the probability that a model in the top X percentile of 
    validation scores is also in the top X percentile of test scores.
    """
    # Convert to numpy arrays
    validation = np.array(validation_scores)
    test = np.array(test_scores)
    
    # Calculate the threshold values for the top percentile
    validation_threshold = np.percentile(validation, percentile)
    test_threshold = np.percentile(test, percentile)
    
    # Identify models in the top percentile for validation and test
    top_validation = validation >= validation_threshold
    top_test = test >= test_threshold
    
    # Calculate how many models in the top validation percentile are also in the top test percentile
    if sum(top_validation) == 0:
        return 0.0  # No models in top validation percentile
    
    probability = sum(top_validation & top_test) / sum(top_validation)
    return probability

def generate_latex_definitions(task_name, correlation, top10_probability, output_path):
    """Generate LaTeX definitions for the statistics."""
    # Normalize task name for LaTeX command (replace spaces with underscores, remove special chars)
    latex_task_name = ''.join(c if c.isalnum() else '' for c in task_name.lower())
    
    # Create LaTeX command definitions
    content = f"\\newcommand{{\\{latex_task_name}Correlation}}{{{correlation:.3f}}}\n"
    content += f"\\newcommand{{\\{latex_task_name}TopTenPercent}}{{{top10_probability:.3f}}}\n"
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(content)

def analyze_scores(config, task_name, image_path, latex_path, verbose=False):
    """
    Analyze the correlation between validation and test scores for all rounds.
    
    Args:
        config: DatasetConfig object
        task_name: Name of the task (for labels)
        image_path: Path to save the scatter plot
        latex_path: Path to save LaTeX definitions
        verbose: Whether to print verbose output
    """
    # Get the latest split ID
    split_id = config.get_latest_split_id()
    
    # Get all rounds for this split
    rounds = config.get_processed_rounds_for_split(split_id)
    
    if verbose:
        print(f"Analyzing {len(rounds)} rounds for split ID {split_id}")
    
    # Collect the scores for each round
    data = []
    for round_id in rounds:
        try:
            # Calculate validation score
            validation_matrix = config.get_confusion_matrix(round_id, on_holdout_data=True, on_test_data=False)
            validation_score = config.calculate_metric(validation_matrix, 'accuracy')
            
            # Calculate test score
            test_matrix = config.get_confusion_matrix(round_id, on_holdout_data=True, on_test_data=True)
            test_score = config.calculate_metric(test_matrix, 'accuracy')
            
            data.append({
                'round_id': round_id,
                'validation_score': validation_score,
                'test_score': test_score
            })
            
            if verbose:
                print(f"Round {round_id}: Validation = {validation_score:.4f}, Test = {test_score:.4f}")
                
        except Exception as e:
            print(f"Error processing round {round_id}: {e}", file=sys.stderr)
    
    if not data:
        print("No valid rounds found for analysis", file=sys.stderr)
        sys.exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate correlation
    correlation = df['validation_score'].corr(df['test_score'])
    
    # Calculate top 10% overlap
    top10_probability = calculate_top_percentile_overlap(
        df['validation_score'], 
        df['test_score'], 
        percentile=90
    )
    
    if verbose:
        print(f"Correlation between validation and test scores: {correlation:.4f}")
        print(f"Probability that a top 10% validation score is also a top 10% test score: {top10_probability:.4f}")
    
    # Generate scatter plot
    generate_scatter_plot(df, task_name, image_path)
    
    # Generate LaTeX definitions
    generate_latex_definitions(task_name, correlation, top10_probability, latex_path)
    
    return correlation, top10_probability

if __name__ == '__main__':
    args = parse_args()
    
    # Connect to the database
    try:
        conn = sqlite3.connect(args.database)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load configuration
    try:
        config = datasetconfig.DatasetConfig(conn, args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        conn.close()
        sys.exit(1)
    
    # Analyze the scores
    try:
        correlation, top10_probability = analyze_scores(
            config, 
            args.task, 
            args.image, 
            args.latex_definitions,
            args.verbose
        )
        
        print(f"Analysis complete for {args.task}")
        print(f"Correlation: {correlation:.3f}")
        print(f"Top 10% overlap probability: {top10_probability:.3f}")
        print(f"Scatter plot saved to: {args.image}")
        print(f"LaTeX definitions saved to: {args.latex_definitions}")
        
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()
