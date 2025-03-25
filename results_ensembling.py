#!/usr/bin/env python

import argparse
import sqlite3
import pandas as pd
import os
import itertools
import datasetconfig
from typing import List, Dict, Tuple
import sys
from statsmodels.stats.proportion import proportion_confint

def get_predictions_for_round(config, round_id):
    """Get predictions and ground truth for a specific round."""
    conn = config.conn
    cur = conn.cursor()
    
    # Get the split ID for this round
    split_id = config.get_split_id(round_id)
    
    # Query to get test predictions for the specified round_id
    query = f"""
        SELECT m.decodex, m.{config.target_field}, i.prediction
        FROM inferences i
        JOIN {config.table_name} m ON i.{config.primary_key} = m.{config.primary_key}
        JOIN {config.splits_table} s ON (s.{config.primary_key} = i.{config.primary_key})
        WHERE i.round_id = ?
        AND s.split_id = ?
        AND s.holdout = 1
        AND s.validation = 0
    """
    
    cur.execute(query, (int(round_id), int(split_id)))
    rows = cur.fetchall()
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['decodex', 'ground_truth', 'prediction'])
    df.set_index('decodex', inplace=True)
    
    return df


def ensemble_predictions(predictions_list, ensemble_name="ensemble"):
    """Combine predictions from multiple models using majority voting."""
    # Merge all predictions
    merged_df = None
    for i, df in enumerate(predictions_list):
        if merged_df is None:
            merged_df = df[['ground_truth', 'prediction']].copy()
            merged_df.rename(columns={'prediction': f'pred_{i+1}'}, inplace=True)
        else:
            merged_df[f'pred_{i+1}'] = df['prediction']
    
    # Get the prediction columns
    pred_cols = [col for col in merged_df.columns if col.startswith('pred_')]
    
    # Majority voting
    merged_df[ensemble_name] = merged_df[pred_cols].mode(axis=1).iloc[:, 0]
    
    return merged_df


def calculate_metrics(predictions_df, ensemble_name="ensemble"):
    """Calculate accuracy and other metrics for the ensemble."""
    total = len(predictions_df)
    correct = (predictions_df['ground_truth'] == predictions_df[ensemble_name]).sum()
    accuracy = correct / total if total > 0 else 0
    
    # Calculate 95% confidence interval
    lower_bound, upper_bound = proportion_confint(count=correct, nobs=total, alpha=0.05, method='beta')
    
    # Calculate confusion matrix
    unique_classes = sorted(set(predictions_df['ground_truth'].unique()))
    confusion_matrix = {
        'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0
    }
    
    # Assuming binary classification
    if len(unique_classes) == 2:
        positive_class = unique_classes[0]
        for _, row in predictions_df.iterrows():
            true_label = row['ground_truth']
            pred_label = row[ensemble_name]
            
            if true_label == positive_class and pred_label == positive_class:
                confusion_matrix['TP'] += 1
            elif true_label != positive_class and pred_label == positive_class:
                confusion_matrix['FP'] += 1
            elif true_label != positive_class and pred_label != positive_class:
                confusion_matrix['TN'] += 1
            else:  # true_label == positive_class and pred_label != positive_class
                confusion_matrix['FN'] += 1
    
    return {
        'accuracy': accuracy,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'total': total,
        'correct': correct,
        'confusion_matrix': confusion_matrix
    }


def process_database_combinations(config_path, db_paths, k=3, verbose=False, progress_bar=False):
    """Process all combinations of k databases and evaluate ensemble performance."""
    results = []
    
    # Generate all combinations of k databases
    db_combinations = list(itertools.combinations(db_paths, k))
    
    if verbose:
        print(f"Processing {len(db_combinations)} combinations of {k} databases...")

    iterator = db_combinations
    if progress_bar:
        import tqdm
        iterator = tqdm.tqdm(db_combinations)

    for combo in iterator:
        if verbose:
            print(f"\nEvaluating combination: {[db for db in combo]}")
        
        predictions_list = []
        models_info = []

        for db_path in combo:
            # Connect to the database
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            
            # Load the configuration
            config = datasetconfig.DatasetConfig(conn, config_path)
            
            # Get the best round ID
            split_id = config.get_latest_split_id()
            best_round_id = config.get_best_round_id(split_id)
            
            # Get model name from database path
            model_name = os.path.basename(db_path)[:-7]
            
            # Get predictions for the best round
            predictions_df = get_predictions_for_round(config, best_round_id)

            print(f"{db_path=} {predictions_df.shape=}")
            
            if not predictions_df.empty:
                predictions_list.append(predictions_df)
                models_info.append({
                    'db_path': db_path,
                    'model_name': model_name,
                    'best_round_id': best_round_id
                })
                
            if verbose:
                # Calculate individual model accuracy
                correct = (predictions_df['ground_truth'] == predictions_df['prediction']).sum()
                accuracy = correct / len(predictions_df)
                print(f"  {model_name} (Round {best_round_id}): Accuracy = {accuracy:.4f}")
                
            conn.close()
        
        # Check if we have enough predictions
        if len(predictions_list) < k:
            print(f"  Warning: Skipping combination due to insufficient valid predictions ({len(predictions_list)} < {k})")
            continue
        
        # Combine predictions
        ensemble_name = "ensemble"
        ensemble_df = ensemble_predictions(predictions_list, ensemble_name)
        
        # Calculate metrics
        metrics = calculate_metrics(ensemble_df, ensemble_name)
        
        # Add to results
        result = {
            'models': [model['model_name'] for model in models_info],
            'model_rounds': [model['best_round_id'] for model in models_info],
            'accuracy': metrics['accuracy'],
            'lower_bound': metrics['lower_bound'],
            'upper_bound': metrics['upper_bound'],
            'total': metrics['total'],
            'correct': metrics['correct'],
            'confusion_matrix': metrics['confusion_matrix']
        }
        
        results.append(result)
        
        if verbose:
            print(f"  Ensemble accuracy: {metrics['accuracy']:.4f} (95% CI: {metrics['lower_bound']:.4f}-{metrics['upper_bound']:.4f})")
    
    return results


def find_best_combinations(results, top_n=5):
    """Find the top N combinations based on accuracy."""
    # Sort by accuracy (descending)
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    
    # Return top N
    return sorted_results[:top_n]


def format_results_summary(results_list, top_n=5):
    """Format the results into a readable summary."""
    best_combinations = find_best_combinations(results_list, top_n)
    
    summary = f"Top {len(best_combinations)} Ensemble Combinations:\n\n"
    
    for i, result in enumerate(best_combinations):
        summary += f"{i+1}. Ensemble of: {', '.join(result['models'])}\n"
        summary += f"   Accuracy: {result['accuracy']:.4f} (95% CI: {result['lower_bound']:.4f}-{result['upper_bound']:.4f})\n"
        summary += f"   Correct predictions: {result['correct']}/{result['total']}\n"
        
        if all(key in result['confusion_matrix'] for key in ['TP', 'FP', 'TN', 'FN']):
            cm = result['confusion_matrix']
            summary += f"   Confusion Matrix: TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}\n"
        
        summary += "\n"
    
    return summary


def export_results_to_csv(results, output_path):
    """Export the results to a CSV file."""
    # Create a flattened list for DataFrame
    flat_results = []
    for result in results:
        flat_result = {
            'models': ','.join(result['models']),
            'model_rounds': ','.join(map(str, result['model_rounds'])),
            'accuracy': result['accuracy'],
            'lower_bound': result['lower_bound'],
            'upper_bound': result['upper_bound'],
            'total': result['total'],
            'correct': result['correct'],
        }
        
        # Add confusion matrix values if available
        cm = result['confusion_matrix']
        for key in ['TP', 'FP', 'TN', 'FN']:
            if key in cm:
                flat_result[key] = cm[key]
        
        flat_results.append(flat_result)
    
    # Create DataFrame and export
    df = pd.DataFrame(flat_results)
    df.sort_values('accuracy', ascending=False, inplace=True)
    df.to_csv(output_path, index=False)
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ensemble performance using majority voting")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--output", help="Output CSV file path for detailed results")
    parser.add_argument("--summary", help="Output text file path for summary")
    parser.add_argument("--k", type=int, default=3, help="Number of models to ensemble (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--progress-bar", action="store_true", help="Show progress bar")    
    parser.add_argument("database", nargs="+", help="SQLite database files to analyze")
    
    args = parser.parse_args()
    
    if len(args.database) < args.k:
        print(f"Error: At least {args.k} databases are required for {args.k}-ensemble")
        sys.exit(1)
    
    # Process all combinations
    results = process_database_combinations(args.config, [x for x in args.database if 'baseline' not in x], args.k, args.verbose, args.progress_bar)
    
    if not results:
        print("No valid ensemble combinations found.")
        sys.exit(1)
    
    # Create summary
    summary = format_results_summary(results)
    
    # Print or save summary
    if args.summary:
        with open(args.summary, 'w') as f:
            f.write(summary)
        print(f"Summary saved to {args.summary}")
    else:
        print("\n" + summary)
    
    # Export detailed results if requested
    if args.output:
        export_results_to_csv(results, args.output)
        print(f"Detailed results saved to {args.output}")
