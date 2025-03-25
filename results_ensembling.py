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

def get_predictions_for_round(config, round_id, validation=False):
    """Get predictions and ground truth for a specific round.
    
    Args:
        config: The dataset configuration.
        round_id: The round ID to get predictions for.
        validation: If True, get validation data; if False, get test data.
    """
    conn = config.conn
    cur = conn.cursor()
    
    # Get the split ID for this round
    split_id = config.get_split_id(round_id)
    
    # Query to get predictions for the specified round_id
    # When validation=True, get validation set data (validation=1)
    # When validation=False, get test set data (holdout=1, validation=0)
    if validation:
        query = f"""
            SELECT m.decodex, m.{config.target_field}, i.prediction
            FROM inferences i
            JOIN {config.table_name} m ON i.{config.primary_key} = m.{config.primary_key}
            JOIN {config.splits_table} s ON (s.{config.primary_key} = i.{config.primary_key})
            WHERE i.round_id = ?
            AND s.split_id = ?
            AND s.validation = 1
        """
    else:
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
    """Process all combinations of k databases and evaluate ensemble performance.
    
    Uses validation data to determine the best combinations, then scores them on test data.
    """
    validation_results = []
    test_results = []
    
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
        
        val_predictions_list = []
        test_predictions_list = []
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
            
            # Get validation predictions for determining best combinations
            val_predictions_df = get_predictions_for_round(config, best_round_id, validation=True)
            
            # Get test predictions for final evaluation
            test_predictions_df = get_predictions_for_round(config, best_round_id, validation=False)

            if verbose:
                print(f"{db_path=} val:{val_predictions_df.shape=} test:{test_predictions_df.shape=}")
            
            if not val_predictions_df.empty and not test_predictions_df.empty:
                val_predictions_list.append(val_predictions_df)
                test_predictions_list.append(test_predictions_df)
                models_info.append({
                    'db_path': db_path,
                    'model_name': model_name,
                    'best_round_id': best_round_id
                })
                
            if verbose:
                # Calculate individual model accuracy on validation set
                if not val_predictions_df.empty:
                    correct = (val_predictions_df['ground_truth'] == val_predictions_df['prediction']).sum()
                    accuracy = correct / len(val_predictions_df)
                    print(f"  {model_name} (Round {best_round_id}): Validation Accuracy = {accuracy:.4f}")
                
            conn.close()
        
        # Check if we have enough predictions
        if len(val_predictions_list) < k or len(test_predictions_list) < k:
            print(f"  Warning: Skipping combination due to insufficient valid predictions (validation: {len(val_predictions_list)}, test: {len(test_predictions_list)})")
            continue
        
        # Combine validation predictions to determine best combinations
        ensemble_name = "ensemble"
        val_ensemble_df = ensemble_predictions(val_predictions_list, ensemble_name)
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(val_ensemble_df, ensemble_name)
        
        # Add to validation results
        val_result = {
            'models': [model['model_name'] for model in models_info],
            'model_rounds': [model['best_round_id'] for model in models_info],
            'accuracy': val_metrics['accuracy'],
            'lower_bound': val_metrics['lower_bound'],
            'upper_bound': val_metrics['upper_bound'],
            'total': val_metrics['total'],
            'correct': val_metrics['correct'],
            'confusion_matrix': val_metrics['confusion_matrix']
        }
        
        validation_results.append(val_result)
        
        # Combine test predictions for final evaluation
        test_ensemble_df = ensemble_predictions(test_predictions_list, ensemble_name)
        
        # Calculate test metrics
        test_metrics = calculate_metrics(test_ensemble_df, ensemble_name)
        
        # Add to test results (with same index as validation results for later matching)
        test_result = {
            'models': [model['model_name'] for model in models_info],
            'model_rounds': [model['best_round_id'] for model in models_info],
            'accuracy': test_metrics['accuracy'],
            'lower_bound': test_metrics['lower_bound'],
            'upper_bound': test_metrics['upper_bound'],
            'total': test_metrics['total'],
            'correct': test_metrics['correct'],
            'confusion_matrix': test_metrics['confusion_matrix']
        }
        
        test_results.append(test_result)
        
        if verbose:
            print(f"  Ensemble validation accuracy: {val_metrics['accuracy']:.4f} (95% CI: {val_metrics['lower_bound']:.4f}-{val_metrics['upper_bound']:.4f})")
            print(f"  Ensemble test accuracy: {test_metrics['accuracy']:.4f} (95% CI: {test_metrics['lower_bound']:.4f}-{test_metrics['upper_bound']:.4f})")
    
    return validation_results, test_results


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


def format_validation_test_results_summary(validation_results, test_results, top_n=5):
    """Format the validation and test results into a readable summary.
    
    First finds the top N combinations based on validation accuracy,
    then reports their test accuracies.
    """
    # Sort by validation accuracy (descending)
    sorted_indices = sorted(range(len(validation_results)), 
                           key=lambda i: validation_results[i]['accuracy'],
                           reverse=True)
    
    # Take the top N
    top_indices = sorted_indices[:top_n]
    
    summary = f"Top {len(top_indices)} Ensemble Combinations (selected based on validation accuracy):\n\n"
    
    for i, idx in enumerate(top_indices):
        val_result = validation_results[idx]
        test_result = test_results[idx]
        
        summary += f"{i+1}. Ensemble of: {', '.join(val_result['models'])}\n"
        summary += f"   Validation Accuracy: {val_result['accuracy']:.4f} (95% CI: {val_result['lower_bound']:.4f}-{val_result['upper_bound']:.4f})\n"
        summary += f"   Test Accuracy: {test_result['accuracy']:.4f} (95% CI: {test_result['lower_bound']:.4f}-{test_result['upper_bound']:.4f})\n"
        summary += f"   Test correct predictions: {test_result['correct']}/{test_result['total']}\n"
        
        if all(key in test_result['confusion_matrix'] for key in ['TP', 'FP', 'TN', 'FN']):
            cm = test_result['confusion_matrix']
            summary += f"   Test Confusion Matrix: TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}\n"
        
        summary += "\n"
    
    return summary


def export_validation_test_results_to_csv(validation_results, test_results, output_path):
    """Export the validation and test results to a CSV file.
    
    Includes both validation and test metrics for each combination.
    """
    # Create a flattened list for DataFrame
    flat_results = []
    for i in range(len(validation_results)):
        val_result = validation_results[i]
        test_result = test_results[i]
        
        flat_result = {
            'models': ','.join(val_result['models']),
            'model_rounds': ','.join(map(str, val_result['model_rounds'])),
            'validation_accuracy': val_result['accuracy'],
            'validation_lower_bound': val_result['lower_bound'],
            'validation_upper_bound': val_result['upper_bound'],
            'validation_total': val_result['total'],
            'validation_correct': val_result['correct'],
            'test_accuracy': test_result['accuracy'],
            'test_lower_bound': test_result['lower_bound'],
            'test_upper_bound': test_result['upper_bound'],
            'test_total': test_result['total'],
            'test_correct': test_result['correct'],
        }
        
        # Add confusion matrix values if available
        val_cm = val_result['confusion_matrix']
        test_cm = test_result['confusion_matrix']
        for key in ['TP', 'FP', 'TN', 'FN']:
            if key in val_cm:
                flat_result[f'validation_{key}'] = val_cm[key]
            if key in test_cm:
                flat_result[f'test_{key}'] = test_cm[key]
        
        flat_results.append(flat_result)
    
    # Create DataFrame and export
    df = pd.DataFrame(flat_results)
    df.sort_values('validation_accuracy', ascending=False, inplace=True)
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
    validation_results, test_results = process_database_combinations(
        args.config, 
        [x for x in args.database if 'baseline' not in x], 
        args.k, 
        args.verbose, 
        args.progress_bar
    )
    
    if not validation_results or not test_results:
        print("No valid ensemble combinations found.")
        sys.exit(1)
    
    # Create summary based on validation results, reporting test performance
    summary = format_validation_test_results_summary(validation_results, test_results)
    
    # Print or save summary
    if args.summary:
        with open(args.summary, 'w') as f:
            f.write(summary)
        print(f"Summary saved to {args.summary}")
    else:
        print("\n" + summary)
    
    # Export detailed results if requested
    if args.output:
        export_validation_test_results_to_csv(validation_results, test_results, args.output)
        print(f"Detailed results saved to {args.output}")
