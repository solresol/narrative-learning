#!/usr/bin/env python

import argparse
import os
import itertools
import pandas as pd
import datasetconfig
from modules.postgres import get_connection
from modules.exceptions import NoProcessedRoundsException
from typing import List, Dict, Tuple, Optional
import sys
from datetime import datetime
from statsmodels.stats.proportion import proportion_confint

def get_predictions_for_round(config, round_id, validation=False, use_decodex=True):
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
    decodex_column = "decodex" if use_decodex else config.primary_key
    
    # Query to get predictions for the specified round_id
    # When validation=True, get validation set data (validation=1)
    # When validation=False, get test set data (holdout=1, validation=0)
    inf_table = f"{config.dataset}_inferences" if config.dataset else "inferences"

    if validation:
        query = f"""
            SELECT m.{decodex_column}, m.{config.target_field}, i.prediction
            FROM {inf_table} i
            JOIN {config.table_name} m ON i.{config.primary_key} = m.{config.primary_key}
            JOIN {config.splits_table} s ON (s.{config.primary_key} = i.{config.primary_key})
            WHERE i.round_id = %s
            AND s.split_id = %s
            AND s.validation = TRUE
        """
    else:
        query = f"""
            SELECT m.{decodex_column}, m.{config.target_field}, i.prediction
            FROM {inf_table} i
            JOIN {config.table_name} m ON i.{config.primary_key} = m.{config.primary_key}
            JOIN {config.splits_table} s ON (s.{config.primary_key} = i.{config.primary_key})
            WHERE i.round_id = %s
            AND s.split_id = %s
            AND s.holdout = TRUE
            AND s.validation = FALSE
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


def process_investigation_combinations(conn, dataset, config_path, investigation_ids, model_names,
                                        k=3, verbose=False, progress_bar=False, use_decodex=True):
    """Process all combinations of ``k`` investigations and evaluate ensemble performance.

    Uses validation data to determine the best combinations, then scores them on test data.

    Args:
        conn: PostgreSQL connection
        dataset: Dataset name for table prefixes
        config_path: Path to the configuration file
        investigation_ids: List of investigation IDs
        model_names: List of model names corresponding to investigations
        k: Number of models to include in each ensemble
        verbose: Whether to print detailed progress
        progress_bar: Whether to show a progress bar
        use_decodex: Whether to use the decodex column for matching predictions
    """
    validation_results = []
    test_results = []

    # Generate all combinations of indices
    indices = list(range(len(investigation_ids)))
    index_combinations = list(itertools.combinations(indices, k))

    if verbose:
        print(f"Processing {len(index_combinations)} combinations of {k} investigations...")

    iterator = index_combinations
    if progress_bar:
        import tqdm
        iterator = tqdm.tqdm(index_combinations)

    for combo_indices in iterator:
        combo_investigations = [investigation_ids[i] for i in combo_indices]
        combo_models = [model_names[i] for i in combo_indices]

        if verbose:
            print(f"\nEvaluating combination: {combo_investigations}")

        val_predictions_list = []
        test_predictions_list = []
        models_info = []

        for i, inv_id in enumerate(combo_investigations):
            # Load the configuration for this investigation
            config = datasetconfig.DatasetConfig(conn, config_path, dataset, inv_id)

            # Get the best round ID
            split_id = config.get_latest_split_id()
            try:
                best_round_id = config.get_best_round_id(split_id)
            except NoProcessedRoundsException:
                if verbose:
                    print(f"  Skipping investigation {inv_id} - no processed rounds")
                continue

            # Investigation/model identifier
            model_name_from_env = combo_models[i]

            # Get validation predictions for determining best combinations
            val_predictions_df = get_predictions_for_round(config, best_round_id, validation=True, use_decodex=use_decodex)

            # Get test predictions for final evaluation
            test_predictions_df = get_predictions_for_round(config, best_round_id, validation=False, use_decodex=use_decodex)

            if verbose:
                print(f"investigation={inv_id} val:{val_predictions_df.shape=} test:{test_predictions_df.shape=}")

            if not val_predictions_df.empty and not test_predictions_df.empty:
                val_predictions_list.append(val_predictions_df)
                test_predictions_list.append(test_predictions_df)
                models_info.append({
                    'investigation_id': inv_id,
                    'model_name_from_env': model_name_from_env,
                    'best_round_id': best_round_id
                })

            if verbose:
                # Calculate individual model accuracy on validation set
                if not val_predictions_df.empty:
                    correct = (val_predictions_df['ground_truth'] == val_predictions_df['prediction']).sum()
                    accuracy = correct / len(val_predictions_df)
                    print(f"  {model_name_from_env} (Round {best_round_id}): Validation Accuracy = {accuracy:.4f}")

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
            'models': [model['model_name_from_env'] for model in models_info],
            'model_names': [model['model_name_from_env'] for model in models_info],
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
            'models': [model['model_name_from_env'] for model in models_info],
            'model_names': [model['model_name_from_env'] for model in models_info],
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




def get_model_release_date(model_name, release_dates_df):
    """Get the release date for a model.
    
    Args:
        model_name: The name of the model
        release_dates_df: DataFrame containing release dates
        
    Returns:
        Release date as a datetime object or None if not found
    """
    if model_name not in release_dates_df['Model Name'].values:
        return None
    
    date_val = release_dates_df.loc[release_dates_df['Model Name'] == model_name, 'Release Date'].iloc[0]
    return pd.to_datetime(date_val).to_pydatetime()


def find_best_combinations_by_date(results, release_dates_df):
    """Find the best combination for each distinct release date.
    
    Args:
        results: List of ensemble results
        release_dates_df: DataFrame containing model release dates
        
    Returns:
        Dictionary mapping release dates to the best ensemble for that date
    """
    # Add release date to each result
    for result in results:
        # Find the max release date among the models in this ensemble
        max_date = None
        for model in result['model_names']:
            date = get_model_release_date(model, release_dates_df)
            if date is None:
                continue
            if max_date is None or date > max_date:
                max_date = date
        result['release_date'] = max_date
    
    # Filter out results with no valid release date
    valid_results = [r for r in results if r['release_date'] is not None]
    
    # Group by date and find best for each date
    best_by_date = {}
    for result in valid_results:
        date_str = result['release_date'].strftime('%Y-%m-%d')
        # For matched_results, the key is 'val_accuracy' instead of 'accuracy'
        accuracy_key = 'val_accuracy' if 'val_accuracy' in result else 'accuracy'
        if date_str not in best_by_date or result[accuracy_key] > best_by_date[date_str][accuracy_key]:
            best_by_date[date_str] = result
    
    return best_by_date


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


def format_validation_test_results_by_date(validation_results, test_results, release_dates_df):
    """Format the validation and test results sorted by release date.
    
    Finds the best ensemble for each distinct release date.
    
    Args:
        validation_results: List of validation results
        test_results: List of test results (matched to validation_results)
        release_dates_df: DataFrame containing model release dates
    """
    # Match validation and test results by index
    matched_results = []
    for i in range(len(validation_results)):
        result = {
            'models': validation_results[i]['models'],
            'model_names': validation_results[i]['model_names'],
            'model_rounds': validation_results[i]['model_rounds'],
            'val_accuracy': validation_results[i]['accuracy'],
            'val_lower_bound': validation_results[i]['lower_bound'],
            'val_upper_bound': validation_results[i]['upper_bound'],
            'test_accuracy': test_results[i]['accuracy'],
            'test_lower_bound': test_results[i]['lower_bound'],
            'test_upper_bound': test_results[i]['upper_bound'],
            'test_total': test_results[i]['total'],
            'test_correct': test_results[i]['correct'],
            'test_confusion_matrix': test_results[i]['confusion_matrix']
        }
        matched_results.append(result)
    
    # Get best combinations by date
    best_by_date = find_best_combinations_by_date(matched_results, release_dates_df)
    
    # Sort by date (most recent first)
    sorted_dates = sorted(best_by_date.keys(), reverse=True)
    
    summary = f"Best Ensemble Combinations by Release Date:\n\n"
    
    for i, date_str in enumerate(sorted_dates):
        result = best_by_date[date_str]
        
        summary += f"{i+1}. Release Date: {date_str}\n"
        summary += f"   Ensemble of: {', '.join(result['models'])}\n"
        summary += f"   Model Names: {', '.join(result['model_names'])}\n"
        summary += f"   Validation Accuracy: {result['val_accuracy']:.4f} (95% CI: {result['val_lower_bound']:.4f}-{result['val_upper_bound']:.4f})\n"
        summary += f"   Test Accuracy: {result['test_accuracy']:.4f} (95% CI: {result['test_lower_bound']:.4f}-{result['test_upper_bound']:.4f})\n"
        summary += f"   Test correct predictions: {result['test_correct']}/{result['test_total']}\n"
        
        if all(key in result['test_confusion_matrix'] for key in ['TP', 'FP', 'TN', 'FN']):
            cm = result['test_confusion_matrix']
            summary += f"   Test Confusion Matrix: TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}\n"
        
        summary += "\n"
    
    return summary


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
        if 'model_names' in val_result:
            summary += f"   Model Names: {', '.join(val_result['model_names'])}\n"
        summary += f"   Validation Accuracy: {val_result['accuracy']:.4f} (95% CI: {val_result['lower_bound']:.4f}-{val_result['upper_bound']:.4f})\n"
        summary += f"   Test Accuracy: {test_result['accuracy']:.4f} (95% CI: {test_result['lower_bound']:.4f}-{test_result['upper_bound']:.4f})\n"
        summary += f"   Test correct predictions: {test_result['correct']}/{test_result['total']}\n"
        
        if all(key in test_result['confusion_matrix'] for key in ['TP', 'FP', 'TN', 'FN']):
            cm = test_result['confusion_matrix']
            summary += f"   Test Confusion Matrix: TP={cm['TP']}, FP={cm['FP']}, TN={cm['TN']}, FN={cm['FN']}\n"
        
        summary += "\n"
    
    return summary


def export_validation_test_results_to_csv(validation_results, test_results, output_path, release_dates_df=None):
    """Export the validation and test results to a CSV file.
    
    Includes both validation and test metrics for each combination.
    
    Args:
        validation_results: List of validation results
        test_results: List of test results (matched to validation_results)
        output_path: Path to save the CSV file
        release_dates_df: Optional DataFrame containing model release dates
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
        
        # Include model names if available
        if 'model_names' in val_result:
            flat_result['model_names'] = ','.join(val_result['model_names'])
        
        # Add release date if release_dates_df is provided
        if release_dates_df is not None and 'model_names' in val_result:
            # Find the max release date among the models in this ensemble
            max_date = None
            for model in val_result['model_names']:
                date = get_model_release_date(model, release_dates_df)
                if date is None:
                    continue
                if max_date is None or date > max_date:
                    max_date = date
            
            if max_date is not None:
                flat_result['release_date'] = max_date.strftime('%Y-%m-%d')
        
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
    
    # Sort by release date (if available) and then by validation accuracy
    if release_dates_df is not None and 'release_date' in df.columns and not df['release_date'].isna().all():
        df.sort_values(['release_date', 'validation_accuracy'], ascending=[False, False], inplace=True)
    else:
        df.sort_values('validation_accuracy', ascending=False, inplace=True)
    
    df.to_csv(output_path, index=False)
    
    return df


def export_best_by_date_to_csv(validation_results, test_results, output_path, release_dates_df):
    """Export the best ensemble for each distinct release date to a CSV file.
    
    Args:
        validation_results: List of validation results
        test_results: List of test results (matched to validation_results)
        output_path: Path to save the CSV file
        release_dates_df: DataFrame containing model release dates
    """
    # Match validation and test results by index
    matched_results = []
    for i in range(len(validation_results)):
        result = {
            'models': validation_results[i]['models'],
            'model_names': validation_results[i]['model_names'],
            'model_rounds': validation_results[i]['model_rounds'],
            'val_accuracy': validation_results[i]['accuracy'],
            'val_lower_bound': validation_results[i]['lower_bound'],
            'val_upper_bound': validation_results[i]['upper_bound'],
            'test_accuracy': test_results[i]['accuracy'],
            'test_lower_bound': test_results[i]['lower_bound'],
            'test_upper_bound': test_results[i]['upper_bound'],
            'test_total': test_results[i]['total'],
            'test_correct': test_results[i]['correct'],
            'test_confusion_matrix': test_results[i]['confusion_matrix']
        }
        matched_results.append(result)
    
    # Get best combinations by date
    best_by_date = find_best_combinations_by_date(matched_results, release_dates_df)
    
    # Create a list for DataFrame
    flat_results = []
    for date_str, result in best_by_date.items():
        flat_result = {
            'release_date': date_str,
            'models': ','.join(result['models']),
            'model_names': ','.join(result['model_names']),
            'model_rounds': ','.join(map(str, result['model_rounds'])),
            'validation_accuracy': result['val_accuracy'],
            'validation_lower_bound': result['val_lower_bound'],
            'validation_upper_bound': result['val_upper_bound'],
            'test_accuracy': result['test_accuracy'],
            'test_lower_bound': result['test_lower_bound'],
            'test_upper_bound': result['test_upper_bound'],
            'test_total': result['test_total'],
            'test_correct': result['test_correct'],
        }
        
        # Add confusion matrix values if available
        cm = result['test_confusion_matrix']
        for key in ['TP', 'FP', 'TN', 'FN']:
            if key in cm:
                flat_result[f'test_{key}'] = cm[key]
        
        flat_results.append(flat_result)
    
    # Create DataFrame and export
    df = pd.DataFrame(flat_results)
    df.sort_values('release_date', ascending=False, inplace=True)
    df.to_csv(output_path, index=False)

    return df


def store_results_in_db(conn, dataset: str, k: int, validation_results, test_results, release_dates_df):
    """Insert ensemble results into PostgreSQL."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ensemble_results (
            dataset TEXT REFERENCES datasets(dataset),
            k INTEGER,
            models TEXT,
            model_names TEXT,
            model_rounds TEXT,
            release_date DATE,
            validation_accuracy DOUBLE PRECISION,
            validation_lower_bound DOUBLE PRECISION,
            validation_upper_bound DOUBLE PRECISION,
            validation_total INTEGER,
            validation_correct INTEGER,
            test_accuracy DOUBLE PRECISION,
            test_lower_bound DOUBLE PRECISION,
            test_upper_bound DOUBLE PRECISION,
            test_total INTEGER,
            test_correct INTEGER,
            created TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (dataset, k, models)
        )
        """
    )

    for val_res, test_res in zip(validation_results, test_results):
        max_date = None
        if release_dates_df is not None:
            for model in val_res.get('model_names', []):
                date = get_model_release_date(model, release_dates_df)
                if date is not None and (max_date is None or date > max_date):
                    max_date = date

        cur.execute(
            """
            INSERT INTO ensemble_results (
                dataset, k, models, model_names, model_rounds, release_date,
                validation_accuracy, validation_lower_bound, validation_upper_bound,
                validation_total, validation_correct,
                test_accuracy, test_lower_bound, test_upper_bound,
                test_total, test_correct
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (dataset, k, models) DO UPDATE SET
                model_names=EXCLUDED.model_names,
                model_rounds=EXCLUDED.model_rounds,
                release_date=EXCLUDED.release_date,
                validation_accuracy=EXCLUDED.validation_accuracy,
                validation_lower_bound=EXCLUDED.validation_lower_bound,
                validation_upper_bound=EXCLUDED.validation_upper_bound,
                validation_total=EXCLUDED.validation_total,
                validation_correct=EXCLUDED.validation_correct,
                test_accuracy=EXCLUDED.test_accuracy,
                test_lower_bound=EXCLUDED.test_lower_bound,
                test_upper_bound=EXCLUDED.test_upper_bound,
                test_total=EXCLUDED.test_total,
                test_correct=EXCLUDED.test_correct,
                created=CURRENT_TIMESTAMP
            """,
            (
                dataset,
                k,
                ','.join(val_res['models']),
                ','.join(val_res.get('model_names', [])),
                ','.join(map(str, val_res['model_rounds'])),
                max_date.strftime('%Y-%m-%d') if max_date else None,
                val_res['accuracy'],
                val_res['lower_bound'],
                val_res['upper_bound'],
                val_res['total'],
                val_res['correct'],
                test_res['accuracy'],
                test_res['lower_bound'],
                test_res['upper_bound'],
                test_res['total'],
                test_res['correct'],
            )
        )

    conn.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate ensemble performance using majority voting")
    parser.add_argument('dataset', help='Dataset name from the datasets table')
    parser.add_argument('--dsn', help='PostgreSQL connection string')
    parser.add_argument('--pg-config', help='JSON file containing postgres_dsn')
    parser.add_argument('--output', help='Output CSV file path for detailed results')
    parser.add_argument('--summary', help='Output text file path for summary')
    parser.add_argument('--k', type=int, default=3, help='Number of models to ensemble (default: 3)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--progress-bar', action='store_true', help='Show progress bar')
    parser.add_argument('--no-decodex', action='store_false', dest='use_decodex', help='Normally we use the decodex column, but for synthetic data there is no decodex')

    args = parser.parse_args()

    conn = get_connection(args.dsn, args.pg_config)
    cur = conn.cursor()

    cur.execute('SELECT config_file FROM datasets WHERE dataset = %s', (args.dataset,))
    row = cur.fetchone()
    if row is None:
        sys.exit(f"Dataset {args.dataset} not found")
    config_path = row[0]

    cur.execute(
        """
        SELECT i.id, m.training_model
          FROM investigations i
          JOIN models m ON i.model = m.model
         WHERE i.dataset = %s
         ORDER BY i.id
        """,
        (args.dataset,),
    )
    rows = cur.fetchall()
    if not rows:
        sys.exit(f"No investigations found for dataset {args.dataset}")

    investigation_ids = [r[0] for r in rows]
    model_names = [r[1] for r in rows]

    release_dates_df = None
    try:
        cur.execute('SELECT training_model, release_date FROM language_models')
        date_rows = cur.fetchall()
        if date_rows:
            release_dates_df = pd.DataFrame(date_rows, columns=['Model Name', 'Release Date'])
            if args.verbose:
                print(f"Loaded {len(release_dates_df)} release dates from database")
    except Exception as e:
        print(f"Warning: Failed to load release dates from database: {e}")

    if release_dates_df is not None:
        valid_invs = []
        valid_models = []
        for inv_id, name in zip(investigation_ids, model_names):
            if name in release_dates_df['Model Name'].values:
                valid_invs.append(inv_id)
                valid_models.append(name)
            else:
                if args.verbose:
                    print(f"Skipping model {name} - not found in release dates")
        investigation_ids = valid_invs
        model_names = valid_models

    if len(investigation_ids) < args.k:
        sys.exit(f"At least {args.k} investigations are required for {args.k}-ensemble, but only {len(investigation_ids)} found")

    validation_results, test_results = process_investigation_combinations(
        conn,
        args.dataset,
        config_path,
        investigation_ids,
        model_names,
        args.k,
        args.verbose,
        args.progress_bar,
        args.use_decodex,
    )
    
    if not validation_results or not test_results:
        print("No valid ensemble combinations found.")
        sys.exit(1)
    
    # Create summaries
    if release_dates_df is not None:
        # Create summary based on release dates
        summary = format_validation_test_results_by_date(validation_results, test_results, release_dates_df)
    else:
        # Create traditional summary
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
        if release_dates_df is not None:
            # Export best by date
            export_best_by_date_to_csv(validation_results, test_results, args.output, release_dates_df)
        else:
            # Export traditional results
            export_validation_test_results_to_csv(validation_results, test_results, args.output)
        print(f"Detailed results saved to {args.output}")

    # Persist results to PostgreSQL
    store_results_in_db(conn, args.dataset, args.k, validation_results, test_results, release_dates_df)

    conn.close()
