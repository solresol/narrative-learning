#!/usr/bin/env python3
import argparse
import sqlite3
import sys
from common import get_confusion_matrix, get_split_id
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def calculate_metric(matrix: Dict, metric_name: str) -> float:
    """Calculate the specified metric from a confusion matrix."""
    tp = matrix['TP']['count']
    fn = matrix['FN']['count']
    fp = matrix['FP']['count']
    tn = matrix['TN']['count']

    if metric_name == 'accuracy':
        total = tp + fn + fp + tn
        return (tp + tn) / total if total > 0 else 0
    elif metric_name == 'precision':
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    elif metric_name == 'recall':
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    elif metric_name == 'f1':
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

def get_latest_split_id(conn: sqlite3.Connection) -> int:
    """Get the split_id from the most recent round."""
    cur = conn.cursor()
    cur.execute("""
        SELECT split_id 
        FROM rounds 
        ORDER BY round_id DESC 
        LIMIT 1
    """)
    row = cur.fetchone()
    if row is None:
        sys.exit("No rounds found in database")
    split_id = row[0]
    return split_id

def get_rounds_for_split(conn: sqlite3.Connection, split_id: int) -> List[int]:
    """Get all round IDs for a given split_id."""
    cur = conn.cursor()
    cur.execute("""
        SELECT round_id
        FROM rounds
        WHERE split_id = ?
        ORDER BY round_id
    """, (split_id,))
    rounds = [row[0] for row in cur.fetchall()]
    print(f"Here are the rounds for {split_id=}: {rounds}")
    return rounds

def check_early_stopping(conn: sqlite3.Connection, split_id: int, metric: str, 
                        patience: int, on_validation: bool = True) -> bool:
    """
    Check if training should be stopped based on validation performance.
    Returns True if training should stop.
    """
    rounds = get_rounds_for_split(conn, split_id)
    if len(rounds) < patience + 1:
        return False

    # Look at last 'patience' + 1 rounds
    relevant_rounds = rounds[-(patience + 1):]
    oldest_round = relevant_rounds[0]

    # Calculate metric for oldest round
    oldest_matrix = get_confusion_matrix(conn, oldest_round, on_holdout_data=on_validation)
    best_score = calculate_metric(oldest_matrix, metric)

    # Check if any later round beat this score
    for round_id in relevant_rounds[1:]:
        matrix = get_confusion_matrix(conn, round_id, on_holdout_data=on_validation)
        score = calculate_metric(matrix, metric)
        if score > best_score:
            return False

    return True

def generate_metrics_data(conn: sqlite3.Connection, split_id: int, 
                         metric: str, data_type: str) -> pd.DataFrame:
    """
    Generate a DataFrame with metrics for all rounds in a split.
    data_type should be 'train', 'validation', or 'test'
    """
    rounds = get_rounds_for_split(conn, split_id)
    data = []

    for round_id in rounds:
        # Set appropriate flags based on data_type
        on_holdout = data_type in ('validation', 'test')
        on_test_data = data_type = 'test'
        matrix = get_confusion_matrix(conn, round_id, on_holdout_data=on_holdout, on_test_data=on_test_data)
        score = calculate_metric(matrix, metric)
        data.append({
            'round_id': round_id,
            'metric': score
        })

    return pd.DataFrame(data)

def create_chart(df: pd.DataFrame, metric: str, data_type: str, output_file: str):
    """Create and save a line chart of the metric over rounds."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['round_id'], df['metric'], marker='o')
    plt.title(f'{metric.capitalize()} over Rounds ({data_type})')
    plt.xlabel('Round ID')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Report on classification metrics across rounds")
    parser.add_argument('--database', default='titanic_medical.sqlite',
                       help="Path to the SQLite database file")
    parser.add_argument('--split-id', type=int, help="Split ID to analyze")
    parser.add_argument('--metric', default='accuracy',
                       choices=['accuracy', 'precision', 'recall', 'f1'],
                       help="Metric to calculate")
    parser.add_argument('--validation', action='store_true',
                       help="Report on validation data")
    parser.add_argument('--test', action='store_true',
                       help="Report on test data")
    parser.add_argument('--train', action='store_true',
                       help="Report on training data")
    parser.add_argument('--patience', type=int,
                       help="Number of rounds to look back for early stopping")
    parser.add_argument('--csv', type=str,
                       help="Output CSV file path")
    parser.add_argument('--chart', type=str,
                       help="Output chart PNG file path")
    
    args = parser.parse_args()
    
    conn = sqlite3.connect(args.database)
    
    # If split_id not provided, use the most recent
    split_id = args.split_id if args.split_id is not None else get_latest_split_id(conn)
    
    # If no data type specified, default to training data
    if not any([args.validation, args.test, args.train]):
        args.train = True
    
    # Handle early stopping check
    if args.validation and args.patience:
        should_stop = check_early_stopping(conn, split_id, args.metric, args.patience)
        if should_stop:
            print(f"Early stopping triggered: No improvement in {args.patience} rounds")
            sys.exit(1)
    
    # Generate and output metrics for each requested data type
    for data_type in ['train', 'validation', 'test']:
        if getattr(args, data_type):
            df = generate_metrics_data(conn, split_id, args.metric, data_type)
            print(data_type)
            print(df)
            
            # Output CSV if requested
            if args.csv:
                csv_path = args.csv.replace('.csv', f'_{data_type}.csv')
                df.to_csv(csv_path, index=False)
                print(f"CSV output written to {csv_path}")
            
            # Create chart if requested
            if args.chart:
                chart_path = args.chart.replace('.png', f'_{data_type}.png')
                create_chart(df, args.metric, data_type, chart_path)
                print(f"Chart saved to {chart_path}")
            
            # Print current metric value
            current_value = df.iloc[-1]['metric']
            print(f"{data_type.capitalize()} {args.metric}: {current_value:.3f}")

if __name__ == '__main__':
    main()
