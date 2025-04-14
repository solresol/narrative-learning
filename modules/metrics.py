#!/usr/bin/env python3
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd

def calculate_metric(matrix: Dict, metric_name: str) -> float:
    """
    Calculate the specified metric from a confusion matrix.

    Args:
        matrix: Confusion matrix dictionary
        metric_name: Name of the metric to calculate ('count', 'accuracy', 'precision', 'recall', 'f1')

    Returns:
        Float value of the calculated metric

    Raises:
        ValueError: If the metric name is unknown
    """
    tp = matrix['TP']['count']
    fn = matrix['FN']['count']
    fp = matrix['FP']['count']
    tn = matrix['TN']['count']

    if metric_name == 'count':
        return tp + fn + fp + tn
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

def get_matrix_label_for_prediction(ground_truth, prediction, positive_label, negative_label):
    """
    Determine the confusion matrix label for a prediction.
    
    Args:
        ground_truth: The actual label
        prediction: The predicted label
        positive_label: The label considered as positive
        negative_label: The label considered as negative
        
    Returns:
        One of 'TP', 'FP', 'TN', or 'FN'
    """
    if ground_truth == positive_label:
        if ground_truth == prediction:
            return 'TP'
        return 'FN'
    if ground_truth == prediction:
        return 'TN'
    else:
        return 'FP'

def format_confusion_matrix(matrix: Dict, round_id: int, prompt: str, negative_label : str, positive_label : str, show_examples: bool = True) -> str:
    """
    Format a confusion matrix as a printable string.

    Args:
        matrix: Confusion matrix dictionary
        round_id: ID of the round
        prompt: The prompt text for the round
        show_examples: Whether to include examples in the output

    Returns:
        Formatted string representation
    """
    answer = ""
    answer += f"Round ID: {round_id}\n"
    answer += "Prompt used:\n\t"
    answer += prompt.replace('\n', '\n\t')
    answer += "\n\nConfusion Matrix:\n"

    predicted_positive_label = f'Predicted {positive_label}'
    predicted_negative_label = f'Predicted {negative_label}'
    # Layout: rows are Actual values; columns are Predicted
    answer += (f"{'':15s} {predicted_positive_label:20s} {predicted_negative_label:20s}\n")

    # For actual positive
    tp = matrix['TP']['count']
    fn = matrix['FN']['count']
    answer += (f"{'Actual {positive_label}':15s} {tp:20d} {fn:20d}\n")

    # For actual negative
    fp = matrix['FP']['count']
    tn = matrix['TN']['count']
    answer += (f"{'Actual {negative_label}':15s} {fp:20d} {tn:20d}\n")
    answer += "\n"

    # Calculate metrics
    total_count = tp + fn + fp + tn
    accuracy = (tp + tn) / total_count if total_count > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Add metrics to output
    answer += f"Accuracy: {accuracy:.3f}\n"
    answer += f"Precision: {precision:.3f}\n"
    answer += f"Recall: {recall:.3f}\n"
    answer += f"F1 Score: {f1_score:.3f}\n\n"

    # Add examples if requested
    if show_examples:
        for cell in ['TP', 'FN', 'FP', 'TN']:
            examples = matrix[cell]['examples']
            if examples:
                cell_full = {
                    'TP': f"Correctly predicted {positive_label}",
                    'FN': f"Falsely predicted {negative_label} when it should have been {positive_label}",
                    'FP': f"Falsely predicted {positive_label} when it should have been {negative_label}",
                    'TN': f"Correctly predicted {negative_label}"
                }[cell]
                ex = examples[0]
                answer += (f"Examples for {cell_full}: (Correct answer: {ex['outcome']}, What the previous set of rules predicted: {ex['prediction']})\n")
                for ex in examples:
                    answer += (f"  Entity Data:\n{ex['features']}\n")
                answer += "\n"

    return answer

def generate_metrics_data(config, split_id: int, metric: str, data_type: str) -> pd.DataFrame:
    """
    Generate a DataFrame with metrics for all rounds in a split.

    Args:
        config: DatasetConfig instance
        split_id: The split ID to analyze
        metric: The metric to calculate ('accuracy', 'precision', 'recall', 'f1')
        data_type: Type of data to analyze ('train', 'validation', or 'test')

    Returns:
        DataFrame with round_id and metric columns
    """
    rounds = config.get_processed_rounds_for_split(split_id)
    data = []

    for round_id in rounds:
        # Set appropriate flags based on data_type
        on_holdout = data_type in ('validation', 'test')
        on_test_data = data_type == 'test'
        matrix = config.get_confusion_matrix(round_id, on_holdout_data=on_holdout, on_test_data=on_test_data)
        score = calculate_metric(matrix, metric)
        data.append({
            'round_id': round_id,
            'metric': score
        })

    return pd.DataFrame(data)

def create_metrics_dataframe(config, split_id: int, metric: str,
                          data_types: List[str]) -> pd.DataFrame:
    """
    Create a DataFrame with metrics for all specified data types.

    Args:
        config: DatasetConfig instance
        split_id: The split ID to analyze
        metric: The metric to calculate
        data_types: List of data types to include ('train', 'validation', 'test')

    Returns:
        DataFrame with columns for each data type metric
    """
    df = pd.DataFrame({})

    for data_type in data_types:
        if data_type not in ['train', 'validation', 'test']:
            continue

        temp_df = generate_metrics_data(config, split_id, metric, data_type)
        if not temp_df.empty:
            temp_df.set_index('round_id', inplace=True)
            column_name = f"{data_type} {metric}"
            df[column_name] = temp_df['metric']

    return df
