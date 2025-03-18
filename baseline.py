#!/usr/bin/env python3

import argparse
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from datasetconfig import DatasetConfig
import sys
import os
import json

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train baseline models using dataset configuration.')
    parser.add_argument('--config', required=True, help='Path to the dataset configuration JSON file')
    parser.add_argument('--database', required=True, help='Path to the SQLite database file')
    parser.add_argument('--output', required=True, help='Path for baseline results JSON file')
    return parser.parse_args()

def load_data(conn, config):
    """
    Load data from database tables based on configuration.
    Return training and test datasets.
    """
    table_name = config.table_name
    primary_key = config.primary_key
    target_field = config.target_field
    splits_table = config.splits_table

    # Query to get all data with their holdout and validation status
    query = f"""
    SELECT t.*, s.holdout, s.validation
    FROM {table_name} t
    JOIN {splits_table} s ON t.{primary_key} = s.{primary_key}
    """

    # Load data into pandas DataFrame
    df = pd.read_sql_query(query, conn)

    # Drop 'decodex' column if it exists
    if 'decodex' in df.columns:
        df = df.drop('decodex', axis=1)

    # Split into training and test sets
    train_df = df[df['holdout'] == 0]
    test_df = df[(df['holdout'] == 1) & (df['validation'] == 0)]

    return train_df, test_df

def preprocess_data(train_df, test_df, config):
    """
    Preprocess data for model training:
    - Handle categorical features with one-hot encoding if needed
    - Prepare feature matrices and target vectors
    """
    primary_key = config.primary_key
    target_field = config.target_field

    # Remove non-feature columns
    non_feature_cols = [primary_key, target_field, 'holdout', 'validation']
    feature_cols = [col for col in train_df.columns if col not in non_feature_cols]

    # Initialize lists to store processed feature data
    X_train_parts = []
    X_test_parts = []
    feature_names = []

    # Process each feature column
    for col in feature_cols:
        train_values = train_df[col].values
        test_values = test_df[col].values

        # Check if the column is numeric
        try:
            # Convert to numeric and check if successful
            train_numeric = pd.to_numeric(train_values)
            test_numeric = pd.to_numeric(test_values)

            # Add as numeric feature
            X_train_parts.append(train_numeric.reshape(-1, 1))
            X_test_parts.append(test_numeric.reshape(-1, 1))
            feature_names.append(col)

        except (ValueError, TypeError):
            # Not numeric, handle as categorical
            distinct_values = train_df[col].nunique()

            if distinct_values < 20:  # One-hot encode only if less than 10 distinct values
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(train_values.reshape(-1, 1))

                train_encoded = encoder.transform(train_values.reshape(-1, 1))
                test_encoded = encoder.transform(test_values.reshape(-1, 1))

                X_train_parts.append(train_encoded)
                X_test_parts.append(test_encoded)

                # Add feature names for each one-hot encoded column
                for category in encoder.categories_[0]:
                    feature_names.append(f"{col}_{category}")
            else:
                print(f"Skipping feature '{col}' with {distinct_values} distinct values (too many to one-hot encode)")

    # Combine all processed features into final feature matrices
    X_train = np.hstack(X_train_parts) if X_train_parts else np.array([])
    X_test = np.hstack(X_test_parts) if X_test_parts else np.array([])

    # Get target vectors
    y_train = train_df[target_field]
    y_test = test_df[target_field]

    return X_train, y_train, X_test, y_test, feature_names

def train_and_evaluate_models(X_train, y_train, X_test, y_test, output_path):
    """
    Train logistic regression and decision tree models.
    Evaluate on test set and save results to specified files.
    Includes 95% confidence lower bound on accuracy.
    """
    from statsmodels.stats.proportion import proportion_confint

    # Train logistic regression model
    lr_model = LogisticRegressionCV()
    lr_model.fit(X_train, y_train)
    lr_accuracy = lr_model.score(X_test, y_test)

    # Train decision tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_accuracy = dt_model.score(X_test, y_test)

    dummy_model = DummyClassifier()
    dummy_model.fit(X_train, y_train)
    dummy_accuracy = dummy_model.score(X_test, y_test)

    # Calculate 95% confidence lower bounds
    n_test = len(y_test)
    lr_correct = round(lr_accuracy * n_test)
    dt_correct = round(dt_accuracy * n_test)
    dummy_correct = round(dummy_accuracy * n_test)

    # Clopper-Pearson method for confidence intervals
    lr_lower_bound, _ = proportion_confint(count=lr_correct, nobs=n_test, alpha=0.05, method='beta')
    dt_lower_bound, _ = proportion_confint(count=dt_correct, nobs=n_test, alpha=0.05, method='beta')
    dummy_lower_bound, _ = proportion_confint(count=dummy_correct, nobs=n_test, alpha=0.05, method='beta')
    
    # Calculate negative log10 of error rates
    import math
    lr_neg_log_error = -math.log10(1 - lr_lower_bound) if lr_lower_bound < 1 else float('inf')
    dt_neg_log_error = -math.log10(1 - dt_lower_bound) if dt_lower_bound < 1 else float('inf')
    dummy_neg_log_error = -math.log10(1 - dummy_lower_bound) if dummy_lower_bound < 1 else float('inf')

    output = {
        'logistic regression': {
            'accuracy': lr_accuracy,
            'lower_bound': lr_lower_bound,
            'neg_log_error': lr_neg_log_error
        },
        'decision trees': {
            'accuracy': dt_accuracy,
            'lower_bound': dt_lower_bound,
            'neg_log_error': dt_neg_log_error
        },
        'dummy': {
            'accuracy': dummy_accuracy,
            'lower_bound': dummy_lower_bound,
            'neg_log_error': dummy_neg_log_error
        }
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Logistic Regression Test Accuracy: {lr_accuracy:.4f} (95% CI Lower Bound: {lr_lower_bound:.4f})")
    print(f"Decision Tree Test Accuracy: {dt_accuracy:.4f} (95% CI Lower Bound: {dt_lower_bound:.4f})")
    print(f"Dummy Accuracy: {dummy_accuracy:.4f} (95% CI Lower Bound: {dummy_lower_bound:.4f})")

def main():
    """Main function to execute the baseline model training and evaluation."""
    args = parse_arguments()

    conn = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    config = DatasetConfig(conn, args.config)
    # Load and preprocess data
    train_df, test_df = load_data(conn, config)

    if train_df.empty or test_df.empty:
        sys.exit("Error: Training or test dataset is empty")

    X_train, y_train, X_test, y_test, feature_names = preprocess_data(train_df, test_df, config)

    if X_train.size == 0 or X_test.size == 0:
        sys.exit("Error: No usable features found after preprocessing")

    # Train and evaluate models
    train_and_evaluate_models(X_train, y_train, X_test, y_test, args.output)

    # Close database connection
    conn.close()
    print("Baseline models trained and evaluated successfully.")

if __name__ == "__main__":
    main()
