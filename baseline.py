#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from imodels import RuleFitClassifier, BayesianRuleListClassifier, OptimalRuleListClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from datasetconfig import DatasetConfig
from modules.postgres import get_connection, get_investigation_settings
import sys
import os
import json

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train baseline models using dataset configuration.'
    )
    parser.add_argument('--config', help='Path to the dataset configuration JSON file')
    parser.add_argument(
        '--dsn',
        help='PostgreSQL DSN (defaults to libpq environment variables)'
    )
    parser.add_argument('--pg-config', help='JSON file containing postgres_dsn')
    parser.add_argument('--investigation-id', type=int, help='Investigation ID when using PostgreSQL')
    parser.add_argument('--dataset', help='Dataset name when using PostgreSQL without investigation ID')
    parser.add_argument('--output', help='Path for baseline results JSON file')
    return parser.parse_args()

def load_data(conn, config):
    """
    Load data from database tables based on configuration.
    Return training and test datasets.
    """
    table_name = config.table_name
    primary_key = config.primary_key.lower()
    target_field = config.target_field.lower()
    splits_table = config.splits_table

    # Query to get all data with their holdout and validation status
    query = f"""
    SELECT t.*, s.holdout, s.validation
    FROM {table_name} t
    JOIN {splits_table} s ON t.{primary_key} = s.{primary_key}
    """

    # Load data into pandas DataFrame without relying on SQLAlchemy
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=columns)
    df.columns = [c.lower() for c in df.columns]

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
    # Work with lower-case column names for robustness across PostgreSQL
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df.columns = [c.lower() for c in train_df.columns]
    test_df.columns = [c.lower() for c in test_df.columns]

    primary_key = config.primary_key.lower()
    target_field = config.target_field.lower()

    # Remove non-feature columns, ignoring case to avoid issues with
    # configuration mismatches
    non_feature_cols = [primary_key, target_field, 'holdout', 'validation']
    non_feature_lower = {c.lower() for c in non_feature_cols}
    feature_cols = [col for col in train_df.columns if col not in non_feature_lower]

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


def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Train multiple baseline models and return accuracy metrics."""
    from statsmodels.stats.proportion import proportion_confint

    models = {
        'logistic regression': LogisticRegressionCV(max_iter=10000),
        'decision trees': DecisionTreeClassifier(),
        'dummy': DummyClassifier(),
        'RuleFit': RuleFitClassifier(),
        'BayesianRuleList': BayesianRuleListClassifier(),
        # imodels 2.x does not support the ``max_depth`` or ``lambda_``
        # parameters used in earlier versions. Use available arguments
        # to get a comparable small model.
        'CORELS': OptimalRuleListClassifier(max_card=3, n_iter=5000, c=0.05),
        'EBM': ExplainableBoostingClassifier(interactions=10),
    }

    accuracies = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        accuracies[name] = clf.score(X_test, y_test)

    n_test = len(y_test)
    correct_counts = {name: round(acc * n_test) for name, acc in accuracies.items()}

    lower_bounds = {}
    for name, count in correct_counts.items():
        lb, _ = proportion_confint(count=count, nobs=n_test, alpha=0.05, method='beta')
        lower_bounds[name] = lb

    import math
    neg_log_errors = {
        name: -math.log10(1 - lb) if lb < 1 else float('inf')
        for name, lb in lower_bounds.items()
    }

    output = {
        name: {
            'accuracy': accuracies[name],
            'lower_bound': lower_bounds[name],
            'neg_log_error': neg_log_errors[name],
        }
        for name in models
    }

    for name in models:
        print(f"{name}: accuracy = {accuracies[name]:.4f} (95% CI Lower Bound: {lower_bounds[name]:.4f})")

    return output


def main():
    """Main function to execute the baseline model training and evaluation."""
    args = parse_arguments()

    if args.investigation_id is not None:
        conn = get_connection(args.dsn, args.pg_config)
        dataset, config_path = get_investigation_settings(conn, args.investigation_id)
        config = DatasetConfig(conn, config_path, dataset, args.investigation_id)
    else:
        conn = get_connection(args.dsn, args.pg_config)
        if not args.dataset:
            sys.exit("Must specify --dataset")
        if not args.config:
            sys.exit("Must specify --config")
        dataset = args.dataset
        config = DatasetConfig(conn, args.config, dataset)
    # Load and preprocess data
    train_df, test_df = load_data(conn, config)

    if train_df.empty or test_df.empty:
        sys.exit("Error: Training or test dataset is empty")

    X_train, y_train, X_test, y_test, feature_names = preprocess_data(train_df, test_df, config)

    if X_train.size == 0 or X_test.size == 0:
        sys.exit("Error: No usable features found after preprocessing")

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    if dataset:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_results (
                dataset TEXT PRIMARY KEY REFERENCES datasets(dataset),
                logistic_regression DOUBLE PRECISION,
                decision_trees DOUBLE PRECISION,
                dummy DOUBLE PRECISION,
                rulefit DOUBLE PRECISION,
                bayesian_rule_list DOUBLE PRECISION,
                corels DOUBLE PRECISION,
                ebm DOUBLE PRECISION
            )
            """
        )
        cur.execute(
            """
            INSERT INTO baseline_results (
                dataset, logistic_regression, decision_trees, dummy,
                rulefit, bayesian_rule_list, corels, ebm
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (dataset) DO UPDATE SET
                logistic_regression=EXCLUDED.logistic_regression,
                decision_trees=EXCLUDED.decision_trees,
                dummy=EXCLUDED.dummy,
                rulefit=EXCLUDED.rulefit,
                bayesian_rule_list=EXCLUDED.bayesian_rule_list,
                corels=EXCLUDED.corels,
                ebm=EXCLUDED.ebm
            """,
            (
                dataset,
                results['logistic regression']['lower_bound'],
                results['decision trees']['lower_bound'],
                results['dummy']['lower_bound'],
                results['RuleFit']['lower_bound'],
                results['BayesianRuleList']['lower_bound'],
                results['CORELS']['lower_bound'],
                results['EBM']['lower_bound'],
            ),
        )
        conn.commit()

    # Close database connection
    conn.close()
    print("Baseline models trained and evaluated successfully.")

if __name__ == "__main__":
    main()
