#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import sys
from modules.postgres import get_connection
from modules.results_loader import load_results_dataframe
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate, KFold
import matplotlib.pyplot as plt

def load_data(conn, datasets):
    """Load and concatenate results from the database for multiple datasets."""
    cur = conn.cursor()
    dfs = []
    for ds in datasets:
        cur.execute("SELECT config_file FROM datasets WHERE dataset = %s", (ds,))
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"dataset {ds} not found")
        df = load_results_dataframe(conn, ds, row[0])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def prepare_data(df):
    """Prepare data for modeling by handling categorical features."""
    # Drop rows with missing or invalid target values
    df = df.dropna(subset=['Neg Log Error'])
    df = df[df['Neg Log Error'] >= 0]  # Filter out negative log error
    
    # Select features
    features = ['Task', 'Sampler', 'Rounds', 'Prompt Word Count', 
                'Reasoning Word Count', 'Cumulative Reasoning Words',
                'Herdan Coefficient', 'Zipf Coefficient', 'Model Size', 'Patience']
    
    # Drop rows with NaN in any of the feature columns
    df = df.dropna(subset=features)
    
    # Create X (features) and y (target)
    X = df[features]
    y = df['Neg Log Error']
    
    # Convert Sampler to a numeric value (it's a count, not a category)
    X['Sampler'] = pd.to_numeric(X['Sampler'], errors='coerce')
    
    # Handle Model Size as numeric
    X['Model Size'] = pd.to_numeric(X['Model Size'], errors='coerce')
    X['Model Size'] = X['Model Size'].fillna(X['Model Size'].median())
    
    return X, y

def train_models(X, y, n_splits=5):
    """Train multiple models and report their performance."""
    # Create a column transformer to one-hot encode categorical features
    categorical_features = ['Task']  # Only Task is categorical now
    numeric_features = ['Sampler', 'Rounds', 'Prompt Word Count', 'Reasoning Word Count', 
                         'Cumulative Reasoning Words', 'Herdan Coefficient', 
                         'Zipf Coefficient', 'Model Size', 'Patience']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])
    
    # Random Forest model
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Linear Regression model
    lr_model = Pipeline([
        ('preprocessor', preprocessor),
        #('regressor', LinearRegression())
        ('regressor', RidgeCV())
    ])
    
    # Polynomial features with Ridge regression
    poly_ridge_model = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Set up cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation for each model
    rf_cv_results = cross_validate(rf_model, X, y, cv=cv, 
                                  scoring='r2', 
                                  return_estimator=True)
    lr_cv_results = cross_validate(lr_model, X, y, cv=cv, 
                                  scoring='r2',
                                  return_estimator=True)
    poly_ridge_cv_results = cross_validate(poly_ridge_model, X, y, cv=cv, 
                                         scoring='r2',
                                         return_estimator=True)
    
    # Calculate average R^2 scores across folds
    rf_r2 = np.mean(rf_cv_results['test_score'])
    lr_r2 = np.mean(lr_cv_results['test_score'])
    poly_ridge_r2 = np.mean(poly_ridge_cv_results['test_score'])
    
    # Get feature names from the first CV fold model
    ohe = rf_cv_results['estimator'][0].named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([cat_feature_names, numeric_features])
    
    # Calculate average feature importances across all folds
    rf_importances = pd.Series(data=np.zeros(len(feature_names)), index=feature_names)
    for estimator in rf_cv_results['estimator']:
        these_cat_feature_names = estimator.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
        these_feature_names = np.concatenate([these_cat_feature_names, numeric_features])
        these_importances = pd.Series(index=these_feature_names, data = estimator.named_steps['regressor'].feature_importances_)
        rf_importances += these_importances
    rf_importances /= len(rf_cv_results['estimator'])
    
    # Calculate average coefficients for linear regression across all folds
    lr_coeffs = pd.Series(data=np.zeros(len(feature_names)), index=feature_names)
    for estimator in lr_cv_results['estimator']:
        these_cat_feature_names = estimator.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
        these_feature_names = np.concatenate([these_cat_feature_names, numeric_features])
        these_importances = pd.Series(index=these_feature_names, data = estimator.named_steps['regressor'].coef_)
        lr_coeffs += these_importances
    lr_coeffs /= len(lr_cv_results['estimator'])
    
    return {
        'Random Forest': {
            'R^2': rf_r2,
            'Importances': rf_importances.to_dict()
        },
        'Linear Regression': {
            'R^2': lr_r2,
            'Coefficients': lr_coeffs.to_dict()
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Model Neg Log Error from features stored in the database')
    parser.add_argument("--remove-extremely-bad", action="store_true", help="Remove narrative learning results that weren't even a match for the dummy regressor")
    parser.add_argument("--crossval", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument('datasets', nargs='+', help='Datasets to include')
    args = parser.parse_args()
    
    conn = get_connection()
    # Load and prepare data
    df = load_data(conn, args.datasets)
    
    # Ignore the really bad models
    if args.remove_extremely_bad:
        df = df[df['Neg Log Error'] > df['dummy']]
    
    print(df.columns)
    X, y = prepare_data(df)
    
    print(f"Loaded data for {', '.join(args.datasets)}")
    print(f"Number of samples: {len(X)}")
    print(f"Using {args.crossval}-fold cross-validation to evaluate models")
    
    # Train models and get results
    results = train_models(X, y, args.crossval)
    
    # Print results
    print("\n=== Random Forest ===")
    print(f"Cross-validated R^2 Score: {results['Random Forest']['R^2']:.4f}")
    print("Average Feature Importances across folds:")
    sorted_importances = sorted(
        results['Random Forest']['Importances'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    for feature, importance in sorted_importances:
        print(f"  {feature}: {importance:.4f}")
    
    print("\n=== Linear Regression ===")
    print(f"Cross-validated R^2 Score: {results['Linear Regression']['R^2']:.4f}")
    print("Average Coefficients across folds:")
    sorted_coeffs = sorted(
        results['Linear Regression']['Coefficients'].items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    for feature, coeff in sorted_coeffs:
        print(f"  {feature}: {coeff:.4f}")
    

if __name__ == "__main__":
    main()
