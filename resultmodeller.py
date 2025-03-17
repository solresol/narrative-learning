#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate, KFold
import matplotlib.pyplot as plt

def load_data(file_paths):
    """Load and concatenate CSV files into a single DataFrame."""
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
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
                'Herdan Coefficient', 'Zipf Coefficient', 'Model Size']
    
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

def train_models(X, y):
    """Train multiple models and report their performance."""
    # Create a column transformer to one-hot encode categorical features
    categorical_features = ['Task']  # Only Task is categorical now
    numeric_features = ['Sampler', 'Rounds', 'Prompt Word Count', 'Reasoning Word Count', 
                         'Cumulative Reasoning Words', 'Herdan Coefficient', 
                         'Zipf Coefficient', 'Model Size']
    
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
        ('regressor', LinearRegression())
    ])
    
    # Polynomial features with Ridge regression
    poly_ridge_model = Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Set up cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
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
    rf_importances = np.zeros(len(feature_names))
    for estimator in rf_cv_results['estimator']:
        rf_importances += estimator.named_steps['regressor'].feature_importances_
    rf_importances /= len(rf_cv_results['estimator'])
    
    # Calculate average coefficients for linear regression across all folds
    lr_coeffs = np.zeros(len(feature_names))
    for estimator in lr_cv_results['estimator']:
        lr_coeffs += estimator.named_steps['regressor'].coef_
    lr_coeffs /= len(lr_cv_results['estimator'])
    
    # Get sample coefficients from the first fold to determine the shape
    sample_poly_coeffs = poly_ridge_cv_results['estimator'][0].named_steps['regressor'].coef_
    
    # Calculate average coefficients for polynomial regression across all folds
    poly_ridge_coeffs = np.zeros(len(sample_poly_coeffs))
    for estimator in poly_ridge_cv_results['estimator']:
        poly_ridge_coeffs += estimator.named_steps['regressor'].coef_
    poly_ridge_coeffs /= len(poly_ridge_cv_results['estimator'])
    
    # Get feature names for polynomial features
    # First get transformed feature names from the preprocessor
    preprocessed_features = list(feature_names)
    
    # Generate polynomial feature names (this is approximate since sklearn doesn't provide this directly)
    poly_features = []
    
    # Add individual features
    poly_features.extend(preprocessed_features)
    
    # Add interaction terms (degree 2)
    for i in range(len(preprocessed_features)):
        for j in range(i, len(preprocessed_features)):
            poly_features.append(f"{preprocessed_features[i]}*{preprocessed_features[j]}")
    
    # Truncate if necessary (in case our calculation doesn't match exactly)
    poly_features = poly_features[:len(poly_ridge_coeffs)]
    
    # Create dictionary of significant coefficients
    significant_coeffs = {}
    for feature, coeff in zip(poly_features, poly_ridge_coeffs):
        if abs(coeff) >= 0.01:  # Only keep coefficients >= 0.01
            significant_coeffs[feature] = coeff
    
    return {
        'Random Forest': {
            'R^2': rf_r2,
            'Importances': dict(zip(feature_names, rf_importances))
        },
        'Linear Regression': {
            'R^2': lr_r2,
            'Coefficients': dict(zip(feature_names, lr_coeffs))
        },
        'Polynomial Ridge': {
            'R^2': poly_ridge_r2,
            'Coefficients': significant_coeffs
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Model Neg Log Error from features in results CSV files')
    parser.add_argument('csv_files', nargs='+', help='CSV files containing results data')
    args = parser.parse_args()
    
    if not args.csv_files:
        print("Please provide at least one CSV file")
        sys.exit(1)
    
    # Load and prepare data
    df = load_data(args.csv_files)
    X, y = prepare_data(df)
    
    print(f"Loaded data from {', '.join(args.csv_files)}")
    print(f"Number of samples: {len(X)}")
    print("Using 5-fold cross-validation to evaluate models")
    
    # Train models and get results
    results = train_models(X, y)
    
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
    
    print("\n=== Polynomial Ridge Regression ===")
    print(f"Cross-validated R^2 Score: {results['Polynomial Ridge']['R^2']:.4f}")
    print("Average Significant Coefficients across folds (abs value >= 0.01):")
    sorted_poly_coeffs = sorted(
        results['Polynomial Ridge']['Coefficients'].items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feature, coeff in sorted_poly_coeffs:
        print(f"  {feature}: {coeff:.4f}")

if __name__ == "__main__":
    main()