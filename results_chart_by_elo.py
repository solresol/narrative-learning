#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import math
import json
import sys
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from chartutils import draw_baselines

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    # Convert accuracy to numeric, handling any missing values
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    return df

def load_elo_data(file_path):
    """Load ELO data from a CSV file."""
    elo_df = pd.read_csv(file_path)
    # Keep only Model and Arena Score columns
    elo_df = elo_df[['Model', 'Arena Score']]
    # Lowercase model names for easier matching
    elo_df['Model'] = elo_df['Model'].str.lower()
    return elo_df

def get_model_elo(model_name, elo_df, translations=None):
    """Get the ELO rating for a model.
    
    Args:
        model_name: The model name from results data
        elo_df: DataFrame with ELO ratings
        translations: Optional dictionary mapping result model names to ELO model names
    
    Returns:
        The ELO rating for the model
        
    Raises:
        ValueError: If the model is not found in ELO data
    """
    # Check if we have a translation for this model
    if translations and model_name in translations:
        elo_name = translations[model_name]
        print(f"Translating '{model_name}' to '{elo_name}' for ELO lookup")
    else:
        elo_name = model_name
    
    # Normalize model name
    normalized_name = elo_name.lower()
    
    # Check if model is in ELO data
    matching_row = elo_df[elo_df['Model'] == normalized_name]
    
    if matching_row.empty:
        # Model not found in ELO data
        raise ValueError(f"Model '{model_name}' (looking for '{normalized_name}' in ELO data) not found")
    
    # Return the ELO score
    return matching_row['Arena Score'].iloc[0]

def plot_elo_vs_log_error(df, output_prefix, dataset_name, pvalue_file=None):
    """Create a plot of ELO rating vs log error rate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a scatter plot with different colors for each model
    sns.scatterplot(x='ELO', y='Neg Log Error',
                   hue='Display_Name',
                   data=df,
                   ax=ax)

    df = df.dropna(subset=['ELO', 'Neg Log Error'])
    
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(df[['ELO']], df['Neg Log Error'])
    
    # Add statsmodels for detailed statistics
    X = sm.add_constant(df[['ELO']])
    model = sm.OLS(df['Neg Log Error'], X).fit()
    print(model.summary())
    
    # Access specific statistics
    slope = model.params['ELO']
    intercept = model.params['const']
    p_value = model.pvalues['ELO']
    r_squared = model.rsquared
    print(f"Slope: {slope:.4f}, p-value: {p_value:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    
    # Save p-value to file if requested
    if pvalue_file:
        with open(pvalue_file, 'w') as f:
            f.write(f"{p_value:.3f}")

    # Add a trend line
    x_range = np.linspace(df['ELO'].min() - 50, df['ELO'].max() + 50, 100)
    ax.plot(x_range, intercept + slope * x_range, linestyle="dashed", color="red", label="Trend")

    # Add title and labels
    ax.set_title(f'{dataset_name} Dataset: ELO Rating vs Error Rate')
    ax.set_xlabel('Arena ELO Rating')
    ax.set_ylabel('Negative Log10 Error Rate')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Draw baseline lines
    draw_baselines(ax, df)

    # Add secondary axis with accuracy percentage
    ax2 = ax.twinx()

    # Create a function to convert negative log error to accuracy
    def neg_log_error_to_accuracy(y):
        # Convert negative log10 error rate to accuracy percentage
        # If y is Neg Log Error_Rate, then error = 10^(-y)
        # And accuracy = 1 - error = 1 - 10^(-y)
        calc = (1 - 10**(-y)) * 100
        print(f"Neg log {y} -> {calc} %")
        return calc

    y1_ticks = ax.get_yticks()
    # Calculate corresponding accuracy values for these specific positions
    y2_ticks = [neg_log_error_to_accuracy(y) for y in y1_ticks]
    ax2.set_yticks(y1_ticks)
    ax2.set_yticklabels([f"{y:.0f}%" for y in y2_ticks])
    y1_min, y1_max = ax.get_ylim()
    ax2.set_ylim(y1_min, y1_max)

    # Set secondary axis label
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='y')

    fig.tight_layout()

    # Save the plot
    fig.savefig(output_prefix)
    print(f"Saved plot to {output_prefix}")

    return output_prefix

def name_model(record):
    """Extract a clean model name from the record."""
    return record['Model']

def calculate_projection(df, elo_threshold=None):
    """
    Calculate parameters needed to beat the best baseline or a specific ELO threshold.
    
    Args:
        df: DataFrame with the data
        elo_threshold: Optional ELO threshold to calculate projection for
        
    Returns:
        Dictionary with projection results
    """
    # Find the best baseline accuracy
    baseline_cols = ['logistic regression', 'decision trees', 'dummy']
    baseline_scores = {col: df[col].mean() for col in baseline_cols if col in df.columns}
    
    if not baseline_scores:
        print("No baseline columns found in the dataset")
        return None
    
    best_baseline = max(baseline_scores.items(), key=lambda x: x[1])
    best_baseline_name, best_baseline_score = best_baseline
    best_baseline_error = 1 - best_baseline_score
    neg_log_best_baseline_error = -math.log10(best_baseline_error)
    
    print(f"Best baseline model: {best_baseline_name} with accuracy {best_baseline_score:.4f}")
    print(f"Best baseline neg log error: {neg_log_best_baseline_error:.4f}")
    
    # Use the linear regression model to estimate ELO needed
    df_clean = df.dropna(subset=['ELO', 'Neg Log Error'])
    if df_clean.empty:
        print("No valid ELO data for projection")
        return None
        
    lr = LinearRegression()
    lr.fit(df_clean[['ELO']], df_clean['Neg Log Error'])
    
    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    # Calculate ELO needed to beat the best baseline
    # neg_log_error = slope * elo + intercept
    # elo = (neg_log_error - intercept) / slope
    elo_needed = (neg_log_best_baseline_error - intercept) / slope
    
    print(f"ELO needed to beat best baseline: {elo_needed:.1f}")
    
    # If a specific ELO threshold is provided, calculate the expected performance
    threshold_projection = None
    if elo_threshold is not None:
        expected_neg_log_error = slope * elo_threshold + intercept
        expected_error = 10 ** (-expected_neg_log_error)
        expected_accuracy = 1 - expected_error
        threshold_projection = {
            'elo_threshold': elo_threshold,
            'expected_accuracy': expected_accuracy,
            'expected_neg_log_error': expected_neg_log_error
        }
        print(f"At ELO {elo_threshold}, expected accuracy: {expected_accuracy:.4f}")
    
    return {
        'best_baseline': best_baseline_name,
        'best_baseline_score': best_baseline_score,
        'elo_needed': elo_needed,
        'threshold_projection': threshold_projection
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create ELO rating vs error rate plot from CSV data.')
    parser.add_argument("--dataset", required=True, help="Dataset name for the plot title")
    parser.add_argument('--input', required=True, help='Path to the input results CSV file')
    parser.add_argument('--elo', required=True, help='Path to the ELO ratings CSV file')
    parser.add_argument('--elo-translation', help='Path to JSON file with model name translations')
    parser.add_argument('--output', required=True, help='Output image file path')
    parser.add_argument('--pvalue', help='File to save the p-value to')
    parser.add_argument('--projection', help='File to save ELO projection to')
    parser.add_argument('--threshold', type=float, help='Specific ELO threshold to calculate projection for')
    args = parser.parse_args()
    
    # Load the data
    df = load_data(args.input)
    
    # Check if data was loaded successfully
    if df.empty:
        print("Error: No data loaded from the results CSV file.")
        sys.exit(1)
    
    # Load ELO data
    try:
        elo_df = load_elo_data(args.elo)
    except Exception as e:
        print(f"Error loading ELO data: {e}")
        sys.exit(1)
    
    # Load model name translations if provided
    translations = None
    if args.elo_translation:
        try:
            with open(args.elo_translation, 'r') as f:
                translations = json.load(f)
            print(f"Loaded {len(translations)} model name translations")
        except Exception as e:
            print(f"Error loading translations file: {e}")
            sys.exit(1)
    
    # Add ELO ratings to the dataframe
    df['Display_Name'] = df.apply(name_model, axis=1)
    
    # Convert model names and add ELO scores
    df['ELO'] = None
    for idx, row in df.iterrows():
        try:
            df.at[idx, 'ELO'] = get_model_elo(row['Model'], elo_df, translations)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Print summary of the data
    print(f"Loaded {len(df)} rows from {args.input}")
    print("\nData Summary:")
    print(df[['Model', 'ELO', 'Accuracy', 'Neg Log Error']].describe())
    
    # Calculate the projection if requested
    if args.projection:
        projection = calculate_projection(df, args.threshold)
        if projection:
            with open(args.projection, 'w') as f:
                f.write(f"{projection['elo_needed']:.1f}")
            print(f"Saved projection to {args.projection}")
    
    # Create the plot
    plot_elo_vs_log_error(df, args.output, args.dataset, args.pvalue)
    
    print("\nPlot created successfully!")
