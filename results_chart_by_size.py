#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import math
import json
import sklearn.linear_model
import statsmodels.api as sm
from chartutils import draw_baselines

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    # Convert accuracy to numeric, handling any missing values
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    return df

def plot_log_model_size_vs_log_error(df, output_prefix, dataset_name, pvalue_file=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a scatter plot with different colors for each model
    sns.scatterplot(x='Log_Model_Size', y='Neg Log Error',
                   hue='Display_Name',
                   data=df,
                   ax=ax)

    df = df.dropna(subset=['Log_Model_Size'])
    print(df[df.Log_Model_Size.isnull()])
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(df[['Log_Model_Size']], df['Neg Log Error'])
    X = sm.add_constant(df[['Log_Model_Size']])
    model = sm.OLS(df['Neg Log Error'], X).fit()
    print(model.summary())
    # Access specific statistics
    slope = model.params['Log_Model_Size']
    intercept = model.params['const']
    p_value = model.pvalues['Log_Model_Size']
    r_squared = model.rsquared
    print(f"Slope: {slope:.4f}, p-value: {p_value:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    
    # Save p-value to file if requested
    if pvalue_file:
        with open(pvalue_file, 'w') as f:
            f.write(f"{p_value:.3f}")

    # Add a trend line
    extrapolation_x = [10,15]
    extrapolation_request = pd.DataFrame({'Log_Model_Size': extrapolation_x})
    extrapolation_y = lr.predict(extrapolation_request)
    ax.plot(extrapolation_x, extrapolation_y, linestyle="dashed", color="red", label="Trend")

    ax.set_title(f'{dataset_name} Dataset: Model Size vs Error Rate\n(Log-Log)')
    ax.set_xlabel('Log10 Model Size (parameters)')
    ax.set_ylabel('Negative Log10 Error Rate')
    ax.grid(True, linestyle='--', alpha=0.7)
    draw_baselines(ax, df)


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

    output_file = output_prefix
    fig.savefig(output_file)
    print(f"Saved plot to {output_file}")

    return output_file

name_lookup = {
    'openai10': 'gpt-4o',
    'openai': 'gpt-4o',
    'openai-o1': 'o1',
    'openai45': 'gpt-4.5-preview',
    'llama': "llama 3.3",
    'anthropic': "claude 3.5",
    'falcon': "falcon3b",
    'gemma': "gemma"
}

def name_model(record):
    return record['Model']

def calculate_projection(df):
    """Calculate parameters needed to beat the best baseline."""
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
    
    # Use the linear regression model to estimate parameters needed
    df_clean = df.dropna(subset=['Log_Model_Size'])
    if df_clean.empty:
        print("No valid model size data for projection")
        return None
        
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(df_clean[['Log_Model_Size']], df_clean['Neg Log Error'])
    
    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    # Calculate log model size needed to beat the best baseline
    # neg_log_error = slope * log_model_size + intercept
    # log_model_size = (neg_log_error - intercept) / slope
    log_model_size_needed = (neg_log_best_baseline_error - intercept) / slope
    model_size_needed = 10**log_model_size_needed / 1000000000.0  # Convert back to billions
    
    print(f"Model size needed to beat best baseline: {model_size_needed:.2f} billion parameters")
    
    return {
        'best_baseline': best_baseline_name,
        'best_baseline_score': best_baseline_score,
        'model_size_needed': model_size_needed,
        'log_model_size_needed': log_model_size_needed
    }

def main():
    parser = argparse.ArgumentParser(description='Create model size vs error rate plot from CSV data.')
    parser.add_argument("--dataset-name", required=True, help="Used in the plot title")
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--image', required=True, help='Output image file path')
    parser.add_argument('--pvalue', help='File to save the p-value to (just the value, nothing else)')
    parser.add_argument('--projection', help='File to save parameter projection to')
    args = parser.parse_args()
    
    # Load the data
    df = load_data(args.input)

    # Check if data was loaded successfully
    if df.empty:
        print("Error: No data loaded from the CSV file.")
        return

    df['Display_Name'] = df.apply(name_model, axis=1)
    df['Log_Model_Size'] = (df['Model Size'] * 1000000000.0).map(math.log10)
    
    # Print summary of the data
    print(f"Loaded {len(df)} rows from {args.input}")
    print("\nData Summary:")
    print(df.describe())

    # Calculate the projection if requested
    if args.projection:
        projection = calculate_projection(df)
        if projection:
            with open(args.projection, 'w') as f:
                f.write(f"$10^{projection['log_model_size_needed']:.1f}$")
            print(f"Saved projection to {args.projection}")

    # Create the plot
    plot_log_model_size_vs_log_error(df, args.image, args.dataset_name, args.pvalue)

    print("\nPlot created successfully!")

if __name__ == "__main__":
    main()
