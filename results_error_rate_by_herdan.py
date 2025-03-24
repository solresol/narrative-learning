#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_error_rate_by_herdan(csv_file, output_dir=None, filter_rsquared=0.8):
    """
    Analyze relationship between Herdan coefficient and error rate (neg log error).
    
    Args:
        csv_file: Path to CSV file with results
        output_dir: Directory to save outputs
        filter_rsquared: Minimum R-squared value for Herdan coefficient to include data points
    
    Returns:
        Tuple of (p_value, DataFrame with filtered data)
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Get dataset name from file
    dataset_name = os.path.basename(csv_file).split('_')[0].capitalize()
    
    # Filter data by R-squared
    filtered_df = df[df['Herdan R-squared'] >= filter_rsquared].copy()
    
    # Remove rows with missing values for key fields
    filtered_df = filtered_df.dropna(subset=['Herdan Coefficient', 'Neg Log Error'])
    
    if filtered_df.empty:
        raise ValueError("No data points remain after filtering. Try lowering filter_rsquared.")
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_df['Herdan Coefficient'], 
        filtered_df['Neg Log Error']
    )
    
    # Create scatter plot with trend line
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Plot scatter
    plt.scatter(
        filtered_df['Herdan Coefficient'],
        filtered_df['Neg Log Error'],
        color='steelblue',
        s=100,
        alpha=0.7
    )
    
    # Add labels to points
    for i, row in filtered_df.iterrows():
        plt.annotate(
            f"{row['Model']}",
            (row['Herdan Coefficient'], row['Neg Log Error']),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    # Add regression line
    x_range = np.linspace(
        filtered_df['Herdan Coefficient'].min() - 0.05, 
        filtered_df['Herdan Coefficient'].max() + 0.05, 
        100
    )
    plt.plot(
        x_range, 
        intercept + slope * x_range, 
        'k--', 
        linewidth=2,
        label=f'Trend: y = {slope:.4f}x + {intercept:.4f}'
    )
    
    # Add baseline lines if available
    baseline_columns = ['logistic regression', 'decision trees', 'dummy']
    baselines = {}
    
    for col in baseline_columns:
        if col in filtered_df.columns and not filtered_df[col].isna().all():
            # Take the first non-NaN value
            value = filtered_df[col].dropna().iloc[0]
            baselines[col] = value
            
            # Calculate the negative log of error
            if value < 1:  # Avoid log of 0
                neg_log_error = -np.log10(1 - value)
                plt.axhline(
                    y=neg_log_error,
                    linestyle=':',
                    color='gray',
                    alpha=0.7,
                    label=f'{col.title()}: {neg_log_error:.3f}'
                )
    
    # Add equation and statistics
    equation_text = (
        f"y = {slope:.4f}x + {intercept:.4f}\n"
        f"R² = {r_value**2:.4f}, p = {p_value:.4f}"
    )
    plt.annotate(
        equation_text, 
        xy=(0.05, 0.95), 
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=12
    )
    
    # Add labels and title
    plt.xlabel("Herdan's Law Coefficient", fontsize=14)
    plt.ylabel("Negative Log Error Rate", fontsize=14)
    plt.title(f"{dataset_name}: Lexical Complexity vs. Error Rate", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot if output_dir is provided
    if output_dir:
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f'{dataset_name.lower()}_error_by_herdan.png')
        plt.savefig(plot_file, dpi=300)
        
        # Save the p-value to a text file (nothing but the p-value, to 3 decimal places)
        p_value_file = os.path.join(output_dir, f'{dataset_name.lower()}_herdan_pvalue.txt')
        with open(p_value_file, 'w') as f:
            f.write(f"{p_value:.3f}")
    
    return p_value, filtered_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze the relationship between Herdan coefficient and error rate')
    parser.add_argument('csv_file', help='CSV file with model results')
    parser.add_argument('--output-dir', default='outputs', help='Directory to save outputs')
    parser.add_argument('--filter-rsquared', type=float, default=0.8, 
                        help='Minimum R-squared value for Herdan coefficient to include data points')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    
    args = parser.parse_args()
    
    try:
        p_value, filtered_df = analyze_error_rate_by_herdan(
            args.csv_file,
            args.output_dir,
            args.filter_rsquared
        )
        
        print(f"Analysis completed successfully:")
        print(f"P-value: {p_value:.4f}")
        print(f"Data points: {len(filtered_df)}")
        
        if args.show:
            plt.show()
        
    except Exception as e:
        raise e
