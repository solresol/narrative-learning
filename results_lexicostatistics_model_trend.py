#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_task_correlations(task_model_data):
    """
    Analyze correlations between tasks for models that are present in multiple tasks.
    
    Args:
        task_model_data: Dictionary with task data by model name
        
    Returns:
        DataFrame with correlation matrix and plot
    """
    # Prepare data for correlation analysis
    corr_data = {}
    
    # Find models that appear in multiple tasks
    all_models = set()
    for task, models in task_model_data.items():
        all_models.update(models.keys())
    
    # For each task, create a series with Herdan coefficients
    for task in task_model_data:
        task_series = {}
        for model in all_models:
            if model in task_model_data[task]:
                task_series[model] = task_model_data[task][model]
        corr_data[task] = pd.Series(task_series)
    
    # Create DataFrame from the series
    corr_df = pd.DataFrame(corr_data)
    
    # Calculate correlation matrix
    correlation_matrix = corr_df.corr(min_periods=3)  # Require at least 3 shared models
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5,
        square=True,
        fmt='.2f'
    )
    plt.title("Correlation of Herdan's Law Coefficients Between Tasks", fontsize=16)
    plt.tight_layout()
    
    return correlation_matrix, corr_df

def analyze_model_size_vs_lexical_complexity(csv_files, output_dir=None, filter_rsquared=0.8, min_model_size=10):
    """
    Analyze relationship between model size and lexical complexity metrics.
    
    Args:
        csv_files: List of CSV file paths to analyze
        output_dir: Directory to save outputs
        filter_rsquared: Minimum R-squared value for Herdan coefficient to include data points
        min_model_size: Minimum model size to include in analysis
    
    Returns:
        Tuple of (results dict, filtered dataframe)
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and combine data from all CSV files
    dfs = []
    for file in csv_files:
        task_name = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file)
        df['Task'] = task_name
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Filter data
    filtered_df = combined_df[
        (combined_df['Herdan R-squared'] >= filter_rsquared) & 
        (combined_df['Model Size'].astype(float) >= min_model_size)
    ].copy()
    
    # Convert model size to numeric if needed
    filtered_df['Model Size'] = pd.to_numeric(filtered_df['Model Size'], errors='coerce')
    
    # Drop rows with missing values
    filtered_df = filtered_df.dropna(subset=['Model Size', 'Herdan Coefficient'])
    
    if filtered_df.empty:
        raise ValueError("No data points remain after filtering. Try lowering filter_rsquared or min_model_size.")
    
    # Log-transform model size for better visualization
    filtered_df['Log Model Size'] = np.log10(filtered_df['Model Size'])
    
    # Create scatter plot with trend line
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Create a single scatter plot without grouping by task
    plt.scatter(
        filtered_df['Log Model Size'],
        filtered_df['Herdan Coefficient'],
        color='steelblue',
        s=100,
        alpha=0.7,
    )
    
    # Add task labels to each point
    for i, row in filtered_df.iterrows():
        plt.annotate(
            row['Task'][:3],  # First 3 letters of task name
            (row['Log Model Size'], row['Herdan Coefficient']),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_df['Log Model Size'], 
        filtered_df['Herdan Coefficient']
    )
    
    # Add regression line
    x_range = np.linspace(
        filtered_df['Log Model Size'].min() - 0.1, 
        filtered_df['Log Model Size'].max() + 0.1, 
        100
    )
    plt.plot(
        x_range, 
        intercept + slope * x_range, 
        'k--', 
        linewidth=2,
        label=f'Trend: y = {slope:.4f}x + {intercept:.4f}'
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
    plt.xlabel('Log10(Model Size in Billions)', fontsize=14)
    plt.ylabel("Herdan's Law Coefficient", fontsize=14)
    plt.title("Model Size vs. Lexical Complexity", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Customize X-axis to show actual model sizes
    x_ticks = [1, 2, 3]
    plt.xticks(
        x_ticks, 
        [f'{10**x}B' for x in x_ticks],
        fontsize=12
    )
    
    # Save the plot if output_dir is provided
    if output_dir:
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_size_vs_herdan.png'), dpi=300)
        
        # Also save the statistics as text
        stats_file = os.path.join(output_dir, 'model_size_vs_herdan_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Linear Regression Results:\n")
            f.write(f"Slope: {slope:.6f}\n")
            f.write(f"Intercept: {intercept:.6f}\n")
            f.write(f"R-squared: {r_value**2:.6f}\n")
            f.write(f"P-value: {p_value:.6f}\n")
            f.write(f"Standard Error: {std_err:.6f}\n")
        
        # Save LaTeX-formatted p-value
        latex_file = os.path.join(output_dir, 'model_size_vs_herdan_pvalue.tex')
        with open(latex_file, 'w') as f:
            if p_value < 0.001:
                f.write(f"$p < 0.001$")
            else:
                f.write(f"$p = {p_value:.4f}$")
    
    # Perform correlation analysis between tasks
    # First, organize data by model name across tasks
    task_model_data = {}
    for task in filtered_df['Task'].unique():
        task_df = filtered_df[filtered_df['Task'] == task]
        task_model_data[task] = {}
        
        # Use Model as the key since it might appear across tasks
        for _, row in task_df.iterrows():
            model_name = row['Model']
            task_model_data[task][model_name] = row['Herdan Coefficient']
    
    # Now analyze correlations between tasks
    if len(task_model_data) > 1:  # Need at least 2 tasks for correlation
        correlation_matrix, corr_df = analyze_task_correlations(task_model_data)
        
        if output_dir:
            # Save the correlation heatmap
            plt.savefig(os.path.join(output_dir, 'task_correlation_heatmap.png'), dpi=300)
            
            # Save correlation matrix as CSV
            correlation_matrix.to_csv(os.path.join(output_dir, 'task_correlations.csv'))
            
            # Save the raw data used for correlation
            corr_df.to_csv(os.path.join(output_dir, 'task_correlation_data.csv'))
    
    # Return results
    results = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'data_points': len(filtered_df),
        'tasks': list(filtered_df['Task'].unique())
    }
    
    return results, filtered_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze model size vs. lexical complexity trends')
    parser.add_argument('csv_files', nargs='+', help='CSV files with model results')
    parser.add_argument('--output-dir', default='outputs', help='Directory to save outputs')
    parser.add_argument('--filter-rsquared', type=float, default=0.8, 
                        help='Minimum R-squared value for Herdan coefficient to include data points')
    parser.add_argument('--min-model-size', type=float, default=10, 
                        help='Minimum model size to include (in billions)')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    parser.add_argument('--correlation-only', action='store_true', 
                        help='Only analyze task correlations without regression analysis')
    
    args = parser.parse_args()
    
    try:
        results, filtered_df = analyze_model_size_vs_lexical_complexity(
            args.csv_files,
            args.output_dir,
            args.filter_rsquared,
            args.min_model_size
        )
        
        print(f"Analysis completed successfully:")
        print(f"Slope: {results['slope']:.6f}")
        print(f"Intercept: {results['intercept']:.6f}")
        print(f"R-squared: {results['r_squared']:.6f}")
        print(f"P-value: {results['p_value']:.6f}")
        print(f"Data points: {results['data_points']}")
        print(f"Tasks: {', '.join(results['tasks'])}")
        
        # If there are multiple tasks, print correlation analysis
        if len(results['tasks']) > 1:
            print("\nTask correlation analysis:")
            corr_file = os.path.join(args.output_dir, 'task_correlations.csv')
            if os.path.exists(corr_file):
                corr_matrix = pd.read_csv(corr_file, index_col=0)
                print(corr_matrix)
                
                # Calculate average correlation
                corr_values = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if not pd.isna(corr_matrix.iloc[i, j]):
                            corr_values.append(corr_matrix.iloc[i, j])
                
                if corr_values:
                    avg_corr = sum(corr_values) / len(corr_values)
                    print(f"\nAverage inter-task correlation: {avg_corr:.4f}")
        
        if args.show:
            plt.show()
        
    except Exception as e:
        raise e
