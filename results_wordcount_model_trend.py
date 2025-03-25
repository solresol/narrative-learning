#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_task_correlations(task_model_data, metric_name):
    """
    Analyze correlations between tasks for models that are present in multiple tasks.
    
    Args:
        task_model_data: Dictionary with task data by model name
        metric_name: Name of the metric being analyzed
        
    Returns:
        DataFrame with correlation matrix and plot
    """
    # Prepare data for correlation analysis
    corr_data = {}
    
    # Find models that appear in multiple tasks
    all_models = set()
    for task, models in task_model_data.items():
        all_models.update(models.keys())
    
    # For each task, create a series with the metric coefficients
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
    plt.title(f"Correlation of {metric_name} Between Tasks", fontsize=16)
    plt.tight_layout()
    
    return correlation_matrix, corr_df

def analyze_model_size_vs_wordcount(csv_files, wordcount_type='prompt', output_image=None, latex_output=None, filter_rsquared=0.8, min_model_size=10):
    """
    Analyze relationship between model size and word count metrics.
    
    Args:
        csv_files: List of CSV file paths to analyze
        wordcount_type: Type of word count to analyze ('prompt', 'reasoning', or 'cumulative')
        output_image: Path to save the output image
        latex_output: Path to save LaTeX macros
        filter_rsquared: Minimum R-squared value for Herdan coefficient to include data points
        min_model_size: Minimum model size to include in analysis
    
    Returns:
        Tuple of (results dict, filtered dataframe)
    """
    # Determine metric column and display name based on wordcount_type
    if wordcount_type == 'prompt':
        metric_column = 'Prompt Word Count'
        metric_name = "Prompt Word Count"
        latex_prefix = "promptwc"
    elif wordcount_type == 'reasoning':
        metric_column = 'Reasoning Word Count'
        metric_name = "Reasoning Word Count"
        latex_prefix = "reasoningwc"
    elif wordcount_type == 'cumulative':
        metric_column = 'Cumulative Reasoning Words'
        metric_name = "Cumulative Reasoning Words"
        latex_prefix = "cumulativewc"
    else:
        raise ValueError("wordcount_type must be 'prompt', 'reasoning', or 'cumulative'")
    
    # Create output directory for the image file if needed
    if output_image:
        output_dir = os.path.dirname(output_image)
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
    filtered_df = filtered_df.dropna(subset=['Model Size', metric_column])
    
    if filtered_df.empty:
        raise ValueError("No data points remain after filtering. Try lowering filter_rsquared or min_model_size.")
    
    # Log-transform model size for better visualization
    filtered_df['Log Model Size'] = np.log10(filtered_df['Model Size'])
    
    # Create scatter plot with trend line
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Get unique models for coloring
    models = filtered_df['Model'].unique()
    color_palette = sns.color_palette("husl", len(models))
    model_colors = dict(zip(models, color_palette))
    
    # Create scatter plot with points colored by model
    for i, row in filtered_df.iterrows():
        plt.scatter(
            row['Log Model Size'],
            row[metric_column],
            color=model_colors[row['Model']],
            s=100,
            alpha=0.7,
            label=row['Model'] if row['Model'] not in plt.gca().get_legend_handles_labels()[1] else ""
        )
    
    # Add task labels to each point (single letter)
    for i, row in filtered_df.iterrows():
        task_label = row['Task'][0].lower()  # Just first letter (t, w, or s)
        plt.annotate(
            task_label,
            (row['Log Model Size'], row[metric_column]),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=10,
            alpha=0.8
        )
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_df['Log Model Size'], 
        filtered_df[metric_column]
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
    plt.ylabel(metric_name, fontsize=14)
    plt.title(f"Model Size vs. {metric_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Customize X-axis to show actual model sizes
    x_ticks = [1, 2, 3]
    plt.xticks(
        x_ticks, 
        [f'{10**x}B' for x in x_ticks],
        fontsize=12
    )
    
    # Handle legend with unique model entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, title="Models")
    
    # Save the plot if output_image is provided
    if output_image:
        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
    
    # Perform correlation analysis between tasks
    # First, organize data by model name across tasks
    task_model_data = {}
    for task in filtered_df['Task'].unique():
        task_df = filtered_df[filtered_df['Task'] == task]
        task_model_data[task] = {}
        
        # Use Model as the key since it might appear across tasks
        for _, row in task_df.iterrows():
            model_name = row['Model']
            task_model_data[task][model_name] = row[metric_column]
    
    # Calculate average correlation
    avg_correlation = None
    
    # Now analyze correlations between tasks
    if len(task_model_data) > 1:  # Need at least 2 tasks for correlation
        correlation_matrix, corr_df = analyze_task_correlations(task_model_data, metric_name)
        
        # Calculate average correlation (excluding diagonal)
        corr_values = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if not pd.isna(correlation_matrix.iloc[i, j]):
                    corr_values.append(correlation_matrix.iloc[i, j])
        
        if corr_values:
            avg_correlation = sum(corr_values) / len(corr_values)
    
    # Save LaTeX macros if requested
    if latex_output:
        with open(latex_output, 'w') as f:
            # Size trend (slope)
            f.write(f"\\newcommand{{\\{latex_prefix}sizetrend}}{{{slope:.3f}}}\n")
            
            # P-value of the trend
            if p_value < 0.001:
                f.write(f"\\newcommand{{\\{latex_prefix}sizetrendpvalue}}{{<0.001}}\n")
            else:
                f.write(f"\\newcommand{{\\{latex_prefix}sizetrendpvalue}}{{{p_value:.3f}}}\n")
            
            # Average correlation
            if avg_correlation is not None:
                f.write(f"\\newcommand{{\\{latex_prefix}averagecorrelation}}{{{avg_correlation:.2f}}}\n")
            else:
                f.write(f"\\newcommand{{\\{latex_prefix}averagecorrelation}}{{N/A}}\n")
    
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
    parser = argparse.ArgumentParser(description='Analyze model size vs. word count metric trends')
    parser.add_argument('csv_files', nargs='+', help='CSV files with model results')
    parser.add_argument('--wordcount-type', choices=['prompt', 'reasoning', 'cumulative'], default='prompt',
                      help='Type of word count to analyze (prompt, reasoning, or cumulative)')
    parser.add_argument('--output', help='Path to save the output image')
    parser.add_argument('--latex', help='Path to save LaTeX macros file')
    parser.add_argument('--filter-rsquared', type=float, default=0.8, 
                        help='Minimum R-squared value for Herdan coefficient to include data points')
    parser.add_argument('--min-model-size', type=float, default=10, 
                        help='Minimum model size to include (in billions)')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    
    args = parser.parse_args()
    
    results, filtered_df = analyze_model_size_vs_wordcount(
            args.csv_files,
            args.wordcount_type,
            args.output,
            args.latex,
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