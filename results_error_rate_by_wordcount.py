#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from chartutils import draw_baselines
from modules.postgres import get_connection

def analyze_error_rate_by_wordcount(conn, table, dataset, x_column='Prompt Word Count', image_output=None,
                              pvalue_output=None, slope_output=None):
    """
    Analyze relationship between word count and error rate (neg log error).
    
    Args:
        conn: Active PostgreSQL connection
        table: Table name containing results data
        dataset: Dataset name to analyze
        x_column: The word count column to use as x-axis ('Prompt Word Count',
                 'Reasoning Word Count', or 'Cumulative Reasoning Words')
        image_output: Path to save the output image
        pvalue_output: Path to save the p-value text file
        slope_output: Path to save the slope value text file
    
    Returns:
        Tuple of (p_value, slope, DataFrame with data)
    """
    # Load data from PostgreSQL
    query = f'SELECT * FROM {table} WHERE "Task" = %s'
    df = pd.read_sql_query(query, conn, params=(dataset,))
    dataset_name = dataset.capitalize()
    
    # Set readable column name for display
    x_column_display = x_column
    
    # Remove rows with missing values for key fields
    filtered_df = df.dropna(subset=[x_column, 'Neg Log Error'])
    
    if filtered_df.empty:
        raise ValueError(f"No data points remain after filtering for {x_column}.")
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_df[x_column], 
        filtered_df['Neg Log Error']
    )
    
    # Create scatter plot with trend line
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Get unique models for color mapping
    unique_models = filtered_df['Model'].unique()
    color_palette = sns.color_palette("husl", len(unique_models))
    model_color_map = dict(zip(unique_models, color_palette))
    
    # Plot scatter points colored by model
    for model in unique_models:
        model_data = filtered_df[filtered_df['Model'] == model]
        plt.scatter(
            model_data[x_column],
            model_data['Neg Log Error'],
            color=model_color_map[model],
            s=100,
            alpha=0.7,
            label=model
        )
    
    # Add labels to points
    for i, row in filtered_df.iterrows():
        plt.annotate(
            row['Model'],  # Full model name
            (row[x_column], row['Neg Log Error']),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    # Add regression line
    x_range = np.linspace(
        filtered_df[x_column].min() - 0.05 * filtered_df[x_column].min(), 
        filtered_df[x_column].max() + 0.05 * filtered_df[x_column].max(), 
        100
    )
    plt.plot(
        x_range, 
        intercept + slope * x_range, 
        'k--', 
        linewidth=2,
        label=f'Trend: y = {slope:.6f}x + {intercept:.4f}'
    )
    
    # Add baseline lines
    draw_baselines(plt.gca(), filtered_df, xpos=0.65)
    
    # Add equation and statistics
    equation_text = (
        f"y = {slope:.6f}x + {intercept:.4f}\n"
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
    plt.xlabel(f"{x_column_display}", fontsize=14)
    plt.ylabel("Negative Log Error Rate", fontsize=14)
    plt.title(f"{dataset_name}: {x_column_display} vs. Negative Log Error Rate", fontsize=16)
    
    # Handle the legend - only show each model once
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    label_indices = {}
    
    for i, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = i
            unique_labels.append(label)
            unique_handles.append(handles[i])
    
    plt.legend(unique_handles, unique_labels, fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save outputs if paths are provided
    if image_output:
        plt.tight_layout()
        plt.savefig(image_output, dpi=300)
    
    if pvalue_output:
        with open(pvalue_output, 'w') as f:
            if p_value < 0.1:
                f.write(f"{p_value:.4f}")
            else:
                f.write(f"{p_value:.3f}")
    
    if slope_output:
        with open(slope_output, 'w') as f:
            f.write(f"{slope:.6f}")

    # Store results in PostgreSQL
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS wordcount_error_rate_results (
            dataset TEXT,
            wordcount_type TEXT,
            p_value DOUBLE PRECISION,
            slope DOUBLE PRECISION
        )
        """
    )
    cur.execute(
        "INSERT INTO wordcount_error_rate_results(dataset, wordcount_type, p_value, slope)"
        " VALUES (%s, %s, %s, %s)",
        (dataset_name, x_column, float(p_value), float(slope))
    )
    conn.commit()

    return p_value, slope, filtered_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze the relationship between word count and error rate')
    parser.add_argument('--dsn', help='PostgreSQL DSN')
    parser.add_argument('--pg-config', help='JSON file containing postgres_dsn')
    parser.add_argument('--table', required=True, help='Table with results data')
    parser.add_argument('--dataset', required=True, help='Dataset name to analyze')
    parser.add_argument('--wordcount-type', choices=['prompt', 'reasoning', 'cumulative'], default='prompt',
                        help='Type of word count to use (prompt, reasoning, or cumulative)')
    parser.add_argument('--image-output', help='Path to save the output image')
    parser.add_argument('--pvalue-output', help='Path to save the p-value text file')
    parser.add_argument('--slope-output', help='Path to save the slope value text file')
    parser.add_argument('--show', action='store_true', help='Show the plot')
    
    args = parser.parse_args()
    
    # Map wordcount type to column name
    wordcount_columns = {
        'prompt': 'Prompt Word Count',
        'reasoning': 'Reasoning Word Count',
        'cumulative': 'Cumulative Reasoning Words'
    }
    
    x_column = wordcount_columns[args.wordcount_type]

    try:
        conn = get_connection(args.dsn, args.pg_config)
        p_value, slope, filtered_df = analyze_error_rate_by_wordcount(
            conn,
            args.table,
            args.dataset,
            x_column,
            args.image_output,
            args.pvalue_output,
            args.slope_output,
        )
        conn.close()
        
        print(f"Analysis completed successfully:")
        print(f"P-value: {p_value:.4f}")
        print(f"Slope: {slope:.6f}")
        print(f"Data points: {len(filtered_df)}")
        
        if args.show:
            plt.show()
        
    except Exception as e:
        raise e
