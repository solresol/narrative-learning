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
import sys

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    # Convert accuracy to numeric, handling any missing values
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    return df

def plot_prompt_word_count_vs_accuracy(df, output_prefix):
    """Plot Prompt Word Count vs Accuracy."""
    plt.figure(figsize=(10, 6))

    # Create a scatter plot with different colors for each model
    sns.scatterplot(x='Prompt Word Count', y='Accuracy',
                   hue='Model', size='Model Size',
                   palette='viridis', data=df)

    # Add a trend line
    if df['Accuracy'].notna().sum() > 1:  # Need at least 2 non-NA points for regression
        sns.regplot(x='Prompt Word Count', y='Accuracy',
                   scatter=False, ci=None, color='red', data=df)

    plt.title('Prompt Word Count vs Accuracy')
    plt.xlabel('Prompt Word Count')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_file = f"{output_prefix}_word_count_vs_accuracy.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

    return output_file

def plot_accuracy_by_model(df, output_prefix):
    """Plot Accuracy by Model and Sampler size."""
    plt.figure(figsize=(12, 7))

    # Filter out rows with missing accuracy values
    df_filtered = df.dropna(subset=['Accuracy'])

    # Create a grouped bar chart
    ax = sns.barplot(x='Model', y='Accuracy', hue='Sampler', data=df_filtered)

    # Add value labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.title('Accuracy by Model and Sampler Size')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_file = f"{output_prefix}_accuracy_by_model.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

    return output_file

def plot_model_size_vs_accuracy(df, output_prefix):
    """Plot Model Size vs Accuracy."""
    plt.figure(figsize=(10, 6))

    # Create a scatter plot with different colors for each model
    sns.scatterplot(x='Log_Model_Size', y='Accuracy',
                   hue='Display_Name',
                    #size='Sampler',
                    #sizes=(100, 200),
                    palette='cool', data=df)

    # Add a trend line
    if df['Accuracy'].notna().sum() > 1:  # Need at least 2 non-NA points for regression
        sns.regplot(x='Log_Model_Size', y='Accuracy',
                   scatter=False, ci=None, color='red', data=df)

    plt.title('Model Size (Billions of Parameters) vs Accuracy')
    plt.xlabel('Log10 Model Size (B)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_file = f"{output_prefix}_model_size_vs_accuracy.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

    return output_file


def plot_log_model_size_vs_log_error(df, output_prefix, baselines, dataset_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a scatter plot with different colors for each model
    sns.scatterplot(x='Log_Model_Size', y='Negative_Log_Error_Rate',
                   hue='Display_Name',
                    #size='Sampler',
                    #sizes=(100, 200),
                    palette='cool', data=df,
                    ax=ax)

    df = df.dropna(subset=['Log_Model_Size'])
    print(df[df.Log_Model_Size.isnull()])
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(df[['Log_Model_Size']], df.Negative_Log_Error_Rate)
    X = sm.add_constant(df[['Log_Model_Size']])
    model = sm.OLS(df.Negative_Log_Error_Rate, X).fit()
    print(model.summary())
    # Access specific statistics
    slope = model.params['Log_Model_Size']
    intercept = model.params['const']
    p_value = model.pvalues['Log_Model_Size']
    r_squared = model.rsquared
    print(f"Slope: {slope:.4f}, p-value: {p_value:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    # Add a trend line
    extrapolation_x = [9,14]
    extrapolation_request = pd.DataFrame({'Log_Model_Size': extrapolation_x})
    extrapolation_y = lr.predict(extrapolation_request)
    ax.plot(extrapolation_x, extrapolation_y, linestyle="dashed", color="red", label="Trend")

    ax.set_title(f'{dataset_name} Dataset: Model Size vs Error Rate\n(Log-Log)')
    ax.set_xlabel('Log10 Model Size (parameters)')
    ax.set_ylabel('Negative Log10 Error Rate')
    ax.grid(True, linestyle='--', alpha=0.7)
    draw_baselines(ax, baselines)


    ax2 = ax.twinx()

    # Create a function to convert negative log error to accuracy
    def neg_log_error_to_accuracy(y):
        # Convert negative log10 error rate to accuracy percentage
        # If y is Negative_Log_Error_Rate, then error = 10^(-y)
        # And accuracy = 1 - error = 1 - 10^(-y)
        calc = (1 - 10**(-y)) * 100
        print(f"Neg log {y} -> {calc} %")
        return calc

    y1_ticks = ax.get_yticks()
    # Calculate corresponding accuracy values for these specific positions
    y2_ticks = [neg_log_error_to_accuracy(y) for y in y1_ticks]
    #ax2.set_yticks(y2_ticks)
    ax2.set_yticks(y1_ticks)
    ax2.set_yticklabels([f"{y:.0f}%" for y in y2_ticks])
    y1_min, y1_max = ax.get_ylim()
    ax2.set_ylim(y1_min, y1_max)
    #ax2.set_ylim(neg_log_error_to_accuracy(y1_min), neg_log_error_to_accuracy(y1_max))

    # Set secondary axis label
    ax2.set_ylabel('Accuracy (%)')
    ax2.tick_params(axis='y')

    fig.tight_layout()

    output_file = f"{output_prefix}_model_size_vs_error_rate.png"
    fig.savefig(output_file)
    print(f"Saved plot to {output_file}")

    return output_file

def plot_model_size_vs_prompt_word_count(df, output_prefix):
    """Plot Model Size vs Prompt Word Count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a scatter plot with bubble size representing accuracy
    accuracy_sizes = df['Accuracy'].fillna(0) * 500  # Scale for visibility

    scatter = ax.scatter(df['Model Size'], df['Prompt Word Count'],
                         s=accuracy_sizes,
                         c=df.groupby('Model').ngroup(),
                         alpha=0.6,
                         cmap='viridis')

    # Add model names as annotations
    for i, row in df.iterrows():
        ax.annotate(f"{row['Model']}-{row['Sampler']}",
                    (row['Model Size'], row['Prompt Word Count']),
                    xytext=(5, 5), textcoords='offset points')

    # Create legend for accuracy size reference
    accuracy_levels = [0.4, 0.6, 0.8]
    legend_bubbles = []
    for acc in accuracy_levels:
        legend_bubbles.append(plt.scatter([], [], s=acc*500, c='gray', alpha=0.6))

    ax.legend(legend_bubbles, [f'Accuracy: {acc}' for acc in accuracy_levels],
              scatterpoints=1, loc='upper right', title='Accuracy Reference')

    ax.set_title('Model Size vs Prompt Word Count')
    ax.set_xlabel('Model Size (B)')
    ax.set_ylabel('Prompt Word Count')
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    output_file = f"{output_prefix}_model_size_vs_word_count.png"
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

    return output_file

def main():
    parser = argparse.ArgumentParser(description='Create model analysis plots from CSV data.')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', default='plot', help='Prefix for output plot files')
    args = parser.parse_args()
    
    # Load the data
    df = load_data(args.input)

    # Check if data was loaded successfully
    if df.empty:
        print("Error: No data loaded from the CSV file.")
        return
    
    # Print summary of the data
    print(f"Loaded {len(df)} rows from {args.input}")
    print("\nData Summary:")
    print(df.describe())

    # Create the plots
    plot_prompt_word_count_vs_accuracy(df, args.output)
    plot_accuracy_by_model(df, args.output)
    plot_model_size_vs_accuracy(df, args.output)
    plot_model_size_vs_prompt_word_count(df, args.output)
    
    print("\nAll plots created successfully!")

if __name__ == "__main__":
    main()
