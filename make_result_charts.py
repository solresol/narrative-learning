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
from modules.postgres import get_connection
from modules.results_loader import load_results_dataframe


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
    sns.scatterplot(x='Log_Model_Size', y='Neg Log Error',
                   hue='Display_Name',
                    #size='Sampler',
                    #sizes=(100, 200),
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

    # Add a trend line
    extrapolation_x = [9,14]
    extrapolation_request = pd.DataFrame({'Log_Model_Size': extrapolation_x})
    extrapolation_y = lr.predict(extrapolation_request)
    ax.plot(extrapolation_x, extrapolation_y, linestyle="dashed", color="red", label="Trend")

    ax.set_title(f'{dataset_name} Dataset: Model Size vs Error Rate\n(Log-Log)')
    ax.set_xlabel('Log10 Model Size (parameters)')
    ax.set_ylabel('Negative Log10 Error Rate')
    ax.grid(True, linestyle='--', alpha=0.7)
    draw_baselines(ax, df, baselines)


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
    fig.savefig(output_file)
    print(f"Saved plot to {output_file}")

    return output_file

def plot_sample_size_effect(df, output_prefix, dataset_name):
    scatter_df = pd.DataFrame()
    scatter_df['3 Samples'] = df[df.Sampler == 3].set_index('Official_Name').Accuracy
    scatter_df['10 Samples'] = df[df.Sampler == 10].set_index('Official_Name').Accuracy
    scatter_df.dropna(inplace=True)
    fig, ax = plt.subplots(figsize = (10,6))
    sns.scatterplot(data=scatter_df, ax=ax, x='3 Samples', y = '10 Samples',
                            hue='Official_Name')
    ax.plot([0,1],[0,1], linestyle="dashed", c='red')
    output_file = f"{output_prefix}_sample_size_effect.png"
    ax.set_title(f"{dataset_name} Dataset\nComparison of accuracy of 10 sample runs vs 3 sample runs using the same model")
    ax.set_xlabel("Accuracy on the 3-sample run")
    ax.set_ylabel("Accuracy on the 10-sample run")
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_file)
    print(f"Saved plot to {output_file}")


def draw_baselines(ax, df, baselines, xpos=12.5):
    colours = {
        'logistic regression': 'teal',
        'decision trees': 'gold',
        'dummy': 'orange'
        }
    for model, colour in colours.items():
        # Don't need to take the mean -- they will all be the same value
        score = df[model].mean()
        ax.axhline(score, linestyle='dotted', c=colours[model])
        ax.annotate(xy=(xpos,score-0.03), text=model.title(), c=colours[model])
        
    #for model, score in baselines.items():
    #    ax.axhline(score, linestyle='dotted', c=colours[model])
    #    ax.annotate(xy=(xpos,score-0.03), text=model.title(), c=colours[model])

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
    global name_lookup
    return record['Model']
    env_name = record['Model']
    sample_size = record['Sampler']
    if env_name not in name_lookup:
        sys.exit(f"Don't know how to interpret {env_name}")
    return f"{name_lookup[env_name]}, {sample_size} samples"

def main():
    parser = argparse.ArgumentParser(description='Create model analysis plots from database results.')
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument('--output', required=True, help='Prefix for output plot files')
    parser.add_argument("--baselines", required=True, help="Path the JSON file of baselines for this data")
    args = parser.parse_args()

    baselines = json.load(open(args.baselines))
    error_rate_baseline = { x: 1 - y for x,y in baselines.items() }
    neg_log_error_rate_baseline = { x : - math.log10(y) if y > 0 else math.log10(0.01) for x,y in error_rate_baseline.items() }
    # Load the data
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT config_file FROM datasets WHERE dataset = %s", (args.dataset,))
    cfg_row = cur.fetchone()
    if not cfg_row:
        raise SystemExit(f"dataset {args.dataset} not found")
    df = load_results_dataframe(conn, args.dataset, cfg_row[0])

    df['Display_Name'] = df.apply(name_model, axis=1)
    df['Official_Name'] = df.Model.map(name_lookup.get)
    df['Log_Model_Size'] = (df['Model Size'] * 1000000000.0).map(math.log10)
    df['Error_Rate'] = 1 - df['Accuracy']

    # Need to fix these. I want to get a 99th percentile confidence that the error
    # rate was above some number
    df = df[df.Accuracy > baselines['dummy']]
    # If the next line errors, it's because the error rate was 0
    # Print summary of the data
    print(f"Loaded {len(df)} rows for {args.dataset}")
    print("\nData Summary:")
    print(df.describe())

    # Create the plots
    plot_prompt_word_count_vs_accuracy(df, args.output)
    plot_accuracy_by_model(df, args.output)
    plot_log_model_size_vs_log_error(df, args.output, neg_log_error_rate_baseline, args.dataset_name)
    plot_model_size_vs_accuracy(df, args.output)
    plot_model_size_vs_prompt_word_count(df, args.output)
    #plot_sample_size_effect(df, args.output, args.dataset_name)

    print("\nAll plots created successfully!")

if __name__ == "__main__":
    main()
