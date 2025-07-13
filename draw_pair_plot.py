#!/usr/bin/env python3
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modules.postgres import get_connection
from modules.results_loader import load_results_dataframe

def parse_args():
    parser = argparse.ArgumentParser(description='Create pairplot visualization from database results')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output', default='pairplot.png', help='Output image file')
    parser.add_argument('--hue', default='Model', help='Column to use for color coding')
    parser.add_argument('--size', type=int, default=15, help='Figure size')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT config_file FROM datasets WHERE dataset = %s", (args.dataset,))
    cfg_row = cur.fetchone()
    if not cfg_row:
        raise SystemExit(f"dataset {args.dataset} not found")
    data = load_results_dataframe(conn, args.dataset, cfg_row[0])
    
    # Convert string columns that should be numeric
    numeric_columns = ['Accuracy', 'Sampler', 'Rounds', 'Prompt Word Count', 'Model Size', 'Patience']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Select only numeric columns for the pairplot
    numeric_data = data.select_dtypes(include=['number'])
    
    # Add the hue column if it's not already in numeric_data
    if args.hue not in numeric_data.columns and args.hue in data.columns:
        numeric_data[args.hue] = data[args.hue]
    
    # Create the pairplot
    plt.figure(figsize=(args.size, args.size))
    sns.set_theme(style="ticks")
    
    # Create and save the pairplot
    g = sns.pairplot(numeric_data, hue=args.hue, diag_kind="kde")
    g.fig.suptitle('Pairwise Relationships in Model Performance Data', y=1.02, fontsize=16)
    
    # Save the figure
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Pairplot saved to {args.output}")

if __name__ == "__main__":
    main()
