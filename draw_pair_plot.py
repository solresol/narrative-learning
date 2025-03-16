#!/usr/bin/env python3
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Create pairplot visualization from CSV data')
    parser.add_argument('--input', required=True, help='Input CSV file (from create_task_csv_file.py)')
    parser.add_argument('--output', default='pairplot.png', help='Output image file')
    parser.add_argument('--hue', default='Model', help='Column to use for color coding')
    parser.add_argument('--size', type=int, default=15, help='Figure size')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    data = pd.read_csv(args.input)
    
    # Convert string columns that should be numeric
    numeric_columns = ['Accuracy', 'Sampler', 'Rounds', 'Prompt Word Count', 'Model Size']
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