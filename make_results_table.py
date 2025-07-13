#!/usr/bin/env python3
import pandas as pd
import argparse
import numpy as np
from modules.postgres import get_connection
from modules.results_loader import load_results_dataframe

def format_accuracy(value):
    """Format accuracy values with proper decimal places and escaped percentage"""
    if pd.isna(value):
        return '---'
    # Use \% to escape the percentage sign in LaTeX
    return f"{value:.2f}\\%"

def csv_to_latex_tabular(conn, dataset: str, output_file=None):
    """
    Convert a CSV file to a LaTeX tabular format without the surrounding table environment.
    Properly escapes percentage signs.
    
    Parameters:
    -----------
    conn : psycopg2 connection
        Database connection
    dataset : str
        Dataset name
    output_file : str, optional
        Path to output LaTeX file. If None, prints to console.
    """
    cur = conn.cursor()
    cur.execute("SELECT config_file FROM datasets WHERE dataset = %s", (dataset,))
    cfg_row = cur.fetchone()
    if not cfg_row:
        raise SystemExit(f"dataset {dataset} not found")
    df = load_results_dataframe(conn, dataset, cfg_row[0])
    
    # Convert accuracy to numeric and format special for the table
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    
    # Sort by Model and then Sampler for better organization
    df = df.sort_values(['Model', 'Sampler'])
    
    # Start building the LaTeX tabular
    latex_lines = []
    
    # Column specification with alignment
    columns = df.columns
    alignment = "l" + "r" * (len(columns) - 1)  # First column left-aligned, rest right-aligned
    
    latex_lines.append(f"\\begin{{tabular}}{{|{alignment}|}}")
    latex_lines.append("\\hline")
    
    # Header row with bold formatting
    header_row = " & ".join([f"\\textbf{{{col}}}" for col in columns])
    latex_lines.append(f"{header_row} \\\\")
    latex_lines.append("\\hline")
    
    # Data rows
    for idx, row in df.iterrows():
        row_values = []
        for col in columns:
            if col == 'Accuracy':
                # Special formatting for accuracy with escaped percentage
                row_values.append(format_accuracy(row[col]))
            elif col == 'Model Size':
                # Add 'B' for billions
                row_values.append(f"{row[col]}B")
            else:
                row_values.append(str(row[col]))
        
        latex_lines.append(f"{' & '.join(row_values)} \\\\")
    
    # Table footer
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    # Combine all lines
    latex_tabular = "\n".join(latex_lines)
    
    # Output handling
    if output_file:
        with open(output_file, 'w') as f:
            f.write(latex_tabular)
        print(f"LaTeX tabular written to {output_file}")
    else:
        print(latex_tabular)
    
    return latex_tabular

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert database results to LaTeX tabular format')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--output', help='Path to output LaTeX file (optional)')
    
    args = parser.parse_args()
    
    # Generate the LaTeX tabular
    conn = get_connection()
    csv_to_latex_tabular(conn, args.dataset, args.output)
    
    print("Conversion completed successfully.")

if __name__ == "__main__":
    main()
