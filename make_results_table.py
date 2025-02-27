#!/usr/bin/env python3
import pandas as pd
import argparse
import numpy as np

def format_accuracy(value):
    """Format accuracy values with proper decimal places and escaped percentage"""
    if pd.isna(value):
        return '---'
    # Use \% to escape the percentage sign in LaTeX
    return f"{value:.2f}\\%"

def csv_to_latex_tabular(input_file, output_file=None):
    """
    Convert a CSV file to a LaTeX tabular format without the surrounding table environment.
    Properly escapes percentage signs.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str, optional
        Path to output LaTeX file. If None, prints to console.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
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
    parser = argparse.ArgumentParser(description='Convert CSV to LaTeX tabular format')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', help='Path to output LaTeX file (optional)')
    
    args = parser.parse_args()
    
    # Generate the LaTeX tabular
    csv_to_latex_tabular(
        args.input, 
        args.output
    )
    
    print("Conversion completed successfully.")

if __name__ == "__main__":
    main()
