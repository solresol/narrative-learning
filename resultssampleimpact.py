#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from typing import List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze impact of sample size on model performance')
    parser.add_argument('csv_files', nargs='+', help='List of CSV result files to process')
    parser.add_argument('--pivot', required=True, help='Output path for pivot table CSV')
    parser.add_argument('--image', required=True, help='Output path for scatter plot image')
    parser.add_argument('--stats-results', required=True, help='Output path for statistical test results')
    return parser.parse_args()

def load_and_combine_data(csv_files: List[str]) -> pd.DataFrame:
    """
    Load and combine data from multiple CSV files
    
    Args:
        csv_files: List of paths to CSV files
        
    Returns:
        Combined DataFrame
    """
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)

def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table with Task and Model as rows, Sampler as columns,
    and Neg Log Error as values
    
    Args:
        df: Input DataFrame
        
    Returns:
        Pivot table DataFrame
    """
    # Create pivot table
    pivot = df.pivot_table(
        index=['Task', 'Model'],
        columns='Sampler',
        values='Neg Log Error'
    )
    
    return pivot

def create_scatter_plot(pivot_df: pd.DataFrame, output_path: str) -> None:
    """
    Create a scatter plot comparing performance with sampler=3 vs sampler=10
    
    Args:
        pivot_df: Pivot table DataFrame
        output_path: Path to save the scatter plot
    """
    # Drop rows with missing values in either column 3 or 10
    plot_df = pivot_df.dropna(subset=[3, 10])
    
    # Get x and y values
    x_values = plot_df[3]
    y_values = plot_df[10]
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x_values, y_values, alpha=0.7)
    
    # Add labels for each point
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        task, model = idx
        plt.annotate(f"{task}-{model}", (x_values.iloc[i], y_values.iloc[i]), 
                    fontsize=8, alpha=0.7)
    
    # Calculate and plot the red dashed line (y=x)
    max_val = max(max(x_values), max(y_values))
    min_val = min(min(x_values), min(y_values))
    line_vals = np.linspace(min_val, max_val, 100)
    plt.plot(line_vals, line_vals, 'r--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Neg Log Error (Sampler=3)')
    plt.ylabel('Neg Log Error (Sampler=10)')
    plt.title('Comparison of Model Performance: Sampler=3 vs Sampler=10')
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Add a note about points above the line being better with sampler=10
    plt.figtext(0.5, 0.01, 
                "Points above the line: Sampler=10 performs better\nPoints below the line: Sampler=3 performs better", 
                ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_wilcoxon_test(pivot_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Run a one-sided Wilcoxon Signed-Rank test to check if sampler=10 performs better than sampler=3
    
    Args:
        pivot_df: Pivot table DataFrame
        
    Returns:
        Tuple of (statistic, p-value)
    """
    # Drop rows with missing values in either column 3 or 10
    test_df = pivot_df.dropna(subset=[3, 10])
    
    # Get the data
    sampler_3 = test_df[3]
    sampler_10 = test_df[10]
    
    # Run the one-sided Wilcoxon test
    # alternative='greater' tests whether sampler_10 is greater than sampler_3
    statistic, p_value = wilcoxon(sampler_10, sampler_3, alternative='greater')
    
    return statistic, p_value

def main():
    args = parse_args()
    
    # Load and combine data
    df = load_and_combine_data(args.csv_files)
    
    # Create pivot table
    pivot_df = create_pivot_table(df)
    
    # Save pivot table to CSV
    pivot_df.to_csv(args.pivot)
    print(f"Pivot table saved to {args.pivot}")
    
    # Create and save scatter plot
    create_scatter_plot(pivot_df, args.image)
    print(f"Scatter plot saved to {args.image}")
    
    # Run Wilcoxon test
    statistic, p_value = run_wilcoxon_test(pivot_df)
    
    # Save test results to file
    alpha = 0.05
    with open(args.stats_results, 'w') as f:
        f.write("Wilcoxon Signed-Rank Test Results\n")
        f.write("================================\n\n")
        f.write(f"Testing hypothesis: Sampler=10 produces higher Neg Log Error than Sampler=3\n\n")
        f.write(f"Statistic: {statistic}\n")
        f.write(f"P-value: {p_value:.6f}\n\n")
        
        if p_value < alpha:
            f.write(f"Result: Reject the null hypothesis (p < {alpha})\n")
            f.write("Conclusion: There is evidence that using 10 examples (Sampler=10) performs better than using 3 examples (Sampler=3)\n")
        else:
            f.write(f"Result: Fail to reject the null hypothesis (p >= {alpha})\n")
            f.write("Conclusion: There is insufficient evidence that using 10 examples (Sampler=10) performs better than using 3 examples (Sampler=3)\n")
        
        # Add some details about the data
        test_df = pivot_df.dropna(subset=[3, 10])
        sample_count = len(test_df)
        f.write(f"\nSample size: {sample_count} models with both Sampler=3 and Sampler=10 data\n")
        
        higher_with_10 = sum(test_df[10] > test_df[3])
        f.write(f"Models performing better with Sampler=10: {higher_with_10} ({higher_with_10/sample_count:.1%})\n")
        f.write(f"Models performing better with Sampler=3: {sample_count - higher_with_10} ({(sample_count - higher_with_10)/sample_count:.1%})\n")
    
    # Print a summary to console
    print(f"\nWilcoxon Signed-Rank Test Results saved to {args.stats_results}")
    print(f"P-value: {p_value:.6f}")
    
    if p_value < alpha:
        print(f"Result: Reject the null hypothesis (p < {alpha})")
        print("Conclusion: There is evidence that using 10 examples performs better than using 3 examples")
    else:
        print(f"Result: Fail to reject the null hypothesis (p >= {alpha})")
        print("Conclusion: There is insufficient evidence that using 10 examples performs better than using 3 examples")

if __name__ == "__main__":
    main()