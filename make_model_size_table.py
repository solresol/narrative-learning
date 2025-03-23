#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a LaTeX table from model_details.json")
    parser.add_argument("--output", type=str, required=True, help="Output file path for LaTeX table")
    return parser.parse_args()


def load_model_details(file_path: str = "model_details.json") -> Dict[str, Any]:
    """Load model details from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def format_model_name(name: str) -> str:
    """Format model name for better readability in the table."""
    # Remove version numbers and other common patterns
    name = name.replace("-preview", "")
    name = name.replace("-20240307", "")
    name = name.replace("-20241022", "")
    name = name.replace("-20250219", "")
    name = name.replace("-2024-05-13", "")
    name = name.replace("-0125", "")
    name = name.replace("-0613", "")
    name = name.replace("-1106", "")
    name = name.replace("-vision", "")
    name = name.replace("-exp", "")
    name = name.replace(":latest", "")
    
    # Split by delimiter and capitalize
    parts = name.replace(":", "-").split("-")
    parts = [p.capitalize() for p in parts]
    
    # Handle special cases
    if "gpt" in name.lower() or "llama" in name.lower() or "phi" in name.lower():
        parts = [p.upper() if p.lower() in ["gpt", "llama", "phi"] else p for p in parts]
    
    # Join and return
    return " ".join(parts)


def get_parameter_size_display(size: int) -> str:
    """Format parameter size for display."""
    if size >= 1000:
        return f"{size/1000:.1f}T"
    else:
        return f"{size}B"


def create_latex_table(model_details: Dict[str, Any]) -> str:
    """Create a LaTeX table from model details."""
    # Start LaTeX table
    latex = "\\begin{table}[ht]\n"
    latex += "\\centering\n"
    latex += "\\caption{Model Size Details}\n"
    latex += "\\label{tab:model-details}\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{Size} & \\textbf{Reasoning} \\\\\n"
    latex += "\\midrule\n"
    
    # Sort models by parameter size (descending)
    sorted_models = sorted(
        model_details.items(), 
        key=lambda x: x[1]["parameters"], 
        reverse=True
    )
    
    # Add each model to the table
    for model_name, details in sorted_models:
        formatted_name = format_model_name(model_name)
        param_size = get_parameter_size_display(details["parameters"])
        reasoning = "Yes" if details.get("reasoning", False) else "No"
        
        latex += f"{formatted_name} & {param_size} & {reasoning} \\\\\n"
    
    # End LaTeX table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def main() -> None:
    """Main function."""
    args = parse_args()
    
    # Load model details
    model_details = load_model_details()
    
    # Create LaTeX table
    latex_table = create_latex_table(model_details)
    
    # Write to output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(latex_table)
    
    print(f"LaTeX table written to {args.output}")


if __name__ == "__main__":
    main()