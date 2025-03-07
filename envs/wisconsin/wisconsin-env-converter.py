#!/usr/bin/env python3

import re
import sys
import os

def convert_env_file(input_file, output_file=None):
    """
    Convert an environment file from the first format to the second format.
    
    Rules:
    * Add NARRATIVE_LEARNING_CONFIG=configs/wisconsin_exoplanets.config.json
    * Change NARRATIVE_LEARNING_DATABASE to say wisconsin_exoplanets instead of titanic-medical
    * Keep the models the same
    * Add .wisconsin after .round-tracking-file in ROUND_TRACKING_FILE
    * Keep the NARRATIVE_LEARNING_EXAMPLE_COUNT the same
    """
    
    # Read input file
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    
    # Make the required changes
    
    # 1. Add NARRATIVE_LEARNING_CONFIG if not present
    config_line = "export NARRATIVE_LEARNING_CONFIG=configs/wisconsin_exoplanets.config.json"
    if "NARRATIVE_LEARNING_CONFIG" not in content:
        lines = content.split("\n")
        # Insert at the beginning
        lines.insert(0, config_line)
        content = "\n".join(lines)
    
    # 2. Change NARRATIVE_LEARNING_DATABASE to use wisconsin_exoplanets
    content = re.sub(
        r'(export NARRATIVE_LEARNING_DATABASE=results/)titanic_medical(-.*\.sqlite)',
        r'\1wisconsin_exoplanets\2',
        content
    )
    
    # 3. Add .wisconsin to ROUND_TRACKING_FILE
    content = re.sub(
        r'(export ROUND_TRACKING_FILE=\.round-tracking-file)(\..*)',
        r'\1.wisconsin\2',
        content
    )
    
    # Remove duplicate lines while preserving order
    lines = content.split("\n")
    unique_lines = []
    seen = set()
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Get the variable name (everything before the = sign)
        var_name = line.split('=')[0] if '=' in line else line
        
        if var_name not in seen:
            seen.add(var_name)
            unique_lines.append(line)
    
    # Write to output file or stdout
    if output_file:
        with open(output_file, 'w') as f:
            f.write("\n".join(unique_lines))
        print(f"Converted file saved to {output_file}")
    else:
        print("\n".join(unique_lines))
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wisconsin-env-converter.py input_file [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_env_file(input_file, output_file)
    sys.exit(0 if success else 1)
