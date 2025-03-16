#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sqlite3
import shutil
from pathlib import Path


def parse_env_file(file_path):
    """Parse the environment file and extract the database path."""
    database_path = None
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('export NARRATIVE_LEARNING_DATABASE='):
                database_path = line.strip().split('=', 1)[1]
                break
    return database_path


def check_job_status(task_name, env_file,  terminal_width=80, use_color=True):
    """Check the status of an ML training job by examining its SQLite database."""
    database_path = parse_env_file(env_file)
    
    if not database_path:
        return f"{task_name}: missing database path in env file"
    
    if not os.path.exists(database_path):
        return f"{task_name}: missing"
    
    try:
        conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Get the latest round with inferences
        cursor.execute("""
        SELECT round_id, count(*),
               min(unixepoch('now') - unixepoch(creation_time)) as seconds_ago
         FROM inferences
        GROUP BY round_id
        ORDER BY round_id DESC""")
        rounds = cursor.fetchall()
        
        # Filter rounds having count > 0
        rounds = [r for r in rounds if r[1] > 0]
        
        if not rounds:
            return f"{task_name}: no inferences found"
        
        latest_round, inference_count, seconds_ago = rounds[0]
        is_recent = False
        if seconds_ago is not None and seconds_ago < 1800:
            is_recent = True
            
        # Get the prompt for the latest round
        cursor.execute("SELECT prompt FROM rounds WHERE round_id = ?", (latest_round,))
        prompt_row = cursor.fetchone()
        
        if not prompt_row:
            return f"{task_name}: {inference_count} inferences, round #{latest_round}, prompt: N/A"
        
        # Format the prompt (first 40 chars, replace newlines with spaces)
        prompt = prompt_row[0]
        prompt = prompt.replace('\n', ' ')
        conn.close()

        # Calculate available space for prompt, accounting for the rest of the output
        base_output = f"{task_name}: {inference_count} inferences, round #{latest_round}, prompt: \""
        remaining_width = terminal_width - len(base_output) - 1  # -1 for closing quote
        
        # Ensure we have some minimum space for the prompt
        if remaining_width < 10:
            remaining_width = 40
            
        prompt_preview = prompt[:remaining_width-3] + "..." if len(prompt) > remaining_width else prompt

        result = f"{task_name}: {inference_count} inferences, round #{latest_round}, prompt: \"{prompt_preview}\""
        # Apply ANSI color codes for recent activity if requested
        if use_color and is_recent:
            return f"\033[1;32m{result}\033[0m"  # Bold green text
        else:
            return result
    except sqlite3.Error as e:
        return f"{task_name}: error accessing database - {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Monitor ML training jobs.')
    parser.add_argument('--envs', default='./envs', 
                      help='Directory containing environment files (default: ./envs)')
    parser.add_argument('--width', type=int, default=None,
                      help='Override terminal width detection with custom width')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output')
    parser.add_argument("--glob-pattern", default="*/*.env")
    args = parser.parse_args()

    terminal_width = args.width
    if terminal_width is None:
        try:
            terminal_width, _ = shutil.get_terminal_size()
        except Exception:
            # Default width if detection fails
            terminal_width = 80

    print(f"Terminal width detected: {terminal_width} columns")
    
    # Find all environment files
    env_pattern = os.path.join(args.envs, args.glob_pattern)
    env_files = glob.glob(env_pattern)
    
    if not env_files:
        print(f"No environment files found in {args.envs}")
        return
    
    # Process each environment file
    for env_file in sorted(env_files):
        # Extract task name from directory structure
        rel_path = os.path.relpath(env_file, args.envs)
        dir_name = os.path.dirname(rel_path)
        base_name = os.path.basename(os.path.dirname(env_file))
        file_name = os.path.basename(env_file).replace('.env', '')
        
        if dir_name == file_name:
            task_name = dir_name
        else:
            task_name = f"{dir_name}/{file_name}"
        
        status = check_job_status(task_name, env_file, terminal_width, not args.no_color)
        print(status)


if __name__ == '__main__':
    main()
