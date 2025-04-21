#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

def generate_env_files(datasets, models, base_dir):
    """Generate environment files for specified datasets and models.
    
    Args:
        datasets: List of dataset names 
        models: List of model names
        base_dir: Base directory for the project
    """
    for dataset in datasets:
        # Create the directory if it doesn't exist
        dataset_dir = os.path.join(base_dir, "envs", dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Config file name is dataset.config.json
        config_file = f"configs/{dataset}.config.json"
        
        for model in models:
            # Determine example count (10 if model ends with '10', otherwise 3)
            example_count = "10" if model.endswith("10") else "3"
            
            # Set model names according to pattern
            database_path = f"results/{dataset}-{model}.sqlite"
            
            # Use gpt-4o-mini for inference across all models
            inference_model = "gpt-4o-mini"
            
            # The training model should closely match the filename
            # Comprehensive model mapping dictionary
            model_mapping = {
                # OpenAI models
                "openai": "gpt-4o",
                "openai10": "gpt-4o", 
                "openai10o1": "o1",
                "openailong": "gpt-4o-mini",
                "openai41": "gpt-4.1",
                "openai4110": "gpt-4.1",
                "openai45": "gpt-4.5-preview",
                "openai4510": "gpt-4.5-preview",
                "openaio1": "o1",
                "openaio3": "o3",
                "openaio310": "o3",
                
                # Anthropic models
                "anthropic": "claude-3-5-haiku-20241022",
                "anthropic10": "claude-3-5-haiku-20241022",
                "anthropic37": "claude-3-7-sonnet-20250219",
                "anthropic3710": "claude-3-7-sonnet-20250219",
                
                # Google models
                "gemini": "gemini-2.0-flash",
                "gemini10": "gemini-2.0-flash",
                "gemini25": "gemini-2.5-pro-exp-03-25",
                "gemini2510": "gemini-2.5-pro-exp-03-25", 
                "geminipro": "gemini-2.0-pro-exp",
                "geminipro10": "gemini-2.0-pro-exp",
                
                # Other models
                "gemma": "gemma3:27b",
                "gemma3": "gemma3:27b",
                "llama": "llama3.3:latest",
                "llamaphi": "llama3.3:latest",
                "phi": "phi4:latest",
                "falcon": "falcon:latest",
                "falcon10": "falcon:latest",
                "deepseek": "deepseek-llm:latest",
                "cogito": "cogito:latest",
                "qwq": "qwq:latest"
            }
            
            # Get the training model from the mapping, with fallback to gpt-4o-mini
            training_model = model_mapping.get(model, "gpt-4o-mini")
            
            # Round tracking file follows the pattern
            round_tracking_file = f".round-tracking-file.{dataset}.{model}"
            
            # Dump file is similar to database but with different extension and directory
            dump_file = f"dumps/{dataset}-{model}.sql"
            
            # Generate the content for the .env file
            env_content = f"""export NARRATIVE_LEARNING_CONFIG={config_file}
export NARRATIVE_LEARNING_DATABASE={database_path}
export NARRATIVE_LEARNING_TRAINING_MODEL={training_model}
export NARRATIVE_LEARNING_INFERENCE_MODEL={inference_model}
export ROUND_TRACKING_FILE={round_tracking_file}
export NARRATIVE_LEARNING_EXAMPLE_COUNT={example_count}
export NARRATIVE_LEARNING_DUMP={dump_file}
"""
            
            # Write the env file
            env_file_path = os.path.join(dataset_dir, f"{model}.env")
            with open(env_file_path, "w") as f:
                f.write(env_content)
            print(f"Created {env_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate environment files for datasets")
    parser.add_argument("--datasets", nargs="+", default=["espionage", "timetravel_insurance", "potions"],
                        help="List of datasets to generate env files for")
    parser.add_argument("--models", nargs="+", 
                        default=["openai", "openai10", "openai10o1", "openai45", "openai4510", 
                                "openailong", "openaio1", "anthropic37", "anthropic3710", 
                                "gemini", "geminipro", "gemini10", "geminipro10", 
                                "anthropic", "anthropic10", "gemma", "gemini25", 
                                "openaio3", "openaio310", "openai41", "openai4110"],
                        help="List of models to generate env files for")
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory of the project")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute if needed
    base_dir = os.path.abspath(args.base_dir)
    
    generate_env_files(args.datasets, args.models, base_dir)
    print(f"Generated environment files for {len(args.datasets)} datasets and {len(args.models)} models")

if __name__ == "__main__":
    main()
