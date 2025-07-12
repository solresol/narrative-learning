# Narrative Learning Project Guide

## Project Overview
This project explores using human-readable explanations as machine learning models themselves. By leveraging Large Language Models (LLMs) to interpret and apply natural language rules, we can create inherently interpretable models that can be refined interactively.

## Available Datasets
- **Wisconsin Breast Cancer Dataset**: Classified as exoplanets scenario
- **Titanic Survival Dataset**: Classified as medical scenario
- **South German Credit**: Financial risk assessment scenario

## Build Commands
- `make`: Build everything (all datasets and analysis)
- `make wisconsin`: Build Wisconsin dataset results
- `make titanic`: Build Titanic dataset results
- `make southgermancredit`: Build South German Credit dataset results
- `make ensembles`: Generate ensemble model results
- `uv run <script.py>`: Run Python scripts with UV package manager
- `sqlite3 <file.sqlite> < file.sql`: Create/update SQLite database

## Model Evaluation
The project evaluates various LLMs including:
- OpenAI models (GPT-3.5, GPT-4, GPT-4o, GPT-4.5)
- Anthropic models (Claude, Claude 3, Claude 3.5, Claude 3.7)
- Google models (Gemini Pro, Gemini 1.5)
- Other models (Gemma, Llama, etc.)

## Analysis Scripts
- `results_chart_by_size.py`: Compare model performance vs. model size
- `results_chart_by_elo.py`: Compare model performance vs. ELO rating
- `results_error_rate_by_wordcount.py`: Analyze error rates relative to prompt complexity
- `results_error_rate_by_herdan.py`: Analyze error rates relative to lexical complexity
- `results_ensembling.py`: Create ensemble models from multiple base models
  - Uses PostgreSQL to read investigations for a dataset
  - Integrates with the `language_models` table to track model release dates
  - Stores results in the `ensemble_results` table for later analysis
  - Example: `python results_ensembling.py titanic --summary outputs/titanic_ensemble_summary.txt`
- `resultssampleimpact.py`: Measure the impact of sample count on model performance

## Code Style
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Docstrings**: Google-style format (Args/Returns/Raises)
- **Typing**: Use type hints extensively (List, Dict, Optional, Union, etc.)
- **Error Handling**: Custom exception classes for specific errors
- **Imports**: Standard library first, then third-party, finally local modules

## Project Structure
- `configs/`: Configuration JSON files for datasets
- `datasets/`: CSV data files
- `dbtemplates/`: SQL templates for database initialization
- `envs/`: Environment files for different models (structure: envs/{dataset}/{model}.env)
- `modules/`: Core functionality modules
- `results/`: Output files and SQLite databases
- `outputs/`: Generated charts, tables, and CSV results
- `obfuscations/`: Dataset obfuscation plans
- `conversions/`: Dataset conversion/encoding guidelines
- `postgres-schemas/model_release_dates.sql`: Table definition and data for language models used in chronological ensembling
- `postgres-schemas/ensemble_results.sql`: Schema for storing ensemble evaluation results

## Key Scripts
- `initialise_database.py`: Set up task databases
- `train.py`: Run training iterations
- `predict.py`: Make predictions using trained models
- `report-script.py`: Generate performance reports
- `make_result_charts.py`: Create visualization charts
- `create_task_csv_file.py`: Generate CSV results from environment files
- `lexicostatistics.py`: Calculate Herdan and Zipf coefficients for a
  language model after all investigations have completed
- `env_settings.py`: Parse and validate model environment settings
- `resultdistribution.py`: Generate distribution charts for model outputs
- `baseline.py`: Create baseline model performance metrics
- `obfuscation_plan_generator.py`: Generate dataset obfuscation plans