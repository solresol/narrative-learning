# Narrative Learning Project Guide

## Build Commands
- `make`: Build everything
- `make wisconsin`: Build Wisconsin dataset results
- `make titanic`: Build Titanic dataset results
- `uv run <script.py>`: Run Python scripts with UV package manager
- `sqlite3 <file.sqlite> < file.sql`: Create/update SQLite database

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
- `envs/`: Environment files for different models
- `results/`: Output files and SQLite databases
- `outputs/`: Generated charts and CSV results

## Key Scripts
- `initialise_database.py`: Set up task databases
- `train.py`: Run training iterations
- `predict.py`: Make predictions using trained models
- `report-script.py`: Generate performance reports
- `make_result_charts.py`: Create visualization charts