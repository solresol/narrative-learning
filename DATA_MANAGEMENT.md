# Data Management Overview

This document describes how datasets, configuration files and generated
artifacts are organised in this repository. It also highlights which
files are kept under version control, which ones are produced by the
programs and some ideas for improving the layout.

## Directory layout

- `datasets/` – raw CSV datasets used for creating the SQLite
  databases. Currently the repository contains `breast_cancer.csv`,
  `southgermancredit.csv` and `titanic.csv`【F:datasets】.
- `configs/` – JSON configuration files describing the table names,
  columns and other metadata for each dataset. The programs read these
  files to know how to operate on a particular database.
- `dbtemplates/` – ready made SQL or SQLite files used to initialise
  new experiment databases. For example `dbtemplates/wisconsin_exoplanets.sql`
  contains the schema and sample rows for the Wisconsin task.
- `envs/` – collections of environment files for every dataset/model
  combination. An environment file provides paths to the config and
  database to use as well as which models should perform training and
  inference. Example:
  ```bash
  export NARRATIVE_LEARNING_CONFIG=configs/titanic_medical.config.json
  export NARRATIVE_LEARNING_DATABASE=results/titanic_medical-anthropic.sqlite
  export NARRATIVE_LEARNING_TRAINING_MODEL=claude-3-5-haiku-20241022
  export NARRATIVE_LEARNING_INFERENCE_MODEL=claude-3-5-sonnet-20241022
  export ROUND_TRACKING_FILE=.round-tracking-file.anthropic
  export NARRATIVE_LEARNING_EXAMPLE_COUNT=3
  export NARRATIVE_LEARNING_DUMP=dumps/titanic_medical-anthropic.sql
  ```【F:envs/titanic/anthropic.env†L1-L9】
- `results/` – directory for generated SQLite databases, best round text
  files and baseline JSON statistics. The `.gitignore` here prevents these
  files from being committed:
  ```
  *.sqlite
  *.txt
  *.csv
  *.json
  *.bak*
  ```【F:results/.gitignore†L1-L5】
- `outputs/` – summary CSV files and charts produced from the databases.
  A local `.gitignore` ignores PNGs and fitted distribution text files【F:outputs/.gitignore†L1-L2】, although many of these artefacts are already checked in.
- `dumps/` – SQL dumps created when running `loop.sh`. The dump path is
  controlled by the `NARRATIVE_LEARNING_DUMP` variable in each env file.
- `obfuscations/` and `conversions/` – directories intended to store
  obfuscation plans and intermediate conversion scripts.

## Programs and their data

- `initialise_titanic.py` and `initialise_database.py` load a CSV from
  `datasets/` and produce an SQLite database under `results/`. They also
  generate or update the dataset configuration in `configs/`.
- `obfuscation_plan_generator.py` takes a CSV and guidelines file and
  writes an obfuscation plan SQLite database (often kept in
  `obfuscations/`).
- `baseline.py` trains baseline ML models using a configuration file and
  writes a JSON summary to `results/`.
- `loop.sh` orchestrates repeated calls to `train.py` and
  `process_round.py` while writing progress to the database defined in
  the environment file and optionally dumping it to `dumps/`.
- `create_task_csv_file.py` scans the env files and their databases to
  produce consolidated CSV files in `outputs/`.
- A collection of `results_*.py` scripts and the Makefile read those CSV
  files to generate charts and LaTeX tables also stored in `outputs/`.

## Version controlled vs generated

The repository keeps source datasets (`datasets/`), configuration JSON
files (`configs/`), environment templates (`envs/`) and database
initialisation scripts (`dbtemplates/`) under git. Generated SQLite
files, dumps and intermediate results are ignored through
`results/.gitignore`. Some output charts and CSVs are committed even
though `outputs/.gitignore` suggests otherwise.

## Possible improvements

- **Separate raw and generated data** – adopting the familiar
  `data/raw`, `data/processed` and `results/` structure would make it
  clearer which artefacts are permanent and which can be reproduced.
- **Avoid committing generated charts** – `outputs/` could remain
  ignored entirely, with the Makefile or a dedicated script responsible
  for regenerating plots as needed.
- **Environment files as templates** – rather than storing every
  combination under `envs/`, provide a template file and generate the
  specific `.env` files on demand (for example using `env_generator.py`).
- **Use one configuration location** – centralising path settings (for
  example via a single `config/` directory or a small YAML/JSON file)
  would reduce duplication across env files.
- **Consider a data versioning tool** – if experiment databases need to
  be retained, tools like DVC or MLflow can track them without storing
  them directly in git.

Following these practices would make it easier to recreate results and to
understand which files should be committed to version control.

## Consolidating into a single database

While each dataset currently lives in its own SQLite file under `results/`,
it is possible to keep all tables in one database. This could be a single
SQLite file or a PostgreSQL instance. In that arrangement:

- `initialise_database.py` would create tables for every dataset in the
  same database rather than separate files.
- Environment files would point to one database path or connection string
  and optionally specify which dataset's tables to operate on.
- `train.py`, `process_round.py` and all results scripts would connect to
  the same database and query the dataset-specific tables.
- Using PostgreSQL would offer concurrent access and easier backups but
  would require credentials in the environment files and minor code
  changes to use `psycopg2` instead of `sqlite3`.

Consolidating the data like this would simplify path management and make it
clearer where results live, though the code would need small updates to
handle schema migrations and multiple datasets in one place.
