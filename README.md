# narrative-learning

What if a text-based explanation was the machine learning model?


## Explanation

Traditional machine learning follows a pattern: train a model, then generate explanations so humans can understand its decision-making process. These explanations are meant to be clear enough that a human could theoretically reproduce the model's decisions. But what if we flipped this paradigm?
With the emergence of Large Language Models (LLMs) that can effectively parse and act on natural language instructions, we can now explore an intriguing possibility: using human-readable explanations as the model itself. Instead of treating explanations as post-hoc justifications of a black-box model, we can iteratively refine natural language rules that directly drive the decision-making process.
This approach offers several potential advantages:

- **Inherent Interpretability**: The model's logic is explicitly encoded in human-readable form from the start, eliminating the need for separate explanation methods

- **Interactive Refinement**: We can leverage LLMs to improve the rules based on performance, while maintaining human-understandable language

- **Verification by Inspection**: Domain experts can directly review, validate, and suggest improvements to the decision-making criteria

- **Flexible Deployment**: The same rules can be interpreted by different LLMs, potentially allowing for trade-offs between accuracy and computational efficiency

### Some research questions



- **Model Complexity**: How sophisticated does an LLM need to be to effectively interpret and apply natural language rules?

- **Training Efficiency**: How does the performance of narrative learning compare to traditional classification techniques in terms of sample efficiency and convergence?

- **Rule Evolution**: What patterns emerge in how the natural language rules evolve through iterations of refinement?

- **Context Dependence**: How sensitive is performance to the amount of historical context or number of examples provided?

- **Generalization**: How well do narrative rules learned from one domain transfer to related problems?

- **Regression**: This code shows how to do classifiers. How would we do narrative learning *regressors*?


## Usage

The old Makefile targets for building dataset databases no longer work.
Use the following sequence to create the PostgreSQL database, populate
each dataset and run the investigations.

1. Install PostgreSQL and ensure the `uv` tool is available.  If you
   want to restore a pre-made dump you can run `./envsetup.sh`
   (see the section on restoring from backup), otherwise install
   `postgresql` and `postgresql-client` manually.

2. Load the schema and default investigations:

```bash
psql "$POSTGRES_DSN" -f postgres-schemas/investigations_schema.sql
psql "$POSTGRES_DSN" -f postgres-schemas/investigations_data.sql
```

3. Generate or load each dataset. Synthetic datasets are produced with
`random_classification_data_generator.py`, while real CSV files are
loaded with `initialise_database.py`. Both write a configuration file in
`configs/` and prepare a SQLite template.

4. Import the dataset into PostgreSQL:

```bash
uv run import_dataset.py --investigation-id <id>
```

5. Calculate baseline metrics:

```bash
PGUSER=root uv run baseline.py --dataset <dataset>
```

6. Launch the investigation loop:

```bash
uv run investigate.py <investigation-id>
```

Copy everything from `outputs/*_results.csv` after each run and then
generate ensemble summaries:

```bash
python results_ensembling.py titanic --summary outputs/titanic_ensemble_summary.txt
python results_ensembling.py wisconsin --summary outputs/wisconsin_ensemble_summary.txt
python results_ensembling.py southgermancredit --summary outputs/southgermancredit_ensemble_summary.txt
python results_ensembling.py potions --summary outputs/potions_ensemble_summary.txt
python results_ensembling.py timetravel_insurance --summary outputs/timetravel_insurance_ensemble_summary.txt
python results_ensembling.py espionage --summary outputs/espionage_ensemble_summary.txt
```

The ensemble script stores results in the `ensemble_results` table and
orders them by model release date from `language_models`.
Some older models such as `gpt-4.5-preview` have been removed from the
OpenAI API and `gemini-2.0` now requires a paid account, so you may need
to substitute newer model names.





## To-do

- Document the usage a bit better, including all the programs we have

- Check to see if it's the overseer model or the underling model that has the most effect on the result

- Prompt complexity over time



## Ex to-do

- _Try phi4-mini as an evaluator_ -- it didn't work well, and gave nonsense results. Pity, it was 
  fast and cheap to run


### PostgreSQL utilities

The environment files under `envs/` can be imported into PostgreSQL using the
SQL statements in `postgres-schemas/investigations_data.sql`. The accompanying
schema is defined in `postgres-schemas/investigations_schema.sql` and creates
three tables:

* `datasets` – `dataset` primary key and `config_file` path for each dataset
* `models` – training and inference model names with optional `example_count`
  and `patience` values
* `investigations` – links a dataset and model, records the `sqlite_database`,
  `round_tracking_file`, optional `dump_file` and the current `round_number`.
* `dataset_provenance` – free-form text describing how each dataset was
  generated or obfuscated
  New investigations no longer default to round 1 – the value should be set
  explicitly from the associated round tracking file.

Load the schema and initial data with:

```bash
psql "$POSTGRES_DSN" -f postgres-schemas/investigations_schema.sql
psql "$POSTGRES_DSN" -f postgres-schemas/investigations_data.sql
```

Run the training loop with:

```bash
uv run investigate.py <investigation-id> [--quiet]
```

Use `--quiet` when running many investigations in parallel to
only print when each subprocess starts and stops.

`investigate.py` reads the settings for the given investigation ID from
PostgreSQL and updates the `round_number` field after each successful round.
The initial value should be loaded from the `round_tracking_file` referenced in
the `investigations` table (no loader script exists yet).  Each dataset has its
own tables such as `espionage_rounds` and `espionage_inferences`; the training
scripts use the dataset name from the configuration to construct these table
names.

The `check_round_consistency.py` helper reports how many investigations have a
`round_number` that does not match an entry in the relevant `*_rounds` table.
It also prints `UPDATE` statements to correct the values.
Run it with libpq environment variables set, for example:

```bash
PGUSER=root PGDATABASE=narrative ./check_round_consistency.py
```

## Website export

Run `./export_website.py` to generate static HTML pages in the `website/` directory. You can publish the site with:

```bash
rsync -av website/ merah.cassia.ifost.org.au:/var/www/vhosts/narrative-learning.symmachus.org/htdocs/
```

which will make it appear at <http://narrative-learning.symmachus.org/>.

## Restoring from a backup

`./envsetup.sh` installs PostgreSQL, creates the `root` and `narrative`
roles and downloads a compressed dump of the `narrative` database from
datadumps.ifost.org.au. Run this script if you want to replicate the
exact database used in previous experiments. It will also install the
`uv` package manager so that project scripts can be executed with
`uv run`.

