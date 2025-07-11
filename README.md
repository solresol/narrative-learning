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

Hopefully, you should be able to say `make` and it should build everything.

In practice, I'm not sure it all works. I tend to do this:

`make potions-databases potions-baseline potions-best outputs/potions_results.csv`


(for each of `titanic`, `southgermancredit`, `wisconsin`, `timetravel` and `espionage`)

After the tables are created, calculate baseline metrics with:

```bash
PGUSER=root uv run baseline.py --dataset <dataset>
```

Run this once per dataset to fill the `baseline_results` table.

Then copy everything from `outputs/*_results.csv`

Then, run the ensemble analysis:

```bash
# Generate ensemble results for each dataset stored in PostgreSQL
python results_ensembling.py titanic --output outputs/titanic_ensemble.csv --summary outputs/titanic_ensemble_summary.txt
python results_ensembling.py wisconsin --output outputs/wisconsin_ensemble.csv --summary outputs/wisconsin_ensemble_summary.txt
python results_ensembling.py southgermancredit --output outputs/southgermancredit_ensemble.csv --summary outputs/southgermancredit_ensemble_summary.txt
python results_ensembling.py potions --output outputs/potions_ensemble.csv --summary outputs/potions_ensemble_summary.txt
python results_ensembling.py timetravel_insurance --output outputs/timetravel_insurance_ensemble.csv --summary outputs/timetravel_insurance_ensemble_summary.txt
python results_ensembling.py espionage --output outputs/espionage_ensemble.csv --summary outputs/espionage_ensemble_summary.txt
```

The ensemble script will automatically organize results by model release dates from the `release-dates.csv` file.

Copy these into the `papers/narrative-learning` directory.





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
  New investigations no longer default to round 1 – the value should be set
  explicitly from the associated round tracking file.

Load the schema and initial data with:

```bash
psql "$POSTGRES_DSN" -f postgres-schemas/investigations_schema.sql
psql "$POSTGRES_DSN" -f postgres-schemas/investigations_data.sql
```

Run the training loop with:

```bash
uv run investigate.py <investigation-id>
```

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

