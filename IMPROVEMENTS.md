# IMPROVEMENTS.md

*Analysis date: 2026-07-11*

Narrative-learning is a research project exploring "the explanation *is* the model": natural-language classification rules iteratively refined by LLMs, benchmarked across many models and obfuscated datasets (Titanic-as-medical, Wisconsin-as-exoplanets, South German Credit, plus synthetic ones like `espionage` and `timetravel_insurance`). It has migrated from SQLite to PostgreSQL (see `POSTGRESQL_MIGRATION.md`), grown a large 1,900-line `standalone.py` TUI, and a static-website exporter (`export_website.py`). The working tree is clean, but recent git history ("Temporarily disable highlighting to fix hang issue", waves of debug-logging commits) shows the standalone TUI is mid-debugging, and the repo root is cluttered with editor backups, debug logs, and multi-MB SQLite artefacts.

## Bugs & Unfinished Work

- **standalone.py highlighting is disabled, not fixed.** Commit `a65012d` "Temporarily disable highlighting to fix hang issue" plus five preceding commits of "extremely detailed logging" indicate an unresolved hang in `highlight_examples`. Either fix the root cause (likely a Textual reactive/refresh loop or O(n²) matching over the data table) or remove the dead highlighting code path and its debug logging. The logging commits (`66dbe63`, `add0f39`, `23bb5cd`) should be reverted once the bug is found — `debug.log` in the repo root is a symptom.
- **README contradicts the Makefile.** README says "The old Makefile targets for building dataset databases no longer work", yet `CLAUDE.md` still documents `make wisconsin` etc. as the build commands. Reconcile: delete dead Makefile targets or update CLAUDE.md so agents don't run broken commands.
- **`TODO.md` is not a TODO list** — it's a pasted LLM conversation about redesigning the website (Plotly, Tabulator, CSS themes, Jinja2 templating). If those recommendations are still wanted, distil them into actual checklist items; the actionable core is: `export_website.py` emits raw unstyled HTML with static Matplotlib PNGs and should get a shared template, a stylesheet, interactive charts, and CSV download links. `draw_baselines` in `chartutils.py` hardcodes baseline names — add a name→label mapping shared between charts and tables.

## Housekeeping (high value, low effort)

- **Delete editor backup files and add `*~` to `.gitignore`**: `llmcall.py~`, `report-script.py~`, `schema.sql~`, `loop.sh~`, `make_results_table.py~`, `obfuscation_plan_generator.py~`, `random_classification_data_generator.py~`, `list_missing_patients.py~`, `show_round_patient_data.py~`, `openai.env~`, `titanic_results.csv~`, `wisconsin_results.csv~`. Some of these (`openai.env~`) are env files — even though a scan found no literal API keys committed, `*.env` and `*.env~` should be gitignored as a class.
- **Remove committed binary databases and logs**: seven `titanic_medical-*.sqlite` files sit in the repo root, plus `dbtemplates/sgc_coral.sqlite` and `debug.log`. Post-Postgres-migration these are legacy artefacts; move them to `legacy-outputs/` or drop them and rely on `backup_narrative.sh` dumps.
- **Repo root sprawl**: ~90 top-level scripts. Group them into packages, e.g. `analysis/` (all `results_*.py`, `make_result_charts.py`, `lexicostatistics.py`), `admin/` (`delete_*.py`, `update_round_*.py`, `check_*.py`, `find_*.py`, `cleanup*.sql`), keeping only the core loop (`train.py`, `predict.py`, `process_round.py`, `investigate.py`, `datasetconfig.py`) at top level. The `modules/` directory already exists — use it.

## Modernization

- **Kill `requirements.txt`.** The project already has `pyproject.toml` and `uv.lock`; `requirements.txt` duplicates (and will drift from) them. Verify every dependency in requirements.txt (`imodels`, `interpret`, `umap-learn`, `psycopg2-binary`…) is captured via `uv add`, then delete the file and update README/CLAUDE.md to say `uv run script.py` throughout. Also review `uvbootstrapper.py` — with a lockfile committed it is probably redundant.
- **Consolidate the SQLite/PostgreSQL split.** `schema.sql`, `dbtemplates/`, `legacy-schemas/`, and `conversions/` coexist with `postgres-schemas/`. If the Postgres migration is done, move all SQLite-era schema/SQL files (`update_rounds.sql`, `delete_duplicate_rounds.sql`, `cleanup-legacy-titanic.sql`, etc.) under `legacy-schemas/` and say so in `DATA_MANAGEMENT.md`. Note the DB host convention lives in personal memory (raksasa) — record connection expectations in README so the project is reproducible by others.

## Testing

- Only three test files exist (`test_env_settings.py`, `test_postgres.py`, `test_standalone.py`) plus a `tests/` dir — but they're split between repo root and `tests/`. Move all tests into `tests/` and wire up `uv run pytest` in CI (a GitHub Actions workflow appears absent).
- The recent hang bug is exactly the kind of thing tests would catch: add a non-TUI unit test for the highlighting/matching logic in `standalone.py` (the pure functions like `split_dataset` and `extract_valid_predictions` are testable today).
- `datasetconfig.py` (692 lines) is the core abstraction every script depends on — it deserves direct test coverage with a fixture database (fixtures dir already exists with `hamsters.sql`).

## Documentation

- README's Usage section is good but stops at setup; add a "run one full investigation end-to-end" example (env file → `process_round.py` → `investigate.py` → results scripts).
- `standalone_spec.md` and `standalone_tui_concepts.md` overlap; merge into one doc once the TUI stabilizes.
- Document what each `envs/<dataset>/*.env` file controls and regenerate them via `env_generator.py` rather than hand-editing ~dozens of near-identical files.

## Quick Wins

1. `git rm` the `*~` files and `debug.log`; extend `.gitignore` (`*~`, `*.log`, `*.env`, root `*.sqlite`).
2. Delete `requirements.txt` after confirming `uv.lock` parity.
3. Fix or remove the disabled highlighting path in `standalone.py` and strip the temporary debug logging.
4. Update `CLAUDE.md` build commands to match the post-Postgres reality.
5. Convert `TODO.md` into a real checklist (the website redesign items are its only substantive content).
