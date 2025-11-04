# Standalone Narrative Learning TUI Specification

## 1. Purpose
The `standalone.py` program provides a colourful terminal user interface (TUI) that demonstrates the narrative learning workflow without requiring the PostgreSQL infrastructure used by the main system. It focuses on a single language-model backend, stores state in a lightweight SQLite database, and limits investigations to two feature variables so that the experience can be shown live.

## 2. Objectives
- Offer an interactive demo that mirrors the core loop of the PostgreSQL-based investigation runner.
- Require minimal environment setup: Python 3.11+, `sqlite3`, and TUI/LLM client libraries.
- Showcase progress updates, intermediate metrics, and narrative model evolution inside the terminal.
- Support importing datasets from DataPainter SQLite files, running iterative rounds, and viewing results for exactly one preconfigured model.

## 3. Out of Scope
- Managing multiple simultaneous models, ensembles, or PostgreSQL integrations.
- Persisting more than two feature variables per dataset row.
- Automated scheduling, background execution, or remote LLM dispatching.

## 4. High-Level Architecture
1. **Configuration Layer**
   - YAML or TOML file describing the single model (name, temperature, API key env var) and dataset path.
   - CLI arguments allow overriding dataset location, max rounds, and verbosity.
2. **Persistence Layer**
   - SQLite file created alongside the dataset, schema inspired by `*_rounds` tables but simplified.
   - Tables:
     - `dataset(feature_a TEXT, feature_b TEXT, label TEXT)`
     - `rounds(id INTEGER PK, created_at, prompt TEXT, accuracy REAL, notes TEXT)`
     - `examples(round_id FK, feature_a TEXT, feature_b TEXT, label TEXT, prediction TEXT, correct INTEGER)`
3. **Model Interface**
   - Thin abstraction for the single supported model (e.g., OpenAI, Anthropic) with retry/backoff.
   - Deterministic stub mode for offline demos.
4. **TUI Layer**
   - Built with `textual` or `rich` to provide panels for dataset info, current prompt, progress bars, and logs.
   - Keyboard shortcuts: `n` for next round, `r` to regenerate, `q` to quit, `s` to save snapshot.
   - Async event loop keeps UI responsive while models train and evaluate, enabling concurrent interaction and progress updates.
   - Overseer panel summarises baseline explainable model metrics (accuracy and Kendall Tau) and colour-codes data point outcomes (correct, incorrect, mixed region).
   - Underling panel tracks active hypothesis evaluation with a progress bar, colour legend, and scrollable coordinate list highlighting the currently processed item.

## 5. User Workflow
1. Launch `uv run standalone.py --dataset data/demo.sqlite --table tablename`.
2. **Landing Screen**: shows dataset summary (row count, label distribution), active model, and instructions.
3. **Round Execution**:
   - User triggers next round (`n`).
   - TUI displays spinner/progress bar while the model evaluates both features on validation split.
   - Results panel updates with accuracy, misclassified examples, and narrative prompt adjustments.
   - Baseline explainable models run alongside the primary model; their accuracy and Kendall Tau scores appear in a comparative table with coloured correctness markers.
   - Underling view shows progress through hypothesis evaluation, including coordinates of the points being processed.
4. **History Browsing**:
   - Left sidebar lists past rounds; selecting one shows stored prompt, metrics, and example breakdown.
   - Overseer history logs every proposed hypothesis; selecting one recolours dataset points to reflect evaluation status (correct, incorrect, not yet evaluated).
5. **Completion**:
   - User exits via `q`; final summary written to stdout and stored in SQLite.

## 6. Progress & Telemetry
- **Round Progress**: real-time bar indicating stages: preparing data → calling model → scoring → saving.
- **Inline Metrics**: accuracy, per-class precision/recall, confusion matrix (2x2) rendered as table.
- **Event Log**: scrolling panel with colour-coded entries (info, warning, error).

## 7. Dataset Handling
- DataPainter SQLite database ingestion with automatic metadata reading.
- Dataset format: tables with (x, y, target) columns representing two-dimensional labeled data.
- Table selection via `--table` argument; defaults to first table in metadata if not specified.
- 80/20 train-validation split performed deterministically.
- Option `--shuffle-seed` controls reproducibility.
- Validation results persisted; training subset fed to prompt builder.

## 8. Narrative Prompt Management
- Template stored in SQLite along with round metadata.
- Each round uses previous prompt plus auto-generated critique from validation errors.
- User can edit prompt inside TUI via modal text editor; saves back to database.

## 9. Error Handling & Resilience
- Graceful handling of API failures with retry UI feedback and ability to switch to stub mode on the fly.
- Validation preventing dataset load if metadata table is missing or specified table doesn't exist.
- Integrity checks ensure only one concurrent run; lock file in same directory as SQLite database.

## 10. Extensibility Hooks
- Modular adapters for future support of additional models.
- CLI flag `--export-json` to dump round history for offline analysis.
- Theme configuration file for customizing colours.

## 11. Testing Strategy
- Unit tests for SQLite persistence, dataset ingestion, and prompt update logic using pytest + temporary files.
- Snapshot tests for TUI layout using `textual`'s pilot mode.
- CLI smoke test verifying round execution in stub mode completes and writes expected database rows.

## 12. Deployment & Distribution
- Package as a self-contained script runnable via `uv run standalone.py`.
- Document environment variables required for API access.
- Provide demo DataPainter dataset and stub model credentials for offline showcases.
- DataPainter format documentation available in `DATAPAINTER_FORMAT.md` and `DATAPAINTER_USAGE.md`.
