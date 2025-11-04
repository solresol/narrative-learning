"""Standalone Narrative Learning TUI implementation.

This module implements a standalone demonstration of the narrative learning
workflow using the Textual terminal user-interface framework.  The design is
guided by the specification in ``standalone_spec.md`` and focuses on
supporting a single language-model backend with lightweight SQLite
persistence.

Run the application with:
    uv run standalone.py --dataset path/to.csv
    uv run standalone.py --dataset path/to.sqlite --table tablename
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import dataclasses
import datetime as dt
import json
import os
import random
import sqlite3
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    tomllib = None  # type: ignore[assignment]

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

from rich.markdown import Markdown as RichMarkdown
from textual import log
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    RichLog,
    Static,
)

try:  # Textual 0.29+
    from textual.widgets import TextArea
except ImportError:  # pragma: no cover - fallback for older Textual
    TextArea = None  # type: ignore[assignment]


###############################################################################
# Data models and configuration
###############################################################################


@dataclass(slots=True)
class DatasetRow:
    """A single dataset record comprising two features and a label."""

    feature_a: str
    feature_b: str
    label: str


@dataclass(slots=True)
class DatasetSplit:
    """Train/validation split for the demo dataset."""

    train: List[DatasetRow]
    validation: List[DatasetRow]


@dataclass(slots=True)
class RoundExample:
    """Evaluation result for a single validation row."""

    feature_a: str
    feature_b: str
    label: str
    prediction: str
    correct: bool


@dataclass(slots=True)
class RoundMetrics:
    """Aggregate metrics produced for a round."""

    accuracy: float
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[Tuple[str, str], int]
    notes: str


@dataclass(slots=True)
class RoundRecord:
    """Round details persisted in SQLite and displayed in the UI."""

    id: int
    created_at: dt.datetime
    prompt: str
    metrics: RoundMetrics
    examples: List[RoundExample]


@dataclass(slots=True)
class BaselineMetrics:
    """Metrics describing a lightweight explainable baseline."""

    name: str
    accuracy: float
    kendall_tau: float


@dataclass(slots=True)
class AppConfig:
    """Configuration loaded from disk and CLI overrides."""

    model_name: str = "Stub Narrative Model"
    temperature: float = 0.2
    api_key_env: Optional[str] = None
    dataset_path: Optional[Path] = None
    stub_mode: bool = True
    theme_path: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[Path]) -> "AppConfig":
        if path is None:
            return cls()
        data: Dict[str, Any]
        text = Path(path).read_bytes()
        if path.suffix in {".toml", ""} and tomllib is not None:
            data = tomllib.loads(text.decode("utf-8"))
        elif path.suffix in {".yaml", ".yml"} and yaml is not None:
            data = yaml.safe_load(text)
        else:
            raise RuntimeError(
                "Unsupported config format. Use TOML or YAML and install PyYAML for YAML support."
            )
        if not isinstance(data, dict):
            raise TypeError("Config root must be a mapping")
        return cls(
            model_name=str(data.get("model_name", cls.model_name)),
            temperature=float(data.get("temperature", cls.temperature)),
            api_key_env=data.get("api_key_env"),
            dataset_path=Path(data["dataset_path"]) if data.get("dataset_path") else None,
            stub_mode=bool(data.get("stub_mode", cls.stub_mode)),
            theme_path=Path(data["theme_path"]) if data.get("theme_path") else None,
        )


###############################################################################
# Dataset ingestion
###############################################################################


def load_dataset_csv(path: Path) -> List[DatasetRow]:
    """Load dataset rows from a CSV file.

    The CSV must contain the columns ``feature_a, feature_b, label`` in that
    order.  Extra columns trigger a validation error.
    """

    rows: List[DatasetRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = ["feature_a", "feature_b", "label"]
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"CSV header must be exactly {expected_fields}, received {reader.fieldnames!r}"
            )
        for entry in reader:
            rows.append(
                DatasetRow(
                    feature_a=str(entry["feature_a"]),
                    feature_b=str(entry["feature_b"]),
                    label=str(entry["label"]),
                )
            )
    if not rows:
        raise ValueError("Dataset is empty; supply at least one row")
    return rows


def load_dataset_datapainter(path: Path, table_name: Optional[str] = None) -> List[DatasetRow]:
    """Load dataset rows from a DataPainter SQLite file.

    DataPainter files contain a metadata table with information about available
    data tables, and separate tables containing (x, y, target) data points.

    Args:
        path: Path to the DataPainter SQLite database file
        table_name: Optional name of the table to load. If None, uses the first
                   table found in metadata.

    Returns:
        List of DatasetRow objects with x/y coordinates as feature_a/feature_b

    Raises:
        ValueError: If the file is not a valid DataPainter database or table not found
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    try:
        # Check for metadata table to verify this is a DataPainter file
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'"
        )
        if not cursor.fetchone():
            raise ValueError(
                f"{path} does not appear to be a DataPainter file (no metadata table)"
            )

        # Get available tables from metadata
        cursor.execute("SELECT table_name FROM metadata")
        available_tables = [row["table_name"] for row in cursor.fetchall()]

        if not available_tables:
            raise ValueError(f"No tables found in metadata table of {path}")

        # Determine which table to use
        if table_name is None:
            table_name = available_tables[0]
        elif table_name not in available_tables:
            raise ValueError(
                f"Table {table_name!r} not found. Available tables: {available_tables}"
            )

        # Load data from the specified table
        cursor.execute(f"SELECT x, y, target FROM {table_name} ORDER BY id")
        rows: List[DatasetRow] = []
        for row in cursor.fetchall():
            rows.append(
                DatasetRow(
                    feature_a=str(row["x"]),
                    feature_b=str(row["y"]),
                    label=str(row["target"]),
                )
            )

        if not rows:
            raise ValueError(f"Table {table_name!r} is empty; supply at least one row")

        return rows

    finally:
        conn.close()


def load_dataset(path: Path, table_name: Optional[str] = None) -> List[DatasetRow]:
    """Load dataset from either CSV or DataPainter SQLite format.

    Automatically detects the file format based on extension:
    - .csv: CSV format with feature_a, feature_b, label columns
    - .sqlite, .sqlite3, .db: DataPainter format with metadata and data tables

    Args:
        path: Path to the dataset file
        table_name: For DataPainter files, optional table name to load

    Returns:
        List of DatasetRow objects

    Raises:
        ValueError: If format is unrecognized or file is invalid
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_dataset_csv(path)
    elif suffix in {".sqlite", ".sqlite3", ".db"}:
        return load_dataset_datapainter(path, table_name)
    else:
        raise ValueError(
            f"Unsupported dataset format: {suffix}. "
            "Use .csv or .sqlite/.sqlite3/.db"
        )


def split_dataset(rows: Sequence[DatasetRow], seed: int, ratio: float = 0.8) -> DatasetSplit:
    """Perform a deterministic train/validation split."""

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    pivot = max(1, int(len(shuffled) * ratio))
    train = shuffled[:pivot]
    validation = shuffled[pivot:]
    if not validation:
        validation = train[-1:]
        train = train[:-1] or train
    return DatasetSplit(train=train, validation=validation)


###############################################################################
# SQLite persistence helpers
###############################################################################


class StandaloneDatabase:
    """Persistence layer storing dataset, rounds, and examples."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self.initialise()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def initialise(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS dataset(
                    feature_a TEXT NOT NULL,
                    feature_b TEXT NOT NULL,
                    label TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rounds(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP NOT NULL,
                    prompt TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS examples(
                    round_id INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
                    feature_a TEXT NOT NULL,
                    feature_b TEXT NOT NULL,
                    label TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    correct INTEGER NOT NULL
                );
                """
            )
            cur.close()

    def store_dataset(self, rows: Sequence[DatasetRow]) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM dataset")
            if cur.fetchone()[0]:
                cur.close()
                return
            cur.executemany(
                "INSERT INTO dataset(feature_a, feature_b, label) VALUES (?, ?, ?)",
                ((row.feature_a, row.feature_b, row.label) for row in rows),
            )
            self.conn.commit()
            cur.close()

    def fetch_rounds(self) -> List[RoundRecord]:
        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                "SELECT id, created_at, prompt, accuracy, notes FROM rounds ORDER BY id"
            ).fetchall()
            records: List[RoundRecord] = []
            for row in rows:
                examples = self.fetch_examples(row["id"])
                metrics = compute_metrics(examples, row["notes"] or "")
                metrics = dataclasses.replace(metrics, accuracy=row["accuracy"])
                created_at = row["created_at"]
                if isinstance(created_at, str):
                    created_at = dt.datetime.fromisoformat(created_at)
                records.append(
                    RoundRecord(
                        id=row["id"],
                        created_at=created_at,
                        prompt=row["prompt"],
                        metrics=metrics,
                        examples=examples,
                    )
                )
            cur.close()
            return records

    def fetch_examples(self, round_id: int) -> List[RoundExample]:
        with self._lock:
            cur = self.conn.cursor()
            entries = cur.execute(
                """
                SELECT feature_a, feature_b, label, prediction, correct
                FROM examples
                WHERE round_id = ?
                ORDER BY rowid
                """,
                (round_id,),
            ).fetchall()
            examples = [
                RoundExample(
                    feature_a=row["feature_a"],
                    feature_b=row["feature_b"],
                    label=row["label"],
                    prediction=row["prediction"],
                    correct=bool(row["correct"]),
                )
                for row in entries
            ]
            cur.close()
            return examples

    def insert_round(
        self, prompt: str, metrics: RoundMetrics, examples: Sequence[RoundExample]
    ) -> Tuple[int, dt.datetime]:
        with self._lock:
            cur = self.conn.cursor()
            created_at = dt.datetime.utcnow()
            cur.execute(
                "INSERT INTO rounds(created_at, prompt, accuracy, notes) VALUES (?, ?, ?, ?)",
                (created_at, prompt, metrics.accuracy, metrics.notes),
            )
            round_id = cur.lastrowid
            cur.executemany(
                """
                INSERT INTO examples(round_id, feature_a, feature_b, label, prediction, correct)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        round_id,
                        ex.feature_a,
                        ex.feature_b,
                        ex.label,
                        ex.prediction,
                        int(ex.correct),
                    )
                    for ex in examples
                ),
            )
            self.conn.commit()
            cur.close()
            return int(round_id), created_at

    def export_json(self, destination: Path) -> None:
        payload = []
        for record in self.fetch_rounds():
            payload.append(
                {
                    "id": record.id,
                    "created_at": record.created_at.isoformat(),
                    "prompt": record.prompt,
                    "accuracy": record.metrics.accuracy,
                    "notes": record.metrics.notes,
                    "examples": [dataclasses.asdict(ex) for ex in record.examples],
                }
            )
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


###############################################################################
# Model abstraction and prompt management
###############################################################################


class ModelAdapter:
    """Interface for the single supported model."""

    name: str

    def generate(self, prompt: str, rows: Sequence[DatasetRow]) -> List[str]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class StubModelAdapter(ModelAdapter):
    """Deterministic pseudo-model suitable for offline demos."""

    def __init__(self, name: str, temperature: float) -> None:
        self.name = f"{name} (stub)"
        self.temperature = temperature

    def generate(self, prompt: str, rows: Sequence[DatasetRow]) -> List[str]:
        base = len(prompt)
        outputs: List[str] = []
        for row in rows:
            score = sum(ord(char) for char in (row.feature_a + row.feature_b)) + base
            rng = (score * (1 + int(self.temperature * 100))) % 997
            prediction = "positive" if rng % 2 == 0 else "negative"
            if "always" in prompt.lower():
                prediction = row.label
            outputs.append(prediction)
        return outputs


class PromptManager:
    """Maintains the evolving narrative prompt across rounds."""

    def __init__(self, initial_prompt: str) -> None:
        self.current_prompt = initial_prompt

    def apply_feedback(self, examples: Sequence[RoundExample]) -> Tuple[str, str]:
        incorrect = [ex for ex in examples if not ex.correct]
        if not incorrect:
            feedback = "Validation was perfect. Reinforce existing strategy."
            updated = self.current_prompt + "\nContinue validating strengths."
        else:
            bullets = []
            for ex in incorrect[:3]:
                bullets.append(
                    f"- Re-evaluate cases like ({ex.feature_a}, {ex.feature_b}) where label {ex.label!r} was missed."
                )
            feedback = "Errors detected:\n" + "\n".join(bullets)
            updated = self.current_prompt + "\n" + feedback
        self.current_prompt = updated
        return updated, feedback

    def set_prompt(self, prompt: str) -> None:
        self.current_prompt = prompt


###############################################################################
# Metric computation helpers
###############################################################################


def compute_metrics(examples: Sequence[RoundExample], notes: str) -> RoundMetrics:
    total = len(examples)
    correct = sum(1 for ex in examples if ex.correct)
    accuracy = correct / total if total else 0.0
    labels = sorted({ex.label for ex in examples})
    per_class: Dict[str, Dict[str, float]] = {}
    confusion: Dict[Tuple[str, str], int] = {}
    for label in labels:
        tp = sum(1 for ex in examples if ex.label == label and ex.prediction == label)
        fp = sum(1 for ex in examples if ex.label != label and ex.prediction == label)
        fn = sum(1 for ex in examples if ex.label == label and ex.prediction != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[label] = {"precision": precision, "recall": recall}
    for ex in examples:
        confusion[(ex.label, ex.prediction)] = confusion.get((ex.label, ex.prediction), 0) + 1
    return RoundMetrics(accuracy=accuracy, per_class=per_class, confusion_matrix=confusion, notes=notes)


def calculate_baseline_metrics(split: DatasetSplit) -> List[BaselineMetrics]:
    """Compute simple baseline metrics for display."""

    def majority_label(rows: Sequence[DatasetRow]) -> str:
        labels = [row.label for row in rows]
        mode = max(set(labels), key=labels.count)
        return mode

    baseline_predictions = []
    majority = majority_label(split.train)
    baseline_predictions.append(("Majority Label", [majority for _ in split.validation]))

    # A deterministic heuristic baseline using feature parity
    def parity(row: DatasetRow) -> str:
        parity_value = (len(row.feature_a) + len(row.feature_b)) % 2
        return "positive" if parity_value == 0 else "negative"

    baseline_predictions.append(("Feature Parity", [parity(row) for row in split.validation]))

    metrics: List[BaselineMetrics] = []
    labels = [row.label for row in split.validation]
    for name, predictions in baseline_predictions:
        accuracy = sum(1 for truth, pred in zip(labels, predictions) if truth == pred) / len(labels)
        tau = kendall_tau(labels, predictions)
        metrics.append(BaselineMetrics(name=name, accuracy=accuracy, kendall_tau=tau))
    return metrics


def kendall_tau(labels: Sequence[str], predictions: Sequence[str]) -> float:
    """Lightweight Kendall Tau approximation for ordinal agreement.

    For binary labels we approximate by treating the two classes as ordered
    categories and computing concordance of pairs.
    """

    unique = sorted(set(labels + list(predictions)))
    index = {label: i for i, label in enumerate(unique)}
    concordant = 0
    discordant = 0
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = index[labels[i]], index[labels[j]]
            pi, pj = index[predictions[i]], index[predictions[j]]
            concordant += int((li - lj) * (pi - pj) >= 0)
            discordant += int((li - lj) * (pi - pj) < 0)
    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs


###############################################################################
# Round execution engine
###############################################################################


class RoundProgress(Message):
    """Message carrying round progress updates to the UI."""

    def __init__(self, stage: str, completed: int, total: int, current_index: Optional[int]) -> None:
        self.stage = stage
        self.completed = completed
        self.total = total
        self.current_index = current_index
        super().__init__()


class RoundFinished(Message):
    """Message signalling that a round has completed."""

    def __init__(self, record: RoundRecord) -> None:
        self.record = record
        super().__init__()


class RoundEngine:
    """Coordinates dataset evaluation using the selected model."""

    def __init__(
        self,
        split: DatasetSplit,
        database: StandaloneDatabase,
        model: ModelAdapter,
        prompt_manager: PromptManager,
    ) -> None:
        self.split = split
        self.database = database
        self.model = model
        self.prompt_manager = prompt_manager

    def run_round(
        self,
        notify: Callable[[RoundProgress], None],
    ) -> RoundRecord:
        notify(RoundProgress("Preparing data", 0, len(self.split.validation), None))
        prompt = self.prompt_manager.current_prompt

        # Stage 1: Generate predictions sequentially to allow progress updates.
        predictions: List[str] = []
        notes_log: List[str] = []
        for idx, row in enumerate(self.split.validation):
            notify(RoundProgress("Calling model", idx, len(self.split.validation), idx))
            prediction = self.model.generate(prompt, [row])[0]
            predictions.append(prediction)
            notes_log.append(
                f"Processed ({row.feature_a}, {row.feature_b}) -> {prediction}"
            )
        notify(RoundProgress("Scoring", len(self.split.validation), len(self.split.validation), None))

        examples = []
        for row, pred in zip(self.split.validation, predictions):
            examples.append(
                RoundExample(
                    feature_a=row.feature_a,
                    feature_b=row.feature_b,
                    label=row.label,
                    prediction=pred,
                    correct=pred == row.label,
                )
            )
        metrics = compute_metrics(examples, notes="\n".join(notes_log))
        updated_prompt, feedback = self.prompt_manager.apply_feedback(examples)
        metrics.notes = feedback + "\n" + metrics.notes
        notify(RoundProgress("Saving", len(self.split.validation), len(self.split.validation), None))
        round_id, created_at = self.database.insert_round(
            prompt=prompt, metrics=metrics, examples=examples
        )
        record = RoundRecord(
            id=round_id,
            created_at=created_at,
            prompt=prompt,
            metrics=metrics,
            examples=examples,
        )
        log.info("Updated prompt to %s", updated_prompt)
        notify(RoundProgress("Complete", len(self.split.validation), len(self.split.validation), None))
        return record


###############################################################################
# TUI widgets
###############################################################################


class DatasetPanel(Static):
    """Displays dataset summary statistics."""

    def update_summary(self, rows: Sequence[DatasetRow], split: DatasetSplit, round_count: int) -> None:
        labels = [row.label for row in rows]
        distribution: Dict[str, int] = {}
        for label in labels:
            distribution[label] = distribution.get(label, 0) + 1
        summary_lines = [f"Total rows: {len(rows)}", f"Training rows: {len(split.train)}", f"Validation rows: {len(split.validation)}"]
        summary_lines.append("Label distribution:")
        for label, count in distribution.items():
            summary_lines.append(f"- {label}: {count}")
        summary_lines.append(f"Stored rounds: {round_count}")
        self.update("\n".join(summary_lines))


class BaselinePanel(DataTable):
    """Table summarising baseline models."""

    def __init__(self) -> None:
        super().__init__(zebra_stripes=True)
        self.add_columns("Model", "Accuracy", "Kendall Tau")
        self.cursor_type = "none"

    def populate(self, metrics: Sequence[BaselineMetrics]) -> None:
        self.clear()
        for metric in metrics:
            self.add_row(metric.name, f"{metric.accuracy:.2%}", f"{metric.kendall_tau:.2f}")


class HistoryList(ListView):
    """List of completed rounds."""

    def set_rounds(self, rounds: Sequence[RoundRecord]) -> None:
        self.clear()
        for record in rounds:
            created = record.created_at.strftime("%H:%M:%S")
            subtitle = f"Accuracy {record.metrics.accuracy:.1%}"
            self.append(ListItem(Label(f"Round {record.id} @ {created} — {subtitle}"), id=str(record.id)))


class RoundDetail(Static):
    """Displays details for a selected round."""

    def show_round(self, record: Optional[RoundRecord]) -> None:
        if record is None:
            self.update("Select a round to view details.")
            return
        matrix_lines = []
        for (label, pred), count in record.metrics.confusion_matrix.items():
            matrix_lines.append(f"{label!r} → {pred!r}: {count}")
        per_class = [
            f"{label}: precision {values['precision']:.2f}, recall {values['recall']:.2f}"
            for label, values in record.metrics.per_class.items()
        ]
        examples = [
            f"{idx+1:02d}. ({ex.feature_a}, {ex.feature_b}) label={ex.label!r} pred={ex.prediction!r} {'✅' if ex.correct else '❌'}"
            for idx, ex in enumerate(record.examples[:10])
        ]
        body = textwrap.dedent(
            f"""
            ### Prompt
            {record.prompt}

            ### Metrics
            Accuracy: {record.metrics.accuracy:.2%}
            {os.linesep.join(per_class)}

            ### Confusion Matrix
            {os.linesep.join(matrix_lines)}

            ### Notes
            {record.metrics.notes}

            ### Sample Examples
            {os.linesep.join(examples)}
            """
        )
        self.update(RichMarkdown(body))


class UnderlingPanel(Static):
    """Tracks active evaluation progress."""

    progress_text = reactive("Idle")

    def __init__(self) -> None:
        super().__init__()
        self.progress_bar = ProgressBar(total=1)
        self.coordinates = ListView(id="underling-coordinates")

    def compose(self) -> ComposeResult:
        yield Label("Underling Progress", id="underling-title")
        yield self.progress_bar
        yield Label(id="underling-status")
        yield self.coordinates

    def reset(self, rows: Sequence[DatasetRow]) -> None:
        self.progress_bar.total = max(1, len(rows))
        self.progress_bar.progress = 0
        self.coordinates.clear()
        for idx, row in enumerate(rows, start=1):
            self.coordinates.append(ListItem(Label(f"{idx:02d}. ({row.feature_a}, {row.feature_b})")))
        status = self.query_one("#underling-status", Label)
        status.update("Awaiting round start…")

    def update_progress(self, message: RoundProgress) -> None:
        self.progress_bar.progress = message.completed
        status = self.query_one("#underling-status", Label)
        status.update(f"{message.stage} ({message.completed}/{message.total})")
        if message.current_index is not None:
            self.coordinates.index = message.current_index


class PromptPanel(Static):
    """Displays current prompt text."""

    def update_prompt(self, prompt: str) -> None:
        self.update(RichMarkdown(f"### Active Prompt\n{prompt}"))


class EventLog(RichLog):
    """Colour-coded log of application events."""

    def __init__(self) -> None:
        super().__init__(highlight=True, max_lines=200)

    def info(self, message: str) -> None:
        self.write(f"[green]INFO[/green] {message}")

    def warning(self, message: str) -> None:
        self.write(f"[yellow]WARN[/yellow] {message}")

    def error(self, message: str) -> None:
        self.write(f"[red]ERROR[/red] {message}")


class PromptEditor(ModalScreen[str]):
    """Modal editor that lets the user modify the active prompt."""

    def __init__(self, initial: str) -> None:
        super().__init__()
        self.initial = initial
        if TextArea is not None:
            self.editor: Any = TextArea(code=False)
            self.editor.value = initial
        else:  # pragma: no cover - fallback for older Textual
            self.editor = Input(value=initial, placeholder="Edit narrative prompt…", password=False)

    def compose(self) -> ComposeResult:
        yield Container(
            Label("Edit Prompt", id="prompt-editor-title"),
            self.editor,
            Horizontal(
                Button("Cancel", id="prompt-cancel"),
                Button("Save", id="prompt-save", variant="primary"),
                id="prompt-buttons",
            ),
            id="prompt-editor",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prompt-save":
            value = getattr(self.editor, "value", self.initial)
            self.dismiss(value)
        else:
            self.dismiss(None)


###############################################################################
# Textual application
###############################################################################


class StandaloneApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #columns {
        height: 1fr;
    }
    #left, #center, #right {
        padding: 1;
        height: 1fr;
    }
    #left {
        width: 30%;
    }
    #center {
        width: 40%;
    }
    #right {
        width: 30%;
    }
    #prompt-editor {
        padding: 2;
        width: 80%;
        height: auto;
        border: heavy $accent;
        background: $panel;
        margin: 2 4;
    }
    #prompt-buttons {
        align-horizontal: right;
        padding-top: 1;
    }
    #prompt-editor-title {
        content-align: center middle;
        padding-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("n", "next_round", "Next round"),
        Binding("r", "regenerate", "Regenerate"),
        Binding("s", "save_snapshot", "Export JSON"),
        Binding("p", "edit_prompt", "Edit prompt"),
        Binding("t", "toggle_stub", "Toggle stub"),
    ]

    def __init__(
        self,
        *,
        dataset: List[DatasetRow],
        split: DatasetSplit,
        database: StandaloneDatabase,
        model: ModelAdapter,
        prompt_manager: PromptManager,
        baseline_metrics: List[BaselineMetrics],
        export_path: Optional[Path],
        max_rounds: Optional[int],
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.database = database
        self.model = model
        self.prompt_manager = prompt_manager
        self.baseline_metrics = baseline_metrics
        self.export_path = export_path
        self.max_rounds = max_rounds
        self.rounds: List[RoundRecord] = database.fetch_rounds()
        self.engine = RoundEngine(split, database, model, prompt_manager)
        self.current_round: Optional[RoundRecord] = None
        self._worker: Optional[asyncio.Task[None]] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="body"):
            with Horizontal(id="columns"):
                with Vertical(id="left"):
                    dataset_panel = DatasetPanel(id="dataset-panel")
                    yield dataset_panel
                    history = HistoryList(id="history")
                    yield history
                with Vertical(id="center"):
                    prompt_panel = PromptPanel(id="prompt-panel")
                    yield prompt_panel
                    underling = UnderlingPanel(id="underling-panel")
                    yield underling
                    round_detail = RoundDetail(id="round-detail")
                    yield round_detail
                with Vertical(id="right"):
                    baseline = BaselinePanel()
                    yield baseline
                    log_widget = EventLog(id="event-log")
                    yield log_widget
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Narrative Learning Standalone"
        dataset_panel = self.query_one(DatasetPanel)
        dataset_panel.update_summary(self.dataset, self.split, len(self.rounds))
        self.query_one(PromptPanel).update_prompt(self.prompt_manager.current_prompt)
        self.query_one(BaselinePanel).populate(self.baseline_metrics)
        self.query_one(EventLog).info(
            f"Loaded dataset with {len(self.dataset)} rows; model {self.model.describe()}"
        )
        self.query_one(UnderlingPanel).reset(self.split.validation)
        self.query_one(HistoryList).set_rounds(self.rounds)
        if self.rounds:
            self.current_round = self.rounds[-1]
            self.query_one(RoundDetail).show_round(self.current_round)

    async def action_next_round(self) -> None:
        if self.max_rounds is not None and len(self.rounds) >= self.max_rounds:
            self.query_one(EventLog).warning("Max rounds reached; cannot create new round.")
            return
        if self._worker and not self._worker.done():
            self.query_one(EventLog).warning("Round already in progress")
            return
        underling = self.query_one(UnderlingPanel)
        underling.reset(self.split.validation)
        self.query_one(EventLog).info("Starting new round")
        loop = asyncio.get_running_loop()

        def notify(message: RoundProgress) -> None:
            self.post_message(message)

        async def worker() -> None:
            record = await loop.run_in_executor(None, self.engine.run_round, notify)
            self.post_message(RoundFinished(record))

        self._worker = asyncio.create_task(worker())

    async def action_regenerate(self) -> None:
        if not self.rounds:
            self.query_one(EventLog).warning("No previous round to regenerate.")
            return
        self.prompt_manager.set_prompt(self.rounds[-1].prompt)
        self.query_one(PromptPanel).update_prompt(self.prompt_manager.current_prompt)
        await self.action_next_round()

    async def action_save_snapshot(self) -> None:
        target = self.export_path or Path("standalone_export.json")
        self.database.export_json(target)
        self.query_one(EventLog).info(f"Exported snapshot to {target}")

    async def action_edit_prompt(self) -> None:
        previous = self.prompt_manager.current_prompt
        modal = PromptEditor(previous)
        result = await self.push_screen_wait(modal)
        if result is not None and result != previous:
            self.prompt_manager.set_prompt(result)
            self.query_one(PromptPanel).update_prompt(result)
            self.query_one(EventLog).info("Prompt updated by user")

    async def action_toggle_stub(self) -> None:
        if not isinstance(self.model, StubModelAdapter):
            self.query_one(EventLog).warning("Only stub mode is available in this build.")
            return
        if "(stub)" in self.model.name:
            self.model.name = self.model.name.replace(" (stub)", "")
            self.query_one(EventLog).info("Stub label hidden; behaviour unchanged.")
        else:
            self.model.name = f"{self.model.name} (stub)"
            self.query_one(EventLog).info("Stub label restored.")

    async def on_round_progress(self, message: RoundProgress) -> None:
        self.query_one(UnderlingPanel).update_progress(message)

    async def on_round_finished(self, message: RoundFinished) -> None:
        record = message.record
        self.rounds.append(record)
        self.current_round = record
        dataset_panel = self.query_one(DatasetPanel)
        dataset_panel.update_summary(self.dataset, self.split, len(self.rounds))
        self.query_one(HistoryList).set_rounds(self.rounds)
        self.query_one(RoundDetail).show_round(record)
        self.query_one(PromptPanel).update_prompt(self.prompt_manager.current_prompt)
        self.query_one(EventLog).info(
            f"Round {record.id} completed with accuracy {record.metrics.accuracy:.1%}"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id == "history" and event.item.id:
            round_id = int(event.item.id)
            for record in self.rounds:
                if record.id == round_id:
                    self.current_round = record
                    self.query_one(RoundDetail).show_round(record)
                    break

    async def on_shutdown(self, event: App.Shutdown) -> None:
        if self.export_path is not None:
            self.database.export_json(self.export_path)
        summary = {
            "rounds": [
                {
                    "id": record.id,
                    "accuracy": record.metrics.accuracy,
                    "prompt": record.prompt,
                }
                for record in self.rounds
            ]
        }
        print(json.dumps(summary, indent=2))


###############################################################################
# CLI interface and application entrypoint
###############################################################################


class LockFile:
    """Simple filesystem lock to prevent multiple concurrent runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Optional[int] = None

    def acquire(self) -> None:
        if self.path.exists():
            raise RuntimeError(f"Lock file {self.path} already exists; another instance may be running.")
        self.path.write_text(str(os.getpid()))

    def release(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()

    def __enter__(self) -> "LockFile":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Narrative Learning demo")
    parser.add_argument("--config", type=Path, help="Path to config TOML/YAML file", default=None)
    parser.add_argument("--dataset", type=Path, help="Dataset path (CSV or DataPainter SQLite)", default=None)
    parser.add_argument("--table", type=str, help="Table name for DataPainter SQLite files", default=None)
    parser.add_argument("--database", type=Path, help="SQLite database path", default=None)
    parser.add_argument("--export-json", dest="export_json", type=Path, help="Export JSON destination", default=None)
    parser.add_argument("--shuffle-seed", type=int, default=13, help="Dataset shuffle seed")
    parser.add_argument("--max-rounds", type=int, default=None, help="Maximum allowed rounds")
    parser.add_argument("--lock", type=Path, default=None, help="Optional lock file path")
    return parser.parse_args(argv)


def determine_paths(config: AppConfig, args: argparse.Namespace) -> Tuple[Path, Path]:
    dataset_path = args.dataset or config.dataset_path
    if dataset_path is None:
        raise SystemExit("Dataset path must be provided via --dataset or config file")
    dataset_path = dataset_path.resolve()
    db_path = args.database or dataset_path.with_suffix(".sqlite3")
    return dataset_path, Path(db_path)


def bootstrap(argv: Optional[Sequence[str]] = None) -> StandaloneApp:
    args = parse_args(argv)
    config = AppConfig.load(args.config)
    dataset_path, db_path = determine_paths(config, args)
    dataset = load_dataset(dataset_path, table_name=args.table)
    split = split_dataset(dataset, args.shuffle_seed)
    database = StandaloneDatabase(db_path)
    database.store_dataset(dataset)
    model = StubModelAdapter(config.model_name, config.temperature)
    prompt_manager = PromptManager("Start with a descriptive hypothesis about the relationship between features and labels.")
    baseline_metrics = calculate_baseline_metrics(split)
    export_path = args.export_json
    app = StandaloneApp(
        dataset=dataset,
        split=split,
        database=database,
        model=model,
        prompt_manager=prompt_manager,
        baseline_metrics=baseline_metrics,
        export_path=export_path,
        max_rounds=args.max_rounds,
    )
    if args.lock is not None:
        app.run = LockingRun(app.run, args.lock)  # type: ignore[method-assign]
    return app


class LockingRun:
    """Wrap the Textual app run method to honour lock files."""

    def __init__(self, runner: Callable[..., Any], lock_path: Path) -> None:
        self.runner = runner
        self.lock_path = lock_path

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with LockFile(self.lock_path):
            return self.runner(*args, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    app = bootstrap(argv)
    app.run()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
