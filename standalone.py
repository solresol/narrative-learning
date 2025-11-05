"""Standalone Narrative Learning TUI implementation.

This module implements a standalone demonstration of the narrative learning
workflow using the Textual terminal user-interface framework.  The design is
guided by the specification in ``standalone_spec.md`` and focuses on
supporting a single language-model backend with lightweight SQLite
persistence.

Datasets must be in DataPainter SQLite format. Run the application with:
    uv run standalone.py --dataset path/to.sqlite --table tablename
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    tomllib = None  # type: ignore[assignment]

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
import llmcall
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from imodels import RuleFitClassifier, BayesianRuleListClassifier, OptimalRuleListClassifier
from interpret.glassbox import ExplainableBoostingClassifier
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
    examples: List[RoundExample]  # Validation examples (for display/metrics)
    training_examples: Optional[List[RoundExample]] = None  # Training examples (for prompt generation)


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


def load_dataset(path: Path, table_name: Optional[str] = None) -> List[DatasetRow]:
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


def extract_valid_predictions(rows: Sequence[DatasetRow]) -> List[str]:
    """Extract the unique labels from the dataset to use as valid predictions."""
    unique_labels = sorted(set(row.label for row in rows))
    return unique_labels


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
                    round_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_uuid TEXT,
                    round_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    prompt TEXT NOT NULL,
                    reasoning_for_this_prompt TEXT,
                    train_accuracy REAL,
                    validation_accuracy REAL NOT NULL,
                    round_completed TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS inferences(
                    round_id INTEGER NOT NULL REFERENCES rounds(round_id) ON DELETE CASCADE,
                    creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_id TEXT NOT NULL,
                    narrative_text TEXT,
                    llm_stderr TEXT,
                    prediction TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    PRIMARY KEY (round_id, data_id)
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
                "SELECT round_id, round_start, prompt, reasoning_for_this_prompt, validation_accuracy FROM rounds ORDER BY round_id"
            ).fetchall()
            records: List[RoundRecord] = []
            for row in rows:
                examples = self.fetch_examples(row["round_id"])
                notes = row["reasoning_for_this_prompt"] or ""
                metrics = compute_metrics(examples, notes)
                metrics = dataclasses.replace(metrics, accuracy=row["validation_accuracy"])
                created_at = row["round_start"]
                if isinstance(created_at, str):
                    created_at = dt.datetime.fromisoformat(created_at)
                records.append(
                    RoundRecord(
                        id=row["round_id"],
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
                SELECT data_id, prediction, correct
                FROM inferences
                WHERE round_id = ?
                ORDER BY creation_time
                """,
                (round_id,),
            ).fetchall()
            examples = []
            for row in entries:
                # data_id format is "feature_a,feature_b,label"
                parts = row["data_id"].split(",", 2)
                if len(parts) == 3:
                    examples.append(
                        RoundExample(
                            feature_a=parts[0],
                            feature_b=parts[1],
                            label=parts[2],
                            prediction=row["prediction"],
                            correct=bool(row["correct"]),
                        )
                    )
            cur.close()
            return examples

    def insert_round(
        self, prompt: str, metrics: RoundMetrics, examples: Sequence[RoundExample]
    ) -> Tuple[int, dt.datetime]:
        with self._lock:
            cur = self.conn.cursor()
            import uuid
            round_start = dt.datetime.utcnow()
            round_uuid = str(uuid.uuid4())
            cur.execute(
                """INSERT INTO rounds(round_uuid, round_start, prompt, reasoning_for_this_prompt,
                   validation_accuracy, round_completed)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (round_uuid, round_start, prompt, metrics.notes, metrics.accuracy, round_start),
            )
            round_id = cur.lastrowid
            cur.executemany(
                """
                INSERT INTO inferences(round_id, data_id, prediction, correct)
                VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        round_id,
                        f"{ex.feature_a},{ex.feature_b},{ex.label}",
                        ex.prediction,
                        int(ex.correct),
                    )
                    for ex in examples
                ),
            )
            self.conn.commit()
            cur.close()
            return int(round_id), round_start

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


class RealLLMAdapter(ModelAdapter):
    """Real LLM adapter that uses llmcall to make actual predictions."""

    def __init__(self, model_name: str, valid_predictions: List[str]) -> None:
        self.name = model_name
        self.valid_predictions = valid_predictions

    def generate(self, prompt: str, rows: Sequence[DatasetRow]) -> List[str]:
        """Generate predictions for the given rows using the LLM.

        The prompt should be the narrative instructions. This method will construct
        the full prompt for each row (instructions + entity data) and call the LLM.
        """
        predictions: List[str] = []

        for row in rows:
            # Construct the full prompt following predict.py pattern
            entity_data = f"Feature A: {row.feature_a}\nFeature B: {row.feature_b}"
            full_prompt = f"""This is an experiment in identifying whether an LLM can predict outcomes. Use the following methodology for predicting the outcome for this entity.

```
{prompt}
```

Entity Data:
{entity_data}
"""

            # Handle the special case of random choice
            if prompt.strip() == "Choose randomly":
                prediction = random.choice(self.valid_predictions)
            else:
                try:
                    prediction_output, run_info = llmcall.dispatch_prediction_prompt(
                        self.name, full_prompt, self.valid_predictions
                    )
                    prediction = prediction_output['prediction']
                except (llmcall.MissingPrediction, llmcall.InvalidPrediction) as e:
                    log.warning(f"LLM prediction failed: {e}, choosing randomly")
                    prediction = random.choice(self.valid_predictions)

            predictions.append(prediction)

        return predictions


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
    """Compute baseline metrics using traditional ML models for display."""
    import numpy as np

    # Prepare training data
    X_train = np.array([[float(row.feature_a), float(row.feature_b)] for row in split.train])
    y_train = np.array([row.label for row in split.train])

    # Prepare validation data
    X_val = np.array([[float(row.feature_a), float(row.feature_b)] for row in split.validation])
    y_val = [row.label for row in split.validation]

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    baseline_predictions = []

    # 1. Majority Label (Dummy Classifier)
    majority = max(set(y_train), key=list(y_train).count)
    baseline_predictions.append(("Majority", [majority for _ in split.validation]))

    # 2. Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_predictions = lr.predict(X_val)
        baseline_predictions.append(("Logistic", list(lr_predictions)))
    except Exception as e:
        log(f"Logistic Regression failed: {e}")

    # 3. Decision Tree
    try:
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        dt_predictions = dt.predict(X_val)
        baseline_predictions.append(("DecTree", list(dt_predictions)))
    except Exception as e:
        log(f"Decision Tree failed: {e}")

    # 4. RuleFit
    try:
        rf = RuleFitClassifier()
        rf.fit(X_train, y_train_encoded)
        rf_predictions = le.inverse_transform(rf.predict(X_val).astype(int))
        baseline_predictions.append(("RuleFit", list(rf_predictions)))
    except Exception as e:
        log(f"RuleFit failed: {e}")

    # 5. Bayesian Rule List
    try:
        brl = BayesianRuleListClassifier(max_iter=500, n_chains=2)
        brl.fit(X_train, y_train_encoded)
        brl_predictions = le.inverse_transform(brl.predict(X_val).astype(int))
        baseline_predictions.append(("BRL", list(brl_predictions)))
    except Exception as e:
        log(f"BayesianRuleList failed: {e}")

    # 6. CORELS
    try:
        corels = OptimalRuleListClassifier(max_card=3, n_iter=5000, c=0.05)
        corels.fit(X_train, y_train_encoded)
        corels_predictions = le.inverse_transform(corels.predict(X_val).astype(int))
        baseline_predictions.append(("CORELS", list(corels_predictions)))
    except Exception as e:
        log(f"CORELS failed: {e}")

    # 7. EBM (Explainable Boosting Machine)
    try:
        ebm = ExplainableBoostingClassifier(interactions=10)
        ebm.fit(X_train, y_train)
        ebm_predictions = ebm.predict(X_val)
        baseline_predictions.append(("EBM", list(ebm_predictions)))
    except Exception as e:
        log(f"EBM failed: {e}")

    # Calculate metrics for all baselines
    metrics: List[BaselineMetrics] = []
    for name, predictions in baseline_predictions:
        accuracy = sum(1 for truth, pred in zip(y_val, predictions) if truth == pred) / len(y_val)
        tau = kendall_tau(y_val, predictions)
        metrics.append(BaselineMetrics(name=name, accuracy=accuracy, kendall_tau=tau))

    return metrics


def calculate_baseline_metrics_incremental(split: DatasetSplit) -> Iterator[BaselineMetrics]:
    """Compute baseline metrics one at a time, yielding results as they complete."""
    import numpy as np

    # Prepare training data
    X_train = np.array([[float(row.feature_a), float(row.feature_b)] for row in split.train])
    y_train = np.array([row.label for row in split.train])

    # Prepare validation data
    X_val = np.array([[float(row.feature_a), float(row.feature_b)] for row in split.validation])
    y_val = [row.label for row in split.validation]

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Helper to calculate and yield a baseline
    def calc_and_yield(name: str, predictions: List[str]) -> BaselineMetrics:
        accuracy = sum(1 for truth, pred in zip(y_val, predictions) if truth == pred) / len(y_val)
        tau = kendall_tau(y_val, predictions)
        return BaselineMetrics(name=name, accuracy=accuracy, kendall_tau=tau)

    # 1. Majority Label
    majority = max(set(y_train), key=list(y_train).count)
    yield calc_and_yield("Majority", [majority for _ in split.validation])

    # 2. Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_predictions = lr.predict(X_val)
        yield calc_and_yield("Logistic", list(lr_predictions))
    except Exception as e:
        log(f"Logistic Regression failed: {e}")

    # 3. Decision Tree
    try:
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        dt_predictions = dt.predict(X_val)
        yield calc_and_yield("DecTree", list(dt_predictions))
    except Exception as e:
        log(f"Decision Tree failed: {e}")

    # 4. RuleFit
    try:
        rf = RuleFitClassifier()
        rf.fit(X_train, y_train_encoded)
        rf_predictions = le.inverse_transform(rf.predict(X_val).astype(int))
        yield calc_and_yield("RuleFit", list(rf_predictions))
    except Exception as e:
        log(f"RuleFit failed: {e}")

    # 5. Bayesian Rule List
    try:
        brl = BayesianRuleListClassifier(max_iter=500, n_chains=2)
        brl.fit(X_train, y_train_encoded)
        brl_predictions = le.inverse_transform(brl.predict(X_val).astype(int))
        yield calc_and_yield("BRL", list(brl_predictions))
    except Exception as e:
        log(f"BayesianRuleList failed: {e}")

    # 6. CORELS
    try:
        corels = OptimalRuleListClassifier(max_card=3, n_iter=5000, c=0.05)
        corels.fit(X_train, y_train_encoded)
        corels_predictions = le.inverse_transform(corels.predict(X_val).astype(int))
        yield calc_and_yield("CORELS", list(corels_predictions))
    except Exception as e:
        log(f"CORELS failed: {e}")

    # 7. EBM (Explainable Boosting Machine)
    try:
        ebm = ExplainableBoostingClassifier(interactions=10)
        ebm.fit(X_train, y_train)
        ebm_predictions = ebm.predict(X_val)
        yield calc_and_yield("EBM", list(ebm_predictions))
    except Exception as e:
        log(f"EBM failed: {e}")


def format_sig_figs(value: str, n: int = 3) -> str:
    """Format a numeric string to n significant figures."""
    try:
        num = float(value)
        if num == 0:
            return "0"
        from math import log10, floor
        magnitude = floor(log10(abs(num)))
        rounded = round(num, -magnitude + n - 1)
        # Format with appropriate decimal places
        if magnitude >= n - 1:
            return f"{int(rounded)}"
        else:
            decimals = max(0, n - magnitude - 1)
            return f"{rounded:.{decimals}f}"
    except (ValueError, TypeError):
        return value


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

    def __init__(
        self,
        stage: str,
        completed: int,
        total: int,
        current_index: Optional[int],
        prediction: Optional[str] = None,
    ) -> None:
        self.stage = stage
        self.completed = completed
        self.total = total
        self.current_index = current_index
        self.prediction = prediction
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
        # Track which indices in the all_rows list have been processed
        self.processed_indices: set[int] = set()
        # Store predictions for each index
        self.predictions: Dict[int, str] = {}

    def get_all_rows(self) -> List[DatasetRow]:
        """Get all data points (training + validation)."""
        return self.split.train + self.split.validation

    def process_unprocessed_points(
        self,
        notify: Callable[[RoundProgress], None],
    ) -> int:
        """Process all unprocessed data points incrementally.

        Returns the number of points processed.
        """
        all_rows = self.get_all_rows()
        prompt = self.prompt_manager.current_prompt

        # Find unprocessed indices
        unprocessed = [i for i in range(len(all_rows)) if i not in self.processed_indices]

        if not unprocessed:
            notify(RoundProgress("Complete", len(all_rows), len(all_rows), None))
            return 0

        processed_count = 0
        for idx in unprocessed:
            row = all_rows[idx]
            total_processed = len(self.processed_indices)
            notify(RoundProgress("Calling model", total_processed, len(all_rows), idx, None))

            prediction = self.model.generate(prompt, [row])[0]
            self.predictions[idx] = prediction
            self.processed_indices.add(idx)
            processed_count += 1

            # Notify with the prediction so UI can update
            notify(RoundProgress("Calling model", total_processed + 1, len(all_rows), idx, prediction))

        notify(RoundProgress("Complete", len(self.processed_indices), len(all_rows), None))
        return processed_count

    def run_round(
        self,
        notify: Callable[[RoundProgress], None],
    ) -> RoundRecord:
        """Complete the current round by processing unprocessed points and saving results."""
        # Process any remaining unprocessed points
        self.process_unprocessed_points(notify)

        notify(RoundProgress("Scoring", len(self.processed_indices), len(self.processed_indices), None))

        all_rows = self.get_all_rows()
        validation_start_idx = len(self.split.train)

        # Build training examples (for prompt generation)
        training_examples = []
        for train_idx, train_row in enumerate(self.split.train):
            if train_idx in self.predictions:
                pred = self.predictions[train_idx]
                training_examples.append(
                    RoundExample(
                        feature_a=train_row.feature_a,
                        feature_b=train_row.feature_b,
                        label=train_row.label,
                        prediction=pred,
                        correct=pred == train_row.label,
                    )
                )

        # Build validation examples (for metrics/display)
        validation_examples = []
        for val_idx, val_row in enumerate(self.split.validation):
            all_idx = validation_start_idx + val_idx
            if all_idx in self.predictions:
                pred = self.predictions[all_idx]
                validation_examples.append(
                    RoundExample(
                        feature_a=val_row.feature_a,
                        feature_b=val_row.feature_b,
                        label=val_row.label,
                        prediction=pred,
                        correct=pred == val_row.label,
                    )
                )

        notes_log = [f"Processed {len(self.processed_indices)}/{len(all_rows)} data points"]
        metrics = compute_metrics(validation_examples, notes="\n".join(notes_log))

        notify(RoundProgress("Saving", len(self.processed_indices), len(self.processed_indices), None))
        round_id, created_at = self.database.insert_round(
            prompt=self.prompt_manager.current_prompt, metrics=metrics, examples=validation_examples
        )
        record = RoundRecord(
            id=round_id,
            created_at=created_at,
            prompt=self.prompt_manager.current_prompt,
            metrics=metrics,
            examples=validation_examples,
            training_examples=training_examples,
        )

        # Reset for next round
        self.processed_indices.clear()
        self.predictions.clear()

        log.info("Round complete, keeping prompt: %s", self.prompt_manager.current_prompt[:50])
        notify(RoundProgress("Complete", len(all_rows), len(all_rows), None))
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(zebra_stripes=True, **kwargs)
        self.add_columns("Model", "Acc", "τ")
        self.cursor_type = "none"

    def populate(self, metrics: Sequence[BaselineMetrics]) -> None:
        self.clear()
        for metric in metrics:
            self.add_row(metric.name, f"{metric.accuracy:.2%}", f"{metric.kendall_tau:.2f}")

    def add_baseline(self, metric: BaselineMetrics) -> None:
        """Add a single baseline result incrementally."""
        self.add_row(metric.name, f"{metric.accuracy:.2%}", f"{metric.kendall_tau:.2f}")


class HistoryList(ListView):
    """List of completed rounds."""

    def set_rounds(self, rounds: Sequence[RoundRecord]) -> None:
        # Remove all existing children to avoid ID conflicts
        for child in list(self.children):
            child.remove()
        # Add new items (no IDs needed to avoid conflicts)
        for record in rounds:
            created = record.created_at.strftime("%H:%M:%S")
            subtitle = f"Accuracy {record.metrics.accuracy:.1%}"
            self.append(ListItem(Label(f"Round {record.id} @ {created} — {subtitle}")))


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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.progress_bar = ProgressBar(total=1)
        self.data_table = DataTable(id="underling-table", zebra_stripes=False)
        self.data_table.cursor_type = "row"
        self.validation_indices: set[int] = set()
        self.validation_to_all_map: dict[int, int] = {}  # Maps validation index to all_rows index
        self.all_rows: List[DatasetRow] = []  # Store original data for matching

    def compose(self) -> ComposeResult:
        yield Label("Underling Progress", id="underling-title")
        yield self.progress_bar
        yield Label(id="underling-status")
        yield self.data_table

    def reset(self, all_rows: Sequence[DatasetRow], validation_rows: Sequence[DatasetRow]) -> None:
        """Reset the panel with all data, marking validation rows specially."""
        # Store original data for later matching
        self.all_rows = list(all_rows)

        # Build validation set for quick lookup
        validation_set = {(r.feature_a, r.feature_b, r.label) for r in validation_rows}

        self.progress_bar.total = max(1, len(validation_rows))
        self.progress_bar.progress = 0

        # Clear and setup table
        self.data_table.clear(columns=True)
        self.data_table.add_columns("", "Feature A", "Feature B", "Truth", "Pred")

        # Track which rows are validation
        self.validation_indices.clear()
        self.validation_to_all_map.clear()

        # Build mapping from validation index to all_rows index
        validation_idx = 0
        for all_idx, row in enumerate(all_rows):
            if (row.feature_a, row.feature_b, row.label) in validation_set:
                self.validation_to_all_map[validation_idx] = all_idx
                validation_idx += 1

        # Add all rows
        for idx, row in enumerate(all_rows):
            is_validation = (row.feature_a, row.feature_b, row.label) in validation_set
            if is_validation:
                self.validation_indices.add(idx)
                marker = Text("V", style="bold yellow on blue")
                style = "on blue"
            else:
                marker = Text("T", style="dim")
                style = ""

            feat_a = format_sig_figs(row.feature_a)
            feat_b = format_sig_figs(row.feature_b)

            # Create styled cells for validation rows
            if is_validation:
                feat_a_text = Text(feat_a, style=style)
                feat_b_text = Text(feat_b, style=style)
                label_text = Text(row.label, style=style)
                pred_text = Text("", style=style)
            else:
                feat_a_text = feat_a
                feat_b_text = feat_b
                label_text = row.label
                pred_text = ""

            self.data_table.add_row(marker, feat_a_text, feat_b_text, label_text, pred_text)

        status = self.query_one("#underling-status", Label)
        status.update("Awaiting round start…")

    def update_progress(self, message: RoundProgress) -> None:
        self.progress_bar.progress = message.completed
        status = self.query_one("#underling-status", Label)
        status.update(f"{message.stage} ({message.completed}/{message.total})")

        # Update cursor and prediction if available
        # message.current_index is already an all_rows index
        if message.current_index is not None:
            all_row_idx = message.current_index
            self.data_table.move_cursor(row=all_row_idx)

            # Update prediction column if we have a prediction
            if message.prediction is not None:
                row_key = self.data_table.get_row_at(all_row_idx)
                if row_key is not None:
                    # Update the prediction column (index 4)
                    is_validation = all_row_idx in self.validation_indices
                    if is_validation:
                        pred_text = Text(message.prediction, style="on blue")
                    else:
                        pred_text = message.prediction
                    self.data_table.update_cell_at((all_row_idx, 4), pred_text)

    def highlight_examples(self, examples: List[RoundExample]) -> None:
        """Highlight rows that match the given examples (used for prompt generation)."""
        log.info(f"highlight_examples: Starting with {len(examples)} examples")

        # Get EventLog for debugging output
        try:
            event_log = self.app.query_one(EventLog)
        except:
            event_log = None

        if event_log:
            event_log.info(f"[DEBUG] highlight_examples: Starting with {len(examples)} examples")

        # Build a set of example signatures for quick lookup
        example_sigs = {(ex.feature_a, ex.feature_b, ex.label) for ex in examples}
        log.info(f"highlight_examples: Built signature set with {len(example_sigs)} unique sigs")
        if event_log:
            event_log.info(f"[DEBUG] Built signature set with {len(example_sigs)} unique sigs")

        highlighted_count = 0

        log.info(f"highlight_examples: About to iterate through {len(self.all_rows)} rows")
        if event_log:
            event_log.info(f"[DEBUG] About to iterate through {len(self.all_rows)} rows")

        # Iterate through stored rows and highlight matching ones
        for row_idx, row in enumerate(self.all_rows):
            # Check if this row matches any example (using original unformatted values)
            if (row.feature_a, row.feature_b, row.label) in example_sigs:
                log.info(f"highlight_examples: Found match at row {row_idx}")
                if event_log:
                    event_log.info(f"[DEBUG] Found match at row {row_idx}")

                row_key = self.data_table.get_row_at(row_idx)
                if row_key is None:
                    log.info(f"highlight_examples: row_key is None for row {row_idx}, skipping")
                    if event_log:
                        event_log.warning(f"[DEBUG] row_key is None for row {row_idx}")
                    continue

                log.info(f"highlight_examples: Getting row data for row {row_idx}")
                # Get the row data (columns: marker, feature_a, feature_b, label, prediction)
                row_data = self.data_table.get_row(row_key)
                if len(row_data) < 4:
                    log.info(f"highlight_examples: row_data too short ({len(row_data)}), skipping")
                    if event_log:
                        event_log.warning(f"[DEBUG] row_data too short")
                    continue

                log.info(f"highlight_examples: Extracting cell values for row {row_idx}")
                # Extract current display values
                feat_a_cell = row_data[1]
                feat_b_cell = row_data[2]
                label_cell = row_data[3]

                feat_a = str(feat_a_cell.plain) if isinstance(feat_a_cell, Text) else str(feat_a_cell)
                feat_b = str(feat_b_cell.plain) if isinstance(feat_b_cell, Text) else str(feat_b_cell)
                label = str(label_cell.plain) if isinstance(label_cell, Text) else str(label_cell)

                # Highlight this row with yellow background
                highlight_style = "bold on yellow"

                log.info(f"highlight_examples: About to update cells for row {row_idx}")
                if event_log:
                    event_log.info(f"[DEBUG] About to update cells for row {row_idx}")

                # Update all cells in the row with highlight
                self.data_table.update_cell_at((row_idx, 0), row_data[0])  # Keep marker as is
                log.info(f"highlight_examples: Updated cell 0")
                self.data_table.update_cell_at((row_idx, 1), Text(feat_a, style=highlight_style))
                log.info(f"highlight_examples: Updated cell 1")
                self.data_table.update_cell_at((row_idx, 2), Text(feat_b, style=highlight_style))
                log.info(f"highlight_examples: Updated cell 2")
                self.data_table.update_cell_at((row_idx, 3), Text(label, style=highlight_style))
                log.info(f"highlight_examples: Updated cell 3")
                if event_log:
                    event_log.info(f"[DEBUG] Updated cells 0-3 for row {row_idx}")

                # Keep prediction styling if it exists
                if len(row_data) > 4:
                    pred_cell = row_data[4]
                    pred_text = str(pred_cell.plain) if isinstance(pred_cell, Text) else str(pred_cell)
                    if pred_text:
                        log.info(f"highlight_examples: About to update cell 4 with prediction")
                        self.data_table.update_cell_at((row_idx, 4), Text(pred_text, style=highlight_style))
                        log.info(f"highlight_examples: Updated cell 4")

                highlighted_count += 1
                log.info(f"highlight_examples: Completed row {row_idx}, total highlighted: {highlighted_count}")
                if event_log:
                    event_log.info(f"[DEBUG] Completed row {row_idx}, total: {highlighted_count}")

        log.info(f"highlight_examples: Finished, highlighted {highlighted_count} rows out of {len(examples)} examples")
        if event_log:
            event_log.info(f"[DEBUG] Finished highlighting {highlighted_count} rows")

    def clear_highlighting(self, all_rows: Sequence[DatasetRow]) -> None:
        """Remove highlighting from all rows, restoring normal appearance."""
        for idx, row in enumerate(all_rows):
            row_key = self.data_table.get_row_at(idx)
            if row_key is None:
                continue

            is_validation = idx in self.validation_indices

            feat_a = format_sig_figs(row.feature_a)
            feat_b = format_sig_figs(row.feature_b)

            # Restore normal styling
            if is_validation:
                style = "on blue"
                feat_a_text = Text(feat_a, style=style)
                feat_b_text = Text(feat_b, style=style)
                label_text = Text(row.label, style=style)
            else:
                feat_a_text = feat_a
                feat_b_text = feat_b
                label_text = row.label

            self.data_table.update_cell_at((idx, 1), feat_a_text)
            self.data_table.update_cell_at((idx, 2), feat_b_text)
            self.data_table.update_cell_at((idx, 3), label_text)

            # Preserve prediction column if it exists
            row_data = self.data_table.get_row(row_key)
            if len(row_data) > 4:
                pred_cell = row_data[4]
                pred_text = str(pred_cell.plain) if isinstance(pred_cell, Text) else str(pred_cell)
                if pred_text:
                    # Restore prediction with appropriate styling
                    if is_validation:
                        self.data_table.update_cell_at((idx, 4), Text(pred_text, style="on blue"))
                    else:
                        self.data_table.update_cell_at((idx, 4), pred_text)


class PromptPanel(Static):
    """Displays current prompt text."""

    def update_prompt(self, prompt: str) -> None:
        self.update(RichMarkdown(f"### Active Prompt\n{prompt}"))


class EventLog(RichLog):
    """Colour-coded log of application events."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(highlight=True, markup=True, max_lines=200, **kwargs)

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
            self.editor: Any = TextArea(text=initial)
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
            # TextArea uses .text, Input uses .value
            value = getattr(self.editor, "text", getattr(self.editor, "value", self.initial))
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
        width: 25%;
    }
    #center {
        width: 40%;
    }
    #right {
        width: 35%;
    }
    #baseline-panel {
        height: auto;
        max-height: 10;
    }
    #underling-table {
        height: 1fr;
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
        Binding("space", "process_unprocessed", "Process data"),
        Binding("r", "finalize_round", "Finalize round"),
        Binding("g", "generate_prompt", "Generate prompt"),
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
        baseline_metrics: Optional[List[BaselineMetrics]] = None,
        export_path: Optional[Path],
        max_rounds: Optional[int],
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = split
        self.database = database
        self.model = model
        self.prompt_manager = prompt_manager
        self.baseline_metrics = baseline_metrics or []
        self.export_path = export_path
        self.max_rounds = max_rounds
        self.rounds: List[RoundRecord] = database.fetch_rounds()
        self.engine = RoundEngine(split, database, model, prompt_manager)
        self.current_round: Optional[RoundRecord] = None
        self._worker: Optional[asyncio.Task[None]] = None
        self._baseline_worker: Optional[asyncio.Task[None]] = None

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
                    baseline = BaselinePanel(id="baseline-panel")
                    yield baseline
                    log_widget = EventLog(id="event-log")
                    yield log_widget
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Narrative Learning Standalone"
        dataset_panel = self.query_one(DatasetPanel)
        dataset_panel.update_summary(self.dataset, self.split, len(self.rounds))
        self.query_one(PromptPanel).update_prompt(self.prompt_manager.current_prompt)

        # Populate existing baselines or start calculating them
        if self.baseline_metrics:
            self.query_one(BaselinePanel).populate(self.baseline_metrics)
        else:
            self.query_one(EventLog).info("Calculating baseline models...")
            self._baseline_worker = asyncio.create_task(self._calculate_baselines())

        self.query_one(EventLog).info(
            f"Loaded dataset with {len(self.dataset)} rows; model {self.model.describe()}"
        )
        # Show all data, highlighting validation rows
        all_rows = self.split.train + self.split.validation
        self.query_one(UnderlingPanel).reset(all_rows, self.split.validation)
        self.query_one(HistoryList).set_rounds(self.rounds)
        if self.rounds:
            self.current_round = self.rounds[-1]
            self.query_one(RoundDetail).show_round(self.current_round)

    async def action_process_unprocessed(self) -> None:
        """Process unprocessed data points (SPACE key)."""
        if self._worker and not self._worker.done():
            self.query_one(EventLog).warning("Processing already in progress")
            return

        # Initialize the underling panel if needed
        underling = self.query_one(UnderlingPanel)
        all_rows = self.engine.get_all_rows()
        if len(self.engine.processed_indices) == 0:
            # Starting fresh
            underling.reset(all_rows, self.split.validation)
            self.query_one(EventLog).info("Starting to process data points")

        unprocessed_count = len(all_rows) - len(self.engine.processed_indices)
        if unprocessed_count == 0:
            self.query_one(EventLog).info("All data points already processed. Press 'r' to finalize round.")
            return

        self.query_one(EventLog).info(f"Processing {unprocessed_count} unprocessed data points")
        loop = asyncio.get_running_loop()

        def notify(message: RoundProgress) -> None:
            self.post_message(message)

        async def worker() -> None:
            await loop.run_in_executor(None, self.engine.process_unprocessed_points, notify)
            self.query_one(EventLog).info("Processing complete. Press 'r' to finalize round or 'p' to edit prompt.")

        self._worker = asyncio.create_task(worker())

    async def action_finalize_round(self) -> None:
        """Finalize the current round and save results (r key)."""
        if self._worker and not self._worker.done():
            self.query_one(EventLog).warning("Wait for processing to complete first")
            return

        if len(self.engine.processed_indices) == 0:
            self.query_one(EventLog).warning("No data points processed yet. Press SPACE to process data.")
            return

        if self.max_rounds is not None and len(self.rounds) >= self.max_rounds:
            self.query_one(EventLog).warning("Max rounds reached; cannot create new round.")
            return

        self.query_one(EventLog).info("Finalizing round and saving results")
        loop = asyncio.get_running_loop()

        def notify(message: RoundProgress) -> None:
            self.post_message(message)

        async def worker() -> None:
            record = await loop.run_in_executor(None, self.engine.run_round, notify)
            self.post_message(RoundFinished(record))

        self._worker = asyncio.create_task(worker())

    def _build_reprompt_prompt(self, record: RoundRecord) -> Tuple[str, List[RoundExample]]:
        """Build a prompt asking gpt-5 to improve the current prompt based on results.

        Uses TRAINING data only (not validation data) to avoid training on the test set.

        Returns:
            Tuple of (prompt_text, examples_used) where examples_used are the specific
            examples shown in the prompt (for UI highlighting).
        """
        log.info("_build_reprompt_prompt: Starting")

        # Use training examples for prompt generation
        train_examples = record.training_examples or []
        log.info(f"_build_reprompt_prompt: training_examples has {len(train_examples)} items")

        if not train_examples:
            # Fallback to validation examples if training examples not available (old data)
            train_examples = record.examples
            log.info(f"_build_reprompt_prompt: Falling back to validation examples: {len(train_examples)} items")

        # Show incorrect examples (up to 5) and correct examples (up to 3)
        incorrect = [ex for ex in train_examples if not ex.correct]
        correct = [ex for ex in train_examples if ex.correct]
        log.info(f"_build_reprompt_prompt: Found {len(incorrect)} incorrect, {len(correct)} correct")

        # Track which examples we're actually showing
        shown_examples = incorrect[:5] + correct[:3]
        log.info(f"_build_reprompt_prompt: Will show {len(shown_examples)} examples")

        # Build confusion matrix from training data
        log.info("_build_reprompt_prompt: Building confusion matrix")
        train_confusion: Dict[Tuple[str, str], int] = {}
        for ex in train_examples:
            key = (ex.label, ex.prediction)
            train_confusion[key] = train_confusion.get(key, 0) + 1

        # Build confusion matrix display
        log.info("_build_reprompt_prompt: Formatting confusion matrix")
        matrix_lines = []
        for (label, pred), count in sorted(train_confusion.items()):
            matrix_lines.append(f"  {label!r} → {pred!r}: {count}")

        # Build examples display
        log.info("_build_reprompt_prompt: Building incorrect examples")
        incorrect_examples = []
        for idx, ex in enumerate(incorrect[:5], 1):
            incorrect_examples.append(
                f"  {idx}. Feature A={ex.feature_a}, Feature B={ex.feature_b}\n"
                f"     True label: {ex.label!r}, Predicted: {ex.prediction!r}"
            )

        log.info("_build_reprompt_prompt: Building correct examples")
        correct_examples = []
        for idx, ex in enumerate(correct[:3], 1):
            correct_examples.append(
                f"  {idx}. Feature A={ex.feature_a}, Feature B={ex.feature_b}\n"
                f"     Label: {ex.label!r} (predicted correctly)"
            )

        # Calculate training accuracy
        log.info("_build_reprompt_prompt: Calculating training accuracy")
        train_correct = len(correct)
        train_total = len(train_examples)
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0

        log.info("_build_reprompt_prompt: Building final prompt string")
        prompt = f"""You are part of a program that is trying to learn inference rules on this dataset. At each round, a prompt is shown to an LLM together with one row of data at a time. It then attempts to predict the outcome based on the rules in the prompt. This process works well if the prompt has very explicit and clear rules: aim for unambiguous thresholds for values, clear criteria for labels and careful wording.

We would like to improve the prompt that is being used.

Please create a new prompt that will reduce the number of false positives and false negatives in this dataset. You can see the prompt that has been used previously, and how effective it was on the TRAINING data. There are also some examples of where the prompt did and didn't work.

Remember: you need to create rules. Don't just waffle about what changes need to happen. Look at the examples where the previous prediction system got it wrong, and try to come up with at least one new rule that would handle one of those situations correctly.

----------------------------

## Current Prompt (Round {record.id})

{record.prompt}

## Training Results

Training Accuracy: {train_accuracy:.1%}
Total training examples: {train_total}
Correct: {train_correct}
Incorrect: {len(incorrect)}

(Note: Validation accuracy is {record.metrics.accuracy:.1%}, but we show TRAINING data below to avoid overfitting to the validation set.)

### Confusion Matrix (Training Data)
{chr(10).join(matrix_lines)}

### Examples Where the Prompt FAILED (Training Data)
{chr(10).join(incorrect_examples) if incorrect_examples else "  (None - all predictions were correct!)"}

### Examples Where the Prompt SUCCEEDED (Training Data)
{chr(10).join(correct_examples) if correct_examples else "  (None)"}

----------------------------

Based on this analysis of the TRAINING data, please generate an improved prompt that will perform better on this dataset.
"""
        log.info("_build_reprompt_prompt: Returning")
        return prompt, shown_examples

    async def action_generate_prompt(self) -> None:
        """Generate a new prompt using gpt-5 based on the latest round results (g key)."""
        if self._worker and not self._worker.done():
            self.query_one(EventLog).warning("Wait for processing to complete first")
            return

        if not self.rounds:
            self.query_one(EventLog).warning("No rounds completed yet. Press SPACE to process data and 'r' to finalize a round first.")
            return

        # Use the most recent round
        latest_round = self.rounds[-1]

        self.query_one(EventLog).info(f"Generating new prompt with gpt-5 based on Round {latest_round.id}...")
        loop = asyncio.get_running_loop()

        async def worker() -> None:
            try:
                # Check if we have training or validation examples
                has_train = latest_round.training_examples and len(latest_round.training_examples) > 0
                has_val = latest_round.examples and len(latest_round.examples) > 0

                if not has_train and not has_val:
                    self.query_one(EventLog).error(
                        f"Round {latest_round.id} has no examples. "
                        "Please finalize a new round (SPACE then 'r') before generating a prompt."
                    )
                    return

                if not has_train:
                    self.query_one(EventLog).warning(
                        f"Round {latest_round.id} has no training examples (old round format). "
                        "Using validation examples instead. Consider finalizing a new round for better results."
                    )

                # Build the reprompt and get the examples that will be shown
                self.query_one(EventLog).info("Building prompt for GPT-5...")
                log.info("About to call _build_reprompt_prompt")
                reprompt_prompt, shown_examples = self._build_reprompt_prompt(latest_round)
                log.info("Returned from _build_reprompt_prompt")
                self.query_one(EventLog).info("Prompt built successfully")

                # Highlight the rows that are being used as examples
                log.info("About to query UnderlingPanel")
                underling = self.query_one(UnderlingPanel)
                log.info(f"Got underling panel, about to highlight {len(shown_examples)} examples")
                self.query_one(EventLog).info(f"About to highlight {len(shown_examples)} example rows...")

                underling.highlight_examples(shown_examples)

                log.info("Highlighting complete")
                self.query_one(EventLog).info(f"Highlighting complete for {len(shown_examples)} rows")

                # Call gpt-5 to generate a new prompt
                self.query_one(EventLog).info("Calling GPT-5 to generate new prompt...")
                log.info(f"Calling dispatch_reprompt_prompt with model=gpt-5")

                new_prompt_data, process_info = await loop.run_in_executor(
                    None,
                    llmcall.dispatch_reprompt_prompt,
                    "gpt-5",
                    reprompt_prompt
                )

                log.info(f"GPT-5 returned: {new_prompt_data}")
                self.query_one(EventLog).info("GPT-5 response received")

                new_prompt = new_prompt_data['updated_prompt']
                reasoning = new_prompt_data.get('reasoning', '')

                # Clear the highlighting
                all_rows = self.engine.get_all_rows()
                underling.clear_highlighting(all_rows)

                # Update the prompt manager and display
                self.prompt_manager.set_prompt(new_prompt)
                self.query_one(PromptPanel).update_prompt(new_prompt)

                # Log the update
                self.query_one(EventLog).info(f"New prompt generated by gpt-5")
                if reasoning:
                    self.query_one(EventLog).info(f"Reasoning: {reasoning[:100]}...")

            except Exception as e:
                # Clear highlighting even on error
                all_rows = self.engine.get_all_rows()
                self.query_one(UnderlingPanel).clear_highlighting(all_rows)
                self.query_one(EventLog).error(f"Failed to generate prompt: {e}")
                log.exception("Prompt generation failed")
                import traceback
                self.query_one(EventLog).error(f"Traceback: {traceback.format_exc()[:200]}")

        self._worker = asyncio.create_task(worker())

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
        if isinstance(self.model, StubModelAdapter):
            if "(stub)" in self.model.name:
                self.model.name = self.model.name.replace(" (stub)", "")
                self.query_one(EventLog).info("Stub label hidden; behaviour unchanged.")
            else:
                self.model.name = f"{self.model.name} (stub)"
                self.query_one(EventLog).info("Stub label restored.")
        else:
            self.query_one(EventLog).info(f"Using real LLM: {self.model.name}")

    async def _calculate_baselines(self) -> None:
        """Calculate baseline models in the background, updating UI as each completes."""
        loop = asyncio.get_running_loop()

        def run_calculations() -> List[BaselineMetrics]:
            """Run baseline calculations in thread pool."""
            results = []
            for baseline in calculate_baseline_metrics_incremental(self.split):
                results.append(baseline)
                # Notify UI from the event loop
                loop.call_soon_threadsafe(self._add_baseline_result, baseline)
            return results

        # Run calculations in thread pool to avoid blocking the UI
        self.baseline_metrics = await loop.run_in_executor(None, run_calculations)
        self.query_one(EventLog).info("All baseline models calculated.")

    def _add_baseline_result(self, baseline: BaselineMetrics) -> None:
        """Add a baseline result to the panel (called from thread)."""
        self.query_one(BaselinePanel).add_baseline(baseline)
        self.query_one(EventLog).info(f"Baseline {baseline.name}: {baseline.accuracy:.2%}")

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
        if event.list_view.id == "history":
            # Use the index to get the corresponding round
            index = event.list_view.index
            if index is not None and 0 <= index < len(self.rounds):
                self.current_round = self.rounds[index]
                self.query_one(RoundDetail).show_round(self.current_round)

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
    parser.add_argument("--dataset", type=Path, help="Path to DataPainter SQLite database file", default=None)
    parser.add_argument("--table", type=str, help="Table name (defaults to first table in metadata)", default=None)
    parser.add_argument("--database", type=Path, help="SQLite database path for results", default=None)
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

    # Extract valid predictions from dataset for tool calling
    valid_predictions = extract_valid_predictions(dataset)

    # Use RealLLMAdapter with hard-coded gpt-5-mini
    model = RealLLMAdapter("gpt-5-mini", valid_predictions)

    prompt_manager = PromptManager("Choose randomly")
    export_path = args.export_json
    # Baselines will be calculated asynchronously when the app starts
    app = StandaloneApp(
        dataset=dataset,
        split=split,
        database=database,
        model=model,
        prompt_manager=prompt_manager,
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
