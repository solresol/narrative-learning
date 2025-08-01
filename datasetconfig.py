#!/usr/bin/env python3
import sqlite3
import sys
import json
import os
import pandas as pd
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from modules.exceptions import TargetClassingException, NonexistentRoundException, MissingConfigElementException, NoProcessedRoundsException
from modules.database import get_database_path
from modules.metrics import (
    calculate_metric, get_matrix_label_for_prediction, format_confusion_matrix,
    generate_metrics_data, create_metrics_dataframe
)
from modules.text_analysis import (
    calculate_zipfs_law, calculate_herdans_law, count_words, get_word_counts
)

class DatasetConfig:
    """
    Class for handling dataset configuration and database connection.
    This class provides methods for retrieving and handling obfuscated data.
    """

    def __init__(self, conn: Any, config_path: str, dataset: str = "", investigation_id: Optional[int] = None):
        """
        Initialize with a database connection and configuration file path.

        Args:
            conn: SQLite or PostgreSQL database connection
            config_path: Path to the JSON configuration file
            investigation_id: Filter rounds by this investigation when using PostgreSQL
        """
        self.conn = conn
        self.dataset = dataset
        self.investigation_id = investigation_id

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Required configuration fields
        required_fields = [
            "table_name",
            "primary_key",
            "target_field",
            "splits_table",
            "rounds_table",
            "columns",
        ]

        # Verify configuration has all required fields
        for field in required_fields:
            if field not in self.config:
                raise MissingConfigElementException(field, config_path)

        # Store key configuration values as attributes for convenience
        self.table_name = self.config["table_name"]
        self.primary_key = self.config["primary_key"]
        self.target_field = self.config["target_field"]
        self.splits_table = self.config["splits_table"]
        self.rounds_table = self.config["rounds_table"]
        self.columns = self.config["columns"]
        self.database_path = get_database_path(self.conn)

        cursor = conn.cursor()
        try:
            cursor.execute(
                f"select distinct {self.target_field} from {self.table_name} order by {self.target_field}"
            )
        except sqlite3.OperationalError as e:
            sys.stderr.write(f"Problem with {self.database_path}: ")
            raise e
        self.valid_predictions = [row[0] for row in cursor]
        if len(self.valid_predictions) != 2:
            raise TargetClassingException(self.target_field, len(self.valid_predictions))

    def _execute(self, cur, query: str, params: tuple = ()) -> None:
        """Execute a query converting placeholders for PostgreSQL if needed."""
        if not isinstance(self.conn, sqlite3.Connection):
            query = query.replace("?", "%s")
        cur.execute(query, params)

    def _ident(self, name: str) -> str:
        """Return a safely quoted identifier depending on the database."""
        if isinstance(self.conn, sqlite3.Connection):
            return f'"{name}"'
        return name


    def get_entity_features(self, entity_id: str) -> str:
        """
        Retrieve the features for a specific entity as a formatted string.

        Args:
            entity_id: The ID of the entity to retrieve

        Returns:
            Formatted string with entity features
        """
        # Build a query that selects all columns except primary key and target
        columns_to_select = []
        for column in self.columns:
            if column != self.primary_key and column != self.target_field:
                columns_to_select.append(column)

        column_list = ", ".join([self._ident(col) for col in columns_to_select])

        query = f"""
        SELECT {column_list}
        FROM {self.table_name}
        WHERE {self.primary_key} = ?
        """

        cur = self.conn.cursor()
        self._execute(cur, query, (entity_id,))
        row = cur.fetchone()

        if not row:
            raise KeyError(f"Entity ID '{entity_id}' not found.")

        # Format the result as a string
        result = ""
        for i, col in enumerate(columns_to_select):
            result += f"\t{col}: {row[i]}\n"

        return result

    def get_entity_by_id(self, entity_id: str) -> Dict[str, Any]:
        """
        Retrieve all data for a specific entity.

        Args:
            entity_id: The ID of the entity to retrieve

        Returns:
            Dictionary with entity data
        """
        # I think this function isn't used. The only difference between
        # this and get_entity_features is that this supplies the entity_id
        # (which the caller already knew) and the target (which is only used
        # in the training phase, when it's an aggregate query and we wouldn't
        # ask for one element).
        column_list = ", ".join([self._ident(col) for col in self.columns])

        query = f"""
        SELECT {column_list}
        FROM {self.table_name}
        WHERE {self.primary_key} = ?
        """

        cur = self.conn.cursor()
        self._execute(cur, query, (entity_id,))
        row = cur.fetchone()

        if not row:
            sys.exit(f"Entity ID '{entity_id}' not found.")

        # Convert to dictionary
        result = {}
        for i, col in enumerate(self.columns):
            result[col] = row[i]

        return result

    def get_random_non_holdout_id(self, split_id: int) -> str:
        """
        Retrieve the ID of a random entity that is not part of the holdout dataset.

        Args:
           split_id: The split ID to check against

        Returns:
           A random entity ID that is not in the holdout set

        Raises:
           ValueError: If no non-holdout data is available for the given split
        """
        cur = self.conn.cursor()
        query = f"""
           SELECT s.{self.primary_key}
           FROM {self.splits_table} s
           WHERE s.split_id = ? AND s.holdout = ?
           ORDER BY RANDOM()
           LIMIT 1
        """

        holdout_false = 0 if isinstance(self.conn, sqlite3.Connection) else False
        self._execute(cur, query, (split_id, holdout_false))
        row = cur.fetchone()

        if row is None:
            raise ValueError(f"No non-holdout data available for split ID {split_id}")

        return row[0]

    def is_holdout_data(self, entity_id: str, split_id: int) -> bool:
        """
        Check if an entity is part of the holdout dataset.

        Args:
            entity_id: The ID of the entity to check
            split_id: The split ID to check against

        Returns:
            True if the entity is holdout data, False otherwise
        """
        cur = self.conn.cursor()
        query = f"""
        SELECT holdout
        FROM {self.splits_table}
        WHERE {self.primary_key} = ? AND split_id = ?
        """

        self._execute(cur, query, (entity_id, split_id))
        row = cur.fetchone()

        if row is None:
            raise KeyError(entity_id)

        return bool(row[0])

    def get_round_prompt(self, round_id: int) -> str:
        """
        Retrieve the prompt text for a given round.

        Args:
            round_id: ID of the round

        Returns:
            Prompt text
        """
        cur = self.conn.cursor()
        self._execute(cur, f"SELECT prompt FROM {self.rounds_table} WHERE round_id = ?", (int(round_id),))
        row = cur.fetchone()
        if row is None:
            raise NonexistentRoundException(round_id, self.database_path)
        return row[0]
        
    def get_round_reasoning(self, round_id: int) -> str:
        """
        Retrieve the reasoning for the prompt text for a given round.

        Args:
            round_id: ID of the round

        Returns:
            Reasoning text for the prompt
        """
        cur = self.conn.cursor()
        self._execute(cur, f"SELECT reasoning_for_this_prompt FROM {self.rounds_table} WHERE round_id = ?", (int(round_id),))
        row = cur.fetchone()
        if row is None:
            raise NonexistentRoundException(round_id, self.database_path)
        return row[0] if row[0] is not None else ""

    def get_split_id(self, round_id: int) -> int:
        """
        Get the split ID associated with a round.

        Args:
            round_id: ID of the round

        Returns:
            Split ID
        """
        cur = self.conn.cursor()
        self._execute(cur, f"SELECT split_id FROM {self.rounds_table} WHERE round_id = ?", (int(round_id),))
        row = cur.fetchone()
        if row is None:
            raise NonexistentRoundException(round_id, self.database_path)
        split_id = row[0]

        # Ensure the split ID actually exists in the splits table. Earlier
        # datasets used an initial value of ``0`` before any splits were
        # created which means there may not be matching rows.  If no entry is
        # found, default to the minimum available split ID and update the round
        # so subsequent calls use the valid value.
        self._execute(cur, f"SELECT 1 FROM {self.splits_table} WHERE split_id = ? LIMIT 1", (split_id,))
        if cur.fetchone() is None:
            self._execute(cur, f"SELECT MIN(split_id) FROM {self.splits_table}")
            row_min = cur.fetchone()
            if row_min is None or row_min[0] is None:
                raise ValueError(f"No splits defined in table {self.splits_table}")
            split_id = row_min[0]
            self._execute(
                cur,
                f"UPDATE {self.rounds_table} SET split_id = ? WHERE round_id = ?",
                (split_id, int(round_id)),
            )
            self.conn.commit()

        return split_id

    def positive_label(self):
        # This ain't great. Fundamentally, we are trying to shoe-horn into a "true positive" / "false positive"
        # labelling system and it's not all that well-defined what positive or negative is here.
        return self.valid_predictions[0]

    def negative_label(self):
        return self.valid_predictions[1]

    def get_matrix_label_for_prediction(self, ground_truth, prediction):
        return get_matrix_label_for_prediction(
            ground_truth, prediction, self.positive_label(), self.negative_label()
        )

    def get_confusion_matrix(self, round_id: int, example_count: int = 0,
                           on_holdout_data: bool = False, on_test_data: bool = False) -> Dict:
        """
        Retrieve the confusion matrix for a given round.

        Args:
            round_id: ID of the round
            example_count: Number of examples to include per cell (0 for none)
            on_holdout_data: Whether to use holdout data
            on_test_data: Whether to use test data (subset of holdout)

        Returns:
            Confusion matrix dictionary
        """
        cur = self.conn.cursor()
        split_id = self.get_split_id(round_id)

        inf_table = f"{self.dataset}_inferences" if self.dataset else "inferences"
        query = f"""
            SELECT m.{self.target_field}, i.prediction, i.{self.primary_key},
                   i.narrative_text, s.holdout, s.validation
              FROM {inf_table} i
              JOIN {self.table_name} m ON i.{self.primary_key} = m.{self.primary_key}
              JOIN {self.splits_table} s ON (s.{self.primary_key} = i.{self.primary_key})
             WHERE i.round_id = ? AND s.split_id = ?
             ORDER BY RANDOM()
        """

        self._execute(cur, query, (round_id, split_id))
        rows = cur.fetchall()

        # Initialize confusion matrix cells
        matrix = {
            'TP': {'count': 0, 'examples': []},
            'FN': {'count': 0, 'examples': []},
            'FP': {'count': 0, 'examples': []},
            'TN': {'count': 0, 'examples': []},
        }

        for outcome, prediction, entity_id, narrative_text, holdout, validation in rows:
            if on_holdout_data and not holdout:
                continue
            if not on_holdout_data and holdout:
                continue
            if on_holdout_data and holdout:
                if validation and on_test_data:
                    continue
                if not validation and not on_test_data:
                    continue

            cell = self.get_matrix_label_for_prediction(outcome, prediction)

            # Update counts and collect examples if needed
            matrix[cell]['count'] += 1
            if example_count > 0 and len(matrix[cell]['examples']) < example_count:
                matrix[cell]['examples'].append({
                    'entity_id': entity_id,
                    'features': self.get_entity_features(entity_id),
                    'narrative_text': narrative_text.replace('\n', '\n\t'),
                    'outcome': outcome,
                    'prediction': prediction
                })

        return matrix

    def get_printable_confusion_matrix_and_examples(self, round_id: int, matrix: Dict, show_examples: bool = True) -> str:
        """
        Format a confusion matrix as a printable string.

        Args:
            round_id: ID of the round
            matrix: Confusion matrix from get_confusion_matrix
            show_examples: Whether to include examples in the output

        Returns:
            Formatted string representation
        """
        prompt = self.get_round_prompt(round_id)
        return format_confusion_matrix(matrix, round_id, prompt,
                                       self.negative_label(),
                                       self.positive_label(),
                                       show_examples)

    def calculate_metric(self, matrix: Dict, metric_name: str) -> float:
        """
        Calculate the specified metric from a confusion matrix.

        Args:
            matrix: Confusion matrix dictionary
            metric_name: Name of the metric to calculate ('count', 'accuracy', 'precision', 'recall', 'f1')

        Returns:
            Float value of the calculated metric

        Raises:
            ValueError: If the metric name is unknown
        """
        return calculate_metric(matrix, metric_name)

    def get_latest_split_id(self) -> int:
        """
        Get the split_id from the most recent round.

        Returns:
            Integer split ID

        Raises:
            SystemExit: If no rounds are found in the database
        """
        cur = self.conn.cursor()
        query = f"SELECT split_id FROM {self.rounds_table}"
        params: list[Any] = []
        if self.investigation_id is not None:
            query += " WHERE investigation_id = ?"
            params.append(self.investigation_id)
        query += " ORDER BY round_id DESC LIMIT 1"
        self._execute(cur, query, tuple(params))
        row = cur.fetchone()
        if row is None:
            sys.exit("No rounds found in database")
        return row[0]

    def get_rounds_for_split(self, split_id: int) -> List[int]:
        """
        Get all round IDs for a given split_id.

        Args:
            split_id: The split ID to query

        Returns:
            List of round IDs
        """
        cur = self.conn.cursor()
        query = f"SELECT round_id FROM {self.rounds_table} WHERE split_id = ?"
        params: list[Any] = [split_id]
        if self.investigation_id is not None:
            query += " AND investigation_id = ?"
            params.append(self.investigation_id)
        query += " ORDER BY round_id"
        self._execute(cur, query, tuple(params))
        rounds = [row[0] for row in cur.fetchall()]
        return rounds

    def get_processed_rounds_for_split(self, split_id: int) -> List[int]:
        """
        Get all round IDs for a given split_id that have inferences.

        Args:
            split_id: The split ID to query

        Returns:
            List of round IDs that have inferences
        """
        cur = self.conn.cursor()
        inf_table = f"{self.dataset}_inferences" if self.dataset else "inferences"
        answer = []
        for r in self.get_rounds_for_split(split_id):
            query = f"select count(*) from {inf_table} where round_id = ?"
            params: list[Any] = [r]
            if self.investigation_id is not None:
                query += " and investigation_id = ?"
                params.append(self.investigation_id)
            self._execute(cur, query, tuple(params))
            row = cur.fetchone()
            if row[0] == 0:
                continue
            answer.append(r)
        return answer

    def check_early_stopping(self, split_id: int, metric: str,
                            patience: int, on_validation: bool = True) -> bool:
        """
        Check if training should be stopped based on validation performance.

        Args:
            split_id: The split ID to check
            metric: The metric to use for evaluation ('accuracy', 'precision', 'recall', 'f1')
            patience: Number of rounds to look back for improvement
            on_validation: Whether to use validation data

        Returns:
            True if training should stop, False otherwise
        """
        rounds = self.get_processed_rounds_for_split(split_id)
        if len(rounds) < patience + 1:
            return False

        # Look at last 'patience' + 1 rounds
        relevant_rounds = rounds[-(patience + 1):]
        oldest_round = relevant_rounds[0]

        # Calculate combined metric for oldest round
        train_matrix = self.get_confusion_matrix(oldest_round)
        val_matrix = self.get_confusion_matrix(oldest_round, on_holdout_data=on_validation)
        train_score = self.calculate_metric(train_matrix, metric)
        val_score = self.calculate_metric(val_matrix, metric)
        best_score = min(train_score, val_score)

        # Check if any later round beat this score
        for round_id in relevant_rounds[1:]:
            train_matrix = self.get_confusion_matrix(round_id)
            val_matrix = self.get_confusion_matrix(round_id, on_holdout_data=on_validation)
            train_score = self.calculate_metric(train_matrix, metric)
            val_score = self.calculate_metric(val_matrix, metric)
            score = min(train_score, val_score)
            if score > best_score:
                return False

        return True

    def generate_metrics_data(self, split_id: int, metric: str, data_type: str) -> pd.DataFrame:
        """
        Generate a DataFrame with metrics for all rounds in a split.

        Args:
            split_id: The split ID to analyze
            metric: The metric to calculate ('accuracy', 'precision', 'recall', 'f1')
            data_type: Type of data to analyze ('train', 'validation', or 'test')

        Returns:
            DataFrame with round_id and metric columns
        """
        return generate_metrics_data(self, int(split_id), metric, data_type)

    def get_data_point_count(self) -> int:
        """
        Get the total number of data points in the dataset.

        Returns:
            Integer count of data points in the table
        """
        cur = self.conn.cursor()
        self._execute(cur, f"SELECT COUNT(*) FROM {self.table_name}")
        row = cur.fetchone()
        return row[0] if row else 0

    def get_best_round_id(self, split_id: int, metric: str = 'accuracy') -> int:
        """
        Find the round with the best combined train/validation performance.

        Args:
            split_id: The split ID to analyze
            metric: The metric to calculate ('accuracy', 'precision', 'recall', 'f1')

        Returns:
            Round ID with the highest ``min(train_score, validation_score)``

        Raises:
            ValueError: If no processed rounds are found
        """
        df = self.create_metrics_dataframe(split_id, metric, ['train', 'validation'])
        if df.empty:
            raise NoProcessedRoundsException(split_id, self.database_path)

        df['combined'] = df[[f'train {metric}', f'validation {metric}']].min(axis=1)
        return df['combined'].idxmax()

    def get_test_metric_for_best_validation_round(self, split_id: int,
                                               validation_metric: str = 'accuracy',
                                               test_metric: Optional[str] = None) -> float:
        """
        Get the test set performance of the round with the best validation performance.

        Args:
            split_id: The split ID to analyze
            validation_metric: The metric to use for finding the best validation round
            test_metric: The metric to calculate on the test set (defaults to same as validation_metric)

        Returns:
            Test metric value for the best validation round

        Raises:
            ValueError: If no processed rounds are found or the best round has no test data
        """
        if test_metric is None:
            test_metric = validation_metric

        best_round_id = self.get_best_round_id(split_id, validation_metric)
        test_data = self.generate_metrics_data(split_id, test_metric, 'test')

        row = test_data[test_data.round_id == best_round_id]
        if row.empty:
            raise ValueError(f"No test data available for best round {best_round_id}")

        return row.metric.iloc[0]

    def create_metrics_dataframe(self, split_id: int, metric: str,
                              data_types: List[str]) -> pd.DataFrame:
        """
        Create a DataFrame with metrics for all specified data types.

        Args:
            split_id: The split ID to analyze
            metric: The metric to calculate
            data_types: List of data types to include ('train', 'validation', 'test')

        Returns:
            DataFrame with columns for each data type metric
        """
        return create_metrics_dataframe(self, split_id, metric, data_types)
        
    def get_all_prompts_and_reasoning(self, split_id: Optional[int] = None, up_to_round: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Retrieve all prompts and reasoning for all rounds in a split.
        
        Args:
            split_id: The split ID to analyze. If None, uses the most recent split.
            up_to_round: If provided, only include rounds up to and including this round ID.
        
        Returns:
            Dictionary with 'prompts' and 'reasoning' lists containing text from all rounds
        """
        if split_id is None:
            split_id = self.get_latest_split_id()
            
        rounds = self.get_processed_rounds_for_split(split_id)
        
        # Filter rounds if up_to_round is specified
        if up_to_round is not None:
            rounds = [r for r in rounds if r <= up_to_round]
            
        prompts = []
        reasoning = []
        
        for round_id in rounds:
            try:
                prompt_text = self.get_round_prompt(round_id)
                reasoning_text = self.get_round_reasoning(round_id)
                
                if prompt_text:
                    prompts.append(prompt_text)
                if reasoning_text:
                    reasoning.append(reasoning_text)
            except NonexistentRoundException:
                continue
                
        return {
            'prompts': prompts,
            'reasoning': reasoning
        }
        
    def get_total_word_count(self, split_id: Optional[int] = None, up_to_round: Optional[int] = None) -> Dict[str, int]:
        """
        Calculate the total word count of all prompts and reasoning up to a specific round.
        
        Args:
            split_id: The split ID to analyze. If None, uses the most recent split.
            up_to_round: If provided, only include rounds up to and including this round ID.
            
        Returns:
            Dictionary with total word counts for prompts, reasoning, and combined
        """
        corpus = self.get_all_prompts_and_reasoning(split_id, up_to_round)
        return get_word_counts(corpus.get('prompts', []), corpus.get('reasoning', []))

    def calculate_zipfs_law(self, split_id: Optional[int] = None) -> Dict[str, Union[float, dict]]:
        """
        Calculate Zipf's law coefficient from all prompts and reasoning combined.
        Zipf's law states that the frequency of a word is inversely proportional to its rank.
        
        Args:
            split_id: The split ID to analyze. If None, uses the most recent split.
            
        Returns:
            Dictionary with Zipf coefficient and related statistics
        """
        corpus = self.get_all_prompts_and_reasoning(split_id)
        text_list = corpus.get('prompts', []) + corpus.get('reasoning', [])
        return calculate_zipfs_law(text_list)
    
    def calculate_herdans_law(self, split_id: Optional[int] = None) -> Dict[str, Union[float, dict]]:
        """
        Calculate Herdan's law (Heaps' law) coefficient from all prompts and reasoning combined.
        Herdan's law describes the relationship between vocabulary size and text length.
        
        Args:
            split_id: The split ID to analyze. If None, uses the most recent split.
            
        Returns:
            Dictionary with Herdan coefficient and related statistics
        """
        corpus = self.get_all_prompts_and_reasoning(split_id)
        text_list = corpus.get('prompts', []) + corpus.get('reasoning', [])
        return calculate_herdans_law(text_list)
