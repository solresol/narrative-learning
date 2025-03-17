#!/usr/bin/env python3
import sqlite3
import sys
import json
import os
import pandas as pd
import numpy as np
import re
from collections import Counter
import math
from typing import Dict, List, Any, Optional, Tuple, Union

class TargetClassingException(Exception):
    def __init__(self, target, num_classes):
        self.message = f"{target} has {num_classes}, but narrative learning can only cope with binary classification at the moment"
        super().__init__(self.message)

    def __str__(self):
        return self.message

class NonexistentRoundException(Exception):
    def __init__(self, round_id, db_path=None):
        if db_path is None:
            self.message = f"Round {round_id} not found"
        else:
            self.message = f"Round {round_id} not found in the sqlite database {db_path}"
        super().__init__(self.message)
    def __str__(self):
        return self.message

class MissingConfigElementException(Exception):
    """Exception raised when a required configuration element is missing.

    Attributes:
        column_name -- The name of the missing configuration element
        config_path -- The path to the configuration file being read
    """

    def __init__(self, column_name, config_path):
        self.column_name = column_name
        self.config_path = config_path
        self.message = f"Missing required configuration element: '{column_name}' in config file: '{config_path}'"
        super().__init__(self.message)

    def __str__(self):
        return self.message

class DatasetConfig:
    """
    Class for handling dataset configuration and database connection.
    This class provides methods for retrieving and handling obfuscated data.
    """

    def __init__(self, conn: sqlite3.Connection, config_path: str):
        """
        Initialize with a database connection and configuration file path.

        Args:
            conn: SQLite database connection
            config_path: Path to the JSON configuration file
        """
        self.conn = conn

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Required configuration fields
        required_fields = [ "table_name", "primary_key", "target_field", "splits_table", "columns" ]

        # Verify configuration has all required fields
        for field in required_fields:
            if field not in self.config:
                raise MissingConfigElementException(field, config_path)

        # Store key configuration values as attributes for convenience
        self.table_name = self.config["table_name"]
        self.primary_key = self.config["primary_key"]
        self.target_field = self.config["target_field"]
        self.splits_table = self.config["splits_table"]
        self.columns = self.config["columns"]

        cursor = conn.cursor()
        cursor.execute(f"select distinct {self.target_field} from {self.table_name} order by {self.target_field}")
        self.valid_predictions = []
        for row in cursor:
            self.valid_predictions.append(row[0])
        if len(self.valid_predictions) != 2:
            raise TargetClassingException(self.target_field, len(self.valid_predictions))

        self.database_path = self.get_database_path()


    def get_database_path(self):
        """Get the file path of an SQLite database from its connection object."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA database_list")
        # The database info is returned as (seq, name, file)
        db_info = cursor.fetchall()

        # The main database is usually the one with name 'main'
        for entry in db_info:
            if entry[1] == 'main':
                return entry[2]

        # If we didn't find one labeled 'main', return the first file path
        if db_info:
            return db_info[0][2]

        return None

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

        column_list = ", ".join([f'"{col}"' for col in columns_to_select])

        query = f"""
        SELECT {column_list}
        FROM {self.table_name}
        WHERE {self.primary_key} = ?
        """

        cur = self.conn.cursor()
        cur.execute(query, (entity_id,))
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
        column_list = ", ".join([f'"{col}"' for col in self.columns])

        query = f"""
        SELECT {column_list}
        FROM {self.table_name}
        WHERE {self.primary_key_field} = ?
        """

        cur = self.conn.cursor()
        cur.execute(query, (entity_id,))
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
           WHERE s.split_id = ? AND s.holdout = 0
           ORDER BY RANDOM()
           LIMIT 1
        """

        cur.execute(query, (split_id,))
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

        cur.execute(query, (entity_id, split_id))
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
        cur.execute("SELECT prompt FROM rounds WHERE round_id = ?", (int(round_id),))
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
        cur.execute("SELECT reasoning_for_this_prompt FROM rounds WHERE round_id = ?", (int(round_id),))
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
        cur.execute("SELECT split_id FROM rounds WHERE round_id = ?", (round_id,))
        row = cur.fetchone()

        if row is None:
            raise NonexistentRoundException(round_id, self.database_path)

        return row[0]

    def positive_label(self):
        # This ain't great. Fundamentally, we are trying to shoe-horn into a "true positive" / "false positive"
        # labelling system and it's not all that well-defined what positive or negative is here.
        return self.valid_predictions[0]

    def negative_label(self):
        return self.valid_predictions[1]

    def get_matrix_label_for_prediction(self, ground_truth, prediction):
        if ground_truth == self.positive_label():
            if ground_truth == prediction:
                return 'TP'
            return 'FN'
        if ground_truth == prediction:
            return 'TN'
        else:
            return 'FP'

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

        query = f"""
            SELECT m.{self.target_field}, i.prediction, i.{self.primary_key},
                   i.narrative_text, s.holdout, s.validation
              FROM inferences i
              JOIN {self.table_name} m ON i.{self.primary_key} = m.{self.primary_key}
              JOIN {self.splits_table} s ON (s.{self.primary_key} = i.{self.primary_key})
             WHERE i.round_id = ? AND s.split_id = ?
             ORDER BY RANDOM()
        """

        cur.execute(query, (round_id, split_id))
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

        answer = ""
        answer += f"Round ID: {round_id}\n"
        answer += "Prompt used:\n\t"
        answer += prompt.replace('\n', '\n\t')
        answer += "\n\nConfusion Matrix:\n"

        # Layout: rows are Actual values; columns are Predicted
        answer += (f"{'':15s} {'Predicted Positive':20s} {'Predicted Negative':20s}\n")

        # For actual positive
        tp = matrix['TP']['count']
        fn = matrix['FN']['count']
        answer += (f"{'Actual Positive':15s} {tp:20d} {fn:20d}\n")

        # For actual negative
        fp = matrix['FP']['count']
        tn = matrix['TN']['count']
        answer += (f"{'Actual Negative':15s} {fp:20d} {tn:20d}\n")
        answer += "\n"

        # Calculate metrics
        total_count = tp + fn + fp + tn
        accuracy = (tp + tn) / total_count if total_count > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Add metrics to output
        answer += f"Accuracy: {accuracy:.3f}\n"
        answer += f"Precision: {precision:.3f}\n"
        answer += f"Recall: {recall:.3f}\n"
        answer += f"F1 Score: {f1_score:.3f}\n\n"

        # Add examples if requested
        if show_examples:
            for cell in ['TP', 'FN', 'FP', 'TN']:
                examples = matrix[cell]['examples']
                if examples:
                    cell_full = {
                        'TP': "True Positives",
                        'FN': "False Negatives",
                        'FP': "False Positives",
                        'TN': "True Negatives"
                    }[cell]
                    ex = examples[0]
                    answer += (f"Examples for {cell_full}: (Correct answer: {ex['outcome']}, What the previous set of rules predicted: {ex['prediction']})\n")
                    for ex in examples:
                        answer += (f"  Entity Data:\n{ex['features']}\n")
                    answer += "\n"

        return answer


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
        tp = matrix['TP']['count']
        fn = matrix['FN']['count']
        fp = matrix['FP']['count']
        tn = matrix['TN']['count']

        if metric_name == 'count':
            return tp + fn + fp + tn
        if metric_name == 'accuracy':
            total = tp + fn + fp + tn
            return (tp + tn) / total if total > 0 else 0
        elif metric_name == 'precision':
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric_name == 'recall':
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        elif metric_name == 'f1':
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def get_latest_split_id(self) -> int:
        """
        Get the split_id from the most recent round.

        Returns:
            Integer split ID

        Raises:
            SystemExit: If no rounds are found in the database
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT split_id
            FROM rounds
            ORDER BY round_id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            sys.exit("No rounds found in database")
        split_id = row[0]
        return split_id

    def get_rounds_for_split(self, split_id: int) -> List[int]:
        """
        Get all round IDs for a given split_id.

        Args:
            split_id: The split ID to query

        Returns:
            List of round IDs
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT round_id
            FROM rounds
            WHERE split_id = ?
            ORDER BY round_id
        """, (split_id,))
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
        answer = []
        for r in self.get_rounds_for_split(split_id):
            cur.execute("select count(*) from inferences where round_id = ?", [r])
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

        # Calculate metric for oldest round
        oldest_matrix = self.get_confusion_matrix(oldest_round, on_holdout_data=on_validation)
        best_score = self.calculate_metric(oldest_matrix, metric)

        # Check if any later round beat this score
        for round_id in relevant_rounds[1:]:
            matrix = self.get_confusion_matrix(round_id, on_holdout_data=on_validation)
            score = self.calculate_metric(matrix, metric)
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
        rounds = self.get_processed_rounds_for_split(split_id)
        data = []

        for round_id in rounds:
            # Set appropriate flags based on data_type
            on_holdout = data_type in ('validation', 'test')
            on_test_data = data_type == 'test'
            matrix = self.get_confusion_matrix(round_id, on_holdout_data=on_holdout, on_test_data=on_test_data)
            score = self.calculate_metric(matrix, metric)
            data.append({
                'round_id': round_id,
                'metric': score
            })

        return pd.DataFrame(data)


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
        tp = matrix['TP']['count']
        fn = matrix['FN']['count']
        fp = matrix['FP']['count']
        tn = matrix['TN']['count']

        if metric_name == 'count':
            return tp + fn + fp + tn
        if metric_name == 'accuracy':
            total = tp + fn + fp + tn
            return (tp + tn) / total if total > 0 else 0
        elif metric_name == 'precision':
            return tp / (tp + fp) if (tp + fp) > 0 else 0
        elif metric_name == 'recall':
            return tp / (tp + fn) if (tp + fn) > 0 else 0
        elif metric_name == 'f1':
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def get_latest_split_id(self) -> int:
        """
        Get the split_id from the most recent round.

        Returns:
            Integer split ID

        Raises:
            SystemExit: If no rounds are found in the database
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT split_id
            FROM rounds
            ORDER BY round_id DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            sys.exit("No rounds found in database")
        split_id = row[0]
        return split_id

    def get_rounds_for_split(self, split_id: int) -> List[int]:
        """
        Get all round IDs for a given split_id.

        Args:
            split_id: The split ID to query

        Returns:
            List of round IDs
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT round_id
            FROM rounds
            WHERE split_id = ?
            ORDER BY round_id
        """, (split_id,))
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
        answer = []
        for r in self.get_rounds_for_split(split_id):
            cur.execute("select count(*) from inferences where round_id = ?", [r])
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

        # Calculate metric for oldest round
        oldest_matrix = self.get_confusion_matrix(oldest_round, on_holdout_data=on_validation)
        best_score = self.calculate_metric(oldest_matrix, metric)

        # Check if any later round beat this score
        for round_id in relevant_rounds[1:]:
            matrix = self.get_confusion_matrix(round_id, on_holdout_data=on_validation)
            score = self.calculate_metric(matrix, metric)
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
        rounds = self.get_processed_rounds_for_split(split_id)
        data = []

        for round_id in rounds:
            # Set appropriate flags based on data_type
            on_holdout = data_type in ('validation', 'test')
            on_test_data = data_type == 'test'
            matrix = self.get_confusion_matrix(round_id, on_holdout_data=on_holdout, on_test_data=on_test_data)
            score = self.calculate_metric(matrix, metric)
            data.append({
                'round_id': round_id,
                'metric': score
            })

        return pd.DataFrame(data)

    def get_data_point_count(self) -> int:
        """
        Get the total number of data points in the dataset.

        Returns:
            Integer count of data points in the table
        """
        cur = self.conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        row = cur.fetchone()
        return row[0] if row else 0

    def get_best_round_id(self, split_id: int, metric: str = 'accuracy') -> int:
        """
        Find the round with the best validation performance for a given metric.

        Args:
            split_id: The split ID to analyze
            metric: The metric to calculate ('accuracy', 'precision', 'recall', 'f1')

        Returns:
            Round ID with the best metric on validation data

        Raises:
            ValueError: If no processed rounds are found
        """
        temp_df = self.generate_metrics_data(split_id, metric, 'validation')
        if temp_df.empty:
            raise ValueError(f"No processed rounds found for split {split_id}")

        temp_df.set_index('round_id', inplace=True)
        return temp_df.metric.idxmax()

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
        df = pd.DataFrame({})

        for data_type in data_types:
            if data_type not in ['train', 'validation', 'test']:
                continue

            temp_df = self.generate_metrics_data(split_id, metric, data_type)
            if not temp_df.empty:
                temp_df.set_index('round_id', inplace=True)
                column_name = f"{data_type} {metric}"
                df[column_name] = temp_df['metric']

        return df
        
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
        prompts = corpus.get('prompts', [])
        reasoning = corpus.get('reasoning', [])
        
        prompt_word_count = sum(len(re.findall(r'\w+', text)) for text in prompts)
        reasoning_word_count = sum(len(re.findall(r'\w+', text)) for text in reasoning)
        total_word_count = prompt_word_count + reasoning_word_count
        
        return {
            'prompt_words': prompt_word_count,
            'reasoning_words': reasoning_word_count,
            'total_words': total_word_count
        }

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for linguistic analysis.
        
        Args:
            text: Raw text string
            
        Returns:
            List of tokens (words)
        """
        # Convert to lowercase and split into words
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        # Split into words and filter out empty strings
        words = [word for word in text.split() if word]
        return words
    
    def _get_word_frequencies(self, text_list: List[str]) -> Counter:
        """
        Calculate word frequencies from a list of texts.
        
        Args:
            text_list: List of text strings
            
        Returns:
            Counter object with word frequencies
        """
        all_words = []
        for text in text_list:
            all_words.extend(self._preprocess_text(text))
        
        return Counter(all_words)
    
    def calculate_zipfs_law(self, split_id: Optional[int] = None) -> Dict[str, Union[float, dict]]:
        """
        Calculate Zipf's law coefficient from all prompts and reasoning combined.
        Zipf's law states that the frequency of a word is inversely proportional to its rank.
        
        This implementation samples 1000 words with replacement 5 times and averages the results
        to handle documents of different lengths.
        
        Args:
            split_id: The split ID to analyze. If None, uses the most recent split.
            
        Returns:
            Dictionary with Zipf coefficient and related statistics
        """
        corpus = self.get_all_prompts_and_reasoning(split_id)
        text_list = corpus.get('prompts', []) + corpus.get('reasoning', [])
        
        if not text_list:
            return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
        
        # Preprocess all text and get a list of all words
        all_words = []
        for text in text_list:
            all_words.extend(self._preprocess_text(text))
            
        if not all_words:
            return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
        
        # Number of runs, sample size and results storage
        num_runs = 5
        sample_size = 1000
        zipf_coefficients = []
        r_squared_values = []
        
        # Run multiple times and average the results
        for _ in range(num_runs):
            # Sample words with replacement
            if len(all_words) == 0:
                continue
                
            # Sample with replacement
            sampled_words = np.random.choice(all_words, size=sample_size, replace=True)
            
            # Count word frequencies in the sample
            word_counts = Counter(sampled_words)
            if not word_counts:
                continue
                
            # Get frequency and rank
            frequencies = []
            ranks = []
            
            sorted_items = word_counts.most_common()
            for rank, (word, count) in enumerate(sorted_items, 1):
                frequencies.append(count)
                ranks.append(rank)
            
            # Convert to numpy arrays and calculate log values
            log_ranks = np.log(ranks)
            log_frequencies = np.log(frequencies)
            
            # Linear regression to find Zipf coefficient
            # Zipf's law: frequency ∝ 1/rank^α where α is the Zipf coefficient
            # In log space: log(frequency) = -α * log(rank) + constant
            slope, intercept = np.polyfit(log_ranks, log_frequencies, 1)
            zipf_coefficient = -slope  # The slope is negative, so we negate it
            
            # Calculate R-squared
            y_pred = slope * log_ranks + intercept
            ss_total = np.sum((log_frequencies - np.mean(log_frequencies))**2)
            ss_residual = np.sum((log_frequencies - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            zipf_coefficients.append(zipf_coefficient)
            r_squared_values.append(r_squared)
        
        # Calculate average values
        if not zipf_coefficients:
            return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
            
        avg_zipf_coefficient = sum(zipf_coefficients) / len(zipf_coefficients)
        avg_r_squared = sum(r_squared_values) / len(r_squared_values)
        
        # Return the average values and some additional information
        return {
            'coefficient': avg_zipf_coefficient,
            'r_squared': avg_r_squared,
            'data': {
                'individual_coefficients': zipf_coefficients,
                'individual_r_squared': r_squared_values
            }
        }
    
    def calculate_herdans_law(self, split_id: Optional[int] = None) -> Dict[str, Union[float, dict]]:
        """
        Calculate Herdan's law (Heaps' law) coefficient from all prompts and reasoning combined.
        Herdan's law describes the relationship between vocabulary size and text length.
        
        This implementation samples 1000 words with replacement 5 times and averages the results
        to handle documents of different lengths.
        
        Args:
            split_id: The split ID to analyze. If None, uses the most recent split.
            
        Returns:
            Dictionary with Herdan coefficient and related statistics
        """
        corpus = self.get_all_prompts_and_reasoning(split_id)
        text_list = corpus.get('prompts', []) + corpus.get('reasoning', [])
        
        if not text_list:
            return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
            
        # Preprocess all text and get a list of all words
        all_words = []
        for text in text_list:
            all_words.extend(self._preprocess_text(text))
            
        if not all_words:
            return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
            
        # Number of runs, sample size and results storage
        num_runs = 5
        sample_size = 1000
        herdan_coefficients = []
        r_squared_values = []
        
        # Run multiple times and average the results
        for _ in range(num_runs):
            if len(all_words) == 0:
                continue
            
            # Sample with replacement
            sampled_words = list(np.random.choice(all_words, size=sample_size, replace=True))
            
            # Calculate vocabulary growth as we read through the sample
            vocab_sizes = []
            text_lengths = []
            
            seen_words = set()
            for i, word in enumerate(sampled_words, 1):
                seen_words.add(word)
                
                # Add a data point for every 10 words or so to get enough data points
                if i % 10 == 0 or i == len(sampled_words):
                    vocab_sizes.append(len(seen_words))
                    text_lengths.append(i)
            
            if len(vocab_sizes) < 3:  # Need at least 3 points for a meaningful regression
                continue
                
            # Convert to numpy arrays and calculate log values
            log_text_lengths = np.log(text_lengths)
            log_vocab_sizes = np.log(vocab_sizes)
            
            # Linear regression to find Herdan coefficient
            # Herdan's law: V = K * N^β where V is vocabulary size, N is text length,
            # K is a constant, and β is the Herdan coefficient
            # In log space: log(V) = β * log(N) + log(K)
            slope, intercept = np.polyfit(log_text_lengths, log_vocab_sizes, 1)
            herdan_coefficient = slope
            
            # Calculate R-squared
            y_pred = slope * log_text_lengths + intercept
            ss_total = np.sum((log_vocab_sizes - np.mean(log_vocab_sizes))**2)
            ss_residual = np.sum((log_vocab_sizes - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            herdan_coefficients.append(herdan_coefficient)
            r_squared_values.append(r_squared)
        
        # Calculate average values
        if not herdan_coefficients:
            return {'coefficient': 0.0, 'r_squared': 0.0, 'data': {}}
            
        avg_herdan_coefficient = sum(herdan_coefficients) / len(herdan_coefficients)
        avg_r_squared = sum(r_squared_values) / len(r_squared_values)
        
        # Return the average values and some additional information
        return {
            'coefficient': avg_herdan_coefficient,
            'r_squared': avg_r_squared,
            'data': {
                'individual_coefficients': herdan_coefficients,
                'individual_r_squared': r_squared_values
            }
        }
